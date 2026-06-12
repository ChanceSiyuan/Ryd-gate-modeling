"""Explicit, lossy Protocol → Sequence discretization (Stage 8).

``sequence_from_protocol`` samples a bound continuous-time protocol onto an
integer-ns ``Sequence`` — the opt-in bridge for exporting research protocols
to hardware-shaped schedules (or Pulser payloads). It inverts the documented
``SequenceProtocol`` lowering exactly (amp = Ω/2, det = −Δ; rad/s ↔ rad/µs)
and refuses anything the (amp, det) waveform pair of a single device channel
cannot represent.

**Lossiness is explicit**: sampling a continuous control at ``dt_ns``
introduces discretization error that is visible at gate-grade fidelities
(~1e-7); the continuous Protocol path stays authoritative for gate metrics.
The sampling parameters are stamped into ``Pulse.metadata``.
"""

from __future__ import annotations

import numpy as np

from ryd_gate.devices import DeviceSpec
from ryd_gate.pulse import Pulse, Waveform
from ryd_gate.sequence import Sequence

__all__ = ["sequence_from_protocol"]

_PHASE_TOL = 1e-12


def sequence_from_protocol(
    system,
    x,
    *,
    device: DeviceSpec | None = None,
    channel_id: str = "rydberg_global",
    dt_ns: int = 1,
) -> Sequence:
    """Sample ``system``'s bound protocol into a single-channel ``Sequence``.

    Parameters
    ----------
    system
        A :class:`~ryd_gate.core.system.RydbergSystem` with a protocol bound
        and a register geometry (``from_lattice`` systems).
    x
        Protocol parameter vector (the same ``x`` you would pass to
        ``simulate``).
    device
        Target :class:`DeviceSpec`; default ``DeviceSpec.virtual_rb87()``.
        The register and level structure must satisfy its constraints.
    channel_id
        Device channel whose (amplitude, detuning) compiler mapping the
        protocol must fit; default ``"rydberg_global"``.
    dt_ns
        Sampling step in integer ns. Smaller is more faithful; the loss is
        the caller's explicit choice.

    Raises
    ------
    ValueError
        ``discretize.protocol_missing`` (no bound protocol),
        ``discretize.channel_not_representable`` (protocol drives channels
        outside the device channel's amp/det pair, e.g. the two-photon
        ``drive_420`` set or per-site addressing),
        ``discretize.phase_not_representable`` (complex coefficients), plus
        the usual device validation codes from ``Sequence``.
    """
    protocol = getattr(system, "protocol", None)
    if protocol is None:
        raise ValueError(
            "discretize.protocol_missing: system has no bound protocol; "
            "construct with protocol=... or call .with_protocol(...)."
        )
    register = getattr(system, "geometry", None)
    if register is None:
        raise ValueError(
            "discretize.register_missing: system has no register geometry; "
            "build it with RydbergSystem.from_lattice."
        )
    if not isinstance(dt_ns, (int, np.integer)) or isinstance(dt_ns, bool) or dt_ns <= 0:
        raise ValueError(f"dt_ns must be a positive integer, got {dt_ns!r}.")

    device = device or DeviceSpec.virtual_rb87()
    spec = system.meta("level_spec")
    channel = device.channels.get(channel_id)
    if channel is None:
        raise ValueError(
            f"discretize.channel_unknown: device {device.name} has no channel "
            f"{channel_id!r}."
        )
    amp_name = channel.amplitude_channels.get(spec.name)
    det_name = channel.detuning_channels.get(spec.name)
    if amp_name is None or det_name is None:
        raise ValueError(
            f"discretize.channel_not_representable: channel {channel_id!r} has no "
            f"compiler mapping for level structure {spec.name!r}."
        )

    params = system.unpack_params(list(x))
    t_gate_s = float(params["t_gate"])
    if t_gate_s <= 0:
        raise ValueError(f"discretize.t_gate: protocol t_gate must be positive, got {t_gate_s}.")
    n_steps = max(1, int(np.ceil(t_gate_s * 1e9 / dt_ns)))
    duration_ns = n_steps * int(dt_ns)

    amp_samples = []
    det_samples = []
    for k in range(n_steps + 1):
        t = min(k * dt_ns * 1e-9, t_gate_s)
        coeffs = protocol.get_drive_coefficients(t, params)
        extras = sorted(
            name for name, value in coeffs.items()
            if name not in (amp_name, det_name) and np.abs(value) > 0.0
        )
        if extras:
            raise ValueError(
                f"discretize.channel_not_representable: protocol drives {extras} "
                f"beyond the ({amp_name}, {det_name}) pair of channel {channel_id!r}; "
                "this protocol stays on the continuous path."
            )
        amp_coeff = complex(coeffs.get(amp_name, 0.0))
        det_coeff = complex(coeffs.get(det_name, 0.0))
        for name, value in (("amplitude", amp_coeff), ("detuning", det_coeff)):
            if abs(value.imag) > _PHASE_TOL * max(1.0, abs(value)):
                raise ValueError(
                    f"discretize.phase_not_representable: complex {name} coefficient "
                    f"{value} at t={t:.3e}s; waveform pairs carry real Ω/Δ only."
                )
        # Invert the SequenceProtocol lowering: amp = Ω/2 rad/s, det = -Δ rad/s.
        amp_samples.append(2.0 * amp_coeff.real / 1e6)
        det_samples.append(-det_coeff.real / 1e6)

    metadata = {
        "discretized_from": type(protocol).__name__,
        "dt_ns": int(dt_ns),
        "t_gate_s": t_gate_s,
        "x": [float(v) for v in x],
    }
    pulse = Pulse(
        amplitude=Waveform.custom(amp_samples, dt_ns=int(dt_ns)),
        detuning=Waveform.custom(det_samples, dt_ns=int(dt_ns)),
        metadata=metadata,
    )
    assert pulse.duration_ns == duration_ns

    sequence = Sequence(register, device, spec)
    sequence.declare_channel("ch0", channel_id)
    sequence.add(pulse, "ch0")
    return sequence
