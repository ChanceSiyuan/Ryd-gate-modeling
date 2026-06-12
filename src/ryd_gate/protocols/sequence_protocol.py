"""SequenceProtocol: lowers a user ``Sequence`` to kernel drive coefficients.

``SequenceProtocol`` *is* a :class:`~ryd_gate.protocols.base.Protocol` — the
backend compiler treats it like any other protocol and never sees the
``Sequence``. This class is the convergence point of the two control
surfaces: after construction there is only one simulation system, over
Protocols. The physical→Hamiltonian conventions live only here, matching
``SweepProtocol``'s lowering for the same compiler channels:

- amplitude channel carries ``(Omega(t) / 2) * e^{-i*phi}`` in rad/s, where
  ``phi`` is the pulse phase plus the channel's accumulated virtual-Z
  ``post_phase_shift`` history (Pulser semantics); the Hermitian conjugate is
  added by the compiler per ``channel_needs_hermitian_conjugate``,
- detuning channel carries ``-Delta(t)`` in rad/s,
- local-channel pulses emit per-site keys ``f"{base}_{site}"`` resolved by
  ``core/level_structures.py``,
- users write physical Ω and δ in rad/us on waveforms; the unit conversion
  (×1e6), the 1/2 factor, and the phase factor happen here and nowhere else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.core.serialization import raise_for_errors
from ryd_gate.core.system import RydbergSystem
from ryd_gate.protocols.base import Protocol

if TYPE_CHECKING:
    from ryd_gate.core.level_structures import InteractionSpec
    from ryd_gate.sequence import Sequence

__all__ = ["SequenceProtocol", "compile_sequence_to_system", "sequence_uses_phase"]


def sequence_uses_phase(sequence: "Sequence") -> bool:
    """True when any pulse carries a phase or virtual-Z shift (Stage 8)."""
    from ryd_gate.sequence import PulseOp

    return any(
        isinstance(op, PulseOp)
        and (op.pulse.phase_rad != 0.0 or op.pulse.post_phase_shift_rad != 0.0)
        for op in sequence.operations
    )


class SequenceProtocol(Protocol):
    """Kernel protocol generated from a ``Sequence`` (parameter-free)."""

    def __init__(self, sequence: "Sequence") -> None:
        from ryd_gate.sequence import PulseOp

        self.sequence = sequence
        register = sequence.register
        model = sequence.level_structure.name
        schedules = []
        channels: set[str] = set()
        for name, channel in sequence.declared_channels.items():
            amp_channel = channel.amplitude_channels[model]
            det_channel = channel.detuning_channels[model]
            pulse_ops = sorted(
                (op for op in sequence.operations
                 if isinstance(op, PulseOp) and op.channel == name),
                key=lambda op: op.t_start_ns,
            )
            intervals = []
            accumulated_shift = 0.0
            for op in pulse_ops:
                phase_eff = op.pulse.phase_rad + accumulated_shift
                accumulated_shift += op.pulse.post_phase_shift_rad
                sites = (
                    tuple(register.index(atom_id) for atom_id in op.targets)
                    if op.targets is not None
                    else None
                )
                intervals.append((
                    op.t_start_ns,
                    op.t_start_ns + op.pulse.duration_ns,
                    op.pulse,
                    phase_eff,
                    sites,
                ))
                if sites is None:
                    channels.update((amp_channel, det_channel))
                else:
                    channels.update(f"{amp_channel}_{site}" for site in sites)
                    channels.update(f"{det_channel}_{site}" for site in sites)
            schedules.append((name, amp_channel, det_channel, tuple(intervals)))
        self._schedules = tuple(schedules)
        self._channels = frozenset(channels)

    # ── Protocol ABC implementation ─────────────────────────────────────

    @property
    def n_params(self) -> int:
        return 0

    def validate_params(self, x) -> None:
        if x is not None and len(x) != 0:
            raise ValueError("SequenceProtocol takes no parameters (x must be empty).")

    def unpack_params(self, x, system) -> dict:
        self.validate_params(x)
        return {
            "t_gate": self.sequence.duration_ns * 1e-9,
            "n_sites": self.sequence.register.n_atoms,
        }

    @property
    def required_channels(self) -> frozenset[str]:
        # Must override: the base default is the two-photon drive_420 set.
        return self._channels

    def drive_channels(self, system) -> frozenset[str]:
        return self._channels

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        """Coefficients at kernel time *t* (seconds); zero outside pulse intervals."""
        t_ns = t * 1e9
        coefficients: dict[str, complex] = {name: 0.0 for name in self._channels}
        for _, amp_channel, det_channel, intervals in self._schedules:
            for t_start, t_end, pulse, phase_eff, sites in intervals:
                if t_start <= t_ns < t_end:
                    local_t = t_ns - t_start
                    omega_rad_s = pulse.amplitude.value_at_ns(local_t) * 1e6
                    delta_rad_s = pulse.detuning.value_at_ns(local_t) * 1e6
                    amp_coeff = omega_rad_s / 2.0
                    if phase_eff != 0.0:
                        amp_coeff = amp_coeff * np.exp(-1j * phase_eff)
                    if sites is None:
                        coefficients[amp_channel] += amp_coeff
                        coefficients[det_channel] += -delta_rad_s
                    else:
                        for site in sites:
                            coefficients[f"{amp_channel}_{site}"] += amp_coeff
                            coefficients[f"{det_channel}_{site}"] += -delta_rad_s
                    break
        return coefficients

    # ── Visualization hook (drives the generic Protocol.plot) ──────────

    def pulse_traces(self, t: float, params: dict) -> dict[str, float]:
        """Physical Ω/δ in rad/us per declared channel (what the user wrote)."""
        t_ns = t * 1e9
        traces: dict[str, float] = {}
        for name, _, _, intervals in self._schedules:
            amplitude = detuning = 0.0
            for t_start, t_end, pulse, _phase_eff, _sites in intervals:
                if t_start <= t_ns < t_end:
                    amplitude = pulse.amplitude.value_at_ns(t_ns - t_start)
                    detuning = pulse.detuning.value_at_ns(t_ns - t_start)
                    break
            traces[f"{name}.amp"] = amplitude
            traces[f"{name}.det"] = detuning
        return traces


def compile_sequence_to_system(
    sequence: "Sequence",
    interaction: "InteractionSpec | None" = None,
) -> RydbergSystem:
    """Compile a ``Sequence`` into a protocol-bound ``RydbergSystem``.

    Deliberately thin: validation, then the existing kernel entry point. No
    ``compile_hamiltonian_ir`` call here — the backend entry (``simulate``)
    already does that. Phase-modulated pulses lower to complex amplitude
    coefficients (exact backend); per-backend phase gating lives in
    ``simulate_sequence``.
    """
    raise_for_errors(sequence.validate())
    if sequence.duration_ns == 0:
        raise ValueError(
            "sequence.empty: schedule at least one pulse or delay before compiling."
        )
    spec = sequence.level_structure
    return RydbergSystem.from_lattice(
        sequence.register,
        spec,
        interaction=interaction,
        protocol=SequenceProtocol(sequence),
        **spec.physical_kwargs(),
    )
