"""SequenceProtocol: lowers a user ``Sequence`` to kernel drive coefficients.

``SequenceProtocol`` *is* a :class:`~ryd_gate.protocols.base.Protocol` — the
backend compiler treats it like any other protocol and never sees the
``Sequence``. The physical→Hamiltonian conventions live only here, matching
``SweepProtocol``'s lowering for the same compiler channels:

- amplitude channel carries ``Omega(t) / 2`` in rad/s,
- detuning channel carries ``-Delta(t)`` in rad/s,
- users write physical Ω and δ in rad/us on waveforms; the unit conversion
  (×1e6) and the 1/2 factor happen here and nowhere else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryd_gate.core.system import RydbergSystem
from ryd_gate.core.validation import raise_for_errors
from ryd_gate.protocols.base import Protocol

if TYPE_CHECKING:
    from ryd_gate.core.level_structures import InteractionSpec
    from ryd_gate.sequence import Sequence

__all__ = ["SequenceProtocol", "compile_sequence_to_system"]


class SequenceProtocol(Protocol):
    """Kernel protocol generated from a ``Sequence`` (parameter-free)."""

    def __init__(self, sequence: "Sequence") -> None:
        from ryd_gate.sequence import PulseOp

        self.sequence = sequence
        model = sequence.level_structure.name
        schedules = []
        channels: set[str] = set()
        for name, channel in sequence.declared_channels.items():
            amp_channel = channel.amplitude_channels[model]
            det_channel = channel.detuning_channels[model]
            intervals = sorted(
                (
                    (op.t_start_ns, op.t_start_ns + op.pulse.duration_ns, op.pulse)
                    for op in sequence.operations
                    if isinstance(op, PulseOp) and op.channel == name
                ),
                key=lambda interval: interval[0],
            )
            schedules.append((name, amp_channel, det_channel, intervals))
            channels.update((amp_channel, det_channel))
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
            for t_start, t_end, pulse in intervals:
                if t_start <= t_ns < t_end:
                    local_t = t_ns - t_start
                    omega_rad_s = pulse.amplitude.value_at_ns(local_t) * 1e6
                    delta_rad_s = pulse.detuning.value_at_ns(local_t) * 1e6
                    coefficients[amp_channel] = omega_rad_s / 2.0
                    coefficients[det_channel] = -delta_rad_s
                    break
        return coefficients

    # ── Visualization hook (drives the generic Protocol.plot) ──────────

    def pulse_traces(self, t: float, params: dict) -> dict[str, float]:
        """Physical Ω/δ in rad/us per declared channel (what the user wrote)."""
        t_ns = t * 1e9
        traces: dict[str, float] = {}
        for name, _, _, intervals in self._schedules:
            amplitude = detuning = 0.0
            for t_start, t_end, pulse in intervals:
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

    Deliberately thin: validation, the Stage 2 phase gate, then the existing
    kernel entry point. No ``compile_hamiltonian_ir`` call here — the backend
    entry (``simulate``) already does that.
    """
    from ryd_gate.sequence import PulseOp

    raise_for_errors(sequence.validate())
    if sequence.duration_ns == 0:
        raise ValueError(
            "sequence.empty: schedule at least one pulse or delay before compiling."
        )
    for op in sequence.operations:
        if isinstance(op, PulseOp) and (
            op.pulse.phase_rad != 0.0 or op.pulse.post_phase_shift_rad != 0.0
        ):
            raise NotImplementedError(
                "sequence.phase_not_stage2: phase-modulated pulses stay on the "
                "gate-protocol path (TOProtocol/ARProtocol) until sequence pair "
                "lowering lands."
            )
    spec = sequence.level_structure
    return RydbergSystem.from_lattice(
        sequence.register,
        spec,
        interaction=interaction,
        protocol=SequenceProtocol(sequence),
        **spec.physical_kwargs(),
    )
