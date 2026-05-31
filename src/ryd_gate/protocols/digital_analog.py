"""Digital-analog protocol for the 0-1-r Rydberg lattice.

Piecewise-constant schedule of four channels:

- ``drive_R``   — hyperfine→Rydberg Rabi amplitude on |1>↔|r| (per atom, Omega_R)
- ``drive_hf``  — hyperfine Rabi amplitude on |0>↔|1| (Omega_hf)
- ``delta_R``   — Rydberg detuning (Delta_R, sign convention: H contains -Delta_R n^r)
- ``delta_hf``  — hyperfine detuning (Delta_hf)

A schedule is a list of :class:`Segment`\\ s; the protocol holds the schedule
internally so the parameter vector ``x`` passed to ``simulate()`` is empty.

Typical MVP use (single constant segment)::

    protocol = DigitalAnalogProtocol.constant(
        omega_R=2*pi*1e6,
        omega_hf=0,
        delta_R=0,
        delta_hf=0,
        t_gate=1e-6,
    )
    result = simulate(model, protocol, [], psi0)

Multi-segment echo / IQP-style sequence::

    protocol = DigitalAnalogProtocol([
        Segment(duration=t_pi2, omega_R=Omega),
        Segment(duration=t_int, omega_R=0),
        Segment(duration=t_pi2, omega_R=-Omega),
    ])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ryd_gate.protocols.base import Protocol


@dataclass(frozen=True)
class Segment:
    """A piecewise-constant slice of the schedule.

    All drive amplitudes and detunings are in rad/s.  Drive amplitudes
    are the full Rabi frequencies (``Omega_R``, ``Omega_hf``), not their
    halves -- the protocol divides by 2 internally to match the convention

        H = (Omega/2) (|a><b| + h.c.).
    """

    duration: float
    omega_R: float = 0.0
    omega_hf: float = 0.0
    delta_R: float = 0.0
    delta_hf: float = 0.0


class DigitalAnalogProtocol(Protocol):
    """Piecewise-constant 0-1-r drive schedule.

    Parameters
    ----------
    segments : iterable of Segment
        Ordered list of piecewise-constant segments.  Total gate time is
        the sum of the segments' durations.
    n_steps : int
        Number of slices that the sparse backend should use to integrate
        the schedule.  Default 200; for multi-segment schedules pick a
        value large enough that segment boundaries are well-resolved
        (e.g. >= 50 × len(segments)).
    """

    def __init__(self, segments: Iterable[Segment], n_steps: int = 200) -> None:
        self.segments: list[Segment] = list(segments)
        if not self.segments:
            raise ValueError("DigitalAnalogProtocol requires at least one segment.")
        self.n_steps = int(n_steps)
        self._t_gate = float(sum(s.duration for s in self.segments))
        # Precompute cumulative end times for fast segment lookup
        cum = 0.0
        self._end_times: list[float] = []
        for s in self.segments:
            cum += s.duration
            self._end_times.append(cum)

    @classmethod
    def constant(
        cls,
        omega_R: float = 0.0,
        omega_hf: float = 0.0,
        delta_R: float = 0.0,
        delta_hf: float = 0.0,
        t_gate: float = 1.0,
        n_steps: int = 200,
    ) -> "DigitalAnalogProtocol":
        """Single-segment schedule with constant drives over [0, t_gate]."""
        return cls(
            [Segment(duration=t_gate, omega_R=omega_R, omega_hf=omega_hf,
                     delta_R=delta_R, delta_hf=delta_hf)],
            n_steps=n_steps,
        )

    # -- Protocol interface ------------------------------------------------

    @property
    def n_params(self) -> int:
        # Schedule lives on the protocol; x is empty
        return 0

    def validate_params(self, x) -> None:
        if len(x) != 0:
            raise ValueError(
                f"DigitalAnalogProtocol takes no x parameters (schedule is on the "
                f"protocol); got {len(x)}."
            )

    def unpack_params(self, x, system) -> dict:
        return {"t_gate": self._t_gate}

    @property
    def required_channels(self) -> frozenset[str]:
        return frozenset({"drive_R", "drive_hf", "delta_R", "delta_hf"})

    def _segment_at(self, t: float) -> Segment:
        """Return the segment active at time t (clamped to the last segment after t_gate)."""
        for end, seg in zip(self._end_times, self.segments):
            if t <= end:
                return seg
        return self.segments[-1]

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        """Return the four channel coefficients at time t.

        Sign / factor conventions matching the compiler:

        - ``drive_R``  -> Omega_R / 2  (compiler adds Hermitian conjugate)
        - ``drive_hf`` -> Omega_hf / 2 (compiler adds Hermitian conjugate)
        - ``delta_R``  -> -Delta_R     (operator is sum_nr; coefficient absorbs minus sign)
        - ``delta_hf`` -> -Delta_hf    (operator is sum_n1)
        """
        seg = self._segment_at(t)
        return {
            "drive_R":  complex(seg.omega_R) / 2.0,
            "drive_hf": complex(seg.omega_hf) / 2.0,
            "delta_R":  complex(-seg.delta_R),
            "delta_hf": complex(-seg.delta_hf),
        }
