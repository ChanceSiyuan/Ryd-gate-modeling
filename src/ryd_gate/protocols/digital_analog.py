"""Digital-analog protocol for the 0-1-r Rydberg lattice.

Piecewise-constant schedule of four channels:

- ``drive_R``   — hyperfine→Rydberg Rabi amplitude on |1>↔|r| (per atom, Omega_R)
- ``drive_hf``  — hyperfine Rabi amplitude on |0>↔|1| (Omega_hf)
- ``delta_R``   — Rydberg detuning (Delta_R, sign convention: H contains -Delta_R n^r)
- ``delta_hf``  — hyperfine detuning (Delta_hf)

A schedule is a list of :class:`Segment`\\ s; the protocol holds the schedule
internally so the parameter vector ``x`` passed to ``simulate()`` is empty.

Each ``Segment`` field accepts either a scalar (uniform on all sites) or a
length-``N`` sequence giving a site-dependent profile.

Typical MVP use (single constant segment)::

    protocol = DigitalAnalogProtocol.constant(
        omega_R=2*pi*1e6,
        omega_hf=0,
        delta_R=0,
        delta_hf=0,
        t_gate=1e-6,
    )
    system = RydbergSystem.from_preset("01r", protocol=protocol, N=2)
    result = simulate(system, [], psi0)

Multi-segment echo / IQP-style sequence::

    protocol = DigitalAnalogProtocol([
        Segment(duration=t_pi2, omega_R=Omega),
        Segment(duration=t_int, omega_R=0),
        Segment(duration=t_pi2, omega_R=-Omega),
    ])
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ryd_gate.protocols.base import Protocol

SiteProfile = float | Sequence[float]


def is_scalar_profile(value: SiteProfile) -> bool:
    """True when *value* is a single scalar (not a per-site profile)."""
    if isinstance(value, (int, float, complex)):
        return True
    arr = np.asarray(value)
    return arr.ndim == 0


def as_site_profile(value: SiteProfile, n_sites: int) -> np.ndarray:
    """Broadcast a scalar or length-``n_sites`` profile to shape ``(n_sites,)``."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_sites, float(arr))
    if arr.shape != (n_sites,):
        raise ValueError(
            f"Site profile must be a scalar or length-{n_sites} sequence; "
            f"got shape {arr.shape}."
        )
    return arr


@dataclass(frozen=True)
class Segment:
    """A piecewise-constant slice of the schedule.

    All drive amplitudes and detunings are in rad/s.  Drive amplitudes
    are the full Rabi frequencies (``Omega_R``, ``Omega_hf``), not their
    halves -- the protocol divides by 2 internally to match the convention

        H = (Omega/2) (|a><b| + h.c.).

    Each field may be a scalar (same on every site) or a length-``N`` sequence
    for site-dependent addressing.
    """

    duration: float
    omega_R: SiteProfile = 0.0
    omega_hf: SiteProfile = 0.0
    delta_R: SiteProfile = 0.0
    delta_hf: SiteProfile = 0.0


# (segment field, global channel, per-site channel prefix, half_factor, negate)
_CHANNEL_SPECS = (
    ("omega_R", "drive_R", "drive_R", True, False),
    ("omega_hf", "drive_hf", "drive_hf", True, False),
    ("delta_R", "delta_R", "delta_R", False, True),
    ("delta_hf", "delta_hf", "delta_hf", False, True),
)


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
        omega_R: SiteProfile = 0.0,
        omega_hf: SiteProfile = 0.0,
        delta_R: SiteProfile = 0.0,
        delta_hf: SiteProfile = 0.0,
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
        return {
            "t_gate": self._t_gate,
            "n_sites": system.basis.n_sites,
        }

    @property
    def required_channels(self) -> frozenset[str]:
        return frozenset({"drive_R", "drive_hf", "delta_R", "delta_hf"})

    def drive_channels(self, system) -> frozenset[str]:
        """All drive/detuning channels used by any segment on this lattice."""
        n_sites = system.basis.n_sites
        channels: set[str] = set()
        for field, global_ch, site_prefix, _, _ in _CHANNEL_SPECS:
            for seg in self.segments:
                val = getattr(seg, field)
                if is_scalar_profile(val):
                    channels.add(global_ch)
                else:
                    channels.update(f"{site_prefix}_{i}" for i in range(n_sites))
        return frozenset(channels)

    def _segment_at(self, t: float) -> Segment:
        """Return the segment active at time t (clamped to the last segment after t_gate)."""
        for end, seg in zip(self._end_times, self.segments):
            if t <= end:
                return seg
        return self.segments[-1]

    def _coeffs_for_field(
        self,
        seg: Segment,
        field: str,
        global_ch: str,
        site_prefix: str,
        half_factor: bool,
        negate: bool,
        n_sites: int,
    ) -> dict[str, complex]:
        val = getattr(seg, field)
        profile = as_site_profile(val, n_sites)
        sign = -1.0 if negate else 1.0
        scale = 0.5 if half_factor else 1.0

        if is_scalar_profile(val):
            return {global_ch: complex(sign * scale * profile[0])}

        return {
            f"{site_prefix}_{i}": complex(sign * scale * profile[i])
            for i in range(n_sites)
        }

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        """Return channel coefficients at time *t* for the active segment.

        Sign / factor conventions matching the compiler:

        - ``drive_R`` / ``drive_R_i``  -> Omega_R / 2  (+ Hermitian conjugate)
        - ``drive_hf`` / ``drive_hf_i`` -> Omega_hf / 2 (+ Hermitian conjugate)
        - ``delta_R`` / ``delta_R_i``  -> -Delta_R on Rydberg projector
        - ``delta_hf`` / ``delta_hf_i`` -> -Delta_hf on |1> projector
        """
        seg = self._segment_at(t)
        n_sites = int(params.get("n_sites", 1))
        coeffs: dict[str, complex] = {}
        for spec in _CHANNEL_SPECS:
            coeffs.update(self._coeffs_for_field(seg, *spec, n_sites))
        return coeffs
