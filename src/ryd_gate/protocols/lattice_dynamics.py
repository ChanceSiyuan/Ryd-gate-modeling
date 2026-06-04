"""Lattice dynamics protocols and TFIM-to-Rydberg control mapping.

The two-level Rydberg lattice Hamiltonian used by :class:`RydbergSystem` is

    H = (Omega/2) sum_i sigma^x_i - sum_i Delta_i n_i
        + sum_{i<j} V_ij n_i n_j,

with ``n_i = (1 + sigma^z_i) / 2``.  Therefore the equivalent Ising
parameters are

    J_ij = V_ij / 4,
    h_x = Omega / 2,
    h_z,i = -Delta_i / 2 + (1/4) sum_j V_ij.

This module keeps that mapping in one place so exact, MPS, TTN, 2D-TN, and
NQS paths can consume the same protocol definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ryd_gate.protocols.base import Protocol


@dataclass(frozen=True)
class TFIMRydbergControls:
    """Rydberg controls that realize a target TFIM field profile."""

    Omega: float
    Delta: float
    pin_deltas: dict[int, float]
    interaction_shifts: np.ndarray
    hz_profile: np.ndarray

    @property
    def delta_profile(self) -> np.ndarray:
        """Per-site total detuning ``Delta_i = Delta + pin_i``."""
        profile = np.full_like(self.interaction_shifts, self.Delta, dtype=float)
        for idx, value in self.pin_deltas.items():
            profile[idx] += value
        return profile


def tfim_to_rydberg_controls(
    system,
    *,
    hx: float,
    hz: float | Iterable[float] = 0.0,
    compensate_site_fields: bool = True,
    pin_cutoff: float = 1e-14,
) -> TFIMRydbergControls:
    """Map target TFIM fields to Rydberg ``Omega``, ``Delta``, and pins.

    Parameters
    ----------
    system
        A :class:`RydbergSystem` or TN protocol context exposing ``N`` and
        interaction metadata.
    hx
        Target transverse field coefficient in ``h_x sum sigma_x``.
    hz
        Target longitudinal field. A scalar gives a uniform field; a length-N
        array gives a site-dependent field.
    compensate_site_fields
        If True, use static per-site detuning pins to cancel open-boundary
        coordination shifts and realize the requested ``hz`` profile.
        If False, use a global detuning chosen from the mean interaction shift.
    pin_cutoff
        Pins smaller than this absolute value are omitted from the dict.
    """
    n_sites = _n_sites(system)
    shifts = interaction_longitudinal_shifts(n_sites, _interaction_pairs(system))
    hz_profile = _as_site_profile(hz, n_sites, name="hz")

    Omega = 2.0 * float(hx)
    if compensate_site_fields:
        delta_profile = 2.0 * (shifts - hz_profile)
        Delta = float(np.mean(delta_profile))
        pin_deltas = {
            int(i): float(delta_profile[i] - Delta)
            for i in range(n_sites)
            if abs(delta_profile[i] - Delta) > pin_cutoff
        }
    else:
        Delta = float(2.0 * (np.mean(shifts) - np.mean(hz_profile)))
        pin_deltas = {}

    return TFIMRydbergControls(
        Omega=Omega,
        Delta=Delta,
        pin_deltas=pin_deltas,
        interaction_shifts=shifts,
        hz_profile=hz_profile,
    )


def interaction_longitudinal_shifts(
    n_sites: int,
    interaction_pairs: Iterable[tuple[int, int, float]],
) -> np.ndarray:
    """Return ``(1/4) sum_j V_ij`` for each site."""
    shifts = np.zeros(n_sites, dtype=float)
    for i, j, strength in interaction_pairs:
        value = 0.25 * float(strength)
        shifts[int(i)] += value
        shifts[int(j)] += value
    return shifts


class TFIMQuenchProtocol(Protocol):
    """Constant-field post-quench dynamics for a two-level Rydberg lattice."""

    def __init__(
        self,
        *,
        hx: float,
        hz: float | Iterable[float] = 0.0,
        t_gate: float,
        compensate_site_fields: bool = True,
    ) -> None:
        self.hx = float(hx)
        self.hz = hz
        self.t_gate = float(t_gate)
        self.compensate_site_fields = compensate_site_fields

    @property
    def n_params(self) -> int:
        return 0

    def validate_params(self, x: list[float]) -> None:
        if len(x) != 0:
            raise ValueError(f"TFIMQuenchProtocol expects no parameters, got {len(x)}.")

    def unpack_params(self, x: list[float], system) -> dict:
        self.validate_params(x)
        controls = tfim_to_rydberg_controls(
            system,
            hx=self.hx,
            hz=self.hz,
            compensate_site_fields=self.compensate_site_fields,
        )
        return {
            "t_gate": self.t_gate,
            "Omega": controls.Omega,
            "Delta": controls.Delta,
            "pin_deltas": controls.pin_deltas,
            "scatter_rates": {},
            "static_overlays": [],
            "_system_type": "lattice",
            "tfim": {
                "protocol": "quench",
                "hx": self.hx,
                "hz": np.asarray(controls.hz_profile, dtype=float),
                "interaction_shifts": controls.interaction_shifts,
            },
        }

    @property
    def required_channels(self) -> frozenset[str]:
        return frozenset({"global_X", "global_n"})

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        return {
            "global_X": 0.5 * params["Omega"],
            "global_n": -params["Delta"],
        }


class TFIMAnnealProtocol(Protocol):
    """Piecewise annealing schedule for square-lattice TFIM benchmarks.

    ``h_x`` rises from ``hx_initial`` to ``hx_peak`` during ``t_rise``, stays
    there through ``t_sweep``, and falls to ``hx_final`` during ``t_fall``.
    ``h_z`` is held at ``hz_initial`` during the rise, linearly swept to
    ``hz_final`` during ``t_sweep``, and then held fixed during the fall.
    """

    def __init__(
        self,
        *,
        hx_peak: float,
        hz_initial: float,
        hz_final: float,
        t_rise: float,
        t_sweep: float,
        t_fall: float,
        hx_initial: float = 0.0,
        hx_final: float = 0.0,
        compensate_site_fields: bool = True,
    ) -> None:
        self.hx_peak = float(hx_peak)
        self.hx_initial = float(hx_initial)
        self.hx_final = float(hx_final)
        self.hz_initial = float(hz_initial)
        self.hz_final = float(hz_final)
        self.t_rise = float(t_rise)
        self.t_sweep = float(t_sweep)
        self.t_fall = float(t_fall)
        self.compensate_site_fields = compensate_site_fields

    @property
    def n_params(self) -> int:
        return 0

    @property
    def t_gate(self) -> float:
        return self.t_rise + self.t_sweep + self.t_fall

    def validate_params(self, x: list[float]) -> None:
        if len(x) != 0:
            raise ValueError(f"TFIMAnnealProtocol expects no parameters, got {len(x)}.")
        if self.t_gate <= 0:
            raise ValueError("Anneal duration must be positive.")

    def unpack_params(self, x: list[float], system) -> dict:
        self.validate_params(x)
        n_sites = _n_sites(system)
        shifts = interaction_longitudinal_shifts(n_sites, _interaction_pairs(system))
        if self.compensate_site_fields:
            pin_profile = 2.0 * (shifts - np.mean(shifts))
            pin_deltas = {
                int(i): float(value)
                for i, value in enumerate(pin_profile)
                if abs(value) > 1e-14
            }
            shift_reference = float(np.mean(shifts))
        else:
            pin_deltas = {}
            shift_reference = float(np.mean(shifts))

        return {
            "t_gate": self.t_gate,
            "t_rise": self.t_rise,
            "t_sweep": self.t_sweep,
            "t_fall": self.t_fall,
            "shift_reference": shift_reference,
            "pin_deltas": pin_deltas,
            "scatter_rates": {},
            "static_overlays": [],
            "_system_type": "lattice",
            "tfim": {
                "protocol": "anneal",
                "hx_peak": self.hx_peak,
                "hz_initial": self.hz_initial,
                "hz_final": self.hz_final,
                "interaction_shifts": shifts,
            },
        }

    @property
    def required_channels(self) -> frozenset[str]:
        return frozenset({"global_X", "global_n"})

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        hx_t = self.hx_at(t)
        hz_t = self.hz_at(t)
        Delta_t = 2.0 * (params["shift_reference"] - hz_t)
        return {
            "global_X": hx_t,
            "global_n": -Delta_t,
        }

    def hx_at(self, t: float) -> float:
        t = float(np.clip(t, 0.0, self.t_gate))
        if self.t_rise > 0 and t < self.t_rise:
            return _lerp(self.hx_initial, self.hx_peak, t / self.t_rise)
        sweep_end = self.t_rise + self.t_sweep
        if t < sweep_end:
            return self.hx_peak
        if self.t_fall > 0:
            return _lerp(self.hx_peak, self.hx_final, (t - sweep_end) / self.t_fall)
        return self.hx_final

    def hz_at(self, t: float) -> float:
        t = float(np.clip(t, 0.0, self.t_gate))
        if t < self.t_rise or self.t_sweep <= 0:
            return self.hz_initial
        sweep_end = self.t_rise + self.t_sweep
        if t < sweep_end:
            return _lerp(self.hz_initial, self.hz_final, (t - self.t_rise) / self.t_sweep)
        return self.hz_final


def _lerp(start: float, end: float, frac: float) -> float:
    frac = float(np.clip(frac, 0.0, 1.0))
    return (1.0 - frac) * start + frac * end


def _n_sites(system) -> int:
    if hasattr(system, "N"):
        return int(system.N)
    if hasattr(system, "basis"):
        return int(system.basis.n_sites)
    n_sites = system.meta("n_sites", None) if hasattr(system, "meta") else None
    if n_sites is None:
        raise ValueError("Cannot infer number of lattice sites from protocol context.")
    return int(n_sites)


def _interaction_pairs(system) -> tuple[tuple[int, int, float], ...]:
    if hasattr(system, "meta"):
        pairs = system.meta("interaction_pairs", None)
        if pairs is not None:
            return tuple((int(i), int(j), float(v)) for i, j, v in pairs)
    spec = getattr(system, "_spec", None)
    if spec is not None:
        return tuple(
            (int(i), int(j), float(spec.V_nn) * float(v_rel))
            for i, j, v_rel in spec.vdw_pairs
        )
    raise ValueError(
        "TFIM lattice protocols require interaction pair metadata. "
        "Use RydbergSystem.from_lattice(...) or create_tn_lattice_spec(...)."
    )


def _as_site_profile(value: float | Iterable[float], n_sites: int, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_sites, float(arr), dtype=float)
    if arr.shape != (n_sites,):
        raise ValueError(f"{name} must be a scalar or length-{n_sites}; got shape {arr.shape}.")
    return arr.astype(float, copy=False)
