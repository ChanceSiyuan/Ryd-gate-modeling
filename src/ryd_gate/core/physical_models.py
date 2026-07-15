"""Physical-model resolvers: turn a preset's physical kwargs into the private
compiler fields of :class:`~ryd_gate.core.level_structures.LevelStructure`.

Three pieces of atomic/interaction physics:

- ARC pair-state C6: an isotropic S-state coefficient (shared by all
  Rydberg-pair channels, I05) and channel-resolved nP₃/₂ providers (I06).
- Rb87 seven-level static structure per manifold — level offsets, laser-leg
  CG/dipole ratios, decay/branching characterization (no Rabi amplitudes and no
  intermediate detuning: those belong to the protocol, P19/P20).
- The ``*_level_fields`` resolvers consumed by ``level_structure()``.

All Hamiltonians are Hermitian: there are no ``-iγ/2`` decay diagonals (E08).
"""

from __future__ import annotations

from functools import lru_cache as _lru_cache
from typing import Any

import numpy as np

_RB87_CLOCK_HYPERFINE = 2 * np.pi * 6.835e9


# ── ARC pair-state C6 providers ──────────────────────────────────────────────


@_lru_cache(maxsize=None)
def _arc_s_state_c6_rad_s_um6(n: int, mj: float) -> float:
    """Isotropic nS₁/₂ pair C6 (rad/s·μm⁶) for the ``|mj, mj⟩`` channel (on-axis)."""
    from ryd_gate.physics import arc_pair_c6_rad_s_um6

    return arc_pair_c6_rad_s_um6(
        n1=n, l1=0, j1=0.5, mj1=mj, mj2=mj, theta=0.0, phi=0.0, degenerate=False
    )


def _arc_p_state_c6_channel(ryd_level: int, mj1: float, mj2: float):
    """Return a cached ``(theta, phi) -> C6`` provider for an nP₃/₂ pair channel."""
    from ryd_gate.physics import arc_pair_c6_rad_s_um6

    @_lru_cache(maxsize=None)
    def c6(theta: float, phi: float) -> float:
        return arc_pair_c6_rad_s_um6(
            n1=ryd_level, l1=1, j1=1.5, mj1=mj1, mj2=mj2, theta=theta, phi=phi
        )

    return c6


def _rb87_297_channel_c6(ryd_level: int) -> tuple:
    """Channel-resolved nP₃/₂ pair-C6 providers (I06).

    ``r`` is mⱼ=-3/2 and ``r_garb`` is mⱼ=-1/2; each channel gets its own
    orientation-dependent ARC C6 (no copying the target rr value to the others).
    """
    return (
        (("r", "r"), _arc_p_state_c6_channel(ryd_level, -1.5, -1.5)),
        (("r", "r_garb"), _arc_p_state_c6_channel(ryd_level, -1.5, -0.5)),
        (("r_garb", "r_garb"), _arc_p_state_c6_channel(ryd_level, -0.5, -0.5)),
    )


# ── effective 1r / 01r ───────────────────────────────────────────────────────


def effective_1r_level_fields(*, ryd_level: int = 70) -> dict[str, Any]:
    """Physical fields for the effective ``1r`` / ``01r`` presets.

    All level energies come from the protocol (bare effective model), so the
    static diagonal and laser legs are empty; the only physics is the isotropic
    ARC S-state pair interaction at ``ryd_level`` (default 70S). Decay is not a
    fixed characteristic of the effective model — scripts define it explicitly.
    """
    ryd_level = int(ryd_level)
    return {
        "ryd_level": ryd_level,
        "magnetic_field_G": None,
        "quantization_axis": None,
        "decay_rates_per_s": {},
        "branching_ratios": {},
        "_static_diag": {},
        "_laser_legs": (),
        "_pair_c6_isotropic": _arc_s_state_c6_rad_s_um6(ryd_level, -0.5),
        "_pair_c6_channels": None,
    }


# ── Rb87 seven-level (mp / pm) ───────────────────────────────────────────────


def _rb87_dipole_ratios(atom, manifold: str, ryd_level: int) -> tuple[float, float]:
    """``(d_mid_ratio, d_ryd_ratio)`` garbage/target dipole ratios for the manifold."""
    if manifold == "mp":  # σ⁻(420)/σ⁺(1013)
        d_mid = atom.getDipoleMatrixElement(5, 0, 0.5, 0.5, 6, 1, 1.5, -0.5, -1) / \
            atom.getDipoleMatrixElement(5, 0, 0.5, -0.5, 6, 1, 1.5, -1.5, -1)
        d_ryd = atom.getDipoleMatrixElement(6, 1, 1.5, -0.5, ryd_level, 0, 0.5, 0.5, 1) / \
            atom.getDipoleMatrixElement(6, 1, 1.5, -1.5, ryd_level, 0, 0.5, -0.5, 1)
    else:  # "pm": σ⁺(420)/σ⁻(1013)
        d_mid = atom.getDipoleMatrixElement(5, 0, 0.5, -0.5, 6, 1, 1.5, 0.5, 1) / \
            atom.getDipoleMatrixElement(5, 0, 0.5, 0.5, 6, 1, 1.5, 1.5, 1)
        d_ryd = atom.getDipoleMatrixElement(6, 1, 1.5, 0.5, ryd_level, 0, 0.5, -0.5, -1) / \
            atom.getDipoleMatrixElement(6, 1, 1.5, 1.5, ryd_level, 0, 0.5, 0.5, -1)
    return float(d_mid), float(d_ryd)


def _rb87_decay_data(manifold: str) -> dict[str, float]:
    """Hardcoded per-manifold Rydberg/intermediate decay rates (s⁻¹)."""
    if manifold == "mp":
        ryd_total = 1 / 151.55e-6
        ryd_rd = 1 / 410.41e-6
        mid_total = 1 / 110.7e-9
    else:  # pm
        ryd_total = 1 / 88e-6
        ryd_rd = 1 / 147.64e-6
        mid_total = 1 / 110e-9
    return {
        "ryd_total": ryd_total,
        "ryd_rd": ryd_rd,
        "ryd_bbr": ryd_total - ryd_rd,
        "mid_total": mid_total,
    }


def rb87_7_level_fields(
    manifold: str,
    *,
    ryd_level: int | None = None,
    magnetic_field_G: float = 20.0,
) -> dict[str, Any]:
    """Physical fields for ``rb87_7_mp`` / ``rb87_7_pm``.

    The static diagonal carries the atomic structure only (clock hyperfine on
    |0>, 6P₃/₂ hyperfine offsets on e1/e2/e3, Zeeman splitting on r_garb) — no
    intermediate detuning (P19) and no decay diagonals (E08). Laser legs are the
    per-transition CG/dipole ratios (already ×½) the 420/1013 lasers expand onto.
    """
    from ryd_gate.physics import (
        _get_atom,
        _mid_branching_ratios,
        _rydberg_branching_ratios,
        _rydberg_zeeman_shift_rad_s,
    )

    if manifold == "mp":
        ryd_level = 70 if ryd_level is None else int(ryd_level)
        mj_r = -0.5
    elif manifold == "pm":
        ryd_level = 53 if ryd_level is None else int(ryd_level)
        mj_r = 0.5
    else:
        raise ValueError(f"Unknown rb87 manifold {manifold!r} (expected 'mp' or 'pm').")

    atom = _get_atom()
    d_mid_ratio, d_ryd_ratio = _rb87_dipole_ratios(atom, manifold, ryd_level)
    ryd_zeeman = _rydberg_zeeman_shift_rad_s(magnetic_field_G, manifold=manifold)

    static_diag = {
        "0": -_RB87_CLOCK_HYPERFINE,
        "e1": -2 * np.pi * 51e6,
        "e3": 2 * np.pi * 87e6,
        "r_garb": ryd_zeeman,
    }

    h420 = _rb87_local_h420(manifold, 1.0, d_mid_ratio)
    h1013 = _rb87_local_h1013(manifold, 1.0, d_ryd_ratio)
    mid = [(2, "e1"), (3, "e2"), (4, "e3")]
    legs = tuple(
        _leg("420", ch, val) for ch, val in _offdiag_ratios(h420, mid, [(1, "1"), (0, "0")]).items()
    ) + tuple(
        _leg("1013", ch, val) for ch, val in _offdiag_ratios(h1013, [(5, "r"), (6, "r_garb")], mid).items()
    )

    decay = _rb87_decay_data(manifold)
    ryd_decay = {"total": decay["ryd_total"], "radiative": decay["ryd_rd"], "blackbody": decay["ryd_bbr"]}
    mid_mF = -1 if manifold == "mp" else 1
    mid_branch = {F: _mid_branching_ratios(atom, F, mF=mid_mF) for F in (1, 2, 3)}
    decay_rates = {
        "r": ryd_decay,
        "r_garb": ryd_decay,
        "e1": {"total": decay["mid_total"]},
        "e2": {"total": decay["mid_total"]},
        "e3": {"total": decay["mid_total"]},
    }
    branching = {
        "r": _rydberg_branching_ratios(atom, ryd_level, manifold),
        "e1": mid_branch[1],
        "e2": mid_branch[2],
        "e3": mid_branch[3],
    }

    return {
        "ryd_level": ryd_level,
        "magnetic_field_G": float(magnetic_field_G),
        "quantization_axis": None,
        "decay_rates_per_s": decay_rates,
        "branching_ratios": branching,
        "_static_diag": static_diag,
        "_laser_legs": legs,
        "_pair_c6_isotropic": _arc_s_state_c6_rad_s_um6(ryd_level, mj_r),
        "_pair_c6_channels": None,
    }


# ── Rb87 297 nm single-photon four-level ─────────────────────────────────────


@_lru_cache(maxsize=None)
def _rb87_297_static_and_legs(magnetic_field_G: float, ryd_level: int) -> dict[str, Any]:
    from ryd_gate.physics import _get_atom, zeeman_shift_rad_s

    atom = _get_atom()
    d_garb_ratio = atom.getDipoleMatrixElement(5, 0, 0.5, 0.5, ryd_level, 1, 1.5, -0.5, -1) / \
        atom.getDipoleMatrixElement(5, 0, 0.5, -0.5, ryd_level, 1, 1.5, -1.5, -1)
    garb_detuning = zeeman_shift_rad_s(magnetic_field_G, l=1, j=1.5, delta_mj=1.0)
    ryd_total = 1.0 / atom.getStateLifetime(
        ryd_level, 1, 1.5, temperature=300, includeLevelsUpTo=ryd_level + 27
    )
    ryd_rd = 1.0 / atom.getStateLifetime(ryd_level, 1, 1.5, temperature=0)
    return {
        "d_garb_ratio": float(d_garb_ratio),
        "garb_detuning": float(garb_detuning),
        "ryd_total": float(ryd_total),
        "ryd_rd": float(ryd_rd),
    }


def rb87_297_level_fields(
    *,
    ryd_level: int = 53,
    magnetic_field_G: float = 20.0,
    quantization_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> dict[str, Any]:
    """Physical fields for ``rb87_297_clock_4`` (σ⁻ 297 nm single-photon).

    |1⟩ is the clock ground state; a σ⁻ beam drives the target (|r⟩) and garbage
    (|r_garb⟩) Zeeman branches, separated by the r_garb Zeeman detuning. |0⟩ is a
    dark spectator carrying only the clock hyperfine energy. The pair interaction
    is channel-resolved via ``quantization_axis`` (I06).
    """
    ryd_level = int(ryd_level)
    axis = _normalize_axis(quantization_axis)
    data = _rb87_297_static_and_legs(float(magnetic_field_G), ryd_level)

    static_diag = {"0": -_RB87_CLOCK_HYPERFINE, "r_garb": data["garb_detuning"]}
    legs = (
        _leg("297", "E[r,1]", 0.5),
        _leg("297", "E[r_garb,1]", 0.5 * data["d_garb_ratio"]),
    )
    ryd_decay = {
        "total": data["ryd_total"],
        "radiative": data["ryd_rd"],
        "blackbody": data["ryd_total"] - data["ryd_rd"],
    }
    return {
        "ryd_level": ryd_level,
        "magnetic_field_G": float(magnetic_field_G),
        "quantization_axis": axis,
        "decay_rates_per_s": {"r": ryd_decay, "r_garb": ryd_decay},
        "branching_ratios": {},
        "_static_diag": static_diag,
        "_laser_legs": legs,
        "_pair_c6_isotropic": None,
        "_pair_c6_channels": _rb87_297_channel_c6(ryd_level),
    }


def _normalize_axis(axis) -> tuple[float, float, float]:
    a = np.asarray(axis, dtype=float)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"quantization_axis must be a finite length-3 vector; got {axis!r}.")
    norm = float(np.linalg.norm(a))
    if norm == 0.0:
        raise ValueError("quantization_axis must be non-zero.")
    return tuple(float(x) for x in a / norm)


# ── single-atom CG/dipole leg matrices (shared with the old model) ───────────


def _leg(group: str, channel: str, factor: complex):
    from ryd_gate.core.level_structures import _LaserLeg

    return _LaserLeg(group=group, channel=channel, factor=complex(factor))


def _offdiag_ratios(matrix, rows, cols) -> dict[str, complex]:
    ratios: dict[str, complex] = {}
    for i, ket in rows:
        for j, bra in cols:
            val = complex(matrix[i, j])
            if val != 0:
                ratios[f"E[{ket},{bra}]"] = val
    return ratios


def _rb87_local_h420(manifold: str, rabi_420: float, rabi_420_garbage: float) -> np.ndarray:
    from arc.wigner import CG

    h = np.zeros((7, 7), dtype=np.complex128)
    if manifold == "mp":  # σ⁻ 420 drive
        for row, F in zip((2, 3, 4), (1, 2, 3)):
            h[row, 1] = (
                rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, F, -1)
                + rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, F, -1)
            ) / 2
    else:  # σ⁺ 420 drive
        for row, F in zip((2, 3, 4), (1, 2, 3)):
            h[row, 1] = (
                rabi_420 * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, F, 1)
                + rabi_420_garbage * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, F, 1)
            ) / 2
    for row, g_i in zip((2, 3, 4), _rb87_zero_420_couplings(manifold, rabi_420, rabi_420_garbage)):
        h[row, 0] = g_i
    return h


def _rb87_local_h1013(manifold: str, rabi_1013: float, rabi_1013_garbage: float) -> np.ndarray:
    from arc.wigner import CG

    h = np.zeros((7, 7), dtype=np.complex128)
    if manifold == "mp":  # σ⁺ 1013 drive
        for col, F in zip((2, 3, 4), (1, 2, 3)):
            h[5, col] = (rabi_1013 / 2) * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, F, -1)
            h[6, col] = (rabi_1013_garbage / 2) * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, F, -1)
    else:  # σ⁻ 1013 drive
        for col, F in zip((2, 3, 4), (1, 2, 3)):
            h[5, col] = (rabi_1013 / 2) * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, F, 1)
            h[6, col] = (rabi_1013_garbage / 2) * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, F, 1)
    return h


def _rb87_zero_420_couplings(manifold: str, rabi_420: float, rabi_420_garbage: float) -> list[complex]:
    """Matrix elements for the off-resonant |0> -> |e_F> 420 leg."""
    from arc.wigner import CG

    if manifold == "mp":
        cg_ratio_main = CG(3 / 2, 1 / 2, 1 / 2, -1 / 2, 1, 0) / CG(3 / 2, 1 / 2, 1 / 2, -1 / 2, 2, 0)
        cg_ratio_garb = CG(3 / 2, -1 / 2, 1 / 2, 1 / 2, 1, 0) / CG(3 / 2, -1 / 2, 1 / 2, 1 / 2, 2, 0)
        return [
            (
                cg_ratio_main * rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, F, -1)
                + cg_ratio_garb * rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, F, -1)
            ) / 2
            for F in (1, 2, 3)
        ]
    cg_ratio_main = CG(3 / 2, -1 / 2, 1 / 2, 1 / 2, 1, 0) / CG(3 / 2, -1 / 2, 1 / 2, 1 / 2, 2, 0)
    cg_ratio_garb = CG(3 / 2, 1 / 2, 1 / 2, -1 / 2, 1, 0) / CG(3 / 2, 1 / 2, 1 / 2, -1 / 2, 2, 0)
    return [
        (
            cg_ratio_main * rabi_420 * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, F, 1)
            + cg_ratio_garb * rabi_420_garbage * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, F, 1)
        ) / 2
        for F in (1, 2, 3)
    ]
