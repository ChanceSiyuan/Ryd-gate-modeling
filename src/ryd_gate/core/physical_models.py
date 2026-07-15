"""Physical-model specialization: interactions, Rb87 parameters, local blocks.

Three pieces of atomic/interaction physics in one module:

- :func:`vdw_couplings` — the standard isotropic VdW pair sum
  V_ij = C6 / R_ij^6 with optional range truncation. Lives in ``core/``
  (not ``lattice/``) because computing interaction strengths is physics;
  the lattice package is reserved for pure geometry.
- Rb87 seven-level static physical parameters per manifold/polarization
  convention — ``rb87_7_mp`` (σ⁻/σ⁺, was ``our``) and ``rb87_7_pm`` (σ⁺/σ⁻,
  was ``lukin``) — covering level energies, decay/branching rates, and VdW
  strengths (no laser Rabi: those belong to the protocol).
- The ``*_level_fields`` resolvers that turn a preset's physical kwargs into
  the typed fields of :class:`~ryd_gate.core.level_structures.LevelStructure`:
  the single-atom static energy matrix (``local_static``), static off-diagonal
  couplings, and the per-channel CG/dipole ratios (``laser_channel_ratios``)
  a protocol multiplies onto its laser coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache as _functools_lru_cache
from typing import Any

import numpy as np

from ryd_gate.core.level_structures import DEFAULT_C6

_RB87_CLOCK_HYPERFINE = 2 * np.pi * 6.835e9


# ── Rydberg-Rydberg interactions ─────────────────────────────────────────────


def vdw_couplings(
    coords_um: np.ndarray,
    C6: float,
    max_range_um: float | None = None,
) -> tuple:
    """Compute all-pairs van der Waals couplings ``V_ij = C6 / R_ij^6``.

    Parameters
    ----------
    coords_um : ndarray, shape (N, 2) or (N, 3)
        Atom positions in microns.
    C6 : float
        Isotropic VdW coefficient in rad/s · μm^6.
    max_range_um : float or None
        If given, omit pairs with separation > max_range_um.

    Returns
    -------
    tuple of (i, j, V_ij)
        Upper-triangular list of pairs with V_ij in rad/s.
    """
    coords_um = np.asarray(coords_um, dtype=float)
    N = len(coords_um)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            r = float(np.linalg.norm(coords_um[i] - coords_um[j]))
            if max_range_um is not None and r > max_range_um:
                continue
            pairs.append((i, j, C6 / r ** 6))
    return tuple(pairs)


def vdw_couplings_from_c6_function(
    coords_um: np.ndarray,
    c6_fn,
    quantization_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    max_range_um: float | None = None,
) -> tuple:
    """All-pairs anisotropic VdW couplings ``V_ij = C6(θ_ij, φ_ij) / R_ij^6``.

    Generalizes :func:`vdw_couplings` to orientation-dependent C6 (e.g. Rydberg
    P states): ``c6_fn(theta, phi)`` returns the pair C6 in rad/s·μm⁶ for a pair
    axis at polar angle ``theta`` from ``quantization_axis`` and azimuth ``phi``
    around it. 2D coordinates are treated as lying in the xy plane, so with the
    default ``(0, 0, 1)`` quantization axis (B perpendicular to the lattice
    plane) every in-plane pair — including 1D chains — has ``theta = pi/2``.

    Parameters
    ----------
    coords_um : ndarray, shape (N, 2) or (N, 3)
        Atom positions in microns.
    c6_fn : callable
        ``(theta, phi) -> C6`` in rad/s · μm^6.
    quantization_axis : tuple of float
        Direction of the quantization (B-field) axis.
    max_range_um : float or None
        If given, omit pairs with separation > max_range_um.

    Returns
    -------
    tuple of (i, j, V_ij)
        Upper-triangular list of pairs with V_ij in rad/s.
    """
    coords_um = np.asarray(coords_um, dtype=float)
    if coords_um.ndim == 2 and coords_um.shape[1] == 2:
        coords_um = np.column_stack([coords_um, np.zeros(len(coords_um))])
    axis = np.asarray(quantization_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    # Orthonormal transverse frame (e1, e2, axis) fixing the azimuth phi.
    seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - axis * (seed @ axis)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    N = len(coords_um)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            d = coords_um[j] - coords_um[i]
            r = float(np.linalg.norm(d))
            if max_range_um is not None and r > max_range_um:
                continue
            theta = float(np.arccos(np.clip((d @ axis) / r, -1.0, 1.0)))
            phi = float(np.arctan2(d @ e2, d @ e1))
            pairs.append((i, j, float(c6_fn(theta, phi)) / r ** 6))
    return tuple(pairs)


# ── Rb87 seven-level physical parameter sets ─────────────────────────────────


@dataclass(frozen=True)
class Rb87SevenLevelParams:
    """Static atom/manifold parameters for the seven-level Rb87 gate model.

    ``manifold`` is ``"mp"`` (σ⁻/σ⁺, was param_set ``"our"``) or ``"pm"``
    (σ⁺/σ⁻, was param_set ``"lukin"``).  This container holds *only* static
    atom physics — no laser 420/1013 Rabi amplitudes (those belong to the
    protocol); the unit-Rabi drive blocks carry only the CG/dipole ratios.
    """

    manifold: str
    ryd_level: int
    Delta: float
    d_mid_ratio: float
    d_ryd_ratio: float
    v_ryd: float
    v_ryd_garb: float
    ryd_zeeman_shift: float
    detuning_sign: int
    mid_state_decay_rate: float
    mid_garb_decay_rate: float
    ryd_state_decay_rate: float
    ryd_RD_rate: float
    ryd_BBR_rate: float
    ryd_garb_decay_rate: float
    ryd_branch: dict
    mid_branch: dict
    t_rise: float
    enable_rydberg_decay: bool
    enable_intermediate_decay: bool
    magnetic_field_G: float
    n_levels: int = 7
    rydberg_indices: tuple[int, ...] = (5, 6)
    n_atoms: int = 2


def _rb87_default_c6(manifold: str) -> float:
    """Default VdW C6 (rad/s · μm⁶) for an rb87 manifold ("mp"/"pm")."""
    if manifold == "pm":  # was param_set="lukin"
        return 2 * np.pi * 450e6 * 3.0**6
    return DEFAULT_C6


def _rb87_physical_params(
    manifold: str,
    *,
    detuning_sign: int,
    enable_rydberg_decay: bool,
    enable_intermediate_decay: bool,
    magnetic_field_G: float = 20.0,
    ryd_level: int | None = None,
    C6_rad_s_um6: float | None = None,
    t_rise: float | None = None,
    Delta_Hz: float | None = None,
) -> Rb87SevenLevelParams:
    """Static seven-level params for ``manifold`` ("mp"=σ⁻/σ⁺, "pm"=σ⁺/σ⁻).

    The per-manifold numbers are defaults; ``ryd_level``, ``C6_rad_s_um6``,
    ``t_rise``, ``Delta_Hz`` (Hz; → ``Delta = detuning_sign·2π·Delta_Hz``)
    override them.  No laser Rabi amplitudes are computed here.
    """
    from ryd_gate.physics import (
        _get_atom,
        _mid_branching_ratios,
        _rydberg_branching_ratios,
        rydberg_zeeman_shift_rad_s,
    )

    atom = _get_atom()
    if manifold == "mp":  # σ⁻(420)/σ⁺(1013); was param_set="our"
        ryd_level = 70 if ryd_level is None else int(ryd_level)
        Delta = detuning_sign * 2 * np.pi * 9.1e9
        d_mid_ratio = atom.getDipoleMatrixElement(5, 0, 0.5, 0.5, 6, 1, 1.5, -0.5, -1) / atom.getDipoleMatrixElement(
            5, 0, 0.5, -0.5, 6, 1, 1.5, -1.5, -1
        )
        d_ryd_ratio = atom.getDipoleMatrixElement(
            6, 1, 1.5, -0.5, ryd_level, 0, 0.5, 0.5, 1
        ) / atom.getDipoleMatrixElement(6, 1, 1.5, -1.5, ryd_level, 0, 0.5, -0.5, 1)
        v_ryd = 2 * np.pi * 874e9 / 3**6
        mid_state_decay_rate = 1 / 110.7e-9
        ryd_state_decay_rate = 1 / 151.55e-6
        ryd_RD_rate = 1 / 410.41e-6
        ryd_branch = _rydberg_branching_ratios(atom, ryd_level, "mp")
        mid_branch = {F: _mid_branching_ratios(atom, F, mF=-1) for F in (1, 2, 3)}
    elif manifold == "pm":  # σ⁺(420)/σ⁻(1013); was param_set="lukin"
        ryd_level = 53 if ryd_level is None else int(ryd_level)
        Delta = detuning_sign * 2 * np.pi * 7.8e9
        d_mid_ratio = atom.getDipoleMatrixElement(5, 0, 0.5, -0.5, 6, 1, 1.5, 0.5, 1) / atom.getDipoleMatrixElement(
            5, 0, 0.5, 0.5, 6, 1, 1.5, 1.5, 1
        )
        d_ryd_ratio = atom.getDipoleMatrixElement(
            6, 1, 1.5, 0.5, ryd_level, 0, 0.5, -0.5, -1
        ) / atom.getDipoleMatrixElement(6, 1, 1.5, 1.5, ryd_level, 0, 0.5, 0.5, -1)
        v_ryd = 2 * np.pi * 450e6
        mid_state_decay_rate = 1 / 110e-9
        ryd_state_decay_rate = 1 / 88e-6
        ryd_RD_rate = 1 / 147.64e-6
        ryd_branch = _rydberg_branching_ratios(atom, ryd_level, "pm")
        mid_branch = {F: _mid_branching_ratios(atom, F, mF=1) for F in (1, 2, 3)}
    else:
        raise ValueError(f"Unknown rb87 manifold '{manifold}' (expected 'mp' or 'pm').")

    # Physical Zeeman splitting of the garbage Rydberg state r_garb (opposite
    # m_j = ±1/2 of r) from the bias field; sets h[6,6] = ryd_zeeman_shift.
    ryd_zeeman_shift = rydberg_zeeman_shift_rad_s(magnetic_field_G, manifold=manifold)

    if Delta_Hz is not None:
        Delta = detuning_sign * 2 * np.pi * float(Delta_Hz)
    if C6_rad_s_um6 is not None:
        v_ryd = float(C6_rad_s_um6) / 3**6  # nearest-pair strength at the nominal 3 μm
    v_ryd_garb = v_ryd
    if t_rise is None:
        t_rise = 20e-9

    ryd_BBR_rate = ryd_state_decay_rate - ryd_RD_rate

    return Rb87SevenLevelParams(
        manifold=manifold,
        ryd_level=ryd_level,
        Delta=Delta,
        d_mid_ratio=d_mid_ratio,
        d_ryd_ratio=d_ryd_ratio,
        v_ryd=v_ryd,
        v_ryd_garb=v_ryd_garb,
        ryd_zeeman_shift=ryd_zeeman_shift,
        detuning_sign=detuning_sign,
        mid_state_decay_rate=mid_state_decay_rate,
        mid_garb_decay_rate=mid_state_decay_rate,
        ryd_state_decay_rate=ryd_state_decay_rate,
        ryd_RD_rate=ryd_RD_rate,
        ryd_BBR_rate=ryd_BBR_rate,
        ryd_garb_decay_rate=ryd_state_decay_rate,
        ryd_branch=ryd_branch,
        mid_branch=mid_branch,
        t_rise=float(t_rise),
        enable_rydberg_decay=enable_rydberg_decay,
        enable_intermediate_decay=enable_intermediate_decay,
        magnetic_field_G=magnetic_field_G,
    )


# ── Single-atom physics → primitive E[ket,bra] terms/ratios ──────────────────


def _offdiag_ratios(
    matrix: np.ndarray,
    rows: list[tuple[int, str]],
    cols: list[tuple[int, str]],
) -> dict[str, complex]:
    """Nonzero ``matrix[i,j]`` entries as an ``E[ket,bra] -> value`` ratio dict.

    ``rows``/``cols`` are ``(matrix_index, level_label)`` pairs, so a row ``ket``
    over a col ``bra`` keys the entry as ``E[ket,bra]``.
    """
    ratios: dict[str, complex] = {}
    for i, ket in rows:
        for j, bra in cols:
            val = complex(matrix[i, j])
            if val != 0:
                ratios[f"E[{ket},{bra}]"] = val
    return ratios


@dataclass(frozen=True, eq=False)
class Analog3Blocks:
    """analog_3 single-atom 3x3 blocks and scalars (shared by exact + TN paths).

    ``h_const``/``h_1013``/``drive_420`` are the analog single-atom matrices (the
    source for both the system's static terms and the TN ``local_blocks``);
    ``static`` is their time-independent sum used by the TN backends, while
    ``drive_420`` is the base operator modulated each step by the protocol's
    (generally complex) ``E[e,g]`` coefficient.
    """

    h_const: np.ndarray
    h_1013: np.ndarray
    drive_420: np.ndarray
    rydberg_index: int
    hermitian: bool
    Delta: float
    rabi_420: float
    rabi_1013: float
    rabi_eff: float
    time_scale: float

    @property
    def static(self) -> np.ndarray:
        """``H_const + H_1013 + H_1013^dag`` — the time-independent local Hamiltonian."""
        return self.h_const + self.h_1013 + self.h_1013.conj().T

    @property
    def drive_420_dag(self) -> np.ndarray:
        return self.drive_420.conj().T


_ANALOG3_MID_DECAY_RATE = 1 / 110.7e-9
_ANALOG3_RYD_DECAY_RATE = 1 / 151.55e-6


def _analog3_blocks(Delta, rabi_420, rabi_1013, mid_decay, ryd_decay, rabi_eff, time_scale) -> Analog3Blocks:
    h_const = np.zeros((3, 3), dtype=np.complex128)
    h_const[1, 1] = Delta - 1j * mid_decay / 2
    h_const[2, 2] = -1j * ryd_decay / 2
    h_1013 = np.zeros((3, 3), dtype=np.complex128)
    h_1013[2, 1] = rabi_1013 / 2
    drive_420 = np.zeros((3, 3), dtype=np.complex128)
    drive_420[1, 0] = rabi_420 / 2
    return Analog3Blocks(
        h_const=h_const, h_1013=h_1013, drive_420=drive_420, rydberg_index=2,
        hermitian=(mid_decay == 0.0 and ryd_decay == 0.0),
        Delta=float(Delta), rabi_420=float(rabi_420), rabi_1013=float(rabi_1013),
        rabi_eff=float(rabi_eff), time_scale=float(time_scale),
    )


def analog_3_local_blocks(
    *,
    Delta_Hz: float | None = None,
    rabi_420_Hz: float | None = None,
    rabi_1013_Hz: float | None = None,
    detuning_sign: int = 1,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
) -> Analog3Blocks:
    """Build the analog_3 single-atom blocks from physical (Hz) knobs.

    Single source of truth for the analog_3 local Hamiltonian: the exact path
    (``_apply_analog_3_lattice_blocks``) and the TN lattice-spec builders both go
    through this, so the matrices stay bit-identical across backends.
    """
    Delta = detuning_sign * 2 * np.pi * (Delta_Hz if Delta_Hz is not None else 9.1e9)
    rabi_420 = 2 * np.pi * (rabi_420_Hz if rabi_420_Hz is not None else 491e6)
    rabi_1013 = 2 * np.pi * (rabi_1013_Hz if rabi_1013_Hz is not None else 491e6)
    rabi_eff = rabi_420 * rabi_1013 / (2 * abs(Delta))
    time_scale = 2 * np.pi / rabi_eff
    mid_decay = _ANALOG3_MID_DECAY_RATE if enable_intermediate_decay else 0.0
    ryd_decay = _ANALOG3_RYD_DECAY_RATE if enable_rydberg_decay else 0.0
    return _analog3_blocks(Delta, rabi_420, rabi_1013, mid_decay, ryd_decay, rabi_eff, time_scale)


def analog_3_local_blocks_from_level_structure(ls) -> Analog3Blocks:
    """Reconstruct analog_3 blocks from a resolved :class:`LevelStructure`.

    Falls back to the default analog_3 constants when *ls* carries no Delta
    (e.g. a bare structural spec in a TN lattice context).
    """
    if ls is None or getattr(ls, "Delta", None) is None:
        return analog_3_local_blocks()
    mid_decay = float(ls.mid_state_decay_rate or 0.0) if ls.enable_intermediate_decay else 0.0
    ryd_decay = float(ls.ryd_state_decay_rate or 0.0) if ls.enable_rydberg_decay else 0.0
    return _analog3_blocks(
        float(ls.Delta), float(ls.rabi_420), float(ls.rabi_1013),
        mid_decay, ryd_decay, float(ls.rabi_eff), float(ls.time_scale),
    )


def analog_3_level_fields(
    *,
    detuning_sign: int = 1,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
    Delta_Hz: float | None = None,
    rabi_420_Hz: float | None = None,
    rabi_1013_Hz: float | None = None,
) -> dict[str, Any]:
    """Resolved analog_3 physical fields for :func:`level_structure`.

    Diagonal g/e/r energies land in ``local_static``; the e-r 1013 leg is a
    static (h.c.-completed) ``E[r,e]`` coupling; the g-e 420 leg is a driveable
    channel whose full Rabi is carried as the ``E[e,g]`` ratio.
    """
    blk = analog_3_local_blocks(
        Delta_Hz=Delta_Hz,
        rabi_420_Hz=rabi_420_Hz,
        rabi_1013_Hz=rabi_1013_Hz,
        detuning_sign=detuning_sign,
        enable_rydberg_decay=enable_rydberg_decay,
        enable_intermediate_decay=enable_intermediate_decay,
    )
    ryd_RD_rate = 1 / 410.41e-6
    c_er = complex(blk.h_1013[2, 1])
    return {
        "local_static": blk.h_const,
        "static_couplings": ((("E[r,e]", c_er),) if c_er != 0 else ()),
        "laser_channel_ratios": {"420": {"E[e,g]": complex(blk.drive_420[1, 0])}},
        "physical_model": "analog_3",
        "Delta": blk.Delta,
        "t_rise": 20e-9,
        "rabi_eff": blk.rabi_eff,
        "time_scale": blk.time_scale,
        "rabi_420": blk.rabi_420,
        "rabi_1013": blk.rabi_1013,
        "ryd_level": 70,
        "ryd_state_decay_rate": _ANALOG3_RYD_DECAY_RATE,
        "ryd_RD_rate": ryd_RD_rate,
        "ryd_BBR_rate": _ANALOG3_RYD_DECAY_RATE - ryd_RD_rate,
        "mid_state_decay_rate": _ANALOG3_MID_DECAY_RATE,
        "ryd_branch": {},
        "mid_branch": {},
        "rydberg_indices": (2,),
        "enable_rydberg_decay": enable_rydberg_decay,
        "enable_intermediate_decay": enable_intermediate_decay,
        "default_c6": DEFAULT_C6,
    }


def rb87_7_level_fields(
    manifold: str,
    *,
    detuning_sign: int = 1,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
    magnetic_field_G: float = 20.0,
    ryd_level: int | None = None,
    C6_rad_s_um6: float | None = None,
    t_rise: float | None = None,
    Delta_Hz: float | None = None,
) -> dict[str, Any]:
    """Resolved rb87 seven-level physical fields for :func:`level_structure`.

    Unit-Rabi, phase-free 420/1013 legs are decomposed into per-channel
    CG/dipole ratios (the off-diagonal entries of the drive matrices, all
    real).  A CZ protocol multiplies its laser coefficient c420(t)/c1013(t)
    onto these; the compiler auto-adds each leg's h.c.  420: |0>,|1> -> |e_F>;
    1013: |e_F> -> |r>,|r_garb>.
    """
    physical = _rb87_physical_params(
        manifold,
        detuning_sign=detuning_sign,
        enable_rydberg_decay=enable_rydberg_decay,
        enable_intermediate_decay=enable_intermediate_decay,
        magnetic_field_G=magnetic_field_G,
        ryd_level=ryd_level,
        C6_rad_s_um6=C6_rad_s_um6,
        t_rise=t_rise,
        Delta_Hz=Delta_Hz,
    )

    h_const = _rb87_local_h_const(
        physical.Delta,
        physical.ryd_zeeman_shift,
        physical.mid_state_decay_rate if enable_intermediate_decay else 0.0,
        physical.ryd_state_decay_rate if enable_rydberg_decay else 0.0,
    )

    h420 = _rb87_local_h420(manifold, 1.0, physical.d_mid_ratio)
    h1013 = _rb87_local_h1013(manifold, 1.0, physical.d_ryd_ratio)
    mid = [(2, "e1"), (3, "e2"), (4, "e3")]
    ratios_420 = _offdiag_ratios(h420, mid, [(1, "1"), (0, "0")])
    ratios_1013 = _offdiag_ratios(h1013, [(5, "r"), (6, "r_garb")], mid)

    default_c6 = (
        float(C6_rad_s_um6) if C6_rad_s_um6 is not None else _rb87_default_c6(manifold)
    )
    return {
        "local_static": h_const,
        "laser_channel_ratios": {"420": ratios_420, "1013": ratios_1013},
        "physical_model": f"rb87_7_{manifold}",
        "Delta": physical.Delta,
        "t_rise": physical.t_rise,
        "ryd_level": physical.ryd_level,
        "ryd_state_decay_rate": physical.ryd_state_decay_rate,
        "ryd_RD_rate": physical.ryd_RD_rate,
        "ryd_BBR_rate": physical.ryd_BBR_rate,
        "ryd_garb_decay_rate": physical.ryd_garb_decay_rate,
        "mid_state_decay_rate": physical.mid_state_decay_rate,
        "ryd_branch": physical.ryd_branch,
        "mid_branch": physical.mid_branch,
        "rydberg_indices": physical.rydberg_indices,
        "magnetic_field_G": physical.magnetic_field_G,
        "ryd_zeeman_shift": physical.ryd_zeeman_shift,
        "enable_rydberg_decay": enable_rydberg_decay,
        "enable_intermediate_decay": enable_intermediate_decay,
        "default_c6": default_c6,
    }


def _rb87_local_h_const(
    Delta: float,
    ryd_zeeman_shift: float,
    middecay: float,
    ryddecay: float,
) -> np.ndarray:
    h = np.zeros((7, 7), dtype=np.complex128)
    h[0, 0] = -_RB87_CLOCK_HYPERFINE
    h[2, 2] = Delta - 2 * np.pi * 51e6 - 1j * middecay / 2
    h[3, 3] = Delta - 1j * middecay / 2
    h[4, 4] = Delta + 2 * np.pi * 87e6 - 1j * middecay / 2
    h[5, 5] = -1j * ryddecay / 2
    h[6, 6] = ryd_zeeman_shift - 1j * ryddecay / 2
    return h


def _rb87_local_h420(
    manifold: str,
    rabi_420: float,
    rabi_420_garbage: float,
) -> np.ndarray:
    from arc.wigner import CG

    h = np.zeros((7, 7), dtype=np.complex128)
    if manifold == "mp":  # σ⁻ 420 drive (was param_set="our")
        for row, F in zip((2, 3, 4), (1, 2, 3)):
            h[row, 1] = (
                rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, F, -1)
                + rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, F, -1)
            ) / 2
    else:  # "pm": σ⁺ 420 drive (was param_set="lukin")
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
    if manifold == "mp":  # σ⁺ 1013 drive (was param_set="our")
        for col, F in zip((2, 3, 4), (1, 2, 3)):
            h[5, col] = (rabi_1013 / 2) * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, F, -1)
            h[6, col] = (rabi_1013_garbage / 2) * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, F, -1)
    else:  # "pm": σ⁻ 1013 drive (was param_set="lukin")
        for col, F in zip((2, 3, 4), (1, 2, 3)):
            h[5, col] = (rabi_1013 / 2) * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, F, 1)
            h[6, col] = (rabi_1013_garbage / 2) * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, F, 1)
    return h


def _rb87_zero_420_couplings(
    manifold: str,
    rabi_420: float,
    rabi_420_garbage: float,
) -> list[complex]:
    """Hamiltonian matrix elements for the off-resonant |0> -> |e_F> 420 leg."""
    from arc.wigner import CG

    if manifold == "mp":  # was param_set="our"
        # The clock-state decomposition is written in the conventional
        # hyperfine order <I m_I, J m_J | F m_F>; swapping to J-first would
        # add a (-1) phase for F=1 and flip the explicit |0> leg.
        cg_ratio_main = CG(3 / 2, 1 / 2, 1 / 2, -1 / 2, 1, 0) / CG(3 / 2, 1 / 2, 1 / 2, -1 / 2, 2, 0)
        cg_ratio_garb = CG(3 / 2, -1 / 2, 1 / 2, 1 / 2, 1, 0) / CG(3 / 2, -1 / 2, 1 / 2, 1 / 2, 2, 0)
        return [
            (
                cg_ratio_main * rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, F, -1)
                + cg_ratio_garb * rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, F, -1)
            )
            / 2
            for F in (1, 2, 3)
        ]

    cg_ratio_main = CG(3 / 2, -1 / 2, 1 / 2, 1 / 2, 1, 0) / CG(3 / 2, -1 / 2, 1 / 2, 1 / 2, 2, 0)
    cg_ratio_garb = CG(3 / 2, 1 / 2, 1 / 2, -1 / 2, 1, 0) / CG(3 / 2, 1 / 2, 1 / 2, -1 / 2, 2, 0)
    return [
        (
            cg_ratio_main * rabi_420 * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, F, 1)
            + cg_ratio_garb * rabi_420_garbage * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, F, 1)
        )
        / 2
        for F in (1, 2, 3)
    ]


# ── Rb87 297 nm single-photon four-level model ───────────────────────────────


def _rb87_297_pair_c6_fn(ryd_level: int):
    """``(theta, phi) -> C6`` (rad/s·μm⁶) for the nP₃/₂ (mⱼ=-3/2, mⱼ=-3/2) pair.

    First-version approximation: this single target-channel C6 feeds the shared
    Rydberg pair-interaction term, i.e. the same strength is applied to *all*
    pair projectors involving ``r`` or ``r_garb`` (r-r, r-r_garb,
    r_garb-r_garb).
    """
    from ryd_gate.physics import arc_pair_c6_rad_s_um6

    def c6_fn(theta: float, phi: float) -> float:
        return arc_pair_c6_rad_s_um6(
            n1=ryd_level, l1=1, j1=1.5, mj1=-1.5, theta=theta, phi=phi
        )

    return c6_fn


@_functools_lru_cache(maxsize=None)
def _rb87_297_level_fields_cached(
    enable_rydberg_decay: bool, magnetic_field_G: float, ryd_level: int
) -> dict[str, Any]:
    from ryd_gate.physics import _get_atom, zeeman_shift_rad_s

    atom = _get_atom()

    # Garbage/target dipole ratio of the two σ⁻ branches (the clock-state
    # 1/sqrt(2) amplitude factors are common to both legs and cancel).
    d_garb_ratio = atom.getDipoleMatrixElement(
        5, 0, 0.5, 0.5, ryd_level, 1, 1.5, -0.5, -1
    ) / atom.getDipoleMatrixElement(5, 0, 0.5, -0.5, ryd_level, 1, 1.5, -1.5, -1)

    # Magnetic garbage-branch detuning for the low-field hyperfine clock state:
    # both clock components share the same |F=2,mF=0> ground-state energy, so
    # only the excited nP3/2 Zeeman splitting separates the two branches.
    garb_detuning = zeeman_shift_rad_s(
        magnetic_field_G, l=1, j=1.5, delta_mj=1.0
    )

    # nP₃/₂ decay from ARC lifetimes: 0 K is radiative-only (RD); 300 K adds BBR
    # (includeLevelsUpTo must exceed n for the finite-temperature calculation).
    ryd_state_decay_rate = 1.0 / atom.getStateLifetime(
        ryd_level, 1, 1.5, temperature=300, includeLevelsUpTo=ryd_level + 27
    )
    ryd_RD_rate = 1.0 / atom.getStateLifetime(ryd_level, 1, 1.5, temperature=0)

    decay = ryd_state_decay_rate if enable_rydberg_decay else 0.0
    h_const = np.zeros((4, 4), dtype=np.complex128)
    h_const[0, 0] = -_RB87_CLOCK_HYPERFINE
    h_const[2, 2] = -1j * decay / 2
    h_const[3, 3] = garb_detuning - 1j * decay / 2

    # Unit-Rabi σ⁻ legs: the protocol's laser coefficient c297(t) = Ω_r(t)
    # multiplies these (Hamiltonian element Ω/2); the compiler adds each h.c.
    ratios_297 = {
        "E[r,1]": complex(0.5),
        "E[r_garb,1]": complex(0.5 * d_garb_ratio),
    }
    return {
        "local_static": h_const,
        "laser_channel_ratios": {"297": ratios_297},
        "physical_model": "rb87_297_clock_4",
        "rydberg_indices": (2, 3),
        "ryd_level": ryd_level,
        "d_garb_ratio": float(d_garb_ratio),
        "garb_zeeman_detuning": float(garb_detuning),
        "magnetic_field_G": float(magnetic_field_G),
        "ryd_state_decay_rate": float(ryd_state_decay_rate),
        "ryd_RD_rate": float(ryd_RD_rate),
        "ryd_BBR_rate": float(ryd_state_decay_rate - ryd_RD_rate),
        "ryd_garb_decay_rate": float(ryd_state_decay_rate),
        "enable_rydberg_decay": bool(enable_rydberg_decay),
    }


def rb87_297_level_fields(
    *,
    enable_rydberg_decay: bool = False,
    magnetic_field_G: float = 20.0,
    ryd_level: int = 53,
) -> dict[str, Any]:
    """Resolved 297 nm single-photon ``("0","1","r","r_garb")`` physical fields.

    ``|1⟩`` is the clock-like ``|F=2, mF=0⟩`` ground state; a σ⁻ 297 nm beam
    drives the target branch ``|5S₁/₂ mⱼ=-1/2⟩ → |nP₃/₂ mⱼ=-3/2⟩`` (``|r⟩``)
    and the garbage branch ``|5S₁/₂ mⱼ=+1/2⟩ → |nP₃/₂ mⱼ=-1/2⟩``
    (``|r_garb⟩``). The logical ``|0⟩`` (``|F=1, mF=0⟩``) is a dark spectator —
    no 297 leg touches it; it carries only the static clock hyperfine energy
    (same ``h[0,0] = -ω_hf`` convention as the seven-level model). The blocks
    are unit-Rabi: the ``E[r,1]``/``E[r_garb,1]`` channel ratios carry only the
    relative branch dipole factor, and the protocol multiplies the physical
    target Rabi onto them.  The pairwise nP₃/₂ interaction defaults to the
    anisotropic ARC C6 via ``pair_c6_fn`` (lazy: evaluated at system build).
    """
    fields = dict(_rb87_297_level_fields_cached(
        bool(enable_rydberg_decay), float(magnetic_field_G), int(ryd_level)
    ))
    fields["pair_c6_fn"] = _rb87_297_pair_c6_fn(int(ryd_level))
    return fields


def _nearest_pair_strength(pairs: tuple) -> float:
    if not pairs:
        return 0.0
    return float(max(abs(strength) for _, _, strength in pairs))


def _reject_unused(unused: dict) -> None:
    if unused:
        names = ", ".join(sorted(unused))
        raise TypeError(f"Unused physical parameter(s): {names}")
