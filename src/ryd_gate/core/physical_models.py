"""Physical-model specialization: interactions, Rb87 parameters, local blocks.

Three pieces of atomic/interaction physics in one module:

- :func:`vdw_couplings` — the standard isotropic VdW pair sum
  V_ij = C6 / R_ij^6 with optional range truncation. Lives in ``core/``
  (not ``lattice/``) because computing interaction strengths is physics;
  the lattice package is reserved for pure geometry.
- Rb87 seven-level static physical parameters per manifold/polarization
  convention — ``rb87_7_mp`` (σ⁻/σ⁺, was ``our``) and ``rb87_7_pm`` (σ⁺/σ⁻,
  was ``lukin``) — covering level energies, decay/branching rates, and VdW
  strengths (no laser Rabi: those belong to the protocol), and the helper
  that flattens them into a system metadata dict.
- Single-atom physics for ``analog_3`` and the Rb87 seven-level models, lowered
  to primitive ``E[ket,bra]`` form: static diagonal energies (and the analog
  static e-r coupling) become ``StaticHamiltonianTerm`` s, and the off-diagonal
  laser legs become per-channel CG/dipole ratios (``laser_channel_ratios``
  metadata) that a protocol multiplies onto its laser coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ryd_gate.core.level_structures import DEFAULT_C6, level_structure
from ryd_gate.core.operators import StaticHamiltonianTerm

if TYPE_CHECKING:
    from ryd_gate.core.system import RydbergSystem


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
    from arc import Rubidium87

    from ryd_gate.physics import (
        _mid_branching_ratios,
        _rydberg_branching_ratios,
        rydberg_zeeman_shift_rad_s,
    )

    atom = Rubidium87()
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


def _metadata_from_rb87_params(system: Rb87SevenLevelParams) -> dict[str, Any]:
    # The laser Rabi scale is not a system property — the unit-normalized blocks
    # carry no Rabi, and the CZ protocol owns the 420/1013 amplitudes.  Static
    # atom/manifold energies (Delta, manifold, decays) stay here.
    return {
        "rb87_manifold": system.manifold,
        "t_rise": system.t_rise,
        "n_atoms": system.n_atoms,
        "n_levels": system.n_levels,
        "Delta": system.Delta,
        "v_ryd": system.v_ryd,
        "v_ryd_garb": system.v_ryd_garb,
        "ryd_state_decay_rate": system.ryd_state_decay_rate,
        "ryd_RD_rate": system.ryd_RD_rate,
        "ryd_BBR_rate": system.ryd_BBR_rate,
        "mid_state_decay_rate": system.mid_state_decay_rate,
        "ryd_branch": system.ryd_branch,
        "mid_branch": system.mid_branch,
        "rydberg_indices": system.rydberg_indices,
        "enable_rydberg_decay": system.enable_rydberg_decay,
        "enable_intermediate_decay": system.enable_intermediate_decay,
        "magnetic_field_G": system.magnetic_field_G,
        "ryd_zeeman_shift": system.ryd_zeeman_shift,
    }


# ── Single-atom physics → primitive E[ket,bra] terms/ratios ──────────────────


def _add_static_diagonals(model, levels: tuple[str, ...], h_const: np.ndarray) -> None:
    """Append a static ``coeff·sum_i E[a,a]_i`` term per nonzero diagonal energy.

    ``levels`` are the basis labels in index order; ``h_const[i,i]`` is the
    (possibly complex, decay-bearing) single-atom energy of level ``i``.
    """
    for i, level in enumerate(levels):
        coeff = complex(h_const[i, i])
        if coeff != 0:
            name = f"E[{level},{level}]"
            model.static_hamiltonian_terms.append(
                StaticHamiltonianTerm(name, model.operators.sum(name), coeff)
            )


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


def analog_3_local_blocks_from_metadata(metadata: dict | None) -> Analog3Blocks:
    """Reconstruct analog_3 blocks from a system/IR metadata dict (angular rad/s scalars).

    Falls back to the default analog_3 constants when the scalars are absent.
    """
    if not metadata or "Delta" not in metadata:
        return analog_3_local_blocks()
    Delta = float(metadata["Delta"])
    rabi_420 = float(metadata["rabi_420"])
    rabi_1013 = float(metadata["rabi_1013"])
    rabi_eff = float(metadata.get("rabi_eff") or rabi_420 * rabi_1013 / (2 * abs(Delta)))
    time_scale = float(metadata.get("time_scale") or 2 * np.pi / rabi_eff)
    mid_decay = float(metadata.get("mid_state_decay_rate", 0.0)) if metadata.get("enable_intermediate_decay") else 0.0
    ryd_decay = float(metadata.get("ryd_state_decay_rate", 0.0)) if metadata.get("enable_rydberg_decay") else 0.0
    return _analog3_blocks(Delta, rabi_420, rabi_1013, mid_decay, ryd_decay, rabi_eff, time_scale)


def _apply_analog_3_lattice_blocks(
    model: "RydbergSystem",
    *,
    detuning_sign: int = 1,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
    Delta_Hz: float | None = None,
    rabi_420_Hz: float | None = None,
    rabi_1013_Hz: float | None = None,
    **unused,
) -> None:
    _reject_unused(unused)
    ryd_level = 70
    blk = analog_3_local_blocks(
        Delta_Hz=Delta_Hz,
        rabi_420_Hz=rabi_420_Hz,
        rabi_1013_Hz=rabi_1013_Hz,
        detuning_sign=detuning_sign,
        enable_rydberg_decay=enable_rydberg_decay,
        enable_intermediate_decay=enable_intermediate_decay,
    )
    ryd_RD_rate = 1 / 410.41e-6
    ryd_BBR_rate = _ANALOG3_RYD_DECAY_RATE - ryd_RD_rate

    # Diagonal g/e/r energies become static E[a,a] terms; the e-r 1013 leg is a
    # static (non-Hermitian, h.c.-completed) E[r,e] coupling; the g-e 420 leg is a
    # driveable channel whose full Rabi is carried as the E[e,g] ratio.
    _add_static_diagonals(model, ("g", "e", "r"), blk.h_const)
    c_er = complex(blk.h_1013[2, 1])
    if c_er != 0:
        model.static_hamiltonian_terms.append(
            StaticHamiltonianTerm(
                "E[r,e]", model.operators.sum("E[r,e]"), c_er, add_hermitian_conjugate=True
            )
        )
    ratios = {"420": {"E[e,g]": complex(blk.drive_420[1, 0])}}

    model.metadata.update(
        {
            "physical_model": "analog_3",
            "laser_channel_ratios": ratios,
            "rabi_eff": blk.rabi_eff,
            "time_scale": blk.time_scale,
            "t_rise": 20e-9,
            "n_atoms": model.N,
            "n_levels": 3,
            "rabi_420": blk.rabi_420,
            "rabi_1013": blk.rabi_1013,
            "rabi_420_garbage": 0.0,
            "rabi_1013_garbage": 0.0,
            "Delta": blk.Delta,
            "v_ryd": _nearest_pair_strength(model.metadata.get("interaction_pairs", ())),
            "v_ryd_garb": 0.0,
            "ryd_level": ryd_level,
            "ryd_state_decay_rate": _ANALOG3_RYD_DECAY_RATE,
            "ryd_RD_rate": ryd_RD_rate,
            "ryd_BBR_rate": ryd_BBR_rate,
            "mid_state_decay_rate": _ANALOG3_MID_DECAY_RATE,
            "ryd_branch": {},
            "mid_branch": {},
            "rydberg_indices": (2,),
            "enable_rydberg_decay": enable_rydberg_decay,
            "enable_intermediate_decay": enable_intermediate_decay,
        }
    )


def _apply_rb87_7_lattice_blocks(
    model: "RydbergSystem",
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
    **unused,
) -> None:
    _reject_unused(unused)
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
    _add_static_diagonals(
        model, ("0", "1", "e1", "e2", "e3", "r", "r_garb"), h_const
    )

    # Unit-Rabi, phase-free 420/1013 legs decomposed into per-channel CG/dipole
    # ratios (the off-diagonal entries of the old drive matrices, all real).  A CZ
    # protocol multiplies its laser coefficient c420(t)/c1013(t) onto these; the
    # compiler auto-adds each leg's h.c.  420: |0>,|1> -> |e_F>; 1013: |e_F> -> |r>,|r_garb>.
    h420 = _rb87_local_h420(manifold, 1.0, physical.d_mid_ratio)
    h1013 = _rb87_local_h1013(manifold, 1.0, physical.d_ryd_ratio)
    mid = [(2, "e1"), (3, "e2"), (4, "e3")]
    ratios_420 = _offdiag_ratios(h420, mid, [(1, "1"), (0, "0")])
    ratios_1013 = _offdiag_ratios(h1013, [(5, "r"), (6, "r_garb")], mid)

    tag = f"rb87_7_{manifold}"
    model.metadata.update(_metadata_from_rb87_params(physical))
    model.metadata.update(
        {
            "physical_model": tag,
            "n_atoms": model.N,
            "n_sites": model.N,
            "level_structure": tag,
            "level_spec": level_structure(tag),
            "laser_channel_ratios": {"420": ratios_420, "1013": ratios_1013},
            "v_ryd": _nearest_pair_strength(model.metadata.get("interaction_pairs", ())),
            "ryd_level": physical.ryd_level,
        }
    )


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


def _nearest_pair_strength(pairs: tuple) -> float:
    if not pairs:
        return 0.0
    return float(max(abs(strength) for _, _, strength in pairs))


def _reject_unused(unused: dict) -> None:
    if unused:
        names = ", ".join(sorted(unused))
        raise TypeError(f"Unused physical parameter(s): {names}")
