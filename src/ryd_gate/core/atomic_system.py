"""Atomic system definitions: Rb87 parameters, Hamiltonians, branching ratios.

This module extracts the physical system description from CZGateSimulator,
providing an immutable AtomicSystem dataclass and factory functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from arc import Rubidium87

# Re-exports for backward compatibility -- these symbols moved to new modules
from ryd_gate.core.ac_stark import (
    FREQ_D2, FREQ_D1, LAMBDA_D2, LAMBDA_D1, GAMMA_D2, GAMMA_D1,
    LAMBDA_PAPER, CALIBRATION_SHIFT_HZ, CALIBRATION_SCATTER_HZ, POWER_REF_UW,
    compute_shift_scatter,
)
from ryd_gate.core.registry import (
    PROTOCOL_REGISTRY, compatible_protocols, check_protocol_compatibility,
)
from ryd_gate.core.operators import (
    build_occ_operator, build_product_state_map, build_sss_state_map,
    build_vdw_unit_operator,
    build_atom_a_projector, build_atom_b_projector, get_nominal_distance,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class AtomicSystem:
    """Container for Rb87 atomic parameters and precomputed Hamiltonians.

    Created via :func:`create_our_system`, :func:`create_lukin_system`, or
    :func:`create_analog_system`. Holds all physical constants,
    precomputed Hamiltonians, decay rates, and branching ratios.
    """

    param_set: str
    ryd_level: int
    atom: Rubidium87 = field(repr=False)

    # Laser parameters
    Delta: float
    rabi_420: float
    rabi_1013: float
    rabi_eff: float
    time_scale: float
    rabi_420_garbage: float
    rabi_1013_garbage: float
    d_mid_ratio: float
    d_ryd_ratio: float

    # Interaction parameters
    v_ryd: float
    v_ryd_garb: float
    ryd_zeeman_shift: float
    detuning_sign: int

    # Decay rates
    mid_state_decay_rate: float
    mid_garb_decay_rate: float
    ryd_state_decay_rate: float
    ryd_RD_rate: float
    ryd_BBR_rate: float
    ryd_garb_decay_rate: float

    # Branching ratios
    ryd_branch: dict = field(repr=False)
    mid_branch: dict = field(repr=False)

    # Precomputed Hamiltonians (49x49 complex matrices)
    tq_ham_const: "NDArray[np.complexfloating]" = field(repr=False)
    tq_ham_420: "NDArray[np.complexfloating]" = field(repr=False)
    tq_ham_1013: "NDArray[np.complexfloating]" = field(repr=False)
    tq_ham_420_conj: "NDArray[np.complexfloating]" = field(repr=False)
    tq_ham_1013_conj: "NDArray[np.complexfloating]" = field(repr=False)
    tq_ham_lightshift_zero: "NDArray[np.complexfloating]" = field(repr=False)

    # Pulse shaping
    t_rise: float
    blackmanflag: bool

    # Feature flags
    enable_rydberg_decay: bool
    enable_intermediate_decay: bool
    enable_0_scattering: bool
    enable_polarization_leakage: bool

    # Level structure
    n_levels: int = 7
    rydberg_indices: tuple[int, ...] = (5, 6)


# ======================================================================
# FACTORY FUNCTIONS
# ======================================================================


def create_our_system(
    detuning_sign: int = 1,
    blackmanflag: bool = True,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
    enable_0_scattering: bool = True,
    enable_polarization_leakage: bool = False,
) -> AtomicSystem:
    """Initialize 'our' lab experimental parameters (n=70 Rydberg).

    Compatible protocols: TOProtocol, ARProtocol.
    """
    from ryd_gate.core.hamiltonian_builders import (
        _build_tq_ham_const, _tq_ham_420_our, _tq_ham_1013_our,
        _build_zero_state_lightshift,
    )
    from ryd_gate.core.branching import _rydberg_branching_ratios, _mid_branching_ratios

    atom = Rubidium87()
    ryd_level = 70
    Delta = detuning_sign * 2 * np.pi * 9.1e9
    rabi_420 = 2 * np.pi * (491) * 10 ** (6)
    rabi_1013 = 2 * np.pi * (185) * 10 ** (6)
    rabi_eff = rabi_420 * rabi_1013 / (2 * abs(Delta))
    time_scale = 2 * np.pi / rabi_eff

    # Dipole matrix element ratios (sigma- polarization)
    d_mid_ratio = atom.getDipoleMatrixElement(
        5, 0, 0.5, 0.5, 6, 1, 1.5, -0.5, -1
    ) / atom.getDipoleMatrixElement(5, 0, 0.5, -0.5, 6, 1, 1.5, -1.5, -1)
    d_ryd_ratio = atom.getDipoleMatrixElement(
        6, 1, 1.5, -0.5, ryd_level, 0, 0.5, 0.5, 1
    ) / atom.getDipoleMatrixElement(
        6, 1, 1.5, -1.5, ryd_level, 0, 0.5, -0.5, 1
    )
    rabi_420_garbage = rabi_420 * d_mid_ratio
    rabi_1013_garbage = rabi_1013 * d_ryd_ratio

    v_ryd = 2 * np.pi * 874e9 / 3**6
    v_ryd_garb = 2 * np.pi * 874e9 / 3**6
    ryd_zeeman_shift = 2 * np.pi * 56e6 if enable_polarization_leakage else 2 * np.pi * 56e9

    mid_state_decay_rate = 1 / (110.7e-9)
    mid_garb_decay_rate = 1 / (110.7e-9)
    ryd_state_decay_rate = 1 / (151.55e-6)
    ryd_RD_rate = 1 / (410.41e-6)
    ryd_BBR_rate = ryd_state_decay_rate - ryd_RD_rate
    ryd_garb_decay_rate = 1 / (151.55e-6)

    # Branching ratios
    ryd_branch = _rydberg_branching_ratios(atom, ryd_level, "our")
    mid_branch = {
        F: _mid_branching_ratios(atom, F, mF=-1)
        for F in (1, 2, 3)
    }

    # Build Hamiltonians
    tq_ham_const = _build_tq_ham_const(
        Delta, v_ryd, ryd_zeeman_shift,
        mid_state_decay_rate if enable_intermediate_decay else 0,
        ryd_state_decay_rate if enable_rydberg_decay else 0,
    )
    tq_ham_420 = _tq_ham_420_our(rabi_420, rabi_420_garbage)
    tq_ham_1013 = _tq_ham_1013_our(rabi_1013, rabi_1013_garbage)
    tq_ham_lightshift_zero = _build_zero_state_lightshift(
        "our", Delta, rabi_420, rabi_420_garbage,
        mid_state_decay_rate, enable_intermediate_decay, enable_0_scattering,
    )

    return AtomicSystem(
        param_set="our", ryd_level=ryd_level, atom=atom,
        Delta=Delta, rabi_420=rabi_420, rabi_1013=rabi_1013,
        rabi_eff=rabi_eff, time_scale=time_scale,
        rabi_420_garbage=rabi_420_garbage, rabi_1013_garbage=rabi_1013_garbage,
        d_mid_ratio=d_mid_ratio, d_ryd_ratio=d_ryd_ratio,
        v_ryd=v_ryd, v_ryd_garb=v_ryd_garb,
        ryd_zeeman_shift=ryd_zeeman_shift, detuning_sign=detuning_sign,
        mid_state_decay_rate=mid_state_decay_rate,
        mid_garb_decay_rate=mid_garb_decay_rate,
        ryd_state_decay_rate=ryd_state_decay_rate,
        ryd_RD_rate=ryd_RD_rate, ryd_BBR_rate=ryd_BBR_rate,
        ryd_garb_decay_rate=ryd_garb_decay_rate,
        ryd_branch=ryd_branch, mid_branch=mid_branch,
        tq_ham_const=tq_ham_const, tq_ham_420=tq_ham_420,
        tq_ham_1013=tq_ham_1013,
        tq_ham_420_conj=tq_ham_420.conj().T,
        tq_ham_1013_conj=tq_ham_1013.conj().T,
        tq_ham_lightshift_zero=tq_ham_lightshift_zero,
        t_rise=20e-9, blackmanflag=blackmanflag,
        enable_rydberg_decay=enable_rydberg_decay,
        enable_intermediate_decay=enable_intermediate_decay,
        enable_0_scattering=enable_0_scattering,
        enable_polarization_leakage=enable_polarization_leakage,
    )


def create_lukin_system(
    detuning_sign: int = 1,
    blackmanflag: bool = True,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
    enable_0_scattering: bool = True,
    enable_polarization_leakage: bool = False,
) -> AtomicSystem:
    """Initialize 'lukin' (Harvard) experimental parameters (n=53 Rydberg).

    Compatible protocols: TOProtocol, ARProtocol.
    """
    from ryd_gate.core.hamiltonian_builders import (
        _build_tq_ham_const, _tq_ham_420_lukin, _tq_ham_1013_lukin,
        _build_zero_state_lightshift,
    )
    from ryd_gate.core.branching import _rydberg_branching_ratios, _mid_branching_ratios

    atom = Rubidium87()
    ryd_level = 53
    Delta = detuning_sign * 2 * np.pi * 7.8e9
    rabi_420 = 2 * np.pi * 237e6
    rabi_1013 = 2 * np.pi * 303e6
    rabi_eff = rabi_420 * rabi_1013 / (2 * abs(Delta))
    time_scale = 2 * np.pi / rabi_eff

    # Dipole matrix element ratios (sigma+ polarization)
    d_mid_ratio = atom.getDipoleMatrixElement(
        5, 0, 0.5, -0.5, 6, 1, 1.5, 0.5, 1
    ) / atom.getDipoleMatrixElement(5, 0, 0.5, 0.5, 6, 1, 1.5, 1.5, 1)
    d_ryd_ratio = atom.getDipoleMatrixElement(
        6, 1, 1.5, 0.5, ryd_level, 0, 0.5, -0.5, -1
    ) / atom.getDipoleMatrixElement(
        6, 1, 1.5, 1.5, ryd_level, 0, 0.5, 0.5, -1
    )
    rabi_420_garbage = rabi_420 * d_mid_ratio
    rabi_1013_garbage = rabi_1013 * d_ryd_ratio

    v_ryd = 2 * np.pi * 450e6
    v_ryd_garb = 2 * np.pi * 450e6
    ryd_zeeman_shift = 2 * np.pi * 2.4e9 if enable_polarization_leakage else 2 * np.pi * 2.4e12

    mid_state_decay_rate = 1 / (110e-9)
    mid_garb_decay_rate = 1 / (110e-9)
    ryd_state_decay_rate = 1 / (88e-6)
    ryd_RD_rate = 1 / (147.64e-6)
    ryd_BBR_rate = ryd_state_decay_rate - ryd_RD_rate
    ryd_garb_decay_rate = 1 / (88e-6)

    ryd_branch = _rydberg_branching_ratios(atom, ryd_level, "lukin")
    mid_branch = {
        F: _mid_branching_ratios(atom, F, mF=1)
        for F in (1, 2, 3)
    }

    tq_ham_const = _build_tq_ham_const(
        Delta, v_ryd, ryd_zeeman_shift,
        mid_state_decay_rate if enable_intermediate_decay else 0,
        ryd_state_decay_rate if enable_rydberg_decay else 0,
    )
    tq_ham_420 = _tq_ham_420_lukin(rabi_420, rabi_420_garbage)
    tq_ham_1013 = _tq_ham_1013_lukin(rabi_1013, rabi_1013_garbage)
    tq_ham_lightshift_zero = _build_zero_state_lightshift(
        "lukin", Delta, rabi_420, rabi_420_garbage,
        mid_state_decay_rate, enable_intermediate_decay, enable_0_scattering,
    )

    return AtomicSystem(
        param_set="lukin", ryd_level=ryd_level, atom=atom,
        Delta=Delta, rabi_420=rabi_420, rabi_1013=rabi_1013,
        rabi_eff=rabi_eff, time_scale=time_scale,
        rabi_420_garbage=rabi_420_garbage, rabi_1013_garbage=rabi_1013_garbage,
        d_mid_ratio=d_mid_ratio, d_ryd_ratio=d_ryd_ratio,
        v_ryd=v_ryd, v_ryd_garb=v_ryd_garb,
        ryd_zeeman_shift=ryd_zeeman_shift, detuning_sign=detuning_sign,
        mid_state_decay_rate=mid_state_decay_rate,
        mid_garb_decay_rate=mid_garb_decay_rate,
        ryd_state_decay_rate=ryd_state_decay_rate,
        ryd_RD_rate=ryd_RD_rate, ryd_BBR_rate=ryd_BBR_rate,
        ryd_garb_decay_rate=ryd_garb_decay_rate,
        ryd_branch=ryd_branch, mid_branch=mid_branch,
        tq_ham_const=tq_ham_const, tq_ham_420=tq_ham_420,
        tq_ham_1013=tq_ham_1013,
        tq_ham_420_conj=tq_ham_420.conj().T,
        tq_ham_1013_conj=tq_ham_1013.conj().T,
        tq_ham_lightshift_zero=tq_ham_lightshift_zero,
        t_rise=20e-9, blackmanflag=blackmanflag,
        enable_rydberg_decay=enable_rydberg_decay,
        enable_intermediate_decay=enable_intermediate_decay,
        enable_0_scattering=enable_0_scattering,
        enable_polarization_leakage=enable_polarization_leakage,
    )


def create_analog_system(
    detuning_sign: int = 1,
    blackmanflag: bool = True,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
    distance_um: float = 3.0,
    Delta_Hz: float | None = None,
    rabi_420_Hz: float | None = None,
    rabi_1013_Hz: float | None = None,
) -> AtomicSystem:
    """Initialize 3-level analog system for stretched-state quantum simulation.

    Compatible protocols: SweepProtocol.

    Level structure (single chain, unit CG prefactors):
        |g> = |5S_{1/2}, F=2, m=-2>  (index 0)
        |e> = |6P_{3/2}, F=3, m=-3>  (index 1)
        |r> = |70S_{1/2}, m_J=-1/2>  (index 2)

    Uses same physical constants as 'our' parameter set.
    """
    atom = Rubidium87()
    ryd_level = 70
    n = 3
    Delta = detuning_sign * 2 * np.pi * (Delta_Hz if Delta_Hz is not None else 9.1e9)
    rabi_420 = 2 * np.pi * (rabi_420_Hz if rabi_420_Hz is not None else 491e6)
    rabi_1013 = 2 * np.pi * (rabi_1013_Hz if rabi_1013_Hz is not None else 491e6)
    rabi_eff = rabi_420 * rabi_1013 / (2 * abs(Delta))
    time_scale = 2 * np.pi / rabi_eff

    v_ryd = 2 * np.pi * 874e9 / distance_um**6
    mid_state_decay_rate = 1 / 110.7e-9
    ryd_state_decay_rate = 1 / 151.55e-6
    ryd_RD_rate = 1 / 410.41e-6
    ryd_BBR_rate = ryd_state_decay_rate - ryd_RD_rate

    mid_decay = mid_state_decay_rate if enable_intermediate_decay else 0.0
    ryd_decay = ryd_state_decay_rate if enable_rydberg_decay else 0.0

    # H_const: 3x3 single-atom -> 9x9 two-atom
    ham_sq = np.zeros((n, n), dtype=np.complex128)
    ham_sq[1, 1] = Delta - 1j * mid_decay / 2     # |e>
    ham_sq[2, 2] = 0 - 1j * ryd_decay / 2         # |r>
    ham_tq = np.kron(np.eye(n), ham_sq) + np.kron(ham_sq, np.eye(n))
    # VdW: |r,r> <-> |r,r>
    ryd_sq = np.zeros((n, n), dtype=np.complex128)
    ryd_sq[2, 2] = 1.0
    ham_tq += v_ryd * np.kron(ryd_sq, ryd_sq)
    tq_ham_const = ham_tq

    # H_1013: |r><e| + h.c. (constant, RWA applied)
    ham_sq = np.zeros((n, n), dtype=np.complex128)
    ham_sq[2, 1] = rabi_1013 / 2   # CG = 1
    tq_ham_1013 = np.kron(np.eye(n), ham_sq) + np.kron(ham_sq, np.eye(n))

    # H_420: |e><g| (phase-modulated by solve_gate)
    ham_sq = np.zeros((n, n), dtype=np.complex128)
    ham_sq[1, 0] = rabi_420 / 2    # CG = 1
    tq_ham_420 = np.kron(np.eye(n), ham_sq) + np.kron(ham_sq, np.eye(n))

    # No dark-state lightshift in 3-level system
    tq_ham_lightshift_zero = np.zeros((n * n, n * n), dtype=np.complex128)

    return AtomicSystem(
        param_set="analog", ryd_level=ryd_level, atom=atom,
        Delta=Delta, rabi_420=rabi_420, rabi_1013=rabi_1013,
        rabi_eff=rabi_eff, time_scale=time_scale,
        rabi_420_garbage=0.0, rabi_1013_garbage=0.0,
        d_mid_ratio=0.0, d_ryd_ratio=0.0,
        v_ryd=v_ryd, v_ryd_garb=0.0,
        ryd_zeeman_shift=0.0, detuning_sign=detuning_sign,
        mid_state_decay_rate=mid_state_decay_rate,
        mid_garb_decay_rate=0.0,
        ryd_state_decay_rate=ryd_state_decay_rate,
        ryd_RD_rate=ryd_RD_rate, ryd_BBR_rate=ryd_BBR_rate,
        ryd_garb_decay_rate=0.0,
        ryd_branch={}, mid_branch={},
        tq_ham_const=tq_ham_const, tq_ham_420=tq_ham_420,
        tq_ham_1013=tq_ham_1013,
        tq_ham_420_conj=tq_ham_420.conj().T,
        tq_ham_1013_conj=tq_ham_1013.conj().T,
        tq_ham_lightshift_zero=tq_ham_lightshift_zero,
        t_rise=20e-9, blackmanflag=blackmanflag,
        enable_rydberg_decay=enable_rydberg_decay,
        enable_intermediate_decay=enable_intermediate_decay,
        enable_0_scattering=False,
        enable_polarization_leakage=False,
        n_levels=3, rydberg_indices=(2,),
    )


# ======================================================================
# LATTICE SYSTEM (N-atom, 2-level)
# ======================================================================


@dataclass(frozen=True)
class LatticeSystem:
    """N-atom 2-level lattice system with precomputed sparse operators.

    Created via :func:`create_lattice_system`. Holds geometry, interaction
    parameters, and precomputed sparse Hamiltonian components.

    Compatible protocols: SweepProtocol.
    """

    param_set: str
    Lx: int
    Ly: int
    N: int
    coords: np.ndarray             # (N, 2) grid coordinates
    sublattice: np.ndarray         # (N,) checkerboard signs +/-1
    vdw_pairs: tuple               # ((i, j, V_rel), ...)
    V_nn: float                    # NN interaction strength
    Omega: float                   # global Rabi frequency

    # Precomputed sparse operators (dim = 2^N)
    sum_X: object                  # csc_matrix: sum_i sigma^x_i
    sum_n: object                  # csc_matrix: sum_i n_i
    n_list: list                   # [csc_matrix, ...] per-site occupation
    H_vdw: object                  # csc_matrix: VdW interaction (diagonal)


def create_lattice_system(
    Lx: int = 3,
    Ly: int = 3,
    V_nn: float = 24.0,
    Omega: float = 1.0,
) -> LatticeSystem:
    """Build a 2-level square lattice system with precomputed operators.

    Compatible protocols: SweepProtocol.

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions.
    V_nn : float
        Nearest-neighbor VdW interaction strength.
    Omega : float
        Global Rabi frequency.
    """
    from ryd_gate.lattice.geometry import make_square_lattice
    from ryd_gate.lattice.operators import build_operators

    sq = make_square_lattice(Lx, Ly)
    ops = build_operators(sq.N, sq.vdw_pairs, V_nn)

    return LatticeSystem(
        param_set="lattice",
        Lx=Lx, Ly=Ly, N=sq.N,
        coords=sq.coords,
        sublattice=sq.sublattice,
        vdw_pairs=sq.vdw_pairs,
        V_nn=V_nn, Omega=Omega,
        sum_X=ops["sum_X"],
        sum_n=ops["sum_n"],
        n_list=ops["n_list"],
        H_vdw=ops["H_vdw"],
    )
