"""Atomic system definitions: Rb87 parameters, Hamiltonians, branching ratios.

This module extracts the physical system description from CZGateSimulator,
providing an immutable AtomicSystem dataclass and factory functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from arc import Rubidium87
from arc.wigner import CG
from scipy.constants import c

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ======================================================================
# ARC-DERIVED ATOMIC CONSTANTS (AC Stark shift physics)
# ======================================================================

_atom = Rubidium87()
FREQ_D2: float = _atom.getTransitionFrequency(5, 0, 0.5, 5, 1, 1.5)
FREQ_D1: float = _atom.getTransitionFrequency(5, 0, 0.5, 5, 1, 0.5)
LAMBDA_D2: float = c / FREQ_D2 * 1e9   # nm
LAMBDA_D1: float = c / FREQ_D1 * 1e9   # nm
GAMMA_D2: float = 2 * np.pi * 6.065e6  # rad/s
GAMMA_D1: float = 2 * np.pi * 5.746e6  # rad/s

# Calibration reference (Manovitz et al.)
LAMBDA_PAPER: float = 784.0                    # nm
CALIBRATION_SHIFT_HZ: float = -12.2e6          # Hz at 160 uW
CALIBRATION_SCATTER_HZ: float = 35.0           # Hz at 160 uW
POWER_REF_UW: float = 160.0                    # uW reference power


def _raw_shift_and_scatter(wavelengths_nm):
    """Grimm formula for D1+D2 contributions (unnormalized, vectorized).

    Parameters
    ----------
    wavelengths_nm : float or ndarray
        Laser wavelength(s) in nm.

    Returns
    -------
    shift, scatter : same shape as input, in consistent arbitrary units.
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    omega = 2 * np.pi * c / (wavelengths_nm * 1e-9)

    shift = np.zeros_like(omega)
    scatter = np.zeros_like(omega)

    for omega_line, Gamma_line, line_strength in [
        (2 * np.pi * FREQ_D2, GAMMA_D2, 2.0 / 3.0),
        (2 * np.pi * FREQ_D1, GAMMA_D1, 1.0 / 3.0),
    ]:
        Dr = omega_line - omega
        Dc = omega_line + omega
        shift += line_strength * Gamma_line / omega_line**3 * (1.0 / Dr + 1.0 / Dc)
        scatter += (line_strength * Gamma_line**2 / omega_line**3
                    * (omega / omega_line)**3 * (1.0 / Dr + 1.0 / Dc)**2)

    return shift, scatter


_s784, _sc784 = _raw_shift_and_scatter(LAMBDA_PAPER)
_SHIFT_CAL: float = float(CALIBRATION_SHIFT_HZ / _s784)
_SCATTER_CAL: float = float(CALIBRATION_SCATTER_HZ / _sc784)


def compute_shift_scatter(wavelengths_nm):
    """Calibrated AC Stark shift (Hz) and scattering rate (Hz).

    Calibrated to the paper's measured values at 784 nm (160 uW, 1 um waist).
    Accepts scalar or array input.

    Returns
    -------
    shift_Hz, scatter_Hz : same shape as input.
    """
    s, sc = _raw_shift_and_scatter(wavelengths_nm)
    return s * _SHIFT_CAL, sc * _SCATTER_CAL


# ======================================================================
# PROTOCOL-SYSTEM COMPATIBILITY REGISTRY
# ======================================================================

# Maps param_set names to the Protocol subclass names they support.
# Uses strings (not class references) to avoid circular imports.
PROTOCOL_REGISTRY: dict[str, list[str]] = {
    "our":     ["TOProtocol", "ARProtocol"],
    "lukin":   ["TOProtocol", "ARProtocol"],
    "analog":  ["SweepProtocol"],
    "lattice": ["SweepProtocol"],
}


def compatible_protocols(param_set: str) -> list[str]:
    """Return protocol names compatible with the given system.

    >>> compatible_protocols("our")
    ['TOProtocol', 'ARProtocol']
    >>> compatible_protocols("analog")
    ['SweepProtocol']
    """
    if param_set not in PROTOCOL_REGISTRY:
        raise ValueError(
            f"Unknown system '{param_set}'. "
            f"Available: {list(PROTOCOL_REGISTRY.keys())}"
        )
    return PROTOCOL_REGISTRY[param_set]


def check_protocol_compatibility(
    system: "AtomicSystem", protocol: object,
) -> None:
    """Raise ValueError if *protocol* is incompatible with *system*."""
    allowed = PROTOCOL_REGISTRY.get(system.param_set, [])
    proto_name = type(protocol).__name__
    if proto_name not in allowed:
        raise ValueError(
            f"Protocol '{proto_name}' is not compatible with system "
            f"'{system.param_set}'. Compatible protocols: {allowed}"
        )


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
    atom = Rubidium87()
    ryd_level = 70
    Delta = detuning_sign * 2 * np.pi * 9.1e9
    rabi_420 = 2 * np.pi * (491) * 10 ** (6)
    rabi_1013 = 2 * np.pi * (185) * 10 ** (6)
    rabi_eff = rabi_420 * rabi_1013 / (2 * abs(Delta))
    time_scale = 2 * np.pi / rabi_eff

    # Dipole matrix element ratios (σ⁻ polarization)
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
    atom = Rubidium87()
    ryd_level = 53
    Delta = detuning_sign * 2 * np.pi * 7.8e9
    rabi_420 = 2 * np.pi * 237e6
    rabi_1013 = 2 * np.pi * 303e6
    rabi_eff = rabi_420 * rabi_1013 / (2 * abs(Delta))
    time_scale = 2 * np.pi / rabi_eff

    # Dipole matrix element ratios (σ⁺ polarization)
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
) -> AtomicSystem:
    """Initialize 3-level analog system for stretched-state quantum simulation.

    Compatible protocols: SweepProtocol.

    Level structure (single chain, unit CG prefactors):
        |g⟩ = |5S₁/₂, F=2, m=-2⟩  (index 0)
        |e⟩ = |6P₃/₂, F=3, m=-3⟩  (index 1)
        |r⟩ = |70S₁/₂, m_J=-1/2⟩  (index 2)

    Uses same physical constants as 'our' parameter set.
    """
    atom = Rubidium87()
    ryd_level = 70
    n = 3
    Delta = detuning_sign * 2 * np.pi * 9.1e9
    rabi_420 = 2 * np.pi * 491e6
    rabi_1013 = 2 * np.pi * 491e6
    rabi_eff = rabi_420 * rabi_1013 / (2 * abs(Delta))
    time_scale = 2 * np.pi / rabi_eff

    v_ryd = 2 * np.pi * 874e9 / 3**6
    mid_state_decay_rate = 1 / 110.7e-9
    ryd_state_decay_rate = 1 / 151.55e-6
    ryd_RD_rate = 1 / 410.41e-6
    ryd_BBR_rate = ryd_state_decay_rate - ryd_RD_rate

    mid_decay = mid_state_decay_rate if enable_intermediate_decay else 0.0
    ryd_decay = ryd_state_decay_rate if enable_rydberg_decay else 0.0

    # H_const: 3x3 single-atom → 9x9 two-atom
    ham_sq = np.zeros((n, n), dtype=np.complex128)
    ham_sq[1, 1] = Delta - 1j * mid_decay / 2     # |e⟩
    ham_sq[2, 2] = 0 - 1j * ryd_decay / 2         # |r⟩
    ham_tq = np.kron(np.eye(n), ham_sq) + np.kron(ham_sq, np.eye(n))
    # VdW: |r,r⟩ ↔ |r,r⟩
    ryd_sq = np.zeros((n, n), dtype=np.complex128)
    ryd_sq[2, 2] = 1.0
    ham_tq += v_ryd * np.kron(ryd_sq, ryd_sq)
    tq_ham_const = ham_tq

    # H_1013: |r⟩⟨e| + h.c. (constant, RWA applied)
    ham_sq = np.zeros((n, n), dtype=np.complex128)
    ham_sq[2, 1] = rabi_1013 / 2   # CG = 1
    tq_ham_1013 = np.kron(np.eye(n), ham_sq) + np.kron(ham_sq, np.eye(n))

    # H_420: |e⟩⟨g| (phase-modulated by solve_gate)
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
    sublattice: np.ndarray         # (N,) checkerboard signs ±1
    vdw_pairs: tuple               # ((i, j, V_rel), ...)
    V_nn: float                    # NN interaction strength
    Omega: float                   # global Rabi frequency

    # Precomputed sparse operators (dim = 2^N)
    sum_X: object                  # csc_matrix: Σ_i σ^x_i
    sum_n: object                  # csc_matrix: Σ_i n_i
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


# ======================================================================
# HAMILTONIAN BUILDERS (module-level, no class dependency)
# ======================================================================


def _build_tq_ham_const(
    Delta: float, v_ryd: float, ryd_zeeman_shift: float,
    middecay: float, ryddecay: float,
) -> "NDArray[np.complexfloating]":
    """Build the time-independent two-atom Hamiltonian."""
    ham_tq_mat = np.zeros((49, 49), dtype=np.complex128)
    ham_sq_mat = np.zeros((7, 7), dtype=np.complex128)
    delta = 0

    # Intermediate state energies with hyperfine splitting
    ham_sq_mat[2][2] = Delta - 2 * np.pi * 51e6 - 1j * middecay / 2
    ham_sq_mat[3][3] = Delta - 1j * middecay / 2
    ham_sq_mat[4][4] = Delta + 2 * np.pi * 87e6 - 1j * middecay / 2

    # Rydberg state energies
    ham_sq_mat[5][5] = delta - 1j * ryddecay / 2
    ham_sq_mat[6][6] = delta + ryd_zeeman_shift - 1j * ryddecay / 2

    # Two-atom Hamiltonian via Kronecker products
    ham_tq_mat = ham_tq_mat + np.kron(np.eye(7), ham_sq_mat)
    ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat, np.eye(7))

    # Rydberg-Rydberg van der Waals interaction
    ham_vdw_mat = np.zeros((7, 7))
    ham_vdw_mat[5][5] = 1
    ham_vdw_mat_garb = np.zeros((7, 7))
    ham_vdw_mat_garb[6][6] = 1
    ham_tq_mat = ham_tq_mat + v_ryd * np.kron(
        ham_vdw_mat + ham_vdw_mat_garb, ham_vdw_mat + ham_vdw_mat_garb
    )
    return ham_tq_mat


def _tq_ham_420_our(rabi_420, rabi_420_garbage):
    """Build 420nm coupling for 'our' (σ⁻ polarization)."""
    ham_tq_mat = np.zeros((49, 49), dtype=np.complex128)
    ham_sq_mat = np.zeros((7, 7), dtype=np.complex128)

    ham_sq_mat[2][1] = (
        rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, 1, -1)
        + rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, 1, -1)
    ) / 2
    ham_sq_mat[3][1] = (
        rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, 2, -1)
        + rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, 2, -1)
    ) / 2
    ham_sq_mat[4][1] = (
        rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, 3, -1)
        + rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, 3, -1)
    ) / 2

    ham_tq_mat = ham_tq_mat + np.kron(np.eye(7), ham_sq_mat)
    ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat, np.eye(7))
    return ham_tq_mat


def _tq_ham_1013_our(rabi_1013, rabi_1013_garbage):
    """Build 1013nm coupling for 'our' parameter set."""
    ham_tq_mat = np.zeros((49, 49), dtype=np.complex128)
    ham_sq_mat = np.zeros((7, 7), dtype=np.complex128)

    ham_sq_mat[5][2] = (rabi_1013 / 2) * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, 1, -1)
    ham_sq_mat[5][3] = (rabi_1013 / 2) * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, 2, -1)
    ham_sq_mat[5][4] = (rabi_1013 / 2) * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, 3, -1)

    ham_sq_mat[6][2] = (rabi_1013_garbage / 2) * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, 1, -1)
    ham_sq_mat[6][3] = (rabi_1013_garbage / 2) * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, 2, -1)
    ham_sq_mat[6][4] = (rabi_1013_garbage / 2) * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, 3, -1)

    ham_tq_mat = ham_tq_mat + np.kron(np.eye(7), ham_sq_mat)
    ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat, np.eye(7))
    return ham_tq_mat


def _tq_ham_420_lukin(rabi_420, rabi_420_garbage):
    """Build 420nm coupling for 'lukin' (σ⁺ polarization)."""
    ham_tq_mat = np.zeros((49, 49), dtype=np.complex128)
    ham_sq_mat = np.zeros((7, 7), dtype=np.complex128)

    ham_sq_mat[2][1] = (
        rabi_420 * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, 1, 1)
        + rabi_420_garbage * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, 1, 1)
    ) / 2
    ham_sq_mat[3][1] = (
        rabi_420 * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, 2, 1)
        + rabi_420_garbage * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, 2, 1)
    ) / 2
    ham_sq_mat[4][1] = (
        rabi_420 * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, 3, 1)
        + rabi_420_garbage * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, 3, 1)
    ) / 2

    ham_tq_mat = ham_tq_mat + np.kron(np.eye(7), ham_sq_mat)
    ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat, np.eye(7))
    return ham_tq_mat


def _tq_ham_1013_lukin(rabi_1013, rabi_1013_garbage):
    """Build 1013nm coupling for 'lukin' parameter set."""
    ham_tq_mat = np.zeros((49, 49), dtype=np.complex128)
    ham_sq_mat = np.zeros((7, 7), dtype=np.complex128)

    ham_sq_mat[5][2] = (rabi_1013 / 2) * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, 1, 1)
    ham_sq_mat[5][3] = (rabi_1013 / 2) * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, 2, 1)
    ham_sq_mat[5][4] = (rabi_1013 / 2) * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, 3, 1)

    ham_sq_mat[6][2] = (rabi_1013_garbage / 2) * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, 1, 1)
    ham_sq_mat[6][3] = (rabi_1013_garbage / 2) * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, 2, 1)
    ham_sq_mat[6][4] = (rabi_1013_garbage / 2) * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, 3, 1)

    ham_tq_mat = ham_tq_mat + np.kron(np.eye(7), ham_sq_mat)
    ham_tq_mat = ham_tq_mat + np.kron(ham_sq_mat, np.eye(7))
    return ham_tq_mat


def _build_zero_state_lightshift(
    param_set, Delta, rabi_420, rabi_420_garbage,
    mid_state_decay_rate, enable_intermediate_decay, enable_0_scattering,
):
    """Build perturbative light-shift Hamiltonian from |0⟩ → |eᵢ⟩ coupling."""
    E_0 = -2 * np.pi * 6.835e9
    mid_energies = np.array([
        Delta - 2 * np.pi * 51e6,
        Delta,
        Delta + 2 * np.pi * 87e6,
    ], dtype=np.float64)

    if param_set == "our":
        cg_ratio_main = CG(1 / 2, -1 / 2, 3 / 2, 1 / 2, 1, 0) / CG(
            1 / 2, -1 / 2, 3 / 2, 1 / 2, 2, 0
        )
        cg_ratio_garb = CG(1 / 2, 1 / 2, 3 / 2, -1 / 2, 1, 0) / CG(
            1 / 2, 1 / 2, 3 / 2, -1 / 2, 2, 0
        )
        couplings = [
            (
                cg_ratio_main * rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, F, -1)
                + cg_ratio_garb * rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, F, -1)
            ) / 2
            for F in (1, 2, 3)
        ]
    else:  # lukin
        cg_ratio_main = CG(1 / 2, 1 / 2, 3 / 2, -1 / 2, 1, 0) / CG(
            1 / 2, 1 / 2, 3 / 2, -1 / 2, 2, 0
        )
        cg_ratio_garb = CG(1 / 2, -1 / 2, 3 / 2, 1 / 2, 1, 0) / CG(
            1 / 2, -1 / 2, 3 / 2, 1 / 2, 2, 0
        )
        couplings = [
            (
                cg_ratio_main * rabi_420 * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, F, 1)
                + cg_ratio_garb * rabi_420_garbage * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, F, 1)
            ) / 2
            for F in (1, 2, 3)
        ]

    ls_sq = np.zeros((7, 7), dtype=np.complex128)
    total_shift = 0.0
    scatter_rate = 0.0
    gamma = mid_state_decay_rate if enable_intermediate_decay else 0
    for idx, (g_i, E_e) in enumerate(zip(couplings, mid_energies), start=2):
        detuning = E_e - E_0
        shift = (np.abs(g_i) ** 2) / detuning
        ls_sq[idx][idx] = shift
        total_shift += shift
        scatter_rate += (np.abs(g_i) ** 2) * gamma / (detuning ** 2)
    ls_sq[0][0] = -total_shift - 1j * scatter_rate / 2 if enable_0_scattering else -total_shift

    ls_tq = np.kron(np.eye(7), ls_sq) + np.kron(ls_sq, np.eye(7))
    return ls_tq


# ======================================================================
# BRANCHING RATIOS
# ======================================================================


def _rydberg_branching_ratios(atom, ryd_level, param_set):
    """Compute branching ratios for Rydberg radiative decay."""
    I = 3 / 2
    mI = 1 / 2
    nr = ryd_level
    lr, jr = 0, 1 / 2
    if param_set == "our":
        mjr = -1 / 2
    else:
        mjr = 1 / 2
    fr_list = [2, 1]
    mfr = mI + mjr

    ne, le = 5, 1
    je_list = [3 / 2, 1 / 2]
    ng, lg, jg = 5, 0, 1 / 2

    a = []
    b = []

    for _je in je_list:
        fe_range = np.arange(abs(I - _je), I + _je + 1, 1)
        for _fe in fe_range:
            mfe_range = np.arange(-_fe, _fe + 1, 1)
            for _mfe in mfe_range:
                t = 0.0
                for _fr in fr_list:
                    if abs(mfr) <= _fr and abs(mfr - _mfe) < 2:
                        t += CG(jr, mjr, I, mI, _fr, mfr) * \
                            atom.getDipoleMatrixElementHFS(
                                ne, le, _je, _fe, _mfe,
                                nr, lr, jr, _fr, mfr,
                                q=mfr - _mfe,
                            )
                a.append(t**2)

                bb = []
                for fg in [2, 1]:
                    mfg_range = np.arange(-fg, fg + 1, 1)
                    for _mfg in mfg_range:
                        if abs(_mfg - _mfe) < 2:
                            bb.append(
                                atom.getDipoleMatrixElementHFS(
                                    ne, le, _je, _fe, _mfe,
                                    ng, lg, jg, fg, _mfg,
                                    q=_mfg - _mfe,
                                ) ** 2
                            )
                        else:
                            bb.append(0.0)
                bb_sum = np.sum(bb)
                bb = [x / bb_sum for x in bb]
                b.append(bb)

    a_sum = np.sum(a)
    a = [x / a_sum for x in a]

    branch_ratio = np.array(
        [a[i] * np.array(b[i]) for i in range(len(a))]
    ).sum(axis=0)

    return {
        "to_0": float(branch_ratio[6]),
        "to_1": float(branch_ratio[2]),
        "to_L0": float(branch_ratio[5] + branch_ratio[7]),
        "to_L1": float(
            branch_ratio[0] + branch_ratio[1]
            + branch_ratio[3] + branch_ratio[4]
        ),
    }


def _mid_branching_ratios(atom, F, mF):
    """Compute branching ratios for 6P3/2 intermediate state decay."""
    ne, le, je, fe, mfe = 6, 1, 3 / 2, F, mF
    ng, lg, jg = 5, 0, 1 / 2

    a = []
    for fg in [2, 1]:
        mfg_range = np.arange(-fg, fg + 1, 1)
        for _mfg in mfg_range:
            if abs(_mfg - mfe) < 2:
                a.append(
                    atom.getDipoleMatrixElementHFS(
                        ne, le, je, fe, mfe,
                        ng, lg, jg, fg, _mfg,
                        q=_mfg - mfe,
                    ) ** 2
                )
            else:
                a.append(0.0)
    a_sum = np.sum(a)
    branch_ratio = [x / a_sum for x in a]

    return {
        "to_0": float(branch_ratio[6]),
        "to_1": float(branch_ratio[2]),
        "to_L0": float(branch_ratio[5] + branch_ratio[7]),
        "to_L1": float(
            branch_ratio[0] + branch_ratio[1]
            + branch_ratio[3] + branch_ratio[4]
        ),
    }


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================


def build_occ_operator(index: int, n_levels: int = 7) -> "NDArray[np.complexfloating]":
    """Build occupation number operator for level `index`.

    Creates |i⟩⟨i| ⊗ I + I ⊗ |i⟩⟨i| for measuring total population
    in level i across both atoms.

    Parameters
    ----------
    index : int
        Single-atom level index.
    n_levels : int
        Number of single-atom levels (default 7).
    """
    oper_sq = np.zeros((n_levels, n_levels), dtype=np.complex128)
    oper_sq[index, index] = 1
    oper_tq = np.kron(np.eye(n_levels), oper_sq) + np.kron(oper_sq, np.eye(n_levels))
    return oper_tq


def build_sss_state_map(n_levels: int = 7) -> "dict[str, NDArray[np.complexfloating]]":
    """Build mapping from state labels to two-atom state vectors.

    Parameters
    ----------
    n_levels : int
        Number of single-atom levels (default 7).
        For n_levels=3, only product states "00","01","10","11" are returned
        (indices 0 and 1 of the 3-level basis).
    """
    s0 = np.zeros(n_levels, dtype=complex)
    s0[0] = 1.0
    s1 = np.zeros(n_levels, dtype=complex)
    s1[1] = 1.0
    state_00 = np.kron(s0, s0)
    state_01 = np.kron(s0, s1)
    state_10 = np.kron(s1, s0)
    state_11 = np.kron(s1, s1)
    states: dict[str, NDArray[np.complexfloating]] = {
        "00": state_00,
        "01": state_01,
        "10": state_10,
        "11": state_11,
    }
    if n_levels >= 7:
        states.update({
            "SSS-0": 0.5 * state_00 + 0.5 * state_01 + 0.5 * state_10 + 0.5 * state_11,
            "SSS-1": 0.5 * state_00 - 0.5 * state_01 - 0.5 * state_10 + 0.5 * state_11,
            "SSS-2": 0.5 * state_00 + 0.5j * state_01 + 0.5j * state_10 - 0.5 * state_11,
            "SSS-3": 0.5 * state_00 - 0.5j * state_01 - 0.5j * state_10 - 0.5 * state_11,
            "SSS-4": state_00,
            "SSS-5": state_11,
            "SSS-6": 0.5 * state_00 + 0.5 * state_01 + 0.5 * state_10 - 0.5 * state_11,
            "SSS-7": 0.5 * state_00 - 0.5 * state_01 - 0.5 * state_10 - 0.5 * state_11,
            "SSS-8": 0.5 * state_00 + 0.5j * state_01 + 0.5j * state_10 + 0.5 * state_11,
            "SSS-9": 0.5 * state_00 - 0.5j * state_01 - 0.5j * state_10 + 0.5 * state_11,
            "SSS-10": state_00 / np.sqrt(2) + 1j * state_11 / np.sqrt(2),
            "SSS-11": state_00 / np.sqrt(2) - 1j * state_11 / np.sqrt(2),
        })
    return states


def build_vdw_unit_operator(
    rydberg_indices: tuple[int, ...] = (5, 6), n_levels: int = 7,
) -> "NDArray[np.complexfloating]":
    """Build the unit van der Waals interaction operator."""
    ryd_proj = np.zeros((n_levels, n_levels), dtype=np.complex128)
    for i in rydberg_indices:
        ryd_proj[i, i] = 1.0
    return np.kron(ryd_proj, ryd_proj)


def build_atom_a_projector(index: int, n_levels: int = 7) -> "NDArray[np.complexfloating]":
    """Build |i⟩⟨i| ⊗ I — projects Atom A (left) onto level index."""
    sq = np.zeros((n_levels, n_levels), dtype=np.complex128)
    sq[index, index] = 1
    return np.kron(sq, np.eye(n_levels, dtype=np.complex128))


def build_atom_b_projector(index: int, n_levels: int = 7) -> "NDArray[np.complexfloating]":
    """Build I ⊗ |i⟩⟨i| — projects Atom B (right) onto level index."""
    sq = np.zeros((n_levels, n_levels), dtype=np.complex128)
    sq[index, index] = 1
    return np.kron(np.eye(n_levels, dtype=np.complex128), sq)


def get_nominal_distance(param_set: str) -> float:
    """Get the nominal interatomic distance in μm."""
    return 3.0
