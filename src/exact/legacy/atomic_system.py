"""Atomic system definitions: Rb87 parameters, Hamiltonians, branching ratios.

This module extracts the physical system description from CZGateSimulator,
providing an immutable AtomicSystem dataclass and factory functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from arc import Rubidium87
    from numpy.typing import NDArray


def _rubidium87() -> "Rubidium87":
    from arc import Rubidium87

    return Rubidium87()


@dataclass
class AtomicSystem:
    """Container for Rb87 atomic parameters and precomputed Hamiltonians.

    Created via :func:`create_our_system` or :func:`create_lukin_system`.
    Holds all physical constants, precomputed Hamiltonians, decay rates,
    and branching ratios.
    """

    param_set: str
    ryd_level: int
    atom: "Rubidium87" = field(repr=False)

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
    n_atoms: int = 2

    def meta(self, name: str, default=None):
        """Return a named physical parameter for protocol code."""
        return getattr(self, name, default)


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
    from exact.legacy.hamiltonian_builders import (
        _build_tq_ham_const, _tq_ham_420_our, _tq_ham_1013_our,
        _build_zero_state_lightshift,
    )
    from ryd_gate.physics.branching import _rydberg_branching_ratios, _mid_branching_ratios

    atom = _rubidium87()
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
    from exact.legacy.hamiltonian_builders import (
        _build_tq_ham_const, _tq_ham_420_lukin, _tq_ham_1013_lukin,
        _build_zero_state_lightshift,
    )
    from ryd_gate.physics.branching import _rydberg_branching_ratios, _mid_branching_ratios

    atom = _rubidium87()
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
