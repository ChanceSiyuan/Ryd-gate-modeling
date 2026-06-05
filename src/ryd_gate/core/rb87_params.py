"""Rb87 seven-level physical parameter sets and metadata assembly.

Holds the atomic-physics parameters for the ``our`` and ``lukin`` Rb87
configurations (level energies, Rabi frequencies, decay/branching rates) and
the helper that flattens them into a system metadata dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ryd_gate.core.level_structures import DEFAULT_C6


@dataclass(frozen=True)
class _RB87PhysicalParams:
    param_set: str
    ryd_level: int
    Delta: float
    rabi_420: float
    rabi_1013: float
    rabi_eff: float
    time_scale: float
    rabi_420_garbage: float
    rabi_1013_garbage: float
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
    blackmanflag: bool
    enable_rydberg_decay: bool
    enable_intermediate_decay: bool
    enable_0_scattering: bool
    enable_polarization_leakage: bool
    n_levels: int = 7
    rydberg_indices: tuple[int, ...] = (5, 6)
    n_atoms: int = 2


def _rb87_default_c6(param_set: str) -> float:
    if param_set == "lukin":
        return 2 * np.pi * 450e6 * 3.0**6
    return DEFAULT_C6


def _rb87_physical_params(
    param_set: str,
    *,
    detuning_sign: int,
    blackmanflag: bool,
    enable_rydberg_decay: bool,
    enable_intermediate_decay: bool,
    enable_0_scattering: bool,
    enable_polarization_leakage: bool,
) -> _RB87PhysicalParams:
    from arc import Rubidium87
    from ryd_gate.physics.branching import _mid_branching_ratios, _rydberg_branching_ratios

    atom = Rubidium87()
    if param_set == "our":
        ryd_level = 70
        Delta = detuning_sign * 2 * np.pi * 9.1e9
        rabi_420 = 2 * np.pi * 491e6
        rabi_1013 = 2 * np.pi * 185e6
        d_mid_ratio = atom.getDipoleMatrixElement(5, 0, 0.5, 0.5, 6, 1, 1.5, -0.5, -1) / atom.getDipoleMatrixElement(
            5, 0, 0.5, -0.5, 6, 1, 1.5, -1.5, -1
        )
        d_ryd_ratio = atom.getDipoleMatrixElement(
            6, 1, 1.5, -0.5, ryd_level, 0, 0.5, 0.5, 1
        ) / atom.getDipoleMatrixElement(6, 1, 1.5, -1.5, ryd_level, 0, 0.5, -0.5, 1)
        v_ryd = 2 * np.pi * 874e9 / 3**6
        v_ryd_garb = v_ryd
        ryd_zeeman_shift = 2 * np.pi * 56e6 if enable_polarization_leakage else 2 * np.pi * 56e9
        mid_state_decay_rate = 1 / 110.7e-9
        ryd_state_decay_rate = 1 / 151.55e-6
        ryd_RD_rate = 1 / 410.41e-6
        ryd_branch = _rydberg_branching_ratios(atom, ryd_level, "our")
        mid_branch = {F: _mid_branching_ratios(atom, F, mF=-1) for F in (1, 2, 3)}
    elif param_set == "lukin":
        ryd_level = 53
        Delta = detuning_sign * 2 * np.pi * 7.8e9
        rabi_420 = 2 * np.pi * 237e6
        rabi_1013 = 2 * np.pi * 303e6
        d_mid_ratio = atom.getDipoleMatrixElement(5, 0, 0.5, -0.5, 6, 1, 1.5, 0.5, 1) / atom.getDipoleMatrixElement(
            5, 0, 0.5, 0.5, 6, 1, 1.5, 1.5, 1
        )
        d_ryd_ratio = atom.getDipoleMatrixElement(
            6, 1, 1.5, 0.5, ryd_level, 0, 0.5, -0.5, -1
        ) / atom.getDipoleMatrixElement(6, 1, 1.5, 1.5, ryd_level, 0, 0.5, 0.5, -1)
        v_ryd = 2 * np.pi * 450e6
        v_ryd_garb = v_ryd
        ryd_zeeman_shift = 2 * np.pi * 2.4e9 if enable_polarization_leakage else 2 * np.pi * 2.4e12
        mid_state_decay_rate = 1 / 110e-9
        ryd_state_decay_rate = 1 / 88e-6
        ryd_RD_rate = 1 / 147.64e-6
        ryd_branch = _rydberg_branching_ratios(atom, ryd_level, "lukin")
        mid_branch = {F: _mid_branching_ratios(atom, F, mF=1) for F in (1, 2, 3)}
    else:
        raise ValueError(f"Unknown rb87_7 parameter set '{param_set}'.")

    rabi_420_garbage = rabi_420 * d_mid_ratio
    rabi_1013_garbage = rabi_1013 * d_ryd_ratio
    rabi_eff = rabi_420 * rabi_1013 / (2 * abs(Delta))
    time_scale = 2 * np.pi / rabi_eff
    ryd_BBR_rate = ryd_state_decay_rate - ryd_RD_rate

    return _RB87PhysicalParams(
        param_set=param_set,
        ryd_level=ryd_level,
        Delta=Delta,
        rabi_420=rabi_420,
        rabi_1013=rabi_1013,
        rabi_eff=rabi_eff,
        time_scale=time_scale,
        rabi_420_garbage=rabi_420_garbage,
        rabi_1013_garbage=rabi_1013_garbage,
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
        t_rise=20e-9,
        blackmanflag=blackmanflag,
        enable_rydberg_decay=enable_rydberg_decay,
        enable_intermediate_decay=enable_intermediate_decay,
        enable_0_scattering=enable_0_scattering,
        enable_polarization_leakage=enable_polarization_leakage,
    )


def _metadata_from_rb87_params(system: _RB87PhysicalParams) -> dict[str, Any]:
    return {
        "rabi_eff": system.rabi_eff,
        "time_scale": system.time_scale,
        "t_rise": system.t_rise,
        "blackmanflag": system.blackmanflag,
        "n_atoms": system.n_atoms,
        "n_levels": system.n_levels,
        "rabi_420": system.rabi_420,
        "rabi_1013": system.rabi_1013,
        "rabi_420_garbage": system.rabi_420_garbage,
        "rabi_1013_garbage": system.rabi_1013_garbage,
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
        "enable_0_scattering": system.enable_0_scattering,
        "enable_polarization_leakage": system.enable_polarization_leakage,
    }
