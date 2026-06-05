"""Local single-atom Hamiltonian blocks for the physical Rydberg models.

Builds the per-site complex matrices (energies, drives, light shifts) for the
``analog_3`` and Rb87 seven-level (``our`` / ``lukin``) physical models and
registers them on a system's block registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.core.blocks import BlockRegistry
from ryd_gate.core.level_structures import level_structure
from ryd_gate.core.operator_spec import LocalMatrixSumSpec
from ryd_gate.core.rb87_params import _metadata_from_rb87_params, _rb87_physical_params

if TYPE_CHECKING:
    from ryd_gate.core.system import RydbergSystem


def _register_local_matrix_block(
    blocks: BlockRegistry,
    name: str,
    matrix: np.ndarray,
    *,
    hermitian: bool = True,
    description: str = "",
) -> None:
    blocks.register(
        name,
        LocalMatrixSumSpec(np.asarray(matrix, dtype=np.complex128)),
        description=description,
        hermitian=hermitian,
    )


def _apply_analog_3_lattice_blocks(
    model: "RydbergSystem",
    *,
    detuning_sign: int = 1,
    blackmanflag: bool = True,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
    Delta_Hz: float | None = None,
    rabi_420_Hz: float | None = None,
    rabi_1013_Hz: float | None = None,
    **unused,
) -> None:
    _reject_unused(unused)
    ryd_level = 70
    Delta = detuning_sign * 2 * np.pi * (Delta_Hz if Delta_Hz is not None else 9.1e9)
    rabi_420 = 2 * np.pi * (rabi_420_Hz if rabi_420_Hz is not None else 491e6)
    rabi_1013 = 2 * np.pi * (rabi_1013_Hz if rabi_1013_Hz is not None else 491e6)
    rabi_eff = rabi_420 * rabi_1013 / (2 * abs(Delta))
    time_scale = 2 * np.pi / rabi_eff
    mid_state_decay_rate = 1 / 110.7e-9
    ryd_state_decay_rate = 1 / 151.55e-6
    ryd_RD_rate = 1 / 410.41e-6
    ryd_BBR_rate = ryd_state_decay_rate - ryd_RD_rate

    mid_decay = mid_state_decay_rate if enable_intermediate_decay else 0.0
    ryd_decay = ryd_state_decay_rate if enable_rydberg_decay else 0.0

    h_const = np.zeros((3, 3), dtype=np.complex128)
    h_const[1, 1] = Delta - 1j * mid_decay / 2
    h_const[2, 2] = -1j * ryd_decay / 2

    h1013 = np.zeros((3, 3), dtype=np.complex128)
    h1013[2, 1] = rabi_1013 / 2
    h420 = np.zeros((3, 3), dtype=np.complex128)
    h420[1, 0] = rabi_420 / 2
    lightshift_zero = np.zeros((3, 3), dtype=np.complex128)

    _register_local_matrix_block(model.blocks, "H_const", h_const, description="single-atom ger energies")
    _register_local_matrix_block(model.blocks, "H_1013", h1013, hermitian=False, description="static e-r coupling")
    _register_local_matrix_block(model.blocks, "H_1013_conj", h1013.conj().T, hermitian=False)
    _register_local_matrix_block(model.blocks, "drive_420", h420, hermitian=False, description="g-e drive")
    _register_local_matrix_block(model.blocks, "drive_420_dag", h420.conj().T, hermitian=False)
    _register_local_matrix_block(model.blocks, "lightshift_zero", lightshift_zero)

    model.metadata.update(
        {
            "physical_model": "analog_3",
            "rabi_eff": rabi_eff,
            "time_scale": time_scale,
            "t_rise": 20e-9,
            "blackmanflag": blackmanflag,
            "n_atoms": model.N,
            "n_levels": 3,
            "rabi_420": rabi_420,
            "rabi_1013": rabi_1013,
            "rabi_420_garbage": 0.0,
            "rabi_1013_garbage": 0.0,
            "Delta": Delta,
            "v_ryd": _nearest_pair_strength(model.metadata.get("interaction_pairs", ())),
            "v_ryd_garb": 0.0,
            "ryd_level": ryd_level,
            "ryd_state_decay_rate": ryd_state_decay_rate,
            "ryd_RD_rate": ryd_RD_rate,
            "ryd_BBR_rate": ryd_BBR_rate,
            "mid_state_decay_rate": mid_state_decay_rate,
            "ryd_branch": {},
            "mid_branch": {},
            "rydberg_indices": (2,),
            "enable_rydberg_decay": enable_rydberg_decay,
            "enable_intermediate_decay": enable_intermediate_decay,
            "enable_0_scattering": False,
            "enable_polarization_leakage": False,
        }
    )


def _apply_rb87_7_lattice_blocks(
    model: "RydbergSystem",
    param_set: str,
    *,
    detuning_sign: int = 1,
    blackmanflag: bool = True,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
    enable_0_scattering: bool = True,
    enable_polarization_leakage: bool = False,
    **unused,
) -> None:
    _reject_unused(unused)
    physical = _rb87_physical_params(
        param_set,
        detuning_sign=detuning_sign,
        blackmanflag=blackmanflag,
        enable_rydberg_decay=enable_rydberg_decay,
        enable_intermediate_decay=enable_intermediate_decay,
        enable_0_scattering=enable_0_scattering,
        enable_polarization_leakage=enable_polarization_leakage,
    )

    h_const = _rb87_local_h_const(
        physical.Delta,
        physical.ryd_zeeman_shift,
        physical.mid_state_decay_rate if enable_intermediate_decay else 0.0,
        physical.ryd_state_decay_rate if enable_rydberg_decay else 0.0,
    )
    h420 = _rb87_local_h420(param_set, physical.rabi_420, physical.rabi_420_garbage)
    h1013 = _rb87_local_h1013(param_set, physical.rabi_1013, physical.rabi_1013_garbage)
    lightshift_zero = _rb87_local_zero_lightshift(
        param_set,
        physical.Delta,
        physical.rabi_420,
        physical.rabi_420_garbage,
        physical.mid_state_decay_rate,
        enable_intermediate_decay,
        enable_0_scattering,
    )

    _register_local_matrix_block(model.blocks, "H_const", h_const, description="single-atom rb87_7 energies")
    _register_local_matrix_block(model.blocks, "H_1013", h1013, hermitian=False, description="static 1013nm coupling")
    _register_local_matrix_block(model.blocks, "H_1013_conj", h1013.conj().T, hermitian=False)
    _register_local_matrix_block(model.blocks, "drive_420", h420, hermitian=False, description="420nm drive")
    _register_local_matrix_block(model.blocks, "drive_420_dag", h420.conj().T, hermitian=False)
    _register_local_matrix_block(model.blocks, "lightshift_zero", lightshift_zero)

    model.metadata.update(_metadata_from_rb87_params(physical))
    model.metadata.update(
        {
            "physical_model": param_set,
            "n_atoms": model.N,
            "n_sites": model.N,
            "level_structure": "rb87_7",
            "level_spec": level_structure("rb87_7"),
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
    h[2, 2] = Delta - 2 * np.pi * 51e6 - 1j * middecay / 2
    h[3, 3] = Delta - 1j * middecay / 2
    h[4, 4] = Delta + 2 * np.pi * 87e6 - 1j * middecay / 2
    h[5, 5] = -1j * ryddecay / 2
    h[6, 6] = ryd_zeeman_shift - 1j * ryddecay / 2
    return h


def _rb87_local_h420(param_set: str, rabi_420: float, rabi_420_garbage: float) -> np.ndarray:
    from arc.wigner import CG

    h = np.zeros((7, 7), dtype=np.complex128)
    if param_set == "our":
        for row, F in zip((2, 3, 4), (1, 2, 3)):
            h[row, 1] = (
                rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, F, -1)
                + rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, F, -1)
            ) / 2
    else:
        for row, F in zip((2, 3, 4), (1, 2, 3)):
            h[row, 1] = (
                rabi_420 * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, F, 1)
                + rabi_420_garbage * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, F, 1)
            ) / 2
    return h


def _rb87_local_h1013(param_set: str, rabi_1013: float, rabi_1013_garbage: float) -> np.ndarray:
    from arc.wigner import CG

    h = np.zeros((7, 7), dtype=np.complex128)
    if param_set == "our":
        for col, F in zip((2, 3, 4), (1, 2, 3)):
            h[5, col] = (rabi_1013 / 2) * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, F, -1)
            h[6, col] = (rabi_1013_garbage / 2) * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, F, -1)
    else:
        for col, F in zip((2, 3, 4), (1, 2, 3)):
            h[5, col] = (rabi_1013 / 2) * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, F, 1)
            h[6, col] = (rabi_1013_garbage / 2) * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, F, 1)
    return h


def _rb87_local_zero_lightshift(
    param_set: str,
    Delta: float,
    rabi_420: float,
    rabi_420_garbage: float,
    mid_state_decay_rate: float,
    enable_intermediate_decay: bool,
    enable_0_scattering: bool,
) -> np.ndarray:
    from arc.wigner import CG

    E_0 = -2 * np.pi * 6.835e9
    mid_energies = np.array(
        [
            Delta - 2 * np.pi * 51e6,
            Delta,
            Delta + 2 * np.pi * 87e6,
        ],
        dtype=np.float64,
    )
    if param_set == "our":
        cg_ratio_main = CG(1 / 2, -1 / 2, 3 / 2, 1 / 2, 1, 0) / CG(1 / 2, -1 / 2, 3 / 2, 1 / 2, 2, 0)
        cg_ratio_garb = CG(1 / 2, 1 / 2, 3 / 2, -1 / 2, 1, 0) / CG(1 / 2, 1 / 2, 3 / 2, -1 / 2, 2, 0)
        couplings = [
            (
                cg_ratio_main * rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, F, -1)
                + cg_ratio_garb * rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, F, -1)
            )
            / 2
            for F in (1, 2, 3)
        ]
    else:
        cg_ratio_main = CG(1 / 2, 1 / 2, 3 / 2, -1 / 2, 1, 0) / CG(1 / 2, 1 / 2, 3 / 2, -1 / 2, 2, 0)
        cg_ratio_garb = CG(1 / 2, -1 / 2, 3 / 2, 1 / 2, 1, 0) / CG(1 / 2, -1 / 2, 3 / 2, 1 / 2, 2, 0)
        couplings = [
            (
                cg_ratio_main * rabi_420 * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, F, 1)
                + cg_ratio_garb * rabi_420_garbage * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, F, 1)
            )
            / 2
            for F in (1, 2, 3)
        ]

    local = np.zeros((7, 7), dtype=np.complex128)
    total_shift = 0.0
    scatter_rate = 0.0
    gamma = mid_state_decay_rate if enable_intermediate_decay else 0.0
    for idx, (g_i, E_e) in enumerate(zip(couplings, mid_energies), start=2):
        detuning = E_e - E_0
        shift = (np.abs(g_i) ** 2) / detuning
        local[idx, idx] = shift
        total_shift += shift
        scatter_rate += (np.abs(g_i) ** 2) * gamma / (detuning**2)
    local[0, 0] = -total_shift - 1j * scatter_rate / 2 if enable_0_scattering else -total_shift
    return local


def _nearest_pair_strength(pairs: tuple) -> float:
    if not pairs:
        return 0.0
    return float(max(abs(strength) for _, _, strength in pairs))


def _reject_unused(unused: dict) -> None:
    if unused:
        names = ", ".join(sorted(unused))
        raise TypeError(f"Unused physical parameter(s): {names}")
