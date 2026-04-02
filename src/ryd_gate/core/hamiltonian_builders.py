"""Hamiltonian builder functions for two-atom Rydberg gate models.

Module-level functions with no class dependency. Builds the time-independent
and laser-coupling Hamiltonians for 'our' and 'lukin' parameter sets.
"""

from __future__ import annotations

import numpy as np
from arc.wigner import CG


# ======================================================================
# HAMILTONIAN BUILDERS (module-level, no class dependency)
# ======================================================================


def _build_tq_ham_const(
    Delta: float, v_ryd: float, ryd_zeeman_shift: float,
    middecay: float, ryddecay: float,
) -> np.ndarray:
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
    """Build 420nm coupling for 'our' (sigma- polarization)."""
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
    """Build 420nm coupling for 'lukin' (sigma+ polarization)."""
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
    """Build perturbative light-shift Hamiltonian from |0> -> |e_i> coupling."""
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
