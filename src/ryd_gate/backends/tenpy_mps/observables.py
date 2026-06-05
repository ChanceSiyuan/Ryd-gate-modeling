"""MPS measurement utilities for tensor network simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ryd_gate.backends.tn_common.lattice_spec import TNLatticeSpec


def measure_site_occupations(
    psi: object,
    spec: TNLatticeSpec,
) -> np.ndarray:
    """Measure Rydberg occupation ``<n_r_i>`` in 2D site order.

    Parameters
    ----------
    psi : tenpy.networks.mps.MPS
        MPS state (in snake order).
    spec : TNLatticeSpec

    Returns
    -------
    occ : ndarray, shape (N,)
        Per-site Rydberg occupation in 2D site order.
    """
    if spec.level_structure == "01r":
        occ_snake = np.array(psi.expectation_value("n_r"))
    else:
        sz_vals = psi.expectation_value("Sz")
        occ_snake = 0.5 + np.array(sz_vals)
    # Reorder from snake to 2D
    occ_2d = np.empty(spec.N)
    occ_2d[spec.snake_to_2d] = occ_snake
    return occ_2d


def measure_level_occupations(
    psi: object,
    spec: TNLatticeSpec,
    level: str,
) -> np.ndarray:
    """Measure ``<n_level_i>`` in 2D site order."""
    if level == "r":
        return measure_site_occupations(psi, spec)

    if spec.level_structure == "1r":
        if level == "1":
            return 1.0 - measure_site_occupations(psi, spec)
        if level == "0":
            return np.zeros(spec.N)
        raise ValueError(f"Unknown level for 1r TN lattice: {level!r}")

    if level not in {"0", "1"}:
        raise ValueError(f"Unknown level for 01r TN lattice: {level!r}")
    occ_snake = np.array(psi.expectation_value(f"n_{level}"))
    occ_2d = np.empty(spec.N)
    occ_2d[spec.snake_to_2d] = occ_snake
    return occ_2d


def measure_staggered_magnetization(
    psi: object,
    spec: TNLatticeSpec,
) -> float:
    """Compute m_s = (1/N) sum_i s_i (2<n_i> - 1).

    Parameters
    ----------
    psi : tenpy.networks.mps.MPS
    spec : TNLatticeSpec

    Returns
    -------
    m_s : float
    """
    occ = measure_site_occupations(psi, spec)
    return float(np.sum(spec.sublattice * (2 * occ - 1)) / spec.N)


def measure_mean_rydberg(
    psi: object,
    spec: TNLatticeSpec,
) -> float:
    """Compute mean Rydberg fraction <n> = (1/N) sum_i <n_i>.

    Parameters
    ----------
    psi : tenpy.networks.mps.MPS
    spec : TNLatticeSpec

    Returns
    -------
    n_mean : float
    """
    occ = measure_site_occupations(psi, spec)
    return float(occ.mean())


def measure_sigma_z(
    psi: object,
    spec: TNLatticeSpec,
) -> np.ndarray:
    """Measure per-site ``<sigma_z_i> = 2<n_r_i> - 1`` in 2D site order."""
    return 2.0 * measure_site_occupations(psi, spec) - 1.0


def measure_correlation(
    psi: object,
    spec: TNLatticeSpec,
    i_2d: int,
    j_2d: int,
) -> float:
    """Compute connected correlator <n_i n_j> - <n_i><n_j>.

    Parameters
    ----------
    psi : tenpy.networks.mps.MPS
    spec : TNLatticeSpec
    i_2d, j_2d : int
        Site indices in 2D site order.

    Returns
    -------
    corr : float
    """
    i_1d = int(spec.inv_snake[i_2d])
    j_1d = int(spec.inv_snake[j_2d])
    if i_1d > j_1d:
        i_1d, j_1d = j_1d, i_1d

    # n_i = 0.5 + Sz_i  =>  n_i n_j = (0.5+Szi)(0.5+Szj)
    if spec.level_structure == "01r":
        n_vals = psi.expectation_value("n_r")
        n_i = float(n_vals[i_1d])
        n_j = float(n_vals[j_1d])
        nn = float(psi.expectation_value_term([("n_r", i_1d), ("n_r", j_1d)]))
    else:
        sz_i = float(psi.expectation_value("Sz")[i_1d])
        sz_j = float(psi.expectation_value("Sz")[j_1d])
        n_i = 0.5 + sz_i
        n_j = 0.5 + sz_j
        szsz = psi.expectation_value_term([("Sz", i_1d), ("Sz", j_1d)])
        nn = float(szsz) + 0.5 * (sz_i + sz_j) + 0.25
    return float(nn - n_i * n_j)


def measure_connected_zz(
    psi: object,
    spec: TNLatticeSpec,
    i_2d: int,
    j_2d: int,
) -> float:
    """Compute connected ``<sigma_z_i sigma_z_j> - <sigma_z_i><sigma_z_j>``."""
    return 4.0 * measure_correlation(psi, spec, i_2d, j_2d)


def measure_centerline_connected_zz(
    psi: object,
    spec: TNLatticeSpec,
    *,
    axis: str = "horizontal",
) -> np.ndarray:
    """Measure connected ``C_zz`` from a center site along a center line."""
    from ryd_gate.analysis.spin_observables import line_pairs_from_reference

    return np.array([
        measure_connected_zz(psi, spec, i, j)
        for i, j in line_pairs_from_reference(spec.Lx, spec.Ly, axis=axis)
    ])
