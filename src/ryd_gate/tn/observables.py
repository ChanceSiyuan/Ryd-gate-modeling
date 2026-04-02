"""MPS measurement utilities for tensor network simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .lattice_spec import TNLatticeSpec


def measure_site_occupations(
    psi: object,
    spec: TNLatticeSpec,
) -> np.ndarray:
    """Measure <n_i> for each site, returned in 2D site order.

    n_i = 0.5 + Sz_i  (since TeNPy Sz = +/- 0.5).

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
    sz_vals = psi.expectation_value("Sz")
    occ_snake = 0.5 + np.array(sz_vals)
    # Reorder from snake to 2D
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
    sz_i = float(psi.expectation_value("Sz")[i_1d])
    sz_j = float(psi.expectation_value("Sz")[j_1d])
    n_i = 0.5 + sz_i
    n_j = 0.5 + sz_j

    szsz = psi.expectation_value_term(
        [("Sz", i_1d), ("Sz", j_1d)]
    )
    nn = float(szsz) + 0.5 * (sz_i + sz_j) + 0.25
    return float(nn - n_i * n_j)
