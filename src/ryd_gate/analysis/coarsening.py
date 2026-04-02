"""Post-processing for coarsening analysis (Manovitz et al.).

Implements single-spin-flip correction and coarse-grained local staggered
magnetization from the Methods section of the paper.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.signal import convolve2d


def build_neighbor_lists(
    coords: np.ndarray,
    nn_tol: float = 1.01,
    nnn_tol: float = 1.42,
) -> tuple[list[list[int]], list[list[int]]]:
    """Build nearest-neighbor and next-nearest-neighbor lists from grid coords.

    Parameters
    ----------
    coords : ndarray, shape (N, 2)
        Integer grid coordinates.
    nn_tol : float
        Distance threshold for nearest neighbors (default 1.01 for dist=1).
    nnn_tol : float
        Distance threshold for next-nearest neighbors (default ~sqrt(2)).

    Returns
    -------
    nn_lists : list of list of int
        nn_lists[i] = indices of nearest neighbors of atom i.
    nnn_lists : list of list of int
        nnn_lists[i] = indices of next-nearest neighbors of atom i.
    """
    N = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, 2)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)

    nn_lists: list[list[int]] = []
    nnn_lists: list[list[int]] = []
    for i in range(N):
        nn = []
        nnn = []
        for j in range(N):
            if i == j:
                continue
            d = dist[i, j]
            if d <= nn_tol:
                nn.append(j)
            elif d <= nnn_tol:
                nnn.append(j)
        nn_lists.append(nn)
        nnn_lists.append(nnn)
    return nn_lists, nnn_lists


def local_staggered_magnetization(
    occ: np.ndarray,
    sublattice: np.ndarray,
    nn_lists: list[list[int]],
) -> np.ndarray:
    r"""Coarse-grained local staggered magnetization (ms.tex method).

    .. math::

        m_i = \frac{(-1)^{x+y}}{N_i} \sum_{\langle j,i \rangle} (n_i - n_j)
            = (-1)^{x+y} \left(n_i - \frac{C_i}{N_i}\right)

    where :math:`C_i = \sum_j n_j` sums over NN neighbors and
    :math:`N_i` is the coordination number of site *i*.

    Parameters
    ----------
    occ : ndarray, shape (N,)
        Per-site occupation (float or binary).
    sublattice : ndarray, shape (N,)
        Checkerboard signs (+1/-1).
    nn_lists : list of list of int
        Nearest-neighbor index lists from :func:`build_neighbor_lists`.

    Returns
    -------
    m : ndarray, shape (N,)
        Continuous local staggered magnetization in [-1, 1].
    """
    N = len(occ)
    m = np.zeros(N, dtype=float)
    for i in range(N):
        Ni = len(nn_lists[i])
        if Ni == 0:
            continue
        Ci = sum(occ[j] for j in nn_lists[i])
        m[i] = sublattice[i] * (occ[i] - Ci / Ni)
    return m


def correct_single_spin_flips(
    occ: np.ndarray,
    sublattice: np.ndarray,
    nn_lists: list[list[int]],
    nnn_lists: list[list[int]],
) -> np.ndarray:
    """Correct single-spin-flip defects in an occupation snapshot.

    A spin flip is identified when an atom's staggered order is opposite
    to that of ALL its nearest AND next-nearest neighbors. Such atoms are
    flipped to match the surrounding bulk order.

    Parameters
    ----------
    occ : ndarray, shape (N,) or (n_snapshots, N)
        Per-site occupation (0 or 1).
    sublattice : ndarray, shape (N,)
        Checkerboard signs (+1/-1).
    nn_lists, nnn_lists : list of list of int
        From :func:`build_neighbor_lists`.

    Returns
    -------
    occ_corrected : ndarray, same shape as occ
        Corrected occupation array.
    """
    single = occ.ndim == 1
    if single:
        occ = occ[np.newaxis, :]

    occ_out = occ.copy()
    N = occ.shape[1]

    for snap in range(occ_out.shape[0]):
        # Staggered sign: +1 if atom matches expected AF order, -1 if opposite
        stag = sublattice * (2 * occ_out[snap] - 1)
        for i in range(N):
            neighbors = nn_lists[i] + nnn_lists[i]
            if not neighbors:
                continue
            sign_i = stag[i]
            # Check if ALL neighbors have opposite staggered sign
            if all(stag[j] * sign_i < 0 for j in neighbors):
                occ_out[snap, i] = 1 - occ_out[snap, i]

    return occ_out[0] if single else occ_out


# Cross kernel for convolution (sums 4 nearest-neighbor occupations)
_CROSS_KERNEL = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]], dtype=float)


def coarsegrained_boundary_mask(
    occ: np.ndarray,
    Lx: int,
    Ly: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify domain boundary atoms via convolution with cross kernel.

    Convolves the Rydberg occupation reshaped to (Lx, Ly) with
    W = [[0,1,0],[1,0,1],[0,1,0]], producing C(x,y) in [0,4].

    Boundary atoms: n=1 and C!=0, or n=0 and C!=4.

    Parameters
    ----------
    occ : ndarray, shape (N,) or (n_snapshots, N)
        Per-site occupation (0 or 1).
    Lx, Ly : int
        Lattice dimensions.

    Returns
    -------
    C : ndarray, same shape as occ
        Convolution values per site (flattened from 2D).
    is_boundary : ndarray of bool, same shape as occ
        True for boundary atoms.
    """
    single = occ.ndim == 1
    if single:
        occ = occ[np.newaxis, :]

    n_snap = occ.shape[0]
    C_out = np.empty_like(occ, dtype=float)
    is_boundary = np.empty_like(occ, dtype=bool)

    for s in range(n_snap):
        grid = occ[s].reshape(Lx, Ly)
        C_grid = convolve2d(grid, _CROSS_KERNEL, mode='same',
                            boundary='fill', fillvalue=0)
        C_flat = C_grid.ravel()
        C_out[s] = C_flat
        # Boundary: excited atom not fully surrounded by same-order neighbors,
        # or ground atom not fully surrounded by opposite-order neighbors
        is_boundary[s] = ((occ[s] == 1) & (C_flat != 0)) | \
                         ((occ[s] == 0) & (C_flat != 4))

    if single:
        return C_out[0], is_boundary[0]
    return C_out, is_boundary


def identify_domains(
    occ_corrected: np.ndarray,
    sublattice: np.ndarray,
    nn_lists: list[list[int]],
) -> np.ndarray:
    """Label connected domains of same AF ordering.

    After single-spin-flip correction, each atom belongs to AF1 or AF2.
    Connected components of the same ordering (via nearest neighbors)
    form domains.

    Parameters
    ----------
    occ_corrected : ndarray, shape (N,)
        Spin-flip-corrected occupation (single snapshot).
    sublattice : ndarray, shape (N,)
        Checkerboard signs (+1/-1).
    nn_lists : list of list of int
        Nearest-neighbor lists.

    Returns
    -------
    labels : ndarray, shape (N,), dtype int
        Domain label per site (0-indexed).
    """
    N = len(occ_corrected)
    # AF type: +1 for AF1-like, -1 for AF2-like
    af_type = sublattice * (2 * occ_corrected - 1)

    labels = -np.ones(N, dtype=int)
    label_id = 0

    for start in range(N):
        if labels[start] >= 0:
            continue
        # BFS from start
        labels[start] = label_id
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in nn_lists[node]:
                if labels[nb] < 0 and af_type[nb] == af_type[node]:
                    labels[nb] = label_id
                    queue.append(nb)
        label_id += 1

    return labels


def domain_area_distribution(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute area-weighted domain size distribution.

    Parameters
    ----------
    labels : ndarray, shape (N,)
        Domain labels from :func:`identify_domains`.

    Returns
    -------
    areas : ndarray
        Unique domain areas (sorted).
    weights : ndarray
        Area-weighted frequencies, normalized to sum to 1.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    # For each unique area, count how many domains have that area,
    # then weight by area
    area_vals, area_counts = np.unique(counts, return_counts=True)
    weighted = area_vals * area_counts
    total = weighted.sum()
    if total > 0:
        weighted = weighted / total
    return area_vals, weighted.astype(float)
