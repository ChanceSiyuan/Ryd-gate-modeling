"""Observable computation for many-body Rydberg states."""

import numpy as np


def precompute_bit_masks(N):
    """Precompute bit masks for fast occupation measurement.

    For the computational basis |b_0 b_1 ... b_{N-1}>, site i is excited
    when bit (N-1-i) of the basis index is 1.

    Returns
    -------
    mask : ndarray, shape (N, 2^N), dtype float64
    """
    dim = 2 ** N
    indices = np.arange(dim, dtype=np.int64)
    mask = np.empty((N, dim), dtype=np.float64)
    for i in range(N):
        mask[i] = (indices >> (N - 1 - i)) & 1
    return mask


def measure_from_states(states, bit_masks, sublattice):
    """Compute observables from state vector(s).

    Parameters
    ----------
    states : ndarray, shape (n_times, dim) or (dim,) for a single state
    bit_masks : ndarray, shape (N, dim)
        From precompute_bit_masks.
    sublattice : ndarray, shape (N,)
        Checkerboard signs from SquareLattice.

    Returns
    -------
    ms : staggered magnetization (scalar or 1D array)
    n_mean : mean Rydberg fraction (scalar or 1D array)
    occ : per-site occupations, shape (N,) or (n_times, N)
    """
    single = states.ndim == 1
    if single:
        states = states[np.newaxis, :]

    probs = np.abs(states) ** 2
    occ = probs @ bit_masks.T

    N = len(sublattice)
    ms = (occ * 2 - 1) @ sublattice / N
    n_mean = occ.mean(axis=1)

    if single:
        return ms[0], n_mean[0], occ[0]
    return ms, n_mean, occ
