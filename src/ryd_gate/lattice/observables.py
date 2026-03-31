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


# ── 3-level observables ──────────────────────────────────────────────


def precompute_trit_masks(N):
    """Precompute masks for 3-level basis states.

    Returns dict with keys 'g', 'e', 'r', each shape (N, 3^N).
    """
    dim = 3**N
    masks = {level: np.zeros((N, dim), dtype=float) for level in ('g', 'e', 'r')}
    level_map = {0: 'g', 1: 'e', 2: 'r'}
    for idx in range(dim):
        remainder = idx
        for i in range(N):
            power = 3**(N - 1 - i)
            trit = remainder // power
            remainder %= power
            masks[level_map[trit]][i, idx] = 1.0
    return masks


def measure_rydberg_occupation(states, trit_masks):
    """Per-atom Rydberg population for 3-level systems.

    Returns shape (N,) for single state or (n_times, N) for batch.
    """
    mask_r = trit_masks['r']
    if states.ndim == 1:
        return mask_r @ np.abs(states)**2
    else:
        return np.abs(states)**2 @ mask_r.T


def staggered_magnetization(rydberg_occ, sublattice):
    """Compute m_s = (1/N) Σ_i s_i (2·n_r_i - 1)."""
    N = len(sublattice)
    if rydberg_occ.ndim == 1:
        return float(np.sum(sublattice * (2 * rydberg_occ - 1)) / N)
    else:
        return np.sum(sublattice[None, :] * (2 * rydberg_occ - 1), axis=1) / N
