"""Observable measurement helpers for Rydberg simulation results.

Three layers in one module:

- Lattice observables: bit-mask / trit-mask routines that compute per-site
  occupations, staggered magnetization, and similar observables directly
  from state-vector amplitudes (no sparse matrix-vector products). They
  live in ``analysis/`` (not ``lattice``) because they assume specific
  local Hilbert dimensions (2-level vs 3-level), and the lattice module is
  reserved for pure geometry.
- Observable-based metrics using ``SystemModel.observables``:
  :func:`measure_observables`, :func:`measure_trajectory`,
  :func:`state_overlap`, :func:`norm_squared`.
- Spin observables and benchmark metrics for two-level Rydberg lattices:
  sigma_z / C_zz conversions, center-line site helpers, and paper-style
  epsilon_z / epsilon_zz discrepancies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ryd_gate.core.model import SystemModel
    from ryd_gate.ir import EvolutionResult

# ─────────────────────────────────────────────────────────────────────
# 2-level (|g>, |r>) observables
# ─────────────────────────────────────────────────────────────────────


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
        From :func:`precompute_bit_masks`.
    sublattice : ndarray, shape (N,)
        Checkerboard signs from lattice geometry.

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


# ─────────────────────────────────────────────────────────────────────
# 3-level (g, e, r) observables
# ─────────────────────────────────────────────────────────────────────


def precompute_trit_masks(N):
    """Precompute masks for 3-level basis states.

    Returns dict with keys 'g', 'e', 'r', each shape (N, 3^N).
    """
    dim = 3 ** N
    masks = {level: np.zeros((N, dim), dtype=float) for level in ('g', 'e', 'r')}
    level_map = {0: 'g', 1: 'e', 2: 'r'}
    for idx in range(dim):
        remainder = idx
        for i in range(N):
            power = 3 ** (N - 1 - i)
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
        return mask_r @ np.abs(states) ** 2
    else:
        return np.abs(states) ** 2 @ mask_r.T


def staggered_magnetization(rydberg_occ, sublattice):
    """Compute m_s = (1/N) Σ_i s_i (2·n_r_i - 1)."""
    N = len(sublattice)
    if rydberg_occ.ndim == 1:
        return float(np.sum(sublattice * (2 * rydberg_occ - 1)) / N)
    else:
        return np.sum(sublattice[None, :] * (2 * rydberg_occ - 1), axis=1) / N


# ─────────────────────────────────────────────────────────────────────
# Observable-registry metrics (SystemModel.observables)
# ─────────────────────────────────────────────────────────────────────


def measure_observables(
    model: SystemModel,
    result: EvolutionResult,
    observable_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute expectation values of observables on the final state.

    Parameters
    ----------
    model : SystemModel
        System model with observable registry.
    result : EvolutionResult
        Evolution result containing psi_final.
    observable_names : list of str, optional
        Which observables to measure. None measures all.

    Returns
    -------
    dict mapping observable names to expectation values.
    """
    if observable_names is None:
        return model.observables.measure_all(result.psi_final)
    return {
        name: model.observables.measure(name, result.psi_final)
        for name in observable_names
    }


def measure_trajectory(
    model: SystemModel,
    states: np.ndarray,
    observable_names: list[str],
) -> dict[str, np.ndarray]:
    """Measure observables at each time step of a state trajectory.

    Parameters
    ----------
    model : SystemModel
        System model with observable registry.
    states : ndarray, shape (dim, n_t) or (n_t, dim)
        State trajectory. Column-major ``(dim, n_t)`` from
        ``DenseODEBackend``, row-major ``(n_t, dim)`` from
        ``SparseExpmBackend``.
    observable_names : list of str
        Observable names registered in ``model.observables``.

    Returns
    -------
    dict mapping observable names to ndarray of shape (n_t,).
    """
    # Detect layout: (dim, n_t) vs (n_t, dim)
    if states.shape[0] == model.basis.total_dim and states.ndim == 2:
        get_psi = lambda k: states[:, k]
        n_t = states.shape[1]
    else:
        get_psi = lambda k: states[k]
        n_t = states.shape[0]

    # ``RydbergSystem.expectation`` resolves both symbolic operator specs and
    # materialized matrices; fall back to the registry for plain models.
    measure = getattr(model, "expectation", None) or model.observables.measure

    result = {name: np.empty(n_t) for name in observable_names}
    for k in range(n_t):
        psi = get_psi(k)
        for name in observable_names:
            result[name][k] = measure(name, psi)
    return result


def state_overlap(psi: np.ndarray, target: np.ndarray) -> float:
    """Compute |<target|psi>|^2."""
    return float(abs(np.vdot(target, psi)) ** 2)


def norm_squared(psi: np.ndarray) -> float:
    """Compute <psi|psi> (useful for non-Hermitian evolution with decay)."""
    return float(np.real(np.vdot(psi, psi)))


# ─────────────────────────────────────────────────────────────────────
# Spin observables / benchmark metrics (two-level Rydberg lattices)
# ─────────────────────────────────────────────────────────────────────


def sigma_z_from_rydberg_occ(occ: np.ndarray) -> np.ndarray:
    """Convert Rydberg occupation ``<n_r>`` to ``<sigma_z>``."""
    return 2.0 * np.asarray(occ, dtype=float) - 1.0


def connected_zz_from_connected_nn(connected_nn: np.ndarray) -> np.ndarray:
    """Convert connected ``n_i n_j`` correlations to connected ``sigma_z sigma_z``."""
    return 4.0 * np.asarray(connected_nn, dtype=float)


def center_line_sites(Lx: int, Ly: int, *, axis: str = "horizontal") -> np.ndarray:
    """Return site indices on a center-most line of a row-major square grid."""
    if axis not in {"horizontal", "vertical"}:
        raise ValueError("axis must be 'horizontal' or 'vertical'.")
    if axis == "horizontal":
        ix = Lx // 2
        return np.array([ix * Ly + iy for iy in range(Ly)], dtype=int)
    iy = Ly // 2
    return np.array([ix * Ly + iy for ix in range(Lx)], dtype=int)


def center_reference_site(Lx: int, Ly: int) -> int:
    """Return one center-most site for an open square/rectangular grid."""
    return (Lx // 2) * Ly + (Ly // 2)


def line_pairs_from_reference(
    Lx: int,
    Ly: int,
    *,
    reference: int | None = None,
    axis: str = "horizontal",
) -> list[tuple[int, int]]:
    """Pair a reference site with all other sites on the chosen center line."""
    ref = center_reference_site(Lx, Ly) if reference is None else int(reference)
    return [(ref, int(site)) for site in center_line_sites(Lx, Ly, axis=axis) if int(site) != ref]


def epsilon_z(
    sigma_z_a: np.ndarray,
    sigma_z_b: np.ndarray,
    *,
    L: int | None = None,
) -> float:
    """Paper-style discrepancy for local magnetization profiles.

    The paper uses ``2 * sum_i |z_i^A - z_i^B| / L`` over a line of length
    ``L``. If ``L`` is omitted, the length of the supplied vector is used.
    """
    a = np.asarray(sigma_z_a, dtype=float)
    b = np.asarray(sigma_z_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"sigma_z profiles must have the same shape, got {a.shape} and {b.shape}.")
    denom = int(L) if L is not None else a.size
    if denom <= 0:
        raise ValueError("L must be positive.")
    return float(2.0 * np.sum(np.abs(a - b)) / denom)


def epsilon_zz(
    corr_a: np.ndarray,
    corr_b: np.ndarray,
    *,
    floor: float = 1e-15,
) -> float:
    """Relative discrepancy for connected-correlation profiles."""
    a = np.asarray(corr_a, dtype=float)
    b = np.asarray(corr_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"correlation profiles must have the same shape, got {a.shape} and {b.shape}.")
    denom = max(float(np.sum(np.abs(a))), floor)
    return float(np.sum(np.abs(a - b)) / denom)
