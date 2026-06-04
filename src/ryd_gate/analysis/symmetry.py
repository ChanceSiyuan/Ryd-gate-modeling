"""Spatial symmetry diagnostics for square-lattice observables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SymmetryError:
    """Absolute and relative D4 symmetry errors."""

    per_site_abs: np.ndarray
    per_site_rel: np.ndarray

    @property
    def max_abs(self) -> float:
        return float(np.max(self.per_site_abs)) if self.per_site_abs.size else 0.0

    @property
    def max_rel(self) -> float:
        return float(np.max(self.per_site_rel)) if self.per_site_rel.size else 0.0


def d4_permutations(Lx: int, Ly: int) -> tuple[np.ndarray, ...]:
    """Return the 8 D4 site permutations for an ``L x L`` row-major grid."""
    if Lx != Ly:
        raise ValueError("D4 symmetry requires a square Lx == Ly grid.")
    L = int(Lx)

    def idx(x: int, y: int) -> int:
        return x * L + y

    transforms = (
        lambda x, y: (x, y),
        lambda x, y: (y, L - 1 - x),
        lambda x, y: (L - 1 - x, L - 1 - y),
        lambda x, y: (L - 1 - y, x),
        lambda x, y: (L - 1 - x, y),
        lambda x, y: (x, L - 1 - y),
        lambda x, y: (y, x),
        lambda x, y: (L - 1 - y, L - 1 - x),
    )

    perms = []
    for transform in transforms:
        perm = np.empty(L * L, dtype=int)
        for x in range(L):
            for y in range(L):
                xp, yp = transform(x, y)
                perm[idx(x, y)] = idx(xp, yp)
        perms.append(perm)
    return tuple(perms)


def d4_symmetry_error(
    observable: np.ndarray,
    Lx: int,
    Ly: int,
    *,
    precision: float = 1e-12,
) -> SymmetryError:
    """Compute the paper-style D4 symmetry error for a per-site observable.

    ``observable`` may be shape ``(N,)`` or ``(n_times, N)``.  The returned
    arrays have the same leading shape.
    """
    obs = np.asarray(observable, dtype=float)
    single = obs.ndim == 1
    if single:
        obs = obs[np.newaxis, :]
    if obs.ndim != 2 or obs.shape[1] != Lx * Ly:
        raise ValueError(
            f"observable must have shape ({Lx * Ly},) or (n_times, {Lx * Ly}); got {observable.shape}."
        )

    perms = d4_permutations(Lx, Ly)
    orbit_values = np.stack([obs[:, perm] for perm in perms], axis=0)
    center = orbit_values[0]
    per_site_abs = np.max(np.abs(orbit_values[1:] - center[np.newaxis, :, :]), axis=0)
    mean_abs = np.mean(np.abs(orbit_values), axis=0)
    denom = np.maximum(mean_abs, float(precision))
    per_site_rel = per_site_abs / denom

    if single:
        return SymmetryError(per_site_abs=per_site_abs[0], per_site_rel=per_site_rel[0])
    return SymmetryError(per_site_abs=per_site_abs, per_site_rel=per_site_rel)


def first_unconverged_time(
    times: np.ndarray,
    symmetry_errors: np.ndarray,
    *,
    threshold: float = 0.4,
) -> float | None:
    """Return the first time whose symmetry error exceeds ``threshold``."""
    ts = np.asarray(times, dtype=float)
    errs = np.asarray(symmetry_errors, dtype=float)
    if ts.shape != errs.shape:
        raise ValueError(f"times and symmetry_errors must have same shape, got {ts.shape} and {errs.shape}.")
    idx = np.flatnonzero(errs > threshold)
    return None if idx.size == 0 else float(ts[int(idx[0])])
