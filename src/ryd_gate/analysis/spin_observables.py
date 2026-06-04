"""Spin observables and benchmark metrics for two-level Rydberg lattices."""

from __future__ import annotations

import numpy as np


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
