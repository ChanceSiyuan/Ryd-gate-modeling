"""Rydberg-Rydberg interactions (van der Waals, dipolar, ...).

Currently provides :func:`vdw_couplings` for the standard isotropic VdW
pair sum V_ij = C6 / R_ij^6 with optional range truncation.

Lives in ``core/`` (not ``lattice/``) because computing interaction
strengths is physics; the lattice package is reserved for pure geometry.
"""

from __future__ import annotations

import numpy as np


def vdw_couplings(
    coords_um: np.ndarray,
    C6: float,
    max_range_um: float | None = None,
) -> tuple:
    """Compute all-pairs van der Waals couplings ``V_ij = C6 / R_ij^6``.

    Parameters
    ----------
    coords_um : ndarray, shape (N, 2) or (N, 3)
        Atom positions in microns.
    C6 : float
        Isotropic VdW coefficient in rad/s · μm^6.
    max_range_um : float or None
        If given, omit pairs with separation > max_range_um.

    Returns
    -------
    tuple of (i, j, V_ij)
        Upper-triangular list of pairs with V_ij in rad/s.
    """
    coords_um = np.asarray(coords_um, dtype=float)
    N = len(coords_um)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            r = float(np.linalg.norm(coords_um[i] - coords_um[j]))
            if max_range_um is not None and r > max_range_um:
                continue
            pairs.append((i, j, C6 / r ** 6))
    return tuple(pairs)
