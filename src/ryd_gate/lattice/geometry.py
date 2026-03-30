"""Square lattice geometry for Rydberg atom arrays."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SquareLattice:
    """Immutable description of a 2D square lattice."""

    Lx: int
    Ly: int
    N: int
    coords: np.ndarray          # (N, 2) grid coordinates
    sublattice: np.ndarray      # (N,) checkerboard signs +1/-1
    vdw_pairs: tuple            # ((i, j, V_rel), ...) up to NNN


def make_square_lattice(Lx, Ly):
    """Build a square lattice with sublattice labels and interaction pairs.

    Returns a SquareLattice with vdw_pairs containing (i, j, V_ij/V_nn)
    for pairs up to next-nearest neighbors.
    """
    N = Lx * Ly
    coords = np.array([(ix, iy) for ix in range(Lx) for iy in range(Ly)])
    sublattice = np.array([(-1) ** (ix + iy) for ix, iy in coords])

    vdw_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist_sq = dx * dx + dy * dy
            if dist_sq <= 2.01:  # NN (dist=1) and NNN (dist=sqrt(2))
                vdw_pairs.append((i, j, 1.0 / dist_sq**3))

    return SquareLattice(Lx=Lx, Ly=Ly, N=N, coords=coords,
                         sublattice=sublattice, vdw_pairs=tuple(vdw_pairs))


def is_in_domain(ix, iy, cx, cy, radius):
    """Check if site (ix, iy) is within a square domain of given radius."""
    return abs(ix - cx) <= radius and abs(iy - cy) <= radius
