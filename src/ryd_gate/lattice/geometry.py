"""Lattice geometry for Rydberg atom arrays (2-level and 3-level)."""

from __future__ import annotations

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


# ── 3-level geometry (physical coordinates + all-pairs VdW) ──────────


@dataclass(frozen=True)
class LatticeGeometry:
    """N-atom geometry with physical positions and VdW couplings."""

    N: int
    coords: np.ndarray            # (N, 2) positions in μm
    sublattice: np.ndarray        # (N,) checkerboard signs ±1
    vdw_couplings: tuple          # ((i, j, V_ij_rad_s), ...)
    lattice_spacing_um: float


def _compute_vdw_pairs(coords_um, C6, max_range_um=None):
    """Compute all-pairs VdW couplings V_ij = C6 / R_ij^6."""
    N = len(coords_um)
    vdw = []
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(coords_um[i] - coords_um[j])
            if max_range_um is not None and r > max_range_um:
                continue
            vdw.append((i, j, C6 / r**6))
    return tuple(vdw)


def make_3level_square_lattice(
    Lx: int,
    Ly: int,
    spacing_um: float = 3.0,
    C6: float = 2 * np.pi * 874e9,
    max_range_um: float | None = None,
) -> LatticeGeometry:
    """Build a square lattice with physical VdW couplings for 3-level systems.

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions.
    spacing_um : float
        Nearest-neighbor distance in μm.
    C6 : float
        C₆ coefficient in rad/s · μm⁶.
    max_range_um : float or None
        Truncate pairs beyond this distance. None = all pairs.
    """
    sq = make_square_lattice(Lx, Ly)
    coords_um = sq.coords.astype(float) * spacing_um

    return LatticeGeometry(
        N=sq.N, coords=coords_um, sublattice=sq.sublattice,
        vdw_couplings=_compute_vdw_pairs(coords_um, C6, max_range_um),
        lattice_spacing_um=spacing_um,
    )


def make_geometry_from_coords(
    coords_um: np.ndarray,
    C6: float = 2 * np.pi * 874e9,
    sublattice: np.ndarray | None = None,
    max_range_um: float | None = None,
) -> LatticeGeometry:
    """Build geometry from arbitrary atom positions."""
    coords_um = np.asarray(coords_um, dtype=float)
    N = len(coords_um)
    if sublattice is None:
        sublattice = np.zeros(N, dtype=int)

    return LatticeGeometry(
        N=N, coords=coords_um, sublattice=np.asarray(sublattice),
        vdw_couplings=_compute_vdw_pairs(coords_um, C6, max_range_um),
        lattice_spacing_um=float(np.min(np.diff(np.sort(coords_um[:, 0])))) if N > 1 else 0.0,
    )
