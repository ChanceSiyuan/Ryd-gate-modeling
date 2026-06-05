"""Pure lattice geometry: shapes, coordinates, and sublattice labels.

Only "where the atoms sit" — no energy-level structure, no Hamiltonians,
no interactions. Energy structures live in ``core/system.py`` (and level_structures/local_blocks) and
van der Waals coupling computation lives in ``core/interactions.py``.

Available shapes
----------------
- :func:`make_chain`               — 1D chain of N atoms
- :func:`make_square_lattice`      — 2D square grid (Lx x Ly)
- :func:`make_triangular_lattice`  — 2D row-staggered triangular grid (Lx x Ly)
- :func:`make_geometry_from_coords`— arbitrary positions

Helpers
-------
- :func:`is_in_domain`             — square-region membership test
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatticeGeometry:
    """N-atom geometry: positions, sublattice signs, characteristic spacing.

    Pure geometry: no operators, no interactions, no level structure.

    Attributes
    ----------
    N : int
        Number of atoms.
    coords : ndarray, shape (N, 2)
        Atom positions in microns.
    sublattice : ndarray, shape (N,)
        Checkerboard signs ±1 where applicable (square / chain); 0 for
        geometries without a natural bipartition (triangular, custom).
    spacing_um : float
        Nearest-neighbor spacing in microns.
    """

    N: int
    coords: np.ndarray
    sublattice: np.ndarray
    spacing_um: float


def make_chain(
    N: int,
    spacing_um: float = 4.0,
) -> LatticeGeometry:
    """1D chain of N atoms along the x-axis with uniform spacing."""
    coords = np.column_stack([
        np.arange(N, dtype=float) * spacing_um,
        np.zeros(N, dtype=float),
    ])
    sublattice = np.array([(-1) ** i for i in range(N)])
    return LatticeGeometry(
        N=N, coords=coords, sublattice=sublattice, spacing_um=spacing_um,
    )


def make_square_lattice(
    Lx: int,
    Ly: int,
    spacing_um: float = 4.0,
) -> LatticeGeometry:
    """``Lx`` × ``Ly`` square lattice with uniform spacing."""
    coords = np.array(
        [(ix * spacing_um, iy * spacing_um) for ix in range(Lx) for iy in range(Ly)],
        dtype=float,
    )
    sublattice = np.array(
        [(-1) ** (ix + iy) for ix in range(Lx) for iy in range(Ly)]
    )
    return LatticeGeometry(
        N=Lx * Ly, coords=coords, sublattice=sublattice, spacing_um=spacing_um,
    )


def make_triangular_lattice(
    Lx: int,
    Ly: int,
    spacing_um: float = 4.0,
) -> LatticeGeometry:
    """``Lx`` × ``Ly`` triangular lattice (row-staggered).

    Odd rows are offset by ``spacing_um / 2`` in x; row separation is
    ``sqrt(3)/2 * spacing_um`` in y.
    """
    coords = []
    for iy in range(Ly):
        x_offset = 0.5 * spacing_um if (iy % 2 == 1) else 0.0
        for ix in range(Lx):
            coords.append([
                ix * spacing_um + x_offset,
                iy * (np.sqrt(3) / 2) * spacing_um,
            ])
    coords = np.asarray(coords, dtype=float)
    N = len(coords)
    sublattice = np.zeros(N, dtype=int)
    return LatticeGeometry(
        N=N, coords=coords, sublattice=sublattice, spacing_um=spacing_um,
    )


def make_geometry_from_coords(
    coords_um: np.ndarray,
    sublattice: np.ndarray | None = None,
) -> LatticeGeometry:
    """Build a geometry from arbitrary atom positions (Nx2 array in microns)."""
    coords_um = np.asarray(coords_um, dtype=float)
    N = len(coords_um)
    if sublattice is None:
        sublattice = np.zeros(N, dtype=int)
    # Estimate spacing from the smallest pairwise distance along x
    if N > 1:
        xs = np.sort(coords_um[:, 0])
        dx = np.diff(xs)
        dx_pos = dx[dx > 1e-12]
        spacing = float(dx_pos.min()) if dx_pos.size else 0.0
    else:
        spacing = 0.0
    return LatticeGeometry(
        N=N, coords=coords_um, sublattice=np.asarray(sublattice),
        spacing_um=spacing,
    )


def is_in_domain(ix, iy, cx, cy, radius):
    """Check if site (ix, iy) is within a square domain of given radius."""
    return abs(ix - cx) <= radius and abs(iy - cy) <= radius


def nn_nnn_relative_pairs(Lx: int, Ly: int) -> tuple:
    """NN + NNN pair list for an Lx × Ly grid (unit spacing).

    Returns upper-triangular ``(i, j, V_ij / V_nn)`` tuples including
    nearest neighbours (relative strength 1) and next-nearest neighbours
    (relative strength 1/8, since ``sqrt(2)^6 = 8``).
    """
    coords = [(ix, iy) for ix in range(Lx) for iy in range(Ly)]
    N = len(coords)
    pairs = []
    for i in range(N):
        xi, yi = coords[i]
        for j in range(i + 1, N):
            xj, yj = coords[j]
            dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
            if dist_sq <= 2.01:
                pairs.append((i, j, 1.0 / dist_sq ** 3))
    return tuple(pairs)
