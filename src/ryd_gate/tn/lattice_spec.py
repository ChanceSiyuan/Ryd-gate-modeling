"""TN-friendly lattice specification without 2^N matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def snake_order_mapping(Lx: int, Ly: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute snake-order mapping for 2D-to-1D MPS ordering.

    Row 0: left-to-right (y=0,1,...,Ly-1),
    Row 1: right-to-left (y=Ly-1,...,1,0), etc.

    Assumes row-major 2D indexing: ``i_2d = ix * Ly + iy``.

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions.

    Returns
    -------
    snake_to_2d : ndarray, shape (N,)
        ``snake_to_2d[i_1d] = i_2d``
    inv_snake : ndarray, shape (N,)
        ``inv_snake[i_2d] = i_1d``
    """
    N = Lx * Ly
    snake_to_2d = np.empty(N, dtype=int)
    pos = 0
    for ix in range(Lx):
        cols = range(Ly) if ix % 2 == 0 else range(Ly - 1, -1, -1)
        for iy in cols:
            snake_to_2d[pos] = ix * Ly + iy
            pos += 1

    inv_snake = np.empty(N, dtype=int)
    inv_snake[snake_to_2d] = np.arange(N)
    return snake_to_2d, inv_snake


@dataclass(frozen=True)
class TNLatticeSpec:
    """TN-friendly lattice description without 2^N matrices.

    Attributes
    ----------
    Lx, Ly : int
        Lattice dimensions.
    N : int
        Total number of sites.
    coords : ndarray, shape (N, 2)
        Grid coordinates (row-major: i = ix*Ly + iy).
    sublattice : ndarray, shape (N,)
        Checkerboard signs: ``(-1)**(ix+iy)``.
    vdw_pairs : tuple of (int, int, float)
        Interaction pairs ``(i, j, V_ij / V_nn)`` up to NNN.
    V_nn : float
        Nearest-neighbor interaction strength.
    Omega : float
        Global Rabi frequency.
    snake_to_2d : ndarray, shape (N,)
        ``snake_to_2d[i_1d] = i_2d``.
    inv_snake : ndarray, shape (N,)
        ``inv_snake[i_2d] = i_1d``.
    bc : str
        Boundary conditions: ``"open"`` or ``"periodic"``.
    """

    Lx: int
    Ly: int
    N: int
    coords: np.ndarray
    sublattice: np.ndarray
    vdw_pairs: tuple
    V_nn: float
    Omega: float
    snake_to_2d: np.ndarray
    inv_snake: np.ndarray
    bc: str = "open"


def create_tn_lattice_spec(
    Lx: int = 4,
    Ly: int = 4,
    V_nn: float = 24.0,
    Omega: float = 1.0,
    bc: str = "open",
) -> TNLatticeSpec:
    """Build a TN-friendly lattice spec reusing geometry conventions.

    Matches the coordinate and sublattice conventions of
    :func:`ryd_gate.lattice.geometry.make_square_lattice` so that
    addressing indices remain consistent across exact and TN paths.
    """
    from ryd_gate.lattice.geometry import make_square_lattice

    sq = make_square_lattice(Lx, Ly)
    snake_to_2d, inv_snake = snake_order_mapping(Lx, Ly)

    return TNLatticeSpec(
        Lx=Lx, Ly=Ly, N=sq.N,
        coords=sq.coords,
        sublattice=sq.sublattice,
        vdw_pairs=sq.vdw_pairs,
        V_nn=V_nn,
        Omega=Omega,
        snake_to_2d=snake_to_2d,
        inv_snake=inv_snake,
        bc=bc,
    )
