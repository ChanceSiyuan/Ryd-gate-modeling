"""TN-friendly lattice specification without 2^N matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.core.level_structures import LevelStructureSpec

from .sites import resolve_level_structure

if TYPE_CHECKING:
    from ryd_gate.core.physical_models import Analog3Blocks


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


def diagonal_order_mapping(Lx: int, Ly: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute a diagonal snake mapping for rectangular square grids."""
    order: list[int] = []
    for diag in range(Lx + Ly - 1):
        cells = [
            (ix, diag - ix)
            for ix in range(Lx)
            if 0 <= diag - ix < Ly
        ]
        if diag % 2 == 1:
            cells.reverse()
        order.extend(ix * Ly + iy for ix, iy in cells)
    order_to_2d = np.asarray(order, dtype=int)
    inv_order = np.empty(Lx * Ly, dtype=int)
    inv_order[order_to_2d] = np.arange(Lx * Ly)
    return order_to_2d, inv_order


def ordering_mapping(Lx: int, Ly: int, ordering: str = "snake") -> tuple[np.ndarray, np.ndarray]:
    """Return ``(order_to_2d, inv_order)`` for a named MPS mapping."""
    if ordering == "snake":
        return snake_order_mapping(Lx, Ly)
    if ordering in {"diagonal", "diagonal_snake"}:
        return diagonal_order_mapping(Lx, Ly)
    raise ValueError("ordering must be 'snake' or 'diagonal'.")


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
    level_spec : LevelStructureSpec
        Shared central local level structure used by the exact sparse path
        and lowered here to TeNPy site/MPO data.
    interaction_mode : str
        Interaction graph mode: ``"nn"`` for nearest-neighbor only or
        ``"nnn"`` for nearest and next-nearest neighbors, or ``"system"``
        when the pair list is lowered directly from a RydbergSystem.
    ordering : str
        One-dimensional tensor-network site ordering. ``"snake"`` is the
        default; ``"diagonal"`` is useful for MPS convergence checks.
    """

    Lx: int
    Ly: int
    N: int
    coords: np.ndarray
    sublattice: np.ndarray
    vdw_pairs: tuple
    V_nn: float
    Omega: float
    level_spec: LevelStructureSpec
    snake_to_2d: np.ndarray
    inv_snake: np.ndarray
    bc: str = "open"
    interaction_mode: str = "nnn"
    ordering: str = "snake"
    bc_y: str = "open"
    local_blocks: "Analog3Blocks | None" = None

    @property
    def level_structure(self) -> str:
        """Name of the shared central local level structure."""
        return self.level_spec.name


def create_tn_lattice_spec(
    Lx: int = 4,
    Ly: int = 4,
    V_nn: float = 24.0,
    Omega: float = 1.0,
    bc: str = "open",
    level_structure: str | LevelStructureSpec = "1r",
    interaction_mode: str = "nnn",
    ordering: str = "snake",
    bc_y: str = "open",
) -> TNLatticeSpec:
    """Build a TN-friendly lattice spec reusing geometry conventions.

    Matches the coordinate and sublattice conventions of
    :meth:`ryd_gate.lattice.Register.rectangle` (with unit
    spacing) and the NN/NNN VdW convention of
    :func:`ryd_gate.lattice.nn_nnn_relative_pairs`, so that
    addressing indices and interactions stay consistent across the
    exact and TN paths.
    """
    from ryd_gate.lattice import (
        Register,
        cylinder_nn_nnn_pairs,
        nn_nnn_relative_pairs,
    )

    level_spec = resolve_level_structure(level_structure)
    local_blocks = None
    if level_spec.name == "analog_3":
        from ryd_gate.core.physical_models import analog_3_local_blocks

        local_blocks = analog_3_local_blocks()
    if interaction_mode not in {"nn", "nnn"}:
        raise ValueError("TN lattice interaction_mode must be 'nn' or 'nnn'.")
    if bc_y not in {"open", "periodic"}:
        raise ValueError("TN lattice bc_y must be 'open' or 'periodic'.")

    geom = Register.rectangle(Lx, Ly, spacing_um=1.0)
    if bc_y == "periodic":
        vdw_pairs = cylinder_nn_nnn_pairs(Lx, Ly)  # open x, periodic y (cylinder)
    else:
        vdw_pairs = nn_nnn_relative_pairs(Lx, Ly)
    if interaction_mode == "nn":
        vdw_pairs = tuple(pair for pair in vdw_pairs if np.isclose(pair[2], 1.0))
    snake_to_2d, inv_snake = ordering_mapping(Lx, Ly, ordering)

    return TNLatticeSpec(
        Lx=Lx, Ly=Ly, N=geom.N,
        coords=geom.coords,
        sublattice=geom.sublattice,
        vdw_pairs=vdw_pairs,
        V_nn=V_nn,
        Omega=Omega,
        level_spec=level_spec,
        snake_to_2d=snake_to_2d,
        inv_snake=inv_snake,
        bc=bc,
        interaction_mode=interaction_mode,
        ordering=ordering,
        bc_y=bc_y,
        local_blocks=local_blocks,
    )
