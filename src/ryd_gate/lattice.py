"""Atom register: 2D positions and shape constructors (pure geometry).

Only "where the atoms sit" — no ids, no sublattice signs, no derived spacing,
no interactions, no level structure. Site order is exactly the row order of
``coords`` (site ``i`` is ``coords[i]``); that same integer order indexes
initial-state labels, observable ``site`` arguments, amplitude label sequences
and sample-outcome tuples.

Energy structures live in ``core/system.py`` /
``core/level_structures.py`` / ``core/physical_models.py`` (the last also holding
the van der Waals coupling computation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = ["Register"]


@dataclass(frozen=True, slots=True)
class _RegisterOrigin:
    """Private factory provenance (never public; PEPS31 grid-shape source).

    ``grid_shape`` is the Cartesian ``(rows, cols)`` for chain/rectangle/square
    and ``None`` for direct construction and triangular. It carries no spacing,
    edge, adjacency, boundary or interaction data.
    """

    factory: Literal["custom", "chain", "rectangle", "square", "triangular"]
    grid_shape: tuple[int, int] | None


class Register:
    """N-atom 2D register: read-only ``(N, 2)`` micrometre coordinates.

    Constructed directly from coordinates or via the shape factories
    :meth:`chain` / :meth:`rectangle` / :meth:`square` / :meth:`triangular`.
    ``coords`` is a truly read-only finite ``(N, 2)`` array (a view of a frozen
    base, so it cannot be re-enabled for writing); ``N`` is derived from the row
    count. There is no ``ids``, ``sublattice`` or derived ``spacing_um``: a
    factory ``spacing_um`` is only an input used to generate the coordinates,
    never a stored property.
    """

    __slots__ = ("_coords", "_origin")

    def __init__(self, coords) -> None:
        arr = np.array(coords, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != 2:
            raise ValueError(
                "coords must have shape (N, 2) with N >= 1 (registers are 2D; "
                f"embed out-of-plane thermal motion as noise, not geometry). Got {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("coords must be finite.")
        arr.setflags(write=False)
        self._coords = arr
        self._origin = _RegisterOrigin("custom", None)

    @classmethod
    def _with_origin(cls, coords, origin: _RegisterOrigin) -> "Register":
        """Construct through the same validation/freezing path, then set provenance."""
        register = cls(coords)  # full public validation + coord freeze
        register._origin = origin
        return register

    @property
    def coords(self) -> np.ndarray:
        """Read-only ``(N, 2)`` coordinate array (a frozen view)."""
        view = self._coords[:]
        view.flags.writeable = False  # view of a frozen base: cannot be re-enabled
        return view

    @property
    def N(self) -> int:
        """Number of atoms (derived from the coordinate row count)."""
        return int(self._coords.shape[0])

    # ── Shape factories (spacing_um is an input, not a stored property) ──────

    @classmethod
    def chain(cls, n_atoms: int, spacing_um: float = 4.0) -> "Register":
        """1D chain of ``n_atoms`` along x with the given spacing (in the 2D plane)."""
        _check_positive_int(n_atoms, "n_atoms")
        _check_positive_spacing(spacing_um)
        coords = np.column_stack([
            np.arange(n_atoms, dtype=float) * spacing_um,
            np.zeros(n_atoms, dtype=float),
        ])
        return cls._with_origin(coords, _RegisterOrigin("chain", (int(n_atoms), 1)))

    @classmethod
    def rectangle(cls, rows: int, cols: int, spacing_um: float = 4.0) -> "Register":
        """``rows × cols`` grid, row-major (``i = row * cols + col``)."""
        _check_positive_int(rows, "rows")
        _check_positive_int(cols, "cols")
        _check_positive_spacing(spacing_um)
        coords = _rectangle_coords(rows, cols, spacing_um)
        return cls._with_origin(coords, _RegisterOrigin("rectangle", (int(rows), int(cols))))

    @classmethod
    def square(cls, side: int, spacing_um: float = 4.0) -> "Register":
        """``side × side`` grid (like ``rectangle(side, side)`` but with square provenance)."""
        _check_positive_int(side, "side")
        _check_positive_spacing(spacing_um)
        coords = _rectangle_coords(side, side, spacing_um)
        return cls._with_origin(coords, _RegisterOrigin("square", (int(side), int(side))))

    @classmethod
    def triangular(cls, rows: int, atoms_per_row: int, spacing_um: float = 4.0) -> "Register":
        """Row-staggered triangular lattice (odd rows offset by ``spacing/2`` in x).

        Row pitch ``sqrt(3)/2 * spacing_um``.
        """
        _check_positive_int(rows, "rows")
        _check_positive_int(atoms_per_row, "atoms_per_row")
        _check_positive_spacing(spacing_um)
        coords = []
        for row in range(rows):
            x_offset = 0.5 * spacing_um if (row % 2 == 1) else 0.0
            for col in range(atoms_per_row):
                coords.append([
                    col * spacing_um + x_offset,
                    row * (np.sqrt(3) / 2) * spacing_um,
                ])
        return cls._with_origin(np.asarray(coords, dtype=float), _RegisterOrigin("triangular", None))


def _rectangle_coords(rows: int, cols: int, spacing_um: float) -> np.ndarray:
    return np.array(
        [(r * spacing_um, c * spacing_um) for r in range(rows) for c in range(cols)],
        dtype=float,
    )


def _check_positive_int(value, name: str) -> None:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _check_positive_spacing(spacing_um) -> None:
    try:
        spacing = float(spacing_um)
    except (TypeError, ValueError):
        raise ValueError(f"spacing_um must be a float, got {spacing_um!r}.") from None
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"spacing_um must be positive, got {spacing}.")
