"""Atom register: ids, positions, sublattice signs, and shape constructors.

Only "where the atoms sit" — no energy-level structure, no Hamiltonians,
no interactions. Energy structures live in ``core/system.py`` (and
``core/level_structures.py`` / ``core/physical_models.py``, the latter also
holding the van der Waals coupling computation).

Contents
--------
- :class:`Register`        — the atom register consumed by
  :class:`~ryd_gate.core.system.RydbergSystem`; constructed via
  ``Register.chain`` / ``Register.square`` / ``Register.rectangle`` /
  ``Register.triangular`` / ``Register.from_coordinates``.
- :func:`is_in_domain`, :func:`nn_nnn_relative_pairs`,
  :func:`cylinder_nn_nnn_pairs` — internal lattice helpers used by the
  TN layers.
- :func:`plot_spatial_rydberg` — visualization of physics quantities on
  lattice coordinates (matplotlib imported lazily inside the function).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, eq=False)
class Register:
    """N-atom register: positions, ids, and optional sublattice signs.

    Pure geometry: no operators, no interactions, no level structure. The
    stable order of ``ids`` defines the site order used by basis states,
    bitstrings, and observables.  Everything else is derived: ``N`` from the
    coordinate count and ``spacing_um`` from the smallest nonzero Euclidean
    pair distance.

    Attributes
    ----------
    coords : ndarray, shape (N, 2) or (N, 3)
        Atom positions in microns.
    ids : tuple[str, ...]
        Unique atom ids in stable order; generated as ``q0..q{N-1}`` when
        omitted.
    sublattice : ndarray, shape (N,)
        Checkerboard signs ±1 where applicable (square / chain); 0 for
        geometries without a natural bipartition (triangular, custom).
    """

    coords: np.ndarray
    ids: tuple[str, ...] | None = None
    sublattice: np.ndarray | None = None

    def __post_init__(self) -> None:
        coords = np.array(self.coords, dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 1 or coords.shape[1] not in (2, 3):
            raise ValueError(
                f"coords must have shape (N, 2) or (N, 3) with N >= 1, got {coords.shape}."
            )
        if not np.all(np.isfinite(coords)):
            raise ValueError("coords must be finite.")
        object.__setattr__(self, "coords", coords)
        n = coords.shape[0]

        sublattice = self.sublattice
        if sublattice is None:
            sublattice = np.zeros(n, dtype=int)
        else:
            sublattice = np.array(sublattice)
            if sublattice.shape != (n,):
                raise ValueError(
                    f"sublattice must have shape ({n},), got {sublattice.shape}."
                )
        object.__setattr__(self, "sublattice", sublattice)

        ids = self.ids
        if ids is None:
            ids = tuple(f"q{i}" for i in range(n))
        else:
            ids = tuple(str(atom_id) for atom_id in ids)
            if len(ids) != n:
                raise ValueError(f"ids must have length {n}, got {len(ids)}.")
            if any(not atom_id for atom_id in ids):
                raise ValueError("ids must be non-empty strings.")
            if len(set(ids)) != len(ids):
                raise ValueError("ids must be unique.")
        object.__setattr__(self, "ids", ids)

    # ── Derived geometry ────────────────────────────────────────────────

    @property
    def N(self) -> int:
        """Number of atoms (derived from the coordinate count)."""
        return int(self.coords.shape[0])

    @property
    def spacing_um(self) -> float:
        """Characteristic spacing: the smallest nonzero Euclidean pair distance.

        ``0.0`` for a single atom.
        """
        if self.N < 2:
            return 0.0
        dists = self.distances_um()
        positive = dists[dists > 1e-12]
        return float(positive.min()) if positive.size else 0.0

    # ── Constructors ────────────────────────────────────────────────────

    @classmethod
    def chain(cls, n_atoms: int, spacing_um: float = 4.0, prefix: str = "q") -> "Register":
        """1D chain along x with alternating ``(-1)**i`` sublattice signs."""
        _check_positive_int(n_atoms, "n_atoms")
        _check_positive_spacing(spacing_um)
        _check_prefix(prefix)
        coords = np.column_stack([
            np.arange(n_atoms, dtype=float) * spacing_um,
            np.zeros(n_atoms, dtype=float),
        ])
        sublattice = np.array([(-1) ** i for i in range(n_atoms)])
        return cls(
            coords=coords,
            ids=tuple(f"{prefix}{i}" for i in range(n_atoms)),
            sublattice=sublattice,
        )

    @classmethod
    def rectangle(cls, rows: int, cols: int, spacing_um: float = 4.0, prefix: str = "q") -> "Register":
        """rows x cols grid, row-major (``i = row * cols + col``), checkerboard signs."""
        _check_positive_int(rows, "rows")
        _check_positive_int(cols, "cols")
        _check_positive_spacing(spacing_um)
        _check_prefix(prefix)
        coords = np.array(
            [(r * spacing_um, c * spacing_um) for r in range(rows) for c in range(cols)],
            dtype=float,
        )
        sublattice = np.array([(-1) ** (r + c) for r in range(rows) for c in range(cols)])
        n = rows * cols
        return cls(
            coords=coords,
            ids=tuple(f"{prefix}{i}" for i in range(n)),
            sublattice=sublattice,
        )

    @classmethod
    def square(cls, side: int, spacing_um: float = 4.0, prefix: str = "q") -> "Register":
        """side x side grid; equal to ``rectangle(side, side, spacing_um, prefix)``."""
        return cls.rectangle(side, side, spacing_um, prefix)

    @classmethod
    def triangular(
        cls, rows: int, atoms_per_row: int, spacing_um: float = 4.0, prefix: str = "q"
    ) -> "Register":
        """Row-staggered triangular lattice (odd rows offset by ``spacing/2`` in x).

        Row pitch ``sqrt(3)/2 * spacing_um``, zero sublattice signs.
        """
        _check_positive_int(rows, "rows")
        _check_positive_int(atoms_per_row, "atoms_per_row")
        _check_positive_spacing(spacing_um)
        _check_prefix(prefix)
        coords = []
        for row in range(rows):
            x_offset = 0.5 * spacing_um if (row % 2 == 1) else 0.0
            for col in range(atoms_per_row):
                coords.append([
                    col * spacing_um + x_offset,
                    row * (np.sqrt(3) / 2) * spacing_um,
                ])
        n = rows * atoms_per_row
        return cls(
            coords=np.asarray(coords, dtype=float),
            ids=tuple(f"{prefix}{i}" for i in range(n)),
        )

    @classmethod
    def from_coordinates(
        cls,
        coords,
        ids: Sequence[str] | None = None,
        prefix: str = "q",
        center: bool = True,
        sublattice=None,
    ) -> "Register":
        """Register from arbitrary positions (e.g. any trap subset).

        Coordinates are centered by default; spacing and ``N`` are derived
        from the coordinates like every register.
        """
        arr = np.array(coords, dtype=float)
        if arr.size == 0:
            raise ValueError("coords must not be empty.")
        if arr.ndim != 2 or arr.shape[1] not in (2, 3):
            raise ValueError(f"coords must be (N, 2) or (N, 3) array-like, got shape {arr.shape}.")
        n = arr.shape[0]
        if center:
            arr = arr - arr.mean(axis=0)
        if ids is None:
            _check_prefix(prefix)
            ids = tuple(f"{prefix}{i}" for i in range(n))
        else:
            ids = tuple(ids)
        return cls(coords=arr, ids=ids, sublattice=sublattice)

    # ── Indexing ────────────────────────────────────────────────────────

    def index(self, atom_id: str) -> int:
        assert self.ids is not None  # normalized in __post_init__
        try:
            return self.ids.index(atom_id)
        except ValueError:
            raise KeyError(f"Unknown atom id {atom_id!r}.") from None

    def id_at(self, index: int) -> str:
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
            raise IndexError(f"index must be an integer, got {index!r}.")
        if index < 0 or index >= self.N:
            raise IndexError(f"atom index {index} out of range for N={self.N}.")
        assert self.ids is not None  # normalized in __post_init__
        return self.ids[int(index)]

    # ── Geometry queries ────────────────────────────────────────────────

    def distances_um(self) -> np.ndarray:
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def distance_pairs(self, cutoff_um: float | None = None) -> tuple[tuple[int, int, float], ...]:
        if cutoff_um is not None:
            cutoff_um = float(cutoff_um)
            if not np.isfinite(cutoff_um) or cutoff_um < 0:
                raise ValueError(f"cutoff_um must be finite and nonnegative, got {cutoff_um}.")
        dists = self.distances_um()
        pairs = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dij = float(dists[i, j])
                if cutoff_um is None or dij <= cutoff_um:
                    pairs.append((i, j, dij))
        return tuple(pairs)

    def blockade_edges(self, radius_um: float) -> tuple[tuple[int, int], ...]:
        radius_um = float(radius_um)
        if not np.isfinite(radius_um) or radius_um < 0:
            raise ValueError(f"radius_um must be finite and nonnegative, got {radius_um}.")
        return tuple((i, j) for i, j, dij in self.distance_pairs() if dij <= radius_um)

    # ── Drawing ─────────────────────────────────────────────────────────

    def draw(
        self,
        blockade_radius_um: float | None = None,
        show_ids: bool = True,
        show: bool = True,
    ):
        """Plot the register (2D only); returns the matplotlib Figure."""
        if self.coords.shape[1] != 2:
            raise NotImplementedError("Register.draw supports 2D registers only.")
        if blockade_radius_um is not None:
            blockade_radius_um = float(blockade_radius_um)
            if not np.isfinite(blockade_radius_um) or blockade_radius_um <= 0:
                raise ValueError("blockade_radius_um must be a positive float.")

        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, ax = plt.subplots()
        xs, ys = self.coords[:, 0], self.coords[:, 1]
        if blockade_radius_um is not None:
            for i, j in self.blockade_edges(blockade_radius_um):
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="0.7", lw=1.0, zorder=1)
            for x, y in zip(xs, ys):
                ax.add_patch(
                    Circle((x, y), blockade_radius_um / 2, fill=False, ls="--", ec="0.6", lw=0.8)
                )
        ax.scatter(xs, ys, s=60, color="C0", zorder=2)
        if show_ids:
            assert self.ids is not None  # normalized in __post_init__
            for atom_id, x, y in zip(self.ids, xs, ys):
                ax.annotate(atom_id, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_aspect("equal")
        ax.set_title(f"Register ({self.N} atoms)")
        if show:
            plt.show()
        return fig


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


def _check_prefix(prefix) -> None:
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("prefix must be a non-empty string.")


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


def cylinder_nn_nnn_pairs(Lx: int, Ly: int) -> tuple:
    """NN + NNN pair list for an ``Lx`` × ``Ly`` cylinder (open x, periodic y).

    Same convention as :func:`nn_nnn_relative_pairs` (upper-triangular
    ``(i, j, V_ij / V_nn)`` with NN strength 1 and NNN strength 1/8), but the
    y-direction wraps: distances use the minimum-image convention in y so that
    sites at ``iy = Ly-1`` and ``iy = 0`` are nearest neighbours. Intended for
    ``Ly >= 4`` (and even ``Ly`` for a frustration-free checkerboard), matching
    the cylinder geometry used for 2D DMRG finite-size scaling.
    """
    coords = [(ix, iy) for ix in range(Lx) for iy in range(Ly)]
    N = len(coords)
    pairs = []
    for i in range(N):
        xi, yi = coords[i]
        for j in range(i + 1, N):
            xj, yj = coords[j]
            dx = xi - xj
            dy = yi - yj
            dy -= Ly * round(dy / Ly)  # minimum image along the periodic y-axis
            dist_sq = dx * dx + dy * dy
            if dist_sq <= 2.01:
                pairs.append((i, j, 1.0 / dist_sq ** 3))
    return tuple(pairs)


# ── Visualization (matplotlib imported lazily) ───────────────────────────────


def plot_spatial_rydberg(
    coords: np.ndarray,
    rydberg_occ: np.ndarray,
    sublattice: np.ndarray | None = None,
    title: str = "",
    ax=None,
):
    """Plot Rydberg population as colored circles at atom positions.

    Parameters
    ----------
    coords : ndarray, shape (N, 2)
        Atom positions.
    rydberg_occ : ndarray, shape (N,)
        Per-atom Rydberg population.
    sublattice : ndarray or None
        If given, use squares for +1 sublattice, circles for -1.
    title : str
        Plot title.
    ax : Axes or None
        Existing axes. Creates new figure if None.
    """
    import matplotlib.pyplot as plt

    fig: Any
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.figure

    if sublattice is not None:
        for sub_val, marker in [(1, 's'), (-1, 'o')]:
            mask = sublattice == sub_val
            sc = ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=rydberg_occ[mask], cmap='coolwarm',
                vmin=0, vmax=1, s=300, marker=marker,
                edgecolors='black', linewidths=1.0,
            )
    else:
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=rydberg_occ, cmap='coolwarm',
            vmin=0, vmax=1, s=300,
            edgecolors='black', linewidths=1.0,
        )

    fig.colorbar(sc, ax=ax, label=r'$P_r$', shrink=0.8)

    for i, (x, y) in enumerate(coords):
        ax.annotate(f'{rydberg_occ[i]:.2f}', (x, y),
                    ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r'$y$ ($\mu$m)')
    if title:
        ax.set_title(title)
    return fig
