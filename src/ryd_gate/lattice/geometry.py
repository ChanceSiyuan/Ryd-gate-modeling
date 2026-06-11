"""Atom register: ids, positions, sublattice signs, and shape constructors.

Only "where the atoms sit" — no energy-level structure, no Hamiltonians,
no interactions. Energy structures live in ``core/system.py`` (and
level_structures/local_blocks) and van der Waals coupling computation lives
in ``core/interactions.py``.

Contents
--------
- :class:`Register`        — the atom register consumed by
  ``RydbergSystem.from_lattice``; constructed via ``Register.chain`` /
  ``Register.square`` / ``Register.rectangle`` / ``Register.triangular`` /
  ``Register.from_coordinates``.
- :class:`RegisterLayout`  — optional trap-pattern provenance metadata.
- :func:`is_in_domain`, :func:`nn_nnn_relative_pairs`,
  :func:`cylinder_nn_nnn_pairs` — internal lattice helpers used by the
  TN/analysis layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ryd_gate.core.serialization import check_schema, json_ready, schema_tag
from ryd_gate.core.validation import ValidationIssue

_LAYOUT_KINDS = ("chain", "square", "rectangle", "triangular", "custom")


@dataclass(frozen=True)
class RegisterLayout:
    """Trap-pattern metadata: which tweezer pattern a register came from.

    Pure provenance in Stage 1 — no atom ids, no level structure, no device.
    Classmethod register constructors never synthesize layouts; a layout is
    attached only explicitly (see ``Register``).
    """

    name: str
    trap_coords_um: tuple[tuple[float, ...], ...]
    kind: Literal["chain", "square", "rectangle", "triangular", "custom"]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("RegisterLayout.name must be a non-empty string.")
        if self.kind not in _LAYOUT_KINDS:
            raise ValueError(
                f"RegisterLayout.kind must be one of {_LAYOUT_KINDS}, got {self.kind!r}."
            )
        coords = tuple(tuple(float(x) for x in c) for c in self.trap_coords_um)
        if not coords:
            raise ValueError("RegisterLayout.trap_coords_um must not be empty.")
        dims = {len(c) for c in coords}
        if len(dims) != 1 or next(iter(dims)) not in (2, 3):
            raise ValueError(
                "RegisterLayout.trap_coords_um entries must all be 2D or all be 3D."
            )
        if not all(np.isfinite(x) for c in coords for x in c):
            raise ValueError("RegisterLayout.trap_coords_um must be finite.")
        object.__setattr__(self, "trap_coords_um", coords)

    def define_register(
        self,
        trap_ids: Sequence[int],
        qubit_ids: Sequence[str] | None = None,
        *,
        center: bool = False,
    ) -> "Register":
        """Fill a subset of traps with atoms and return the resulting register.

        Interop provenance entry point (Decision Log D10): Pulser layouts
        carry a trap → qubit mapping, and this is where it lands. ``center``
        defaults to ``False`` so layout coordinates are preserved exactly.
        The returned register has ``layout=self`` attached and records the
        trap indices in its metadata.
        """
        traps = [int(t) for t in trap_ids]
        if not traps:
            raise ValueError("trap_ids must not be empty.")
        if len(set(traps)) != len(traps):
            raise ValueError(f"trap_ids must be unique, got {traps}.")
        n_traps = len(self.trap_coords_um)
        for trap in traps:
            if trap < 0 or trap >= n_traps:
                raise ValueError(
                    f"trap id {trap} out of range for layout with {n_traps} traps."
                )
        if qubit_ids is not None and len(qubit_ids) != len(traps):
            raise ValueError(
                f"qubit_ids length {len(qubit_ids)} != trap_ids length {len(traps)}."
            )
        coords = [self.trap_coords_um[trap] for trap in traps]
        base = Register.from_coordinates(coords, ids=qubit_ids, center=center)
        return Register(
            N=base.N,
            coords=base.coords,
            sublattice=base.sublattice,
            spacing_um=base.spacing_um,
            ids=base.ids,
            layout=self,
            metadata={"trap_ids": traps},
        )

    def to_dict(self) -> dict:
        return {
            "schema": schema_tag("register-layout"),
            "name": self.name,
            "trap_coords_um": [list(c) for c in self.trap_coords_um],
            "kind": self.kind,
            "metadata": json_ready(dict(self.metadata), "layout.metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegisterLayout":
        check_schema(data, "register-layout")
        return cls(
            name=data["name"],
            trap_coords_um=tuple(tuple(float(x) for x in c) for c in data["trap_coords_um"]),
            kind=data["kind"],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, eq=False)
class Register:
    """N-atom register: ids, positions, sublattice signs, characteristic spacing.

    Pure geometry: no operators, no interactions, no level structure. The
    stable order of ``ids`` defines the site order used by basis states,
    bitstrings, and observables.

    Attributes
    ----------
    N : int
        Number of atoms.
    coords : ndarray, shape (N, 2) or (N, 3)
        Atom positions in microns.
    sublattice : ndarray, shape (N,)
        Checkerboard signs ±1 where applicable (square / chain); 0 for
        geometries without a natural bipartition (triangular, custom).
    spacing_um : float
        Nearest-neighbor spacing in microns.
    ids : tuple[str, ...]
        Unique atom ids in stable order; generated as ``q0..q{N-1}`` when
        omitted.
    layout : RegisterLayout | None
        Optional trap-pattern provenance; ``None`` unless attached explicitly.
    """

    N: int
    coords: np.ndarray
    sublattice: np.ndarray
    spacing_um: float
    ids: tuple[str, ...] | None = None
    layout: RegisterLayout | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.N, (int, np.integer))
            or isinstance(self.N, bool)
            or self.N <= 0
        ):
            raise ValueError(f"N must be a positive integer, got {self.N!r}.")
        object.__setattr__(self, "N", int(self.N))

        coords = np.array(self.coords, dtype=float)
        if coords.ndim != 2 or coords.shape[0] != self.N or coords.shape[1] not in (2, 3):
            raise ValueError(
                f"coords must have shape ({self.N}, 2) or ({self.N}, 3), got {coords.shape}."
            )
        if not np.all(np.isfinite(coords)):
            raise ValueError("coords must be finite.")
        object.__setattr__(self, "coords", coords)

        sublattice = np.array(self.sublattice)
        if sublattice.shape != (self.N,):
            raise ValueError(
                f"sublattice must have shape ({self.N},), got {sublattice.shape}."
            )
        object.__setattr__(self, "sublattice", sublattice)

        try:
            spacing = float(self.spacing_um)
        except (TypeError, ValueError):
            raise ValueError(f"spacing_um must be a float, got {self.spacing_um!r}.") from None
        if not np.isfinite(spacing) or spacing < 0:
            raise ValueError(f"spacing_um must be finite and nonnegative, got {spacing}.")
        object.__setattr__(self, "spacing_um", spacing)

        ids = self.ids
        if ids is None:
            ids = tuple(f"q{i}" for i in range(self.N))
        else:
            ids = tuple(str(atom_id) for atom_id in ids)
            if len(ids) != self.N:
                raise ValueError(f"ids must have length {self.N}, got {len(ids)}.")
            if any(not atom_id for atom_id in ids):
                raise ValueError("ids must be non-empty strings.")
            if len(set(ids)) != len(ids):
                raise ValueError("ids must be unique.")
        object.__setattr__(self, "ids", ids)

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
            N=n_atoms,
            coords=coords,
            sublattice=sublattice,
            spacing_um=float(spacing_um),
            ids=tuple(f"{prefix}{i}" for i in range(n_atoms)),
        )

    @classmethod
    def rectangle(cls, rows: int, cols: int, spacing_um: float = 4.0, prefix: str = "q") -> "Register":
        """rows x cols grid, row-major (``i = row * cols + col``), checkerboard signs.

        Reproduces the coordinates and atom order of the removed
        ``make_square_lattice(Lx=rows, Ly=cols, spacing_um)``.
        """
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
            N=n,
            coords=coords,
            sublattice=sublattice,
            spacing_um=float(spacing_um),
            ids=tuple(f"{prefix}{i}" for i in range(n)),
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

        Reproduces the removed ``make_triangular_lattice(Lx=atoms_per_row,
        Ly=rows, spacing_um)``: row pitch ``sqrt(3)/2 * spacing_um``, zero
        sublattice signs.
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
            N=n,
            coords=np.asarray(coords, dtype=float),
            sublattice=np.zeros(n, dtype=int),
            spacing_um=float(spacing_um),
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
        """Register from arbitrary positions; spacing inferred from sorted x.

        Spacing is the smallest positive difference between sorted
        x-coordinates (the removed ``make_geometry_from_coords`` rule), 0.0
        for a single atom. Unlike that factory, coordinates are centered by
        default (``center=True``).
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
        if sublattice is None:
            sublattice = np.zeros(n, dtype=int)
        if n > 1:
            xs = np.sort(arr[:, 0])
            dx = np.diff(xs)
            dx_pos = dx[dx > 1e-12]
            spacing = float(dx_pos.min()) if dx_pos.size else 0.0
        else:
            spacing = 0.0
        return cls(N=n, coords=arr, sublattice=sublattice, spacing_um=spacing, ids=ids)

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def n_atoms(self) -> int:
        return self.N

    @property
    def dimensions(self) -> int:
        return int(self.coords.shape[1])

    @property
    def coords_array(self) -> np.ndarray:
        return self.coords.copy()

    @property
    def coords_um(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(x) for x in row) for row in self.coords)

    # ── Indexing ────────────────────────────────────────────────────────

    def index(self, atom_id: str) -> int:
        try:
            return self.ids.index(atom_id)
        except ValueError:
            raise KeyError(f"Unknown atom id {atom_id!r}.") from None

    def id_at(self, index: int) -> str:
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
            raise IndexError(f"index must be an integer, got {index!r}.")
        if index < 0 or index >= self.N:
            raise IndexError(f"atom index {index} out of range for N={self.N}.")
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
        if self.dimensions != 2:
            raise NotImplementedError("Register.draw supports 2D registers only in Stage 1.")
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
            for atom_id, x, y in zip(self.ids, xs, ys):
                ax.annotate(atom_id, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_aspect("equal")
        ax.set_title(f"Register ({self.N} atoms)")
        if show:
            plt.show()
        return fig

    # ── Validation / serialization ──────────────────────────────────────

    def validate(self, device: Any) -> list[ValidationIssue]:
        """Delegate to ``device.validate_register(self)`` (rules live on the device)."""
        return device.validate_register(self)

    def to_dict(self) -> dict:
        return {
            "schema": schema_tag("register"),
            "ids": list(self.ids),
            "coords_um": [list(map(float, row)) for row in self.coords],
            "sublattice": self.sublattice.tolist(),
            "spacing_um": float(self.spacing_um),
            "layout": self.layout.to_dict() if self.layout is not None else None,
            "metadata": json_ready(dict(self.metadata), "register.metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Register":
        check_schema(data, "register")
        ids = tuple(data["ids"])
        layout_data = data.get("layout")
        layout = RegisterLayout.from_dict(layout_data) if layout_data is not None else None
        return cls(
            N=len(ids),
            coords=data["coords_um"],
            sublattice=data["sublattice"],
            spacing_um=data["spacing_um"],
            ids=ids,
            layout=layout,
            metadata=dict(data.get("metadata", {})),
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
