"""Finite dense PEPS container for the rydtn engine.

Leg convention (locked, matches YASTN ``_peps.py``): rank-5 site tensor
``A[t, l, b, r, p]`` = (top, left, bottom, right, physical).  Sites are
``(nx, ny)`` = (row, col).  Directions: ``t=(-1,0)``, ``l=(0,-1)``, ``b=(1,0)``,
``r=(0,1)``.  All virtual legs (including at the lattice boundary) have dim 1 in a
product state and grow only through bond updates.
"""

from __future__ import annotations

import numpy as np

from .operators import PEPSOps
from .tensors import ArrayBackend, RydTNError

# direction -> (drow, dcol)
_DIRS = {"t": (-1, 0), "l": (0, -1), "b": (1, 0), "r": (0, 1)}
# virtual leg axis carrying each direction's bond
AXIS = {"t": 0, "l": 1, "b": 2, "r": 3}


class FinitePEPS:
    """Open-boundary square-lattice PEPS of dense rank-5 tensors."""

    def __init__(self, Lx: int, Ly: int, backend: ArrayBackend) -> None:
        self.Lx = int(Lx)
        self.Ly = int(Ly)
        self.backend = backend
        self.tensors: dict[tuple[int, int], object] = {}

    def __getitem__(self, site):
        return self.tensors[tuple(site)]

    def __setitem__(self, site, value) -> None:
        self.tensors[tuple(site)] = value

    def sites(self) -> list[tuple[int, int]]:
        return [(nx, ny) for nx in range(self.Lx) for ny in range(self.Ly)]

    def nn_site(self, site, d):
        dx, dy = _DIRS[d]
        x, y = site[0] + dx, site[1] + dy
        if 0 <= x < self.Lx and 0 <= y < self.Ly:
            return (x, y)
        return None

    def bonds(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Horizontal bonds then vertical bonds (matches YASTN ``bonds()`` order)."""
        h, v = [], []
        for s in self.sites():
            sr = self.nn_site(s, "r")
            if sr is not None:
                h.append((s, sr))
            sb = self.nn_site(s, "b")
            if sb is not None:
                v.append((s, sb))
        return h + v

    def nn_bond_dirn(self, s0, s1) -> str:
        """Bond orientation: 'lr'/'tb' (s0 before s1) or 'rl'/'bt'."""
        if self.nn_site(s0, "r") == tuple(s1):
            return "lr"
        if self.nn_site(s0, "b") == tuple(s1):
            return "tb"
        if self.nn_site(s0, "l") == tuple(s1):
            return "rl"
        if self.nn_site(s0, "t") == tuple(s1):
            return "bt"
        raise RydTNError(f"{s0}, {s1} are not nearest-neighbor sites.")

    def bond_dim(self) -> int:
        out = 1
        for A in self.tensors.values():
            out = max(out, max(int(d) for d in A.shape[:4]))
        return out

    def copy(self) -> "FinitePEPS":
        out = FinitePEPS(self.Lx, self.Ly, self.backend)
        for site, A in self.tensors.items():
            out.tensors[site] = A.clone() if self.backend.kind == "torch" else A.copy()
        return out


def product_peps(payload: dict, ops: PEPSOps, backend: ArrayBackend) -> FinitePEPS:
    """Build a product-state PEPS from the payload (mirror ``_yastn_product_peps``).

    Each site tensor is ``vec`` placed with four dim-1 virtual legs:
    shape ``(1, 1, 1, 1, d)``.
    """
    lat = payload["lattice"]
    Lx, Ly = int(lat["Lx"]), int(lat["Ly"])
    snake_to_2d = np.asarray(lat["snake_to_2d"], dtype=int)
    superposition = payload.get("initial_superposition")
    psi = FinitePEPS(Lx, Ly, backend)
    for pos in range(len(snake_to_2d)):
        site_2d = int(snake_to_2d[pos])
        coord = (site_2d // Ly, site_2d % Ly)
        if superposition is not None:
            vec = ops.superposition_vector(superposition)
        else:
            vec = ops.vector(str(payload["initial_labels_1d"][pos]))
        psi[coord] = backend.asarray(vec.reshape(1, 1, 1, 1, ops.dim))
    if any(psi.tensors.get(s) is None for s in psi.sites()):
        raise RydTNError("product_peps did not initialize some PEPS tensor.")
    return psi
