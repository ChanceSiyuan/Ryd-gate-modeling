"""Register → PEPS lattice spec + Cartesian-NN topology validation. No YASTN import.

Only ``Register.chain`` / ``Register.rectangle`` / ``Register.square`` factory
provenance is accepted (PEPS31); direct ``Register(coords)`` and
``Register.triangular`` are PEPS capability errors even when the coordinates
happen to form a grid — no shape is inferred from floats. Interactions must lie
on Cartesian nearest-neighbour graph edges (PEPS28/29); this is the only place
that filters pairs, and it filters nothing for exact/MPS/DMRG.
"""

from __future__ import annotations

from dataclasses import dataclass

from ryd_gate.backends.peps._numerics import PEPSError

_ACCEPTED_FACTORIES = ("chain", "rectangle", "square")


@dataclass(frozen=True, slots=True)
class _PEPSLatticeSpec:
    shape: tuple[int, int]
    site_to_coord: tuple[tuple[int, int], ...]
    allowed_edges: frozenset[tuple[int, int]]


def peps_lattice_spec(register) -> _PEPSLatticeSpec:
    """Derive the PEPS lattice spec from a register's factory provenance (PEPS31)."""
    origin = register._origin
    if origin.factory not in _ACCEPTED_FACTORIES or origin.grid_shape is None:
        raise PEPSError(
            "backend='peps' / method='peps_imaginary_time' accepts only registers built by "
            "Register.chain(...), Register.rectangle(...) or Register.square(...); this register "
            f"was built by {origin.factory!r}. Use backend='exact_ode' or 'mps' for "
            "direct-coordinate or triangular geometry."
        )
    rows, cols = origin.grid_shape
    if rows * cols != register.N:
        raise PEPSError(
            f"PEPS grid shape {origin.grid_shape} does not match register.N={register.N}."
        )
    site_to_coord = tuple(divmod(i, cols) for i in range(register.N))
    allowed_edges = _cartesian_edges(site_to_coord)
    return _PEPSLatticeSpec(shape=(int(rows), int(cols)), site_to_coord=site_to_coord, allowed_edges=allowed_edges)


def _cartesian_edges(site_to_coord: tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    """Sorted ``(i, j)`` index pairs that are Cartesian nearest neighbours (PEPS29)."""
    n = len(site_to_coord)
    edges = set()
    for i in range(n):
        ri, ci = site_to_coord[i]
        for j in range(i + 1, n):
            rj, cj = site_to_coord[j]
            if abs(ri - rj) + abs(ci - cj) == 1:
                edges.add((i, j))
    return frozenset(edges)


def validate_and_map_pairs(spec: _PEPSLatticeSpec, terms):
    """Validate compiled ``terms.pairs`` against the PEPS graph; return coord-mapped NN bonds.

    Returns ``((coord_i, coord_j, V), ...)`` for each nonzero pair, coefficient
    unchanged. Self-pairs, out-of-range indices, non-finite coefficients, and any
    nonzero non-graph-edge pair are capability/validity errors (PEPS29). Exact
    zero-coefficient pairs are ignored. Nothing is inserted, dropped or truncated.
    """
    if getattr(terms, "n_sites", None) != len(spec.site_to_coord):
        raise PEPSError(
            f"compiled terms.n_sites={getattr(terms, 'n_sites', None)} does not match the "
            f"PEPS lattice ({len(spec.site_to_coord)} sites)."
        )
    out = []
    for i, j, V in terms.pairs:
        i, j = int(i), int(j)
        if not (0 <= i < j < len(spec.site_to_coord)):
            raise PEPSError(f"invalid Rydberg pair index ({i}, {j}); need 0 <= i < j < N.")
        Vf = float(V)
        if Vf != Vf or Vf in (float("inf"), float("-inf")):
            raise PEPSError(f"pair ({i}, {j}) has a non-finite interaction coefficient {V!r}.")
        if Vf == 0.0:
            continue
        if (i, j) not in spec.allowed_edges:
            raise PEPSError(
                f"backend='peps' supports only Cartesian nearest-neighbour interactions; the "
                f"nonzero pair {i}->{spec.site_to_coord[i]} , {j}->{spec.site_to_coord[j]} is not a "
                "grid edge. Use interaction_cutoff_um to keep only nearest-neighbour pairs, or the "
                "exact/MPS backend for longer-range interactions."
            )
        out.append((spec.site_to_coord[i], spec.site_to_coord[j], Vf))
    return tuple(out)
