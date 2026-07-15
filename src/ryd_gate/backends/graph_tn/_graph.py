"""Compiled ``TNTerms.pairs`` → arbitrary-geometry interaction graph. No quimb import.

Accepts ANY register (triangular / random coordinates / grid alike): the graph is
just the set of atom pairs the compiler kept within ``interaction_cutoff_um``.
Self-pairs, out-of-range indices and non-finite coefficients are rejected; exact
zero-coefficient pairs are dropped. Because the graph-PEPS simple-update requires
every site to be covered by at least one edge, an isolated atom (degree 0) is a
capability error.
"""

from __future__ import annotations

from dataclasses import dataclass

from ryd_gate.backends.graph_tn._options import GraphTNError


@dataclass(frozen=True, slots=True)
class _GraphSpec:
    n_sites: int
    edges: tuple[tuple[int, int], ...]        # sorted (i, j) with i < j
    couplings: dict[tuple[int, int], float]   # (i, j) -> V_ij (rad/s)


def build_graph(terms) -> _GraphSpec:
    """Build the arbitrary interaction graph from compiled ``terms.pairs``."""
    n = int(terms.n_sites)
    if n < 2:
        raise GraphTNError(
            f"backend='graph_peps' needs at least 2 interacting atoms; got n_sites={n}."
        )
    couplings: dict[tuple[int, int], float] = {}
    for i, j, V in terms.pairs:
        i, j = int(i), int(j)
        if not (0 <= i < j < n):
            raise GraphTNError(f"invalid Rydberg pair index ({i}, {j}); need 0 <= i < j < N={n}.")
        Vf = float(V)
        if Vf != Vf or Vf in (float("inf"), float("-inf")):
            raise GraphTNError(f"pair ({i}, {j}) has a non-finite interaction coefficient {V!r}.")
        if Vf == 0.0:
            continue
        couplings[(i, j)] = couplings.get((i, j), 0.0) + Vf

    edges = tuple(sorted(couplings))
    covered = {s for e in edges for s in e}
    missing = sorted(set(range(n)) - covered)
    if missing:
        raise GraphTNError(
            f"atoms {missing} have no interaction edge within the cutoff; the graph-PEPS "
            "simple update requires every atom to share at least one edge. Widen "
            "interaction_cutoff_um or use backend='exact_ode'/'mps'."
        )
    return _GraphSpec(n_sites=n, edges=edges, couplings=couplings)
