"""Private shared ObservableExpr preflight for tensor-network engines.

Both TN engines consume the lowered form ``expr._terms`` (a sum of
``coefficient x product of per-site local matrices``).  Before any evolution
they run :func:`preflight_tn_expressions` — the TN mirror of the exact
backend's ``_preflight_expressions``: every expression must match the compiled
lattice shape, and terms must respect the engine's capability floor (MPS
evaluates any finite product; PEPS is limited to local terms, sums, and
two-point products).  :func:`tn_basis` builds the frozen :class:`BasisSpec`
that labels TN results (``EvolutionResult.basis``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryd_gate.core.model import BasisSpec

if TYPE_CHECKING:
    from ryd_gate.backends.tn_common.lattice_spec import TNLatticeSpec
    from ryd_gate.core.observables import ObservableExpr

# Engine capability floors: the maximum number of distinct sites a single
# product term may act on (``None`` = any finite product).  These constants
# are the single source of truth — the engines pass them to
# :func:`preflight_tn_expressions` and the generated capability matrix
# (docs/_scripts/build_capability_matrix.py) derives its observable section
# from them.
MPS_MAX_TERM_SITES: int | None = None
PEPS_MAX_TERM_SITES: int | None = 2


def tn_basis(spec: "TNLatticeSpec") -> BasisSpec:
    """The frozen basis (register site order, level labels) of a TN lattice."""
    d = int(spec.level_spec.local_dim)
    n = int(spec.N)
    return BasisSpec(
        site_labels=tuple(str(i) for i in range(n)),
        local_levels=tuple(spec.level_spec.levels),
        local_dim=d,
        total_dim=d**n,
    )


def preflight_tn_expressions(
    exprs: "dict[str, ObservableExpr]",
    spec: "TNLatticeSpec",
    *,
    backend: str,
    max_term_sites: int | None = None,
) -> None:
    """Shape + capability preflight for TN observable expressions.

    Raises ``ValueError`` before any evolution when an expression was built
    for a differently shaped system, or when a term is a product over more
    distinct sites than the engine supports (``max_term_sites``; ``None``
    means any finite product is evaluable, the MPS floor).
    """
    shape = (int(spec.N), int(spec.level_spec.local_dim))
    for label, expr in exprs.items():
        if (expr._n_sites, expr._local_dim) != shape:
            raise ValueError(
                f"observables[{label!r}] was built for (n_sites, local_dim)="
                f"{(expr._n_sites, expr._local_dim)} but this lattice has "
                f"{shape}; build expressions from the matching system's "
                "system.observables factory."
            )
        if max_term_sites is None:
            continue
        for term in expr._terms:
            if len(term.factors) > max_term_sites:
                raise ValueError(
                    f"observables[{label!r}] contains a product acting on "
                    f"{len(term.factors)} distinct sites; the {backend} backend "
                    f"supports local terms, sums, and products over at most "
                    f"{max_term_sites} sites."
                )
