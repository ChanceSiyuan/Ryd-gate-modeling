"""Shared observable preflight for the tensor-network engines (O09/E26).

Every ``ObservableExpr`` is a finite sum of terms
``coefficient x product of per-site local matrices`` (``expr._terms``).  Before
any evolution the engines run :func:`preflight_tn_observables`: each expression
must match the system shape, and each term must touch no more distinct sites
than the backend's measurement supports (E26):

- exact / MPS: any finite number of sites;
- PEPS ``belief_propagation``: at most one distinct site per term;
- PEPS ``ctm``: at most two distinct sites per term.

There is no silent BP -> CTM fallback: an unsupported term raises a capability
error before evolving.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ryd_gate.core.observables import ObservableExpr


def preflight_tn_observables(
    exprs: "dict[str, ObservableExpr]",
    *,
    n_sites: int,
    local_dim: int,
    backend: str,
    max_term_sites: int | None,
) -> None:
    """Shape + site-count capability preflight (raises before any evolution)."""
    shape = (int(n_sites), int(local_dim))
    for label, expr in exprs.items():
        if (expr._n_sites, expr._local_dim) != shape:
            raise ValueError(
                f"observables[{label!r}] was built for (n_sites, local_dim)="
                f"{(expr._n_sites, expr._local_dim)} but this system has {shape}; "
                "build expressions from the matching system.observables factory."
            )
        if max_term_sites is None:
            continue
        for term in expr._terms:
            n_touched = len({site for site, _ in term.factors})
            if n_touched > max_term_sites:
                raise ValueError(
                    f"observables[{label!r}] contains a term acting on {n_touched} "
                    f"distinct sites; the {backend} backend supports at most "
                    f"{max_term_sites} distinct site(s) per term (E26). There is no "
                    "silent measurement fallback."
                )
