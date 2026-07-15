"""Immutable scalar observable expressions and their system-bound factory.

:class:`ObservableExpr` is the only observable currency in ``ryd_gate``:
``simulate(..., observables={label: expr})`` takes a plain dict of named
expressions, and every expression is built from the read-only factory exposed
as ``system.observables`` (an :class:`ObservableFactory` bound to the system's
level structure / basis)::

    obs = system.observables
    simulate(system, t_eval=t_eval, observables={
        "n_r_total": obs.level_sum("r"),
        "coh_01":    obs.E("0", "1", site=0),
        "P_11":      obs.product_projector(["1", "1"]),
    })

The closed algebra (design: no user matrices, no callables, no vector values):

- ``E(ket, bra, site)`` — transition ``|ket><bra|`` on one site
- ``n(level, site)`` — local level projector
- ``level_sum(level)`` / ``weighted_level_sum(level, weights)``
- ``product_projector(labels)`` — full product-state projector
- ``identity()`` — expectation = raw survival norm ``<psi|psi>``
- ``+``, ``-``, scalar ``*``, operator product ``@``, ``.dagger()``

Private lowered representation (backends may rely on it): every expression is
a finite sum of terms ``coefficient x product of per-site local matrices``,
stored as ``expr._terms`` — a tuple of :class:`_Term` with ``factors`` sorted
by site index and at most one ``(d, d)`` matrix per site.  Expectations are
raw ``<psi|O|psi>`` — complex, never divided by the norm.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np

from ryd_gate.core.model import BasisSpec

__all__ = ["ObservableExpr", "ObservableFactory"]


@dataclass(frozen=True, eq=False)
class _Term:
    """One lowered term: ``coefficient x prod_site M_site`` (private)."""

    coefficient: complex
    factors: tuple[tuple[int, np.ndarray], ...]  # ((site, d x d matrix), ...), sites strictly increasing


def _frozen(matrix: np.ndarray) -> np.ndarray:
    out = np.array(matrix, dtype=complex, copy=True)
    out.setflags(write=False)
    return out


class ObservableExpr:
    """An immutable scalar observable expression.

    Constructed only through :class:`ObservableFactory` (``system.observables``)
    and the closed algebra: ``a + b``, ``a - b``, ``scalar * a``, ``a @ b``
    (operator product), ``a.dagger()``.  Expressions remember the level
    structure shape ``(n_sites, local_dim)`` they were built for; combining
    expressions from differently shaped systems is an error.
    """

    __slots__ = ("_terms", "_n_sites", "_local_dim")

    def __init__(self, terms: tuple[_Term, ...], n_sites: int, local_dim: int) -> None:
        # Private constructor — use ObservableFactory / the algebra.
        object.__setattr__(self, "_terms", tuple(terms))
        object.__setattr__(self, "_n_sites", int(n_sites))
        object.__setattr__(self, "_local_dim", int(local_dim))

    def __setattr__(self, name, value):  # immutability guard
        raise AttributeError("ObservableExpr is immutable.")

    # -- algebra -----------------------------------------------------------

    def _check_compatible(self, other: "ObservableExpr") -> None:
        if not isinstance(other, ObservableExpr):
            raise TypeError(
                f"expected an ObservableExpr, got {type(other).__name__}; build "
                "observables from system.observables (no raw matrices/callables)."
            )
        if (other._n_sites, other._local_dim) != (self._n_sites, self._local_dim):
            raise ValueError(
                "cannot combine ObservableExpr from differently shaped systems: "
                f"(n_sites, local_dim) {(self._n_sites, self._local_dim)} vs "
                f"{(other._n_sites, other._local_dim)}."
            )

    def __add__(self, other: "ObservableExpr") -> "ObservableExpr":
        self._check_compatible(other)
        return ObservableExpr(self._terms + other._terms, self._n_sites, self._local_dim)

    def __sub__(self, other: "ObservableExpr") -> "ObservableExpr":
        self._check_compatible(other)
        return self + (-1.0) * other

    def __neg__(self) -> "ObservableExpr":
        return (-1.0) * self

    def __mul__(self, scalar) -> "ObservableExpr":
        if isinstance(scalar, ObservableExpr) or not isinstance(scalar, numbers.Number):
            raise TypeError(
                "`*` is scalar multiplication only; use `a @ b` for the operator "
                f"product (got {type(scalar).__name__})."
            )
        c = complex(scalar)
        return ObservableExpr(
            tuple(_Term(c * t.coefficient, t.factors) for t in self._terms),
            self._n_sites,
            self._local_dim,
        )

    __rmul__ = __mul__

    def __matmul__(self, other: "ObservableExpr") -> "ObservableExpr":
        """Operator product ``self @ other`` (finite product of expressions)."""
        self._check_compatible(other)
        terms = []
        for a in self._terms:
            for b in other._terms:
                merged: dict[int, np.ndarray] = {site: m for site, m in a.factors}
                for site, m in b.factors:
                    merged[site] = _frozen(merged[site] @ m) if site in merged else m
                factors = tuple(sorted(merged.items()))
                terms.append(_Term(a.coefficient * b.coefficient, factors))
        return ObservableExpr(tuple(terms), self._n_sites, self._local_dim)

    def dagger(self) -> "ObservableExpr":
        """Hermitian adjoint of the expression."""
        terms = tuple(
            _Term(
                np.conj(t.coefficient),
                tuple((site, _frozen(m.conj().T)) for site, m in t.factors),
            )
            for t in self._terms
        )
        return ObservableExpr(terms, self._n_sites, self._local_dim)

    def __repr__(self) -> str:
        return (
            f"ObservableExpr(n_terms={len(self._terms)}, n_sites={self._n_sites}, "
            f"local_dim={self._local_dim})"
        )


class ObservableFactory:
    """Immutable, read-only :class:`ObservableExpr` factory bound to a basis.

    Exposed as ``system.observables``.  There is no registration and no
    string-name router: every observable is an expression built here.
    """

    __slots__ = ("_basis",)

    def __init__(self, basis: BasisSpec) -> None:
        object.__setattr__(self, "_basis", basis)

    def __setattr__(self, name, value):  # immutability guard
        raise AttributeError("ObservableFactory is read-only.")

    # -- internals -----------------------------------------------------------

    def _expr(self, terms: tuple[_Term, ...]) -> ObservableExpr:
        return ObservableExpr(terms, self._basis.n_sites, self._basis.local_dim)

    def _site(self, site: int) -> int:
        site = int(site)
        if not 0 <= site < self._basis.n_sites:
            raise ValueError(f"site {site} out of range for {self._basis.n_sites} sites.")
        return site

    def _local_E(self, ket: str, bra: str) -> np.ndarray:
        d = self._basis.local_dim
        m = np.zeros((d, d), dtype=complex)
        m[self._basis.level_index(ket), self._basis.level_index(bra)] = 1.0
        m.setflags(write=False)
        return m

    # -- constructors ----------------------------------------------------------

    def E(self, ket: str, bra: str, site: int) -> ObservableExpr:
        """Transition operator ``|ket><bra|`` on ``site``."""
        site = self._site(site)
        return self._expr((_Term(1.0 + 0.0j, ((site, self._local_E(ket, bra)),)),))

    def n(self, level: str, site: int) -> ObservableExpr:
        """Level projector ``|level><level|`` on ``site``."""
        return self.E(level, level, site)

    def level_sum(self, level: str) -> ObservableExpr:
        """Total level population ``sum_i |level><level|_i``."""
        m = self._local_E(level, level)
        terms = tuple(_Term(1.0 + 0.0j, ((i, m),)) for i in range(self._basis.n_sites))
        return self._expr(terms)

    def weighted_level_sum(self, level: str, weights) -> ObservableExpr:
        """Weighted level population ``sum_i w_i |level><level|_i``."""
        w = [float(x) for x in weights]
        if len(w) != self._basis.n_sites:
            raise ValueError(
                f"weighted_level_sum needs {self._basis.n_sites} weights, got {len(w)}."
            )
        m = self._local_E(level, level)
        terms = tuple(
            _Term(complex(wi), ((i, m),)) for i, wi in enumerate(w) if wi != 0.0
        )
        return self._expr(terms)

    def product_projector(self, labels) -> ObservableExpr:
        """Product-state projector ``|labels><labels|`` (one label per site)."""
        labels = list(labels)
        if len(labels) != self._basis.n_sites:
            raise ValueError(
                f"product_projector needs {self._basis.n_sites} per-site labels, "
                f"got {len(labels)}."
            )
        factors = tuple(
            (site, self._local_E(label, label)) for site, label in enumerate(labels)
        )
        return self._expr((_Term(1.0 + 0.0j, factors),))

    def identity(self) -> ObservableExpr:
        """Identity — its raw expectation is the survival norm ``<psi|psi>``."""
        return self._expr((_Term(1.0 + 0.0j, ()),))

    def site_populations(self, level: str) -> dict[str, ObservableExpr]:
        """Per-site population profile: ``{"n_<level>_<i>": n(level, i)}``."""
        return {
            f"n_{level}_{i}": self.n(level, i) for i in range(self._basis.n_sites)
        }


# ── private dense evaluation (used by the exact backend) ─────────────────────


def _dense_expectation(expr: ObservableExpr, states: np.ndarray):
    """Raw complex ``<psi|O|psi>`` per state — no norm division (private).

    ``states`` is one dense vector ``(dim,)`` or a time-major batch
    ``(n_times, dim)``.  Returns a complex scalar or a ``(n_times,)`` complex
    array respectively.
    """
    arr = np.asarray(states, dtype=complex)
    single = arr.ndim == 1
    if single:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"states must be (dim,) or (n_times, dim); got shape {arr.shape}.")
    d, n = expr._local_dim, expr._n_sites
    if arr.shape[1] != d**n:
        raise ValueError(
            f"state dimension {arr.shape[1]} does not match the expression's "
            f"basis (local_dim {d} ** n_sites {n} = {d**n})."
        )
    n_t = arr.shape[0]
    out = np.zeros(n_t, dtype=complex)
    for term in expr._terms:
        phi = arr.reshape((n_t,) + (d,) * n)
        for site, m in term.factors:
            axis = 1 + site
            phi = np.moveaxis(np.tensordot(m, phi, axes=([1], [axis])), 0, axis)
        out += term.coefficient * np.einsum("td,td->t", arr.conj(), phi.reshape(n_t, -1))
    return out[0] if single else out
