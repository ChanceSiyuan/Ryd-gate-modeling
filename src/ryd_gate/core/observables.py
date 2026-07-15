"""Immutable scalar observable expressions and their system-bound factory.

:class:`ObservableExpr` is the only observable currency in ``ryd_gate``:
``simulate(..., observables={label: expr})`` takes a plain dict of named
expressions, and every expression is built from the read-only factory exposed as
``system.observables`` — exactly two physical primitives (O11)::

    obs = system.observables
    n_r = sum(obs.n("r", i) for i in range(system.N))     # total Rydberg population
    P11 = obs.n("1", 0) @ obs.n("1", 1)                   # product-state projector
    A   = obs.E("0", "1", site=0)
    X   = A + A.dagger()                                  # Hermitian coherence

The closed algebra: ``+``, ``-``, unary ``-``, scalar ``*`` (real or complex),
operator product ``@``, ``.dagger()``, and the Python ``sum()`` seed (O13).
Scalars may be complex intermediates, but the final expression handed to
``simulate`` must be Hermitian (O06/O12); its expectation is real.

Private lowered representation (backends rely on it): every expression is a
finite sum of terms ``coefficient x product of per-site local matrices``, stored
as ``expr._terms`` — a tuple of :class:`_Term` with ``factors`` sorted by site
index and at most one ``(d, d)`` matrix per site.
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
    """An immutable scalar observable expression (built via ``system.observables``)."""

    __slots__ = ("_terms", "_n_sites", "_local_dim")

    def __init__(self, terms: tuple[_Term, ...], n_sites: int, local_dim: int) -> None:
        object.__setattr__(self, "_terms", tuple(terms))
        object.__setattr__(self, "_n_sites", int(n_sites))
        object.__setattr__(self, "_local_dim", int(local_dim))

    def __setattr__(self, name, value):  # immutability guard
        raise AttributeError("ObservableExpr is immutable.")

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

    def __radd__(self, other):
        # Support Python sum(): only the additive seed 0 + expr is accepted (O13).
        if isinstance(other, numbers.Number) and not isinstance(other, bool) and other == 0:
            return self
        raise TypeError(
            "ObservableExpr only supports `0 + expr` (the sum() seed); a nonzero "
            "number is not an identity operator."
        )

    def __sub__(self, other: "ObservableExpr") -> "ObservableExpr":
        self._check_compatible(other)
        return self + (-1.0) * other

    def __neg__(self) -> "ObservableExpr":
        return (-1.0) * self

    def __mul__(self, scalar) -> "ObservableExpr":
        if isinstance(scalar, ObservableExpr) or not isinstance(scalar, numbers.Number) or isinstance(scalar, bool):
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
    """Read-only :class:`ObservableExpr` factory bound to a basis (``system.observables``).

    Exactly two physical primitives (O11): ``E(ket, bra, site)`` and
    ``n(level, site)``. Every other scalar operator is composed by the caller
    with ``+``, ``-``, scalar ``*``, ``@``, ``.dagger()`` and ``sum()``.
    """

    __slots__ = ("_basis",)

    def __init__(self, basis: BasisSpec) -> None:
        object.__setattr__(self, "_basis", basis)

    def __setattr__(self, name, value):  # immutability guard
        raise AttributeError("ObservableFactory is read-only.")

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

    def E(self, ket: str, bra: str, site: int) -> ObservableExpr:
        """Transition operator ``|ket><bra|`` on ``site``."""
        site = self._site(site)
        return self._expr((_Term(1.0 + 0.0j, ((site, self._local_E(ket, bra)),)),))

    def n(self, level: str, site: int) -> ObservableExpr:
        """Level projector ``|level><level|`` on ``site`` (== ``E(level, level, site)``)."""
        return self.E(level, level, site)


# ── private dense evaluation + Hermiticity (used by the preflight/backends) ──


def _support_operator(expr: ObservableExpr):
    """Return ``(sites, matrix)`` for the dense operator on the touched sites only.

    ``sites`` is the sorted union of sites appearing in any term; ``matrix`` is
    the ``d^k x d^k`` operator on those ``k`` sites (identity on untouched ones).
    Cheap because observables touch few sites even when ``N`` is large.
    """
    d = expr._local_dim
    sites = sorted({site for term in expr._terms for site, _ in term.factors})
    k = len(sites)
    dim = d**k
    pos = {s: i for i, s in enumerate(sites)}
    eye = np.eye(d, dtype=complex)
    op = np.zeros((dim, dim), dtype=complex)
    for term in expr._terms:
        mats = [eye] * k
        for site, m in term.factors:
            mats[pos[site]] = m
        kron = np.array([[1.0 + 0.0j]])
        for m in mats:
            kron = np.kron(kron, m)
        op += term.coefficient * kron
    return sites, op


def _is_hermitian(expr: ObservableExpr, atol: float = 1e-9) -> bool:
    """Whether the expression is Hermitian (checked on its finite site support)."""
    _sites, op = _support_operator(expr)
    scale = max(1.0, float(np.max(np.abs(op)))) if op.size else 1.0
    return bool(np.allclose(op, op.conj().T, atol=atol * scale, rtol=0.0))


def _dense_expectation(expr: ObservableExpr, states: np.ndarray):
    """Raw complex ``<psi|O|psi>`` per state (private).

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
