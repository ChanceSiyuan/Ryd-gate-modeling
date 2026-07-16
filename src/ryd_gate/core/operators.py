"""Operator construction for Rydberg gate and many-body lattice models.

Two layers in one module:

- Concrete builders: the two-atom dense occupation operator plus the generic
  many-body Kronecker embedding helper :func:`embed_site_op` shared by all
  N-atom lattice models.
- Symbolic operator specs: descriptions of local and pair operators that
  avoid constructing ``d**N`` matrices. Exact matrix compilers materialize
  them only when a state-vector backend is selected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import diags as spdiags
from scipy.sparse import eye as speye
from scipy.sparse import kron as spkron

from ryd_gate.core.model import BasisSpec

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ======================================================================
# MANY-BODY KRONECKER EMBEDDING
# ======================================================================


def embed_site_op(op, site: int, N: int, d: int | None = None):
    """Embed a d×d single-atom operator on ``site`` into the d^N Hilbert space.

    Uses I_{d^site} ⊗ op ⊗ I_{d^(N-1-site)} via sparse Kronecker products.
    The local dimension ``d`` is inferred from ``op.shape[0]`` if not given.
    """
    if d is None:
        d = op.shape[0]
    result = op
    if site > 0:
        result = spkron(speye(d ** site, format='csc', dtype=complex),
                        result, format='csc')
    if site < N - 1:
        result = spkron(result,
                        speye(d ** (N - 1 - site), format='csc', dtype=complex),
                        format='csc')
    return result


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================


def build_occ_operator(index: int, n_levels: int = 7) -> "NDArray[np.complexfloating]":
    """Build occupation number operator for level `index`.

    Creates |i><i| x I + I x |i><i| for measuring total population
    in level i across both atoms.

    Parameters
    ----------
    index : int
        Single-atom level index.
    n_levels : int
        Number of single-atom levels (default 7).
    """
    oper_sq = np.zeros((n_levels, n_levels), dtype=np.complex128)
    oper_sq[index, index] = 1
    oper_tq = np.kron(np.eye(n_levels), oper_sq) + np.kron(oper_sq, np.eye(n_levels))
    return oper_tq


# ======================================================================
# SYMBOLIC OPERATOR SPECS
# ======================================================================


@dataclass(frozen=True)
class LocalProjectorSpec:
    """Projector ``|level><level|`` on one site."""

    level: str
    site: int


@dataclass(frozen=True)
class SumProjectorSpec:
    """Sum of projectors ``sum_i |level><level|_i``."""

    level: str


@dataclass(frozen=True)
class TransitionOperatorSpec:
    """Transition ``|upper><lower|`` on one site, or summed over all sites."""

    lower: str
    upper: str
    site: int | None = None


@dataclass(frozen=True)
class RydbergPairInteractionSpec:
    """Isotropic Rydberg pair interaction ``sum_ij V_ij n_i^R n_j^R`` (S-state, I05).

    Every Rydberg-level combination on a pair shares the same ``V_ij``.
    """

    pairs: tuple[tuple[int, int, float], ...]
    rydberg_levels: tuple[str, ...]


@dataclass(frozen=True)
class ChannelResolvedPairSpec:
    """Channel-resolved Rydberg pair interaction (P-state, I06).

    ``channel_terms`` is a flat list of ``(site_i, site_j, level_a, level_b, V)``
    contributing ``V * |a><a|_i ⊗ |b><b|_j`` to the diagonal. The resolver emits
    both orderings for ``a != b`` so the pair term is symmetric.
    """

    channel_terms: tuple[tuple[int, int, str, str, float], ...]


OperatorSpec = (
    LocalProjectorSpec
    | SumProjectorSpec
    | TransitionOperatorSpec
    | RydbergPairInteractionSpec
    | ChannelResolvedPairSpec
)


# ======================================================================
# PRIMITIVE OPERATOR FACTORY:  E[ket,bra] = |ket><bra|
# ======================================================================

_E_NAME = re.compile(r"^E\[(.+),(.+)\]$")


def parse_E(name: str) -> tuple[str, str]:
    """Parse an ``E[ket,bra]`` operator name into ``(ket, bra)`` labels.

    The bracket form is required (not ``E_ket_bra``) because level labels may
    contain underscores such as ``r_garb``.
    """
    m = _E_NAME.match(name)
    if m is None:
        raise ValueError(
            f"Operator name {name!r} is not of the form 'E[ket,bra]' "
            "(e.g. 'E[r,r]', 'E[r_garb,e1]')."
        )
    return m.group(1), m.group(2)


@dataclass(frozen=True)
class StaticHamiltonianTerm:
    """A static (protocol-independent) Hamiltonian term.

    ``operator`` is an :data:`OperatorSpec` (built via a system's operator
    factory). ``name`` is the ``E[ket,bra]`` (or pair) identity, used so a bound
    protocol's drive on the same channel can override the static term.  Diagonal
    ``E[a,a]`` terms are Hermitian; off-diagonal static couplings set
    ``add_hermitian_conjugate=True``.
    """

    name: str
    operator: object
    coefficient: complex = 1.0
    add_hermitian_conjugate: bool = False


class BasisOperatorFactory:
    """Primitive operator factory generated from a :class:`BasisSpec`.

    The single source of truth for constructing operator specs by ``E[ket,bra]``
    name (``E[ket,bra] = |ket><bra|``).  Diagonal ``E[a,a]`` map to projectors;
    off-diagonal ``E[a,b]`` (``a != b``) map to ``|a><b|`` transitions.
    """

    def __init__(self, basis: BasisSpec) -> None:
        self.basis = basis

    def _parse(self, name: str) -> tuple[str, str]:
        ket, bra = parse_E(name)
        levels = self.basis.local_levels
        for label in (ket, bra):
            if label not in levels:
                raise ValueError(
                    f"Operator {name!r} references level {label!r} not in basis "
                    f"levels {levels}."
                )
        return ket, bra

    def local(self, name: str, site: int):
        """``E[ket,bra]`` on a single ``site``."""
        ket, bra = self._parse(name)
        if ket == bra:
            return LocalProjectorSpec(ket, site)
        return TransitionOperatorSpec(lower=bra, upper=ket, site=site)

    def sum(self, name: str):
        """``sum_i E[ket,bra]_i`` over all sites."""
        ket, bra = self._parse(name)
        if ket == bra:
            return SumProjectorSpec(ket)
        return TransitionOperatorSpec(lower=bra, upper=ket, site=None)

    def pair_projector(self, pairs, rydberg_levels) -> RydbergPairInteractionSpec:
        """Rydberg van der Waals pair-interaction spec ``sum_ij V_ij n_i^R n_j^R``."""
        return RydbergPairInteractionSpec(tuple(pairs), tuple(rydberg_levels))

    @staticmethod
    def is_hermitian(name: str) -> bool:
        """True when ``E[ket,bra]`` is diagonal (``ket == bra``)."""
        ket, bra = parse_E(name)
        return ket == bra


def materialize_sparse_operator(
    spec: OperatorSpec,
    basis: BasisSpec,
    *,
    max_dim: int | None = None,
):
    """Materialize an operator spec as a scipy sparse matrix."""
    _check_exact_dim(basis, max_dim)
    if isinstance(spec, LocalProjectorSpec):
        return _local_projector(spec.level, spec.site, basis)
    if isinstance(spec, SumProjectorSpec):
        return _sum_projector(spec.level, basis)
    if isinstance(spec, TransitionOperatorSpec):
        return _transition_operator(spec.lower, spec.upper, spec.site, basis)
    if isinstance(spec, RydbergPairInteractionSpec):
        return _rydberg_pair_interaction(spec, basis)
    if isinstance(spec, ChannelResolvedPairSpec):
        return _channel_resolved_pair_interaction(spec, basis)
    raise TypeError(f"Unsupported operator spec: {type(spec).__name__}")


def _check_exact_dim(basis: BasisSpec, max_dim: int | None) -> None:
    if max_dim is not None and basis.total_dim > max_dim:
        raise ValueError(
            "Exact sparse compilation requires a full state vector with "
            f"dimension {basis.total_dim}. This exceeds max_dim={max_dim}. "
            "Use a tensor-network compiler/backend for this system size."
        )


def _local_projector(level: str, site: int, basis: BasisSpec):
    d = basis.local_dim
    local = np.zeros((d, d), dtype=complex)
    local[basis.level_index(level), basis.level_index(level)] = 1.0
    return embed_site_op(csc_matrix(local), site, basis.n_sites, d=d).tocsc()


def _sum_projector(level: str, basis: BasisSpec):
    ops = [_local_projector(level, i, basis) for i in range(basis.n_sites)]
    return _sum_sparse(ops, basis.total_dim)


def _transition_operator(lower: str, upper: str, site: int | None, basis: BasisSpec):
    d = basis.local_dim
    local = np.zeros((d, d), dtype=complex)
    local[basis.level_index(upper), basis.level_index(lower)] = 1.0
    if site is not None:
        return embed_site_op(csc_matrix(local), site, basis.n_sites, d=d).tocsc()
    ops = [embed_site_op(csc_matrix(local), i, basis.n_sites, d=d).tocsc() for i in range(basis.n_sites)]
    return _sum_sparse(ops, basis.total_dim)


def _rydberg_pair_interaction(spec: RydbergPairInteractionSpec, basis: BasisSpec):
    indices = np.arange(basis.total_dim, dtype=np.int64)
    ryd_mask_by_site = [
        _site_level_mask(indices, basis, site, spec.rydberg_levels).astype(float) for site in range(basis.n_sites)
    ]
    h_diag = np.zeros(basis.total_dim, dtype=complex)
    for i, j, V_ij in spec.pairs:
        h_diag += V_ij * (ryd_mask_by_site[i] * ryd_mask_by_site[j])
    return spdiags([h_diag], [0], shape=(basis.total_dim, basis.total_dim), format="csc")


def _channel_resolved_pair_interaction(spec: ChannelResolvedPairSpec, basis: BasisSpec):
    indices = np.arange(basis.total_dim, dtype=np.int64)
    mask_cache: dict[tuple[int, str], np.ndarray] = {}

    def mask(site: int, level: str) -> np.ndarray:
        key = (site, level)
        if key not in mask_cache:
            mask_cache[key] = _site_level_mask(indices, basis, site, (level,)).astype(float)
        return mask_cache[key]

    h_diag = np.zeros(basis.total_dim, dtype=complex)
    for i, j, level_a, level_b, V_ij in spec.channel_terms:
        h_diag += V_ij * (mask(i, level_a) * mask(j, level_b))
    return spdiags([h_diag], [0], shape=(basis.total_dim, basis.total_dim), format="csc")


def _site_level_mask(indices: np.ndarray, basis: BasisSpec, site: int, levels: tuple[str, ...]):
    divisor = basis.local_dim ** (basis.n_sites - 1 - site)
    digits = (indices // divisor) % basis.local_dim
    level_indices = [basis.level_index(level) for level in levels]
    return np.isin(digits, level_indices)


def _sum_sparse(ops: list, dim: int):
    if not ops:
        return csc_matrix((dim, dim), dtype=complex)
    return sum(ops[1:], start=ops[0]).tocsc()
