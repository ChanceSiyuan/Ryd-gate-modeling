"""Symbolic operator descriptions for lattice systems.

These specs describe local and pair operators without constructing
``d**N`` matrices. Exact matrix compilers materialize them only when a
state-vector backend is selected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix, diags as spdiags

from ryd_gate.core.basis import BasisSpec
from ryd_gate.core.operators import embed_site_op


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
class WeightedProjectorSumSpec:
    """Weighted sum of projectors on one level."""

    level: str
    weights: tuple[float, ...]


@dataclass(frozen=True)
class TransitionOperatorSpec:
    """Transition ``|upper><lower|`` on one site, or summed over all sites."""

    lower: str
    upper: str
    site: int | None = None


@dataclass(frozen=True)
class RydbergPairInteractionSpec:
    """Pair interaction ``sum_ij V_ij n_i^R n_j^R``."""

    pairs: tuple[tuple[int, int, float], ...]
    rydberg_levels: tuple[str, ...]


OperatorSpec = (
    LocalProjectorSpec
    | SumProjectorSpec
    | WeightedProjectorSumSpec
    | TransitionOperatorSpec
    | RydbergPairInteractionSpec
)


def is_operator_spec(value) -> bool:
    """Return True when *value* is a symbolic operator spec."""
    return isinstance(
        value,
        (
            LocalProjectorSpec,
            SumProjectorSpec,
            WeightedProjectorSumSpec,
            TransitionOperatorSpec,
            RydbergPairInteractionSpec,
        ),
    )


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
    if isinstance(spec, WeightedProjectorSumSpec):
        return _weighted_projector_sum(spec.level, spec.weights, basis)
    if isinstance(spec, TransitionOperatorSpec):
        return _transition_operator(spec.lower, spec.upper, spec.site, basis)
    if isinstance(spec, RydbergPairInteractionSpec):
        return _rydberg_pair_interaction(spec, basis)
    raise TypeError(f"Unsupported operator spec: {type(spec).__name__}")


def measure_state_vector_operator(spec: OperatorSpec, basis: BasisSpec, psi: np.ndarray) -> float:
    """Measure a symbolic observable against an exact state vector."""
    if psi.shape[0] != basis.total_dim:
        raise ValueError(
            f"State dimension {psi.shape[0]} does not match basis dimension {basis.total_dim}."
        )

    if isinstance(spec, LocalProjectorSpec):
        return _projector_expectation(spec.level, spec.site, basis, psi)
    if isinstance(spec, SumProjectorSpec):
        return float(
            sum(_projector_expectation(spec.level, i, basis, psi) for i in range(basis.n_sites))
        )
    if isinstance(spec, WeightedProjectorSumSpec):
        return float(
            sum(
                weight * _projector_expectation(spec.level, i, basis, psi)
                for i, weight in enumerate(spec.weights)
            )
        )

    op = materialize_sparse_operator(spec, basis)
    return float(np.real(np.vdot(psi, op @ psi)))


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


def _weighted_projector_sum(level: str, weights: tuple[float, ...], basis: BasisSpec):
    if len(weights) != basis.n_sites:
        raise ValueError(f"Expected {basis.n_sites} weights, got {len(weights)}.")
    terms = [
        weight * _local_projector(level, i, basis)
        for i, weight in enumerate(weights)
        if abs(weight) > 1e-15
    ]
    return _sum_sparse(terms, basis.total_dim)


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
        _site_level_mask(indices, basis, site, spec.rydberg_levels).astype(float)
        for site in range(basis.n_sites)
    ]
    h_diag = np.zeros(basis.total_dim, dtype=complex)
    for i, j, V_ij in spec.pairs:
        h_diag += V_ij * (ryd_mask_by_site[i] * ryd_mask_by_site[j])
    return spdiags([h_diag], [0], shape=(basis.total_dim, basis.total_dim), format="csc")


def _projector_expectation(level: str, site: int, basis: BasisSpec, psi: np.ndarray) -> float:
    indices = np.arange(basis.total_dim, dtype=np.int64)
    mask = _site_level_mask(indices, basis, site, (level,))
    probs = np.abs(psi[mask]) ** 2
    return float(np.real(np.sum(probs)))


def _site_level_mask(indices: np.ndarray, basis: BasisSpec, site: int, levels: tuple[str, ...]):
    divisor = basis.local_dim ** (basis.n_sites - 1 - site)
    digits = (indices // divisor) % basis.local_dim
    level_indices = [basis.level_index(level) for level in levels]
    return np.isin(digits, level_indices)


def _sum_sparse(ops: list, dim: int):
    if not ops:
        return csc_matrix((dim, dim), dtype=complex)
    return sum(ops[1:], start=ops[0]).tocsc()
