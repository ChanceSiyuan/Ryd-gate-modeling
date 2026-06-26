"""Operator construction for Rydberg gate and many-body lattice models.

Two layers in one module:

- Concrete builders: two-atom dense operators (occupation, state maps,
  projectors) plus the generic many-body Kronecker embedding helper
  :func:`embed_site_op` shared by all N-atom lattice models.
- Symbolic operator specs: descriptions of local and pair operators that
  avoid constructing ``d**N`` matrices. Exact matrix compilers materialize
  them only when a state-vector backend is selected.
"""

from __future__ import annotations

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


def build_sss_state_map(n_levels: int = 7) -> "dict[str, NDArray[np.complexfloating]]":
    """Build mapping from qubit state labels to two-atom state vectors.

    For the 7-level system, qubit states use levels 0 (|g⟩) and 1 (|1⟩),
    plus SSS (symmetric subspace) superpositions used for gate characterization.

    Parameters
    ----------
    n_levels : int
        Number of single-atom levels. Must be >= 7.

    Raises
    ------
    ValueError
        If n_levels < 7. For 3-level systems, use
        :func:`build_product_state_map` instead.
    """
    if n_levels < 7:
        raise ValueError(
            f"build_sss_state_map requires n_levels >= 7, got {n_levels}. "
            "For 3-level systems, use build_product_state_map(n_levels=3)."
        )
    s0 = np.zeros(n_levels, dtype=complex)
    s0[0] = 1.0
    s1 = np.zeros(n_levels, dtype=complex)
    s1[1] = 1.0
    state_00 = np.kron(s0, s0)
    state_01 = np.kron(s0, s1)
    state_10 = np.kron(s1, s0)
    state_11 = np.kron(s1, s1)
    return {
        "00": state_00,
        "01": state_01,
        "10": state_10,
        "11": state_11,
        "SSS-0": 0.5 * state_00 + 0.5 * state_01 + 0.5 * state_10 + 0.5 * state_11,
        "SSS-1": 0.5 * state_00 - 0.5 * state_01 - 0.5 * state_10 + 0.5 * state_11,
        "SSS-2": 0.5 * state_00 + 0.5j * state_01 + 0.5j * state_10 - 0.5 * state_11,
        "SSS-3": 0.5 * state_00 - 0.5j * state_01 - 0.5j * state_10 - 0.5 * state_11,
        "SSS-4": state_00,
        "SSS-5": state_11,
        "SSS-6": 0.5 * state_00 + 0.5 * state_01 + 0.5 * state_10 - 0.5 * state_11,
        "SSS-7": 0.5 * state_00 - 0.5 * state_01 - 0.5 * state_10 - 0.5 * state_11,
        "SSS-8": 0.5 * state_00 + 0.5j * state_01 + 0.5j * state_10 + 0.5 * state_11,
        "SSS-9": 0.5 * state_00 - 0.5j * state_01 - 0.5j * state_10 + 0.5 * state_11,
        "SSS-10": state_00 / np.sqrt(2) + 1j * state_11 / np.sqrt(2),
        "SSS-11": state_00 / np.sqrt(2) - 1j * state_11 / np.sqrt(2),
    }


def build_product_state_map(
    n_levels: int = 3,
    level_labels: tuple[str, ...] | None = None,
) -> "dict[str, NDArray[np.complexfloating]]":
    """Build mapping from product-state labels to two-atom state vectors.

    Generates all ``n_levels^2`` product states ``|i,j⟩`` for a
    two-atom system and labels them using the provided level names.

    Parameters
    ----------
    n_levels : int
        Number of single-atom levels (default 3).
    level_labels : tuple of str or None
        Human-readable names for each level.  If None, defaults to
        ``("g", "e", "r")`` for n_levels=3 or ``("0", "1", ...)``
        otherwise.

    Returns
    -------
    dict mapping "AB" labels to state vectors of dimension n_levels^2.

    Examples
    --------
    >>> m = build_product_state_map(n_levels=3)
    >>> m["gg"]   # |g,g⟩
    >>> m["gr"]   # |g,r⟩
    >>> m["rr"]   # |r,r⟩
    """
    if level_labels is None:
        if n_levels == 3:
            level_labels = ("g", "e", "r")
        else:
            level_labels = tuple(str(i) for i in range(n_levels))

    states: dict[str, NDArray[np.complexfloating]] = {}
    for i in range(n_levels):
        si = np.zeros(n_levels, dtype=complex)
        si[i] = 1.0
        for j in range(n_levels):
            sj = np.zeros(n_levels, dtype=complex)
            sj[j] = 1.0
            label = level_labels[i] + level_labels[j]
            states[label] = np.kron(si, sj)
    return states


def build_vdw_unit_operator(
    rydberg_indices: tuple[int, ...] = (5, 6), n_levels: int = 7,
) -> "NDArray[np.complexfloating]":
    """Build the unit van der Waals interaction operator."""
    ryd_proj = np.zeros((n_levels, n_levels), dtype=np.complex128)
    for i in rydberg_indices:
        ryd_proj[i, i] = 1.0
    return np.kron(ryd_proj, ryd_proj)


def build_atom_a_projector(index: int, n_levels: int = 7) -> "NDArray[np.complexfloating]":
    """Build |i><i| x I -- projects Atom A (left) onto level index."""
    sq = np.zeros((n_levels, n_levels), dtype=np.complex128)
    sq[index, index] = 1
    return np.kron(sq, np.eye(n_levels, dtype=np.complex128))


def build_atom_b_projector(index: int, n_levels: int = 7) -> "NDArray[np.complexfloating]":
    """Build I x |i><i| -- projects Atom B (right) onto level index."""
    sq = np.zeros((n_levels, n_levels), dtype=np.complex128)
    sq[index, index] = 1
    return np.kron(np.eye(n_levels, dtype=np.complex128), sq)


def get_nominal_distance(param_set: str) -> float:
    """Get the nominal interatomic distance in um."""
    return 3.0


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
class LocalMatrixSumSpec:
    """Sum of the same local matrix over all sites."""

    matrix: np.ndarray


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
    | LocalMatrixSumSpec
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
            LocalMatrixSumSpec,
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
    if isinstance(spec, LocalMatrixSumSpec):
        return _local_matrix_sum(spec.matrix, basis)
    if isinstance(spec, RydbergPairInteractionSpec):
        return _rydberg_pair_interaction(spec, basis)
    raise TypeError(f"Unsupported operator spec: {type(spec).__name__}")


def measure_state_vector_operator(spec: OperatorSpec, basis: BasisSpec, psi: np.ndarray) -> float:
    """Measure a symbolic observable against an exact state vector."""
    if psi.shape[0] != basis.total_dim:
        raise ValueError(f"State dimension {psi.shape[0]} does not match basis dimension {basis.total_dim}.")

    if isinstance(spec, LocalProjectorSpec):
        return _projector_expectation(spec.level, spec.site, basis, psi)
    if isinstance(spec, SumProjectorSpec):
        return float(sum(_projector_expectation(spec.level, i, basis, psi) for i in range(basis.n_sites)))
    if isinstance(spec, WeightedProjectorSumSpec):
        return float(
            sum(weight * _projector_expectation(spec.level, i, basis, psi) for i, weight in enumerate(spec.weights))
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
    terms = [weight * _local_projector(level, i, basis) for i, weight in enumerate(weights) if abs(weight) > 1e-15]
    return _sum_sparse(terms, basis.total_dim)


def _transition_operator(lower: str, upper: str, site: int | None, basis: BasisSpec):
    d = basis.local_dim
    local = np.zeros((d, d), dtype=complex)
    local[basis.level_index(upper), basis.level_index(lower)] = 1.0
    if site is not None:
        return embed_site_op(csc_matrix(local), site, basis.n_sites, d=d).tocsc()
    ops = [embed_site_op(csc_matrix(local), i, basis.n_sites, d=d).tocsc() for i in range(basis.n_sites)]
    return _sum_sparse(ops, basis.total_dim)


def _local_matrix_sum(matrix: np.ndarray, basis: BasisSpec):
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.shape != (basis.local_dim, basis.local_dim):
        raise ValueError(
            "LocalMatrixSumSpec matrix shape must match basis.local_dim; "
            f"got {matrix.shape}, expected {(basis.local_dim, basis.local_dim)}."
        )
    local = csc_matrix(matrix)
    ops = [embed_site_op(local, i, basis.n_sites, d=basis.local_dim).tocsc() for i in range(basis.n_sites)]
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
