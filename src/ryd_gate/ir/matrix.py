"""Matrix Hamiltonian intermediate representation.

Provides a backend-facing description of a time-dependent Hamiltonian:

    H(t) = sum(static_terms) + sum(coeff_k(t) * operator_k)

The matrix IR is produced by an exact matrix compiler and consumed by exact
state-vector backends such as dense ODE and sparse matrix exponential.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class HamiltonianTerm:
    """A single term in the Hamiltonian."""

    name: str
    operator: Any
    coefficient: Callable[[float], complex] | complex = 1.0
    add_hermitian_conjugate: bool = False


@dataclass
class HamiltonianIR:
    """Exact matrix Hamiltonian intermediate representation."""

    static_terms: list[HamiltonianTerm]
    drive_terms: list[HamiltonianTerm]
    dim: int
    is_sparse: bool = False
    metadata: dict = field(default_factory=dict)
