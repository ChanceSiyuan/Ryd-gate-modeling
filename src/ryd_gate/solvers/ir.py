"""Hamiltonian Intermediate Representation (IR).

Provides a solver-agnostic description of a time-dependent Hamiltonian:

    H(t) = sum(static_terms) + sum(coeff_k(t) * operator_k)

The IR is produced by a Compiler and consumed by a SolverBackend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class HamiltonianTerm:
    """A single term in the Hamiltonian.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "drive_420", "H_const").
    operator : Any
        Matrix (dense ndarray or sparse) for this term.
    coefficient : callable or complex
        Time-dependent coefficient function ``t -> complex``, or a constant.
    add_hermitian_conjugate : bool
        If True, the solver adds coeff(t)* @ operator.T.conj() automatically.
    """

    name: str
    operator: Any
    coefficient: Callable[[float], complex] | complex = 1.0
    add_hermitian_conjugate: bool = False


@dataclass
class HamiltonianIR:
    """Solver-agnostic Hamiltonian intermediate representation.

    Attributes
    ----------
    static_terms : list of HamiltonianTerm
        Terms with constant coefficients (assembled once).
    drive_terms : list of HamiltonianTerm
        Terms with time-dependent coefficients (evaluated per step).
    dim : int
        Hilbert space dimension.
    is_sparse : bool
        Whether operators are scipy sparse matrices.
    metadata : dict
        Extra info (t_gate, amplitude_scale, etc.) passed to the solver.
    """

    static_terms: list[HamiltonianTerm]
    drive_terms: list[HamiltonianTerm]
    dim: int
    is_sparse: bool = False
    metadata: dict = field(default_factory=dict)
