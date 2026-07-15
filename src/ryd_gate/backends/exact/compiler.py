"""Exact state-vector backend interface and matrix lowering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from ryd_gate.core.operators import is_operator_spec, materialize_sparse_operator
from ryd_gate.ir import EvolutionResult, HamiltonianIR, HamiltonianTerm, compile_hamiltonian_ir


class SolverBackend(ABC):
    """Abstract simulation backend."""

    @abstractmethod
    def evolve(
        self,
        ir: HamiltonianIR,
        psi0: Any,
        t_gate: float,
        t_eval: np.ndarray | None = None,
        observables: dict | None = None,
    ) -> EvolutionResult:
        """Evolve initial state under a compiled IR, recording expectations."""
        ...


@dataclass
class ExactCompiler:
    """Lower unified Hamiltonian IR into exact matrix-backed terms.

    Parameters
    ----------
    max_dim:
        Maximum Hilbert-space dimension allowed for exact state-vector
        matrix materialization. Use ``None`` to disable the guard.
    """

    max_dim: int | None = 2_000_000

    def compile(self, system_or_ir) -> HamiltonianIR:
        """Lower unified Hamiltonian IR into matrix-backed HamiltonianIR."""
        source_ir = (
            system_or_ir
            if isinstance(system_or_ir, HamiltonianIR)
            else compile_hamiltonian_ir(system_or_ir)
        )
        if source_ir.basis is None:
            raise ValueError("Exact lowering requires HamiltonianIR.basis.")

        cache: dict[int, Any] = {}
        static_terms = [
            self._materialize_term(term, source_ir.basis, cache, make_dense=not source_ir.is_sparse)
            for term in source_ir.static_terms
        ]
        drive_terms = [
            self._materialize_term(term, source_ir.basis, cache, make_dense=not source_ir.is_sparse)
            for term in source_ir.drive_terms
        ]
        metadata = dict(source_ir.metadata)
        metadata["source_compiler"] = metadata.get("compiler", "unknown")
        metadata["compiler"] = "exact"
        return HamiltonianIR(
            static_terms=static_terms,
            drive_terms=drive_terms,
            dim=source_ir.dim,
            is_sparse=source_ir.is_sparse,
            metadata=metadata,
            basis=source_ir.basis,
            geometry=source_ir.geometry,
            level_spec=source_ir.level_spec,
            protocol=source_ir.protocol,
            params=source_ir.params,
        )

    def materialize_operator(self, system, operator):
        """Return the exact matrix for an operator spec (or pass a matrix through)."""
        if is_operator_spec(operator):
            return materialize_sparse_operator(operator, system.basis, max_dim=self.max_dim)
        return operator

    def _materialize_term(
        self,
        term: HamiltonianTerm,
        basis,
        cache: dict[int, Any],
        *,
        make_dense: bool,
    ) -> HamiltonianTerm:
        operator = term.operator
        if is_operator_spec(operator):
            cache_key = id(operator)
            if cache_key not in cache:
                materialized = materialize_sparse_operator(
                    operator,
                    basis,
                    max_dim=self.max_dim,
                )
                cache[cache_key] = materialized.toarray() if make_dense else materialized
            operator = cache[cache_key]
        return HamiltonianTerm(
            name=term.name,
            operator=operator,
            coefficient=term.coefficient,
            add_hermitian_conjugate=term.add_hermitian_conjugate,
            channel=term.channel,
            metadata=dict(term.metadata),
        )


def _compile_exact_ir(
    system_or_ir,
    *,
    max_dim: int | None = 2_000_000,
) -> HamiltonianIR:
    """Lower unified Hamiltonian IR into exact matrix form (private compiler seam).

    Research scripts with specialized solvers import this directly.
    """
    return ExactCompiler(max_dim=max_dim).compile(system_or_ir)
