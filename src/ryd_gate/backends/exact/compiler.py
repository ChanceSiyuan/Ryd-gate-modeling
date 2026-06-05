"""Exact matrix lowering for state-vector sparse/dense backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ryd_gate.core.operator_spec import is_operator_spec, materialize_sparse_operator
from ryd_gate.ir import HamiltonianIR, HamiltonianTerm, compile_hamiltonian_ir


@dataclass
class ExactSparseCompiler:
    """Lower unified Hamiltonian IR into exact matrix-backed terms.

    Parameters
    ----------
    max_dim:
        Maximum Hilbert-space dimension allowed for exact state-vector
        matrix materialization. Use ``None`` to disable the guard.
    """

    max_dim: int | None = 2_000_000

    def compile(self, system_or_ir, params: dict | None = None) -> HamiltonianIR:
        """Lower unified Hamiltonian IR into matrix-backed HamiltonianIR."""
        source_ir = (
            system_or_ir
            if isinstance(system_or_ir, HamiltonianIR)
            else compile_hamiltonian_ir(system_or_ir, _require_params(params))
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

    def materialize_block(self, system, name: str, cache: dict[str, Any] | None = None):
        """Return the exact matrix for a registered block."""
        cache = cache if cache is not None else {}
        if name in cache:
            return cache[name]
        operator = system.blocks.get(name)
        if is_operator_spec(operator):
            operator = materialize_sparse_operator(operator, system.basis, max_dim=self.max_dim)
        cache[name] = operator
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


def _require_params(params: dict | None) -> dict:
    if params is None:
        raise TypeError("compile() requires params when given a system instead of HamiltonianIR.")
    return params


def compile_expm_ir(
    system_or_ir,
    params: dict | None = None,
    *,
    max_dim: int | None = 2_000_000,
) -> HamiltonianIR:
    """Lower unified Hamiltonian IR into exact matrix form for exact backends."""
    return ExactSparseCompiler(max_dim=max_dim).compile(system_or_ir, params)
