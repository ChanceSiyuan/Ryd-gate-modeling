"""Dense piecewise-constant matrix-exponential backend.

This mirrors :class:`SparseExpmBackend` step for step (same midpoint freeze, same
per-step renormalization, same ``t_eval`` recording) but evolves with a dense
``scipy.linalg.expm`` instead of ``scipy.sparse.linalg.expm_multiply``. It is a
drop-in with identical numerics (validated to ~1e-12).

Why it exists: ``expm_multiply``'s cost grows with ``||dt*H||``. The physical-ladder
models (``rb87_7``, ``analog_3``) keep the intermediate manifold at ~GHz, so over a
microsecond pulse ``||dt*H|| ~ 10^3`` and ``expm_multiply`` becomes pathologically
slow (minutes per 2-atom run). Dense ``expm`` uses scaling-and-squaring, whose cost
is ~constant in the norm for a small matrix, so it is ~100x faster there. The auto
selector in :func:`ryd_gate.backends.exact.simulate.simulate` routes those models
(below a dimension cap) here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.backends.exact.compiler import SolverBackend, record_steps
from ryd_gate.ir import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.ir import HamiltonianIR


def _to_dense(op) -> np.ndarray:
    return np.asarray(op.toarray() if hasattr(op, "toarray") else op, dtype=complex)


class DenseExpmBackend(SolverBackend):
    """Dense piecewise-constant backend using ``scipy.linalg.expm`` per step."""

    def __init__(self, n_steps: int = 200) -> None:
        self.n_steps = n_steps

    def evolve(
        self,
        ir: "HamiltonianIR",
        psi0: np.ndarray,
        t_gate: float,
        t_eval: np.ndarray | bool | None = None,
    ) -> EvolutionResult:
        """Evolve one state (dense expm per step)."""
        return self.evolve_many(ir, [psi0], t_gate, t_eval)[0]

    def evolve_many(
        self,
        ir: "HamiltonianIR",
        psi0_list,
        t_gate: float,
        t_eval: np.ndarray | bool | None = None,
    ) -> list[EvolutionResult]:
        """Evolve several initial states under the *same* Hamiltonian, sharing each
        step's propagator (computed once, applied to all states as one matmul).

        This is the big win over per-state ``simulate`` calls for the high-norm ladder
        models: the expensive part is the per-step ``expm``, so N states cost the same
        as one. :func:`ryd_gate.backends.exact.simulate.simulate_states` routes here.
        """
        from scipy.linalg import expm

        psis = np.stack([np.asarray(p, dtype=complex).ravel() for p in psi0_list], axis=1)
        dim, n_states = psis.shape
        dt = t_gate / self.n_steps
        record_at = record_steps(self.n_steps, t_eval, t_gate, dt)

        H_static = np.zeros((dim, dim), dtype=complex)
        for term in ir.static_terms:
            coeff = term.coefficient(0) if callable(term.coefficient) else term.coefficient
            H_static = H_static + coeff * _to_dense(term.operator)

        # Pre-bind drive terms once: operators and their Hermitian conjugates are
        # constant, so the dense conversion and transpose happen here, not per step.
        drive = [
            (
                _to_dense(term.operator),
                term.coefficient,
                _to_dense(term.operator.conj().T) if term.add_hermitian_conjugate else None,
            )
            for term in ir.drive_terms
        ]

        rec_times: list[float] = []
        rec_states: list[list[np.ndarray]] = [[] for _ in range(n_states)]
        if record_at is not None and 0 in record_at:
            rec_times.append(0.0)
            for j in range(n_states):
                rec_states[j].append(psis[:, j].copy())

        for k in range(self.n_steps):
            t_mid = (k + 0.5) * dt
            H = H_static.copy()
            for operator, coefficient, op_dag in drive:
                coeff = coefficient(t_mid) if callable(coefficient) else coefficient
                H = H + coeff * operator
                if op_dag is not None:
                    H = H + np.conj(coeff) * op_dag
            psis = expm(-1j * dt * H) @ psis
            psis = psis / np.linalg.norm(psis, axis=0)

            if record_at is not None and (k + 1) in record_at:
                rec_times.append((k + 1) * dt)
                for j in range(n_states):
                    rec_states[j].append(psis[:, j].copy())

        times = np.array(rec_times) if record_at is not None else None
        results = []
        for j in range(n_states):
            result = EvolutionResult(psi_final=psis[:, j], metadata=ir.metadata)
            if record_at is not None:
                result.times = times
                result.states = (
                    np.array(rec_states[j]) if rec_states[j] else np.empty((0, dim), dtype=complex)
                )
            results.append(result)
        return results
