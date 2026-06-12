"""Dense ODE state-vector backend using scipy.integrate.solve_ivp."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.integrate

from ryd_gate.backends.exact.compiler import SolverBackend
from ryd_gate.ir import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.ir import HamiltonianIR


def _as_dense(operator) -> np.ndarray:
    """Densify an IR operator; ``np.asarray`` alone wraps scipy sparse in an object array."""
    if hasattr(operator, "toarray"):
        return np.asarray(operator.toarray())
    return np.asarray(operator)


class DenseODEBackend(SolverBackend):
    """Dense ODE backend using scipy DOP853."""

    def __init__(self, rtol: float = 1e-8, atol: float = 1e-12) -> None:
        self.rtol = rtol
        self.atol = atol

    def evolve(
        self,
        ir: HamiltonianIR,
        psi0: np.ndarray,
        t_gate: float,
        t_eval: np.ndarray | None = None,
    ) -> EvolutionResult:
        """Evolve using adaptive ODE integration."""
        H_static = np.zeros((ir.dim, ir.dim), dtype=np.complex128)
        for term in ir.static_terms:
            coeff = term.coefficient(0) if callable(term.coefficient) else term.coefficient
            H_static += coeff * _as_dense(term.operator)

        # Pre-bind drive terms once: (dense operator, coefficient, op_dag), so the
        # rhs (called many times by solve_ivp) never re-densifies or re-transposes.
        drive = []
        for term in ir.drive_terms:
            op = _as_dense(term.operator)
            op_dag = op.conj().T if term.add_hermitian_conjugate else None
            drive.append((op, term.coefficient, op_dag))

        def rhs(t, y):
            H = H_static.copy()
            for operator, coefficient, op_dag in drive:
                coeff = coefficient(t) if callable(coefficient) else coefficient
                H += coeff * operator
                if op_dag is not None:
                    H += np.conj(coeff) * op_dag
            return -1j * H @ y

        solve_kwargs = dict(method="DOP853", rtol=self.rtol, atol=self.atol)
        if t_eval is not None:
            solve_kwargs["t_eval"] = t_eval

        result = scipy.integrate.solve_ivp(rhs, [0, t_gate], psi0, **solve_kwargs)

        if t_eval is not None:
            return EvolutionResult(
                psi_final=result.y[:, -1],
                times=result.t,
                # ``solve_ivp`` returns y as (dim, n_times); transpose to the row-major
                # (n_times, dim) convention used by SparseExpmBackend and asserted by tests.
                states=result.y.T,
                metadata=ir.metadata,
            )
        return EvolutionResult(psi_final=result.y[:, -1], metadata=ir.metadata)
