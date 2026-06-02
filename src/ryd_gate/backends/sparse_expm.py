"""Sparse piecewise-constant matrix exponential backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.backends.base import EvolutionResult, SolverBackend

if TYPE_CHECKING:
    from ryd_gate.ir.matrix import HamiltonianIR


class SparseExpmBackend(SolverBackend):
    """Sparse piecewise-constant backend using ``expm_multiply``."""

    def __init__(self, n_steps: int = 200) -> None:
        self.n_steps = n_steps

    def evolve(
        self,
        ir: HamiltonianIR,
        psi0: np.ndarray,
        t_gate: float,
        t_eval: np.ndarray | None = None,
    ) -> EvolutionResult:
        """Evolve by freezing H at each interval midpoint."""
        from scipy.sparse.linalg import expm_multiply

        psi = psi0.copy().astype(complex)
        dt = t_gate / self.n_steps

        H_static = None
        for term in ir.static_terms:
            coeff = term.coefficient(0) if callable(term.coefficient) else term.coefficient
            contrib = coeff * term.operator
            H_static = contrib if H_static is None else H_static + contrib
        if H_static is None:
            H_static = 0

        stored_times: list[float] = []
        stored_states: list[np.ndarray] = []

        for k in range(self.n_steps):
            t_mid = (k + 0.5) * dt
            H = H_static.copy() if hasattr(H_static, "copy") else H_static
            for term in ir.drive_terms:
                coeff = term.coefficient(t_mid) if callable(term.coefficient) else term.coefficient
                H = H + coeff * term.operator
                if term.add_hermitian_conjugate:
                    H = H + np.conj(coeff) * term.operator.conj().T
            psi = expm_multiply(-1j * dt * H, psi)
            psi /= np.linalg.norm(psi)

            if t_eval is not None:
                stored_times.append((k + 1) * dt)
                stored_states.append(psi.copy())

        result = EvolutionResult(psi_final=psi, metadata=ir.metadata)
        if t_eval is not None:
            result.times = np.array(stored_times)
            result.states = np.array(stored_states)
        return result
