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
        t_eval: np.ndarray | bool | None = None,
    ) -> EvolutionResult:
        """Evolve by freezing H at each interval midpoint."""
        from scipy.sparse.linalg import expm_multiply

        psi = psi0.copy().astype(complex)
        dt = t_gate / self.n_steps
        record_at = self._record_steps(t_eval, t_gate, dt)

        H_static = None
        for term in ir.static_terms:
            coeff = term.coefficient(0) if callable(term.coefficient) else term.coefficient
            contrib = coeff * term.operator
            H_static = contrib if H_static is None else H_static + contrib
        if H_static is None:
            H_static = 0

        stored_times: list[float] = []
        stored_states: list[np.ndarray] = []

        if record_at is not None and 0 in record_at:
            stored_times.append(0.0)
            stored_states.append(psi.copy())

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

            step_num = k + 1
            if record_at is not None and step_num in record_at:
                stored_times.append((k + 1) * dt)
                stored_states.append(psi.copy())

        result = EvolutionResult(psi_final=psi, metadata=ir.metadata)
        if record_at is not None:
            result.times = np.array(stored_times)
            if stored_states:
                result.states = np.array(stored_states)
            else:
                result.states = np.empty((0, psi.size), dtype=complex)
        return result

    def _record_steps(
        self,
        t_eval: np.ndarray | bool | None,
        t_gate: float,
        dt: float,
    ) -> set[int] | None:
        """Map requested times to discrete sparse-expm step indices."""
        if t_eval is None:
            return None
        if isinstance(t_eval, (bool, np.bool_)):
            return set(range(1, self.n_steps + 1)) if bool(t_eval) else set()

        times = np.asarray(t_eval, dtype=float)
        if times.ndim != 1:
            raise ValueError("t_eval must be a one-dimensional array of times.")
        tol = max(1e-12, abs(t_gate) * 1e-12)
        if np.any(times < -tol) or np.any(times > t_gate + tol):
            raise ValueError("t_eval entries must lie within [0, t_gate].")
        steps = set()
        for t_req in times:
            step = int(round(float(t_req) / dt)) if dt != 0 else 0
            steps.add(max(0, min(step, self.n_steps)))
        return steps
