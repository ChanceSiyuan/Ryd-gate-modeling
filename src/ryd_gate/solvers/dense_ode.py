"""Dense ODE solver backend using scipy.integrate.solve_ivp (DOP853).

Suitable for small systems (dim <= ~200).  Wraps the same logic as
``evolve_ir()`` in ``schrodinger.py`` but returns an :class:`EvolutionResult`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.integrate

from ryd_gate.solvers.base import EvolutionResult, SolverBackend

if TYPE_CHECKING:
    from ryd_gate.compilers.ir import HamiltonianIR


class DenseODEBackend(SolverBackend):
    """Dense ODE solver backend using scipy.integrate.solve_ivp (DOP853).

    Suitable for small systems (dim <= ~200).

    Parameters
    ----------
    rtol : float
        Relative tolerance for the ODE solver.
    atol : float
        Absolute tolerance for the ODE solver.
    """

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
        """Evolve using DOP853 adaptive ODE integration.

        Precomputes the static Hamiltonian once, then evaluates
        time-dependent drive terms at each RHS call.
        """
        # Precompute static Hamiltonian (sum of all constant-coefficient terms)
        H_static = np.zeros((ir.dim, ir.dim), dtype=np.complex128)
        for term in ir.static_terms:
            coeff = term.coefficient(0) if callable(term.coefficient) else term.coefficient
            H_static += coeff * np.asarray(term.operator)

        def rhs(t, y):
            H = H_static.copy()
            for term in ir.drive_terms:
                coeff = term.coefficient(t) if callable(term.coefficient) else term.coefficient
                H += coeff * np.asarray(term.operator)
                if term.add_hermitian_conjugate:
                    H += np.conj(coeff) * np.asarray(term.operator).conj().T
            return -1j * H @ y

        solve_kwargs = dict(
            method="DOP853",
            rtol=self.rtol,
            atol=self.atol,
        )
        if t_eval is not None:
            solve_kwargs["t_eval"] = t_eval

        result = scipy.integrate.solve_ivp(
            rhs,
            [0, t_gate],
            psi0,
            **solve_kwargs,
        )

        if t_eval is not None:
            return EvolutionResult(
                psi_final=result.y[:, -1],
                times=result.t,
                states=result.y,
                metadata=ir.metadata,
            )
        return EvolutionResult(
            psi_final=result.y[:, -1],
            metadata=ir.metadata,
        )
