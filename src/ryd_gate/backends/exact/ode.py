"""The exact state-vector backend: adaptive ODE integration (scipy DOP853).

The only exact solver.  ``hamiltonian_format`` selects the Hamiltonian
storage / matvec strategy:

- ``"dense"`` — prebuild dense operators; each RHS call assembles the dense
  ``H(t)`` and applies one matmul.  Fastest for small stiff ladders (the rb87
  seven-level ~GHz intermediate manifold), unaffordable for large lattices.
- ``"sparse"`` — keep the compiled sparse operators and accumulate
  ``H(t) @ y`` term by term as sparse matvecs.  Never densifies anything
  (no ``.toarray()`` on this path).
- ``"auto"`` — dense while the estimated dense-Hamiltonian memory
  (``16 * dim**2`` bytes) stays below :data:`_DENSE_MEMORY_LIMIT_BYTES`,
  sparse above it.

The integration is adaptive (no ``n_steps``): the fast optical phases are
resolved with error control, which is what makes this backend free of the
step-commensurability aliasing the old piecewise-``expm`` solvers suffered
from.  Solve diagnostics (selected format, estimated dense bytes, tolerances,
RHS evaluation count) land in ``EvolutionResult.metadata``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.integrate

from ryd_gate.backends.exact.compiler import SolverBackend
from ryd_gate.core.observables import _dense_expectation
from ryd_gate.ir import EvolutionResult, validate_evolution_request

if TYPE_CHECKING:
    from ryd_gate.ir import HamiltonianIR

# Above this estimated dense-Hamiltonian size, "auto" must take the sparse
# matvec path (a dense dim x dim complex matrix costs 16 * dim**2 bytes).
_DENSE_MEMORY_LIMIT_BYTES = 256 * 2**20

_FORMATS = ("auto", "dense", "sparse")


def _dense_bytes_estimate(dim: int) -> int:
    return 16 * int(dim) * int(dim)


def _as_dense(operator) -> np.ndarray:
    """Densify an IR operator (dense path only; never called on the sparse path)."""
    if hasattr(operator, "toarray"):
        return np.asarray(operator.toarray())
    return np.asarray(operator)


def _as_sparse(operator):
    """CSR view of an IR operator (matvec-friendly; no densification)."""
    import scipy.sparse

    if scipy.sparse.issparse(operator):
        return operator.tocsr()
    return scipy.sparse.csr_matrix(np.asarray(operator))


class ExactODEBackend(SolverBackend):
    """Adaptive DOP853 state-vector solver with a storage/matvec format choice."""

    def __init__(
        self,
        rtol: float = 1e-8,
        atol: float = 1e-12,
        hamiltonian_format: str = "auto",
    ) -> None:
        if hamiltonian_format not in _FORMATS:
            raise ValueError(
                f"hamiltonian_format must be one of {_FORMATS}; got {hamiltonian_format!r}."
            )
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.hamiltonian_format = hamiltonian_format

    # -- format selection ----------------------------------------------------

    def _select_format(self, ir: "HamiltonianIR") -> tuple[str, int]:
        estimate = _dense_bytes_estimate(ir.dim)
        if self.hamiltonian_format != "auto":
            return self.hamiltonian_format, estimate
        if estimate > _DENSE_MEMORY_LIMIT_BYTES:
            return "sparse", estimate
        return "dense", estimate

    # -- RHS construction ------------------------------------------------------

    def _build_rhs(self, ir: "HamiltonianIR", fmt: str):
        if fmt == "dense":
            H_static = np.zeros((ir.dim, ir.dim), dtype=np.complex128)
            for term in ir.static_terms:
                coeff = term.coefficient(0) if callable(term.coefficient) else term.coefficient
                op = _as_dense(term.operator)
                H_static += coeff * op
                if term.add_hermitian_conjugate:
                    H_static += np.conj(coeff) * op.conj().T

            # Pre-bind drive terms once: (dense operator, coefficient, op_dag), so
            # the rhs (called many times by solve_ivp) never re-densifies.
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
                return -1j * (H @ y)

            return rhs

        # Sparse / matrix-free path: accumulate H(t) @ y as sparse matvecs.
        import scipy.sparse

        H_static = scipy.sparse.csr_matrix((ir.dim, ir.dim), dtype=np.complex128)
        for term in ir.static_terms:
            coeff = term.coefficient(0) if callable(term.coefficient) else term.coefficient
            op = _as_sparse(term.operator)
            H_static = H_static + coeff * op
            if term.add_hermitian_conjugate:
                H_static = H_static + np.conj(coeff) * op.conj().T
        H_static = H_static.tocsr()

        drive = []
        for term in ir.drive_terms:
            op = _as_sparse(term.operator)
            op_dag = op.conj().T.tocsr() if term.add_hermitian_conjugate else None
            drive.append((op, term.coefficient, op_dag))

        def rhs(t, y):
            out = H_static.dot(y)
            for operator, coefficient, op_dag in drive:
                coeff = coefficient(t) if callable(coefficient) else coefficient
                out = out + coeff * operator.dot(y)
                if op_dag is not None:
                    out = out + np.conj(coeff) * op_dag.dot(y)
            return -1j * out

        return rhs

    # -- evolution ---------------------------------------------------------------

    def evolve(
        self,
        ir: "HamiltonianIR",
        psi0: np.ndarray,
        t_gate: float,
        t_eval: np.ndarray | None = None,
        observables: dict | None = None,
    ) -> EvolutionResult:
        return self.evolve_many(ir, [psi0], t_gate, t_eval, observables)[0]

    def evolve_many(
        self,
        ir: "HamiltonianIR",
        psis: list[np.ndarray],
        t_gate: float,
        t_eval: np.ndarray | None = None,
        observables: dict | None = None,
        *,
        t_start: float = 0.0,
    ) -> list[EvolutionResult]:
        """Evolve several initial states over one compiled RHS.

        The Hamiltonian/RHS construction is shared; each initial state runs its
        own independent adaptive DOP853 solve (per-state error control).
        Requested-time expectations of the ``observables`` expressions are
        computed from the internally held trajectory, which is then discarded
        (results expose no state trajectory).  ``t_start`` is the private
        continuation seam: integrate ``[t_start, t_gate]`` instead of
        ``[0, t_gate]`` (used to resume from a dense state mid-protocol).
        """
        t_gate = float(t_gate)
        t_start = float(t_start)
        requested, exprs = validate_evolution_request(
            t_gate, t_eval, observables, t_start=t_start
        )
        _preflight_expressions(exprs, ir)
        fmt, estimate = self._select_format(ir)
        rhs = self._build_rhs(ir, fmt)

        # Internal endpoint handling stays private: solve on the requested grid
        # extended by t_gate when needed, and slice the extension back off.
        internal_t_eval = None
        has_endpoint = True
        if requested is not None:
            has_endpoint = bool(np.isclose(requested[-1], t_gate))
            internal_t_eval = (
                requested if has_endpoint else np.append(requested, t_gate)
            )

        results = []
        for psi0 in psis:
            solve_kwargs = dict(method="DOP853", rtol=self.rtol, atol=self.atol)
            if internal_t_eval is not None:
                solve_kwargs["t_eval"] = internal_t_eval
            solution = scipy.integrate.solve_ivp(
                rhs, [t_start, t_gate], np.asarray(psi0, dtype=complex), **solve_kwargs
            )
            if not solution.success:
                raise RuntimeError(
                    f"exact_ode solve_ivp failed: {solution.message} "
                    f"(format={fmt}, rtol={self.rtol}, atol={self.atol})"
                )
            metadata = dict(ir.metadata)
            metadata.update(
                {
                    "solver": "DOP853",
                    "hamiltonian_format": fmt,
                    "estimated_dense_hamiltonian_bytes": estimate,
                    "rtol": self.rtol,
                    "atol": self.atol,
                    "nfev": int(solution.nfev),
                    "solver_success": bool(solution.success),
                }
            )
            final_state = solution.y[:, -1]
            if requested is not None:
                # Temporarily held requested-time states, time-major; discarded
                # after the expectation evaluation below.
                states = solution.y[:, : requested.size].T
                times = requested.copy()
            else:
                states = final_state[np.newaxis, :]
                times = np.array([t_gate])
            expectations = {
                label: _dense_expectation(expr, states) for label, expr in exprs.items()
            }
            results.append(
                EvolutionResult(
                    final_state=final_state,
                    times=times,
                    expectations=expectations,
                    metadata=metadata,
                    basis=ir.basis,
                )
            )
        return results


def _preflight_expressions(exprs: dict, ir: "HamiltonianIR") -> None:
    """Exact-backend observable preflight (before any evolution).

    The exact backend evaluates every finite structured expression; the only
    capability requirement is that the expressions were built for this
    system's level structure (same site count and local dimension).
    """
    basis = ir.basis
    for label, expr in exprs.items():
        if (expr._n_sites, expr._local_dim) != (basis.n_sites, basis.local_dim):
            raise ValueError(
                f"observables[{label!r}] was built for (n_sites, local_dim)="
                f"{(expr._n_sites, expr._local_dim)} but this system has "
                f"{(basis.n_sites, basis.local_dim)}; build expressions from "
                "this system's system.observables factory."
            )
