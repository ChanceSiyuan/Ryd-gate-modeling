"""Public simulation dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.backends.base import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.backends.base import SolverBackend


def simulate(
    system,
    x,
    psi0,
    t_eval: np.ndarray | None = None,
    backend: "SolverBackend | str | None" = None,
    compiler=None,
    backend_options: dict | None = None,
    method: str = "tdvp",
    observables: list[str] | None = None,
) -> EvolutionResult:
    """Compile + evolve in one call.

    ``system`` must have a protocol bound. The default backend is
    sparse-exponential exact state-vector evolution. Passing
    ``backend="tenpy"`` or ``backend="gputn"`` selects the TN compiler and
    the corresponding TN simulator.
    """
    from ryd_gate.backends.dense_ode import DenseODEBackend
    from ryd_gate.backends.sparse_expm import SparseExpmBackend
    from ryd_gate.compilers.exact_sparse import ExactSparseCompiler
    from ryd_gate.model.system import RydbergSystem

    if not isinstance(system, RydbergSystem):
        raise TypeError(
            "simulate() requires RydbergSystem instances. "
            "Build one with RydbergSystem.from_lattice(...) or from_preset(...)."
        )

    params = system.unpack_params(x)
    opts = backend_options or {}

    if isinstance(backend, str):
        backend_key = backend.lower()
        if backend_key in {"sparse", "sparse_expm", "exact"}:
            compiler = compiler or ExactSparseCompiler()
            ir = compiler.compile(system, params)
            n = getattr(system.protocol, "n_steps", 200)
            solver = SparseExpmBackend(n_steps=opts.get("n_steps", n))
            return solver.evolve(ir, _exact_initial_state(system, psi0), params["t_gate"], t_eval)

        if backend_key in {"tn", "tenpy", "gputn"}:
            from ryd_gate.tn.compiler import TNCompiler
            from ryd_gate.tn.simulate import simulate_tn_ir

            tn_compiler = compiler or TNCompiler(method=method)
            ir = tn_compiler.compile(system, params)
            tn_backend = "gputn" if backend_key == "gputn" else "tenpy"
            return simulate_tn_ir(
                ir,
                psi0,
                backend=tn_backend,
                t_eval=t_eval,
                observables=observables,
                backend_options=opts,
            )

        raise ValueError(
            f"Unknown backend {backend!r}. Use 'sparse_expm', 'tenpy', or 'gputn'."
        )

    compiler = compiler or ExactSparseCompiler()
    ir = compiler.compile(system, params)

    if backend is None:
        if ir.is_sparse:
            n = getattr(system.protocol, "n_steps", 200)
            backend = SparseExpmBackend(n_steps=n)
        else:
            backend = DenseODEBackend()

    return backend.evolve(ir, _exact_initial_state(system, psi0), params["t_gate"], t_eval)


def _exact_initial_state(system, psi0):
    if isinstance(psi0, np.ndarray):
        return psi0
    if isinstance(psi0, str):
        if psi0 in {"all_ground", "all_1"}:
            return system.product_state(["1"] * system.N)
        if psi0 in {"all_0", "all_zero"}:
            return system.product_state(["0"] * system.N)
        if psi0 == "all_r":
            return system.product_state(["r"] * system.N)
    if isinstance(psi0, (list, tuple)):
        return system.product_state(list(psi0))
    return psi0
