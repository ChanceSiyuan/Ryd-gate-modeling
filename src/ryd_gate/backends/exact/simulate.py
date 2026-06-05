"""Exact state-vector simulation entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.backends._options import as_backend_options
from ryd_gate.backends.exact.options import ExactOptions
from ryd_gate.ir.evolution import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.backends.exact.base import SolverBackend


def simulate(
    system,
    x,
    psi0,
    t_eval: np.ndarray | bool | None = None,
    backend: "SolverBackend | str | None" = None,
    compiler=None,
    backend_options: "dict | ExactOptions | None" = None,
) -> EvolutionResult:
    """Compile a protocol-bound Rydberg system and evolve exactly.

    ``system`` must have a protocol bound. The default backend is sparse
    piecewise-exponential exact state-vector evolution. Tensor-network and
    external algorithms live in their own backend packages under
    ``ryd_gate.backends``. ``backend_options`` accepts a dict or an
    :class:`ExactOptions`.
    """
    from ryd_gate.backends.exact.compiler import ExactSparseCompiler
    from ryd_gate.backends.exact.dense_ode import DenseODEBackend
    from ryd_gate.backends.exact.sparse_expm import SparseExpmBackend
    from ryd_gate.core.system import RydbergSystem

    if not isinstance(system, RydbergSystem):
        raise TypeError(
            "simulate() requires RydbergSystem instances. "
            "Build one with RydbergSystem.from_lattice(..., protocol=...) "
            "or call .with_protocol(...) before simulation."
        )

    params = system.unpack_params(x)
    opts = as_backend_options(backend_options)

    if isinstance(backend, str):
        backend_key = backend.lower()
        if backend_key in {"sparse", "sparse_expm", "exact"}:
            compiler = compiler or ExactSparseCompiler()
            ir = compiler.compile(system, params)
            n = getattr(system.protocol, "n_steps", 200)
            solver = SparseExpmBackend(n_steps=opts.get("n_steps", n))
            return solver.evolve(ir, _exact_initial_state(system, psi0), params["t_gate"], t_eval)

        raise ValueError(
            f"Unknown exact backend {backend!r}. Use 'sparse_expm'/'exact', "
            "or call tn_common.simulate_tn(...) for tensor-network algorithms."
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
        if psi0 == "all_ground":
            label = "1" if "1" in system.basis.local_levels else system.basis.local_levels[0]
            return system.product_state([label] * system.N)
        if psi0 == "all_1":
            return system.product_state(["1"] * system.N)
        if psi0 in {"all_0", "all_zero"}:
            return system.product_state(["0"] * system.N)
        if psi0 == "all_r":
            return system.product_state(["r"] * system.N)
    if isinstance(psi0, (list, tuple)):
        return system.product_state(list(psi0))
    return psi0
