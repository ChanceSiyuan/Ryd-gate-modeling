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
    backend: "SolverBackend | None" = None,
    compiler=None,
) -> EvolutionResult:
    """Compile + evolve in one call.

    ``system`` must have a protocol bound. The default compiler is
    ``ExactSparseCompiler``; large systems should pass a tensor-network
    compiler/backend pair.
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
    compiler = compiler or ExactSparseCompiler()
    ir = compiler.compile(system, params)

    if backend is None:
        if ir.is_sparse:
            n = getattr(system.protocol, "n_steps", 200)
            backend = SparseExpmBackend(n_steps=n)
        else:
            backend = DenseODEBackend()

    return backend.evolve(ir, psi0, params["t_gate"], t_eval)
