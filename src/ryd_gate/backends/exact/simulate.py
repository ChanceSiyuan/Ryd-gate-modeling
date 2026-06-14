"""Exact state-vector simulation entry point and typed options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.backends._options import as_backend_options
from ryd_gate.ir import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.backends.exact.compiler import SolverBackend


@dataclass(frozen=True)
class ExactOptions:
    """Options for :func:`ryd_gate.backends.exact.simulate`.

    ``None`` means "use the backend default".
    """

    n_steps: int | None = None


def simulate(
    system,
    x=(),
    psi0="all_ground",
    t_eval: np.ndarray | bool | None = None,
    backend: "SolverBackend | None" = None,
    compiler=None,
    backend_options: "dict | ExactOptions | None" = None,
) -> EvolutionResult:
    """Compile a protocol-bound Rydberg system and evolve exactly.

    ``system`` must have a protocol bound. With ``backend=None`` the solver is
    selected automatically: sparse piecewise-exponential evolution for a sparse
    Hamiltonian IR, otherwise a dense ODE integrator. Pass a concrete
    :class:`~ryd_gate.backends.exact.compiler.SolverBackend` to override. Backend
    dispatch by name lives in :func:`ryd_gate.simulate`; tensor-network and
    external algorithms live under ``ryd_gate.backends``. ``backend_options``
    accepts a dict or an :class:`ExactOptions`.
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

    compiler = compiler or ExactSparseCompiler()
    ir = compiler.compile(system, params)

    if backend is None:
        if ir.is_sparse:
            n = getattr(system.protocol, "n_steps", 200)
            backend = SparseExpmBackend(n_steps=opts.get("n_steps", n))
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
        if psi0 == "plus":
            from ryd_gate.core.states import plus_local_amplitudes, product_superposition_state

            return product_superposition_state(
                plus_local_amplitudes(system.basis.local_levels), system.N
            )
    if isinstance(psi0, (list, tuple)):
        return system.product_state(list(psi0))
    return psi0
