"""High-level simulation entry point.

Provides :func:`simulate`, a convenience function that compiles a
system + protocol into a HamiltonianIR and evolves the state in one call.
Automatically selects an appropriate compiler and solver backend based on
the system type if not explicitly provided.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.solvers.base import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.compilers.base import Compiler
    from ryd_gate.protocols.base import Protocol
    from ryd_gate.solvers.base import SolverBackend


def simulate(
    model_or_system,
    protocol: Protocol,
    x: list[float],
    psi0: np.ndarray,
    t_eval: np.ndarray | None = None,
    backend: SolverBackend | None = None,
    compiler: Compiler | None = None,
) -> EvolutionResult:
    """Compile + evolve in one call.

    Automatically selects compiler and backend based on system type
    if not provided.

    Parameters
    ----------
    model_or_system : SystemModel, AtomicSystem, or LatticeSystem
        The quantum system. If a SystemModel, the underlying system
        is unwrapped for ``protocol.unpack_params()``.
    protocol : Protocol
        Pulse protocol providing drive coefficients.
    x : list of float
        Protocol parameter vector.
    psi0 : ndarray
        Initial state vector.
    t_eval : ndarray or None
        Time points at which to store intermediate states.
    backend : SolverBackend or None
        Solver backend. Auto-selected if None.
    compiler : Compiler or None
        Hamiltonian compiler. Auto-selected if None.

    Returns
    -------
    EvolutionResult
        Result containing psi_final and optionally times/states.
    """
    from ryd_gate.compilers.dense_atomic import DenseAtomicCompiler
    from ryd_gate.compilers.sparse_lattice import SparseLatticeCompiler
    from ryd_gate.core.atomic_system import AtomicSystem, LatticeSystem
    from ryd_gate.core.system_model import SystemModel
    from ryd_gate.solvers.dense_ode import DenseODEBackend
    from ryd_gate.solvers.sparse_expm import SparseExpmBackend

    # Unwrap SystemModel to get underlying system for unpack_params
    if isinstance(model_or_system, SystemModel):
        system = model_or_system.system
    else:
        system = model_or_system

    params = protocol.unpack_params(x, system)

    # Auto-select compiler
    if compiler is None:
        if isinstance(system, LatticeSystem):
            compiler = SparseLatticeCompiler()
        else:
            compiler = DenseAtomicCompiler()

    ir = compiler.compile(system, protocol, params)

    # Auto-select backend
    if backend is None:
        if ir.is_sparse:
            n = getattr(protocol, "n_steps", 200)
            backend = SparseExpmBackend(n_steps=n)
        else:
            backend = DenseODEBackend()

    return backend.evolve(ir, psi0, params["t_gate"], t_eval)
