"""High-level simulation entry point.

Provides :func:`simulate`, which evolves a :class:`RydbergSystem`
(with a protocol already bound) under its time-dependent Hamiltonian
for the duration encoded in the protocol parameters.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.solvers.base import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.protocols.base import Protocol
    from ryd_gate.solvers.base import SolverBackend


def simulate(
    system,
    x_or_protocol,
    psi0_or_x=None,
    psi0_legacy=None,
    t_eval: np.ndarray | None = None,
    backend: "SolverBackend | None" = None,
    compiler=None,
) -> EvolutionResult:
    """Compile + evolve in one call.

    Preferred signature::

        simulate(system, x, psi0, t_eval=None, backend=None)

    where ``system`` is a :class:`RydbergSystem` constructed with a
    ``protocol=`` argument.

    The legacy signature ``simulate(system, protocol, x, psi0, ...)`` is
    accepted for one release and emits a DeprecationWarning.
    """
    from ryd_gate.core.rydberg_system import RydbergSystem
    from ryd_gate.core.system_model import SystemModel
    from ryd_gate.solvers.dense_ode import DenseODEBackend
    from ryd_gate.solvers.sparse_expm import SparseExpmBackend
    from ryd_gate.protocols.base import Protocol

    # Detect legacy (system, protocol, x, psi0) form
    if isinstance(x_or_protocol, Protocol):
        warnings.warn(
            "simulate(system, protocol, x, psi0, ...) is deprecated. "
            "Bind the protocol on the system "
            "(RydbergSystem(..., protocol=protocol)) and call "
            "simulate(system, x, psi0).",
            DeprecationWarning,
            stacklevel=2,
        )
        protocol = x_or_protocol
        x = psi0_or_x
        psi0 = psi0_legacy
        if isinstance(system, SystemModel):
            system = system.system
        if not isinstance(system, RydbergSystem):
            raise TypeError(
                "simulate() now requires RydbergSystem instances. "
                "Build one with RydbergSystem.from_lattice(...) or from_preset(...)."
            )
        system = system.with_protocol(protocol)
    else:
        x = x_or_protocol
        psi0 = psi0_or_x
        if isinstance(system, SystemModel):
            system = system.system
        if not isinstance(system, RydbergSystem):
            raise TypeError(
                "simulate() now requires RydbergSystem instances. "
                "Build one with RydbergSystem.from_lattice(...) or from_preset(...)."
            )

    if compiler is not None:
        warnings.warn(
            "`compiler=` is ignored; RydbergSystem.compile_ir is used instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    params = system.unpack_params(x)
    ir = system.compile_ir(params)

    if backend is None:
        if ir.is_sparse:
            n = getattr(system.protocol, "n_steps", 200)
            backend = SparseExpmBackend(n_steps=n)
        else:
            backend = DenseODEBackend()

    return backend.evolve(ir, psi0, params["t_gate"], t_eval)
