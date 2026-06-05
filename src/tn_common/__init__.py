"""Shared tensor-network IR, lattice specs, and dispatch helpers.

Concrete numerical kernels live in sibling packages such as ``tenpy_mps``,
``ttn``, ``gputn``, and ``itensor``.
"""

from .compiler import (
    TNCompiler,
    TNEvolutionIR,
    tn_lattice_spec_from_hamiltonian_ir,
    tn_lattice_spec_from_system,
)
from .external_backends import (
    External2DTNBPBackend,
    ExternalNQSTVMCBackend,
    ExternalSolverDependencyError,
    ExternalTTNTDVPBackend,
    available_external_solver_packages,
    build_external_solver_payload,
)
from .lattice_spec import (
    TNLatticeSpec,
    create_tn_lattice_spec,
    diagonal_order_mapping,
    ordering_mapping,
    snake_order_mapping,
)
from .simulate import simulate_tn, simulate_tn_ir
from .sites import local_levels, resolve_level_structure

__all__ = [
    "External2DTNBPBackend",
    "ExternalNQSTVMCBackend",
    "ExternalSolverDependencyError",
    "ExternalTTNTDVPBackend",
    "TNCompiler",
    "TNEvolutionIR",
    "TNLatticeSpec",
    "available_external_solver_packages",
    "build_external_solver_payload",
    "create_tn_lattice_spec",
    "diagonal_order_mapping",
    "local_levels",
    "ordering_mapping",
    "resolve_level_structure",
    "simulate_tn",
    "simulate_tn_ir",
    "snake_order_mapping",
    "tn_lattice_spec_from_system",
    "tn_lattice_spec_from_hamiltonian_ir",
]
