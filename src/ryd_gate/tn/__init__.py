"""Optional tensor-network path for large-scale lattice systems.

Uses MPS (matrix product state) evolution via the TeNPy library and exposes
stable adapter boundaries for TTN, 2D-TN/BP, and NQS external solvers.

Install the optional dependency: ``pip install physics-tenpy``
GPU TN dispatch is available via ``simulate_tn(..., backend="gputn")`` once
CUDA tensor-network dependencies and a GPU engine are configured. Julia
ITensors dispatch is available with ``backend="itensors"``. Julia
TensorNetworkQuantumSimulator.jl dispatch is available with ``backend="2dtn"``.
External solver dispatch is available with ``backend="ttn"`` or ``"nqs"`` using
supported optional Python packages; ``backend="2dtn"`` can still use a Python
external solver when ``backend_options`` supplies ``engine`` or
``engine_package="yastn"/"quimb"``.

Contents
--------
- ``lattice_spec`` — TNLatticeSpec: lattice specification and 1D orderings
- ``simulate``     — simulate_tn(): MPS time evolution entry point
- ``model``        — TeNPy Hamiltonian model builder
- ``state``        — product_state_mps, domain_state_mps constructors
- ``observables``  — MPS observable measurements
- ``external_backends`` — adapter boundaries for TTN/2D-TN/NQS engines
"""

from .compiler import TNCompiler, TNEvolutionIR, tn_lattice_spec_from_system
from .external_backends import (
    External2DTNBPBackend,
    ExternalNQSTVMCBackend,
    ExternalSolverDependencyError,
    ExternalTTNTDVPBackend,
    available_external_solver_packages,
    build_external_solver_payload,
)
from .gpu_backends import GPUTNDependencyError, GPUTNTDVPBackend, gputn_available
from .itensors_bridge import ITensorsJuliaBackend, ITensorsJuliaError
from .lattice_spec import (
    TNLatticeSpec,
    create_tn_lattice_spec,
    diagonal_order_mapping,
    ordering_mapping,
    snake_order_mapping,
)
from .simulate import simulate_tn, simulate_tn_ir
from .state import mps_fidelity, product_superposition_mps
from .tnqs_bridge import TNQSJulia2DTNBackend, TNQSJuliaError

__all__ = [
    "GPUTNDependencyError",
    "GPUTNTDVPBackend",
    "ITensorsJuliaBackend",
    "ITensorsJuliaError",
    "TNQSJulia2DTNBackend",
    "TNQSJuliaError",
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
    "gputn_available",
    "ordering_mapping",
    "snake_order_mapping",
    "simulate_tn",
    "simulate_tn_ir",
    "tn_lattice_spec_from_system",
    "product_superposition_mps",
    "mps_fidelity",
]
