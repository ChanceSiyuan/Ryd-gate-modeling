"""Optional tensor-network path for large-scale lattice systems.

Uses MPS (matrix product state) evolution via the TeNPy library to simulate
Rydberg lattices beyond the dense-matrix regime (N ≳ 20 atoms).

Install the optional dependency: ``pip install physics-tenpy``
GPU TN dispatch is available via ``simulate_tn(..., backend="gputn")`` once
CUDA tensor-network dependencies and a GPU engine are configured.

Contents
--------
- ``lattice_spec`` — TNLatticeSpec: MPS lattice specification and snake ordering
- ``simulate``     — simulate_tn(): MPS time evolution entry point
- ``model``        — TeNPy Hamiltonian model builder
- ``state``        — product_state_mps, domain_state_mps constructors
- ``observables``  — MPS observable measurements
"""

from .compiler import TNCompiler, TNEvolutionIR, tn_lattice_spec_from_system
from .gpu_backends import GPUTNDependencyError, GPUTNTDVPBackend, gputn_available
from .lattice_spec import TNLatticeSpec, create_tn_lattice_spec, snake_order_mapping
from .simulate import simulate_tn, simulate_tn_ir
from .state import mps_fidelity, product_superposition_mps

__all__ = [
    "GPUTNDependencyError",
    "GPUTNTDVPBackend",
    "TNCompiler",
    "TNEvolutionIR",
    "TNLatticeSpec",
    "create_tn_lattice_spec",
    "gputn_available",
    "snake_order_mapping",
    "simulate_tn",
    "simulate_tn_ir",
    "tn_lattice_spec_from_system",
    "product_superposition_mps",
    "mps_fidelity",
]
