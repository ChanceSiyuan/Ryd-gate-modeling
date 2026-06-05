"""Local GPU tensor-network kernels for ryd_gate.

The package lives next to :mod:`ryd_gate` so the high-level project can keep
its model/protocol code separate from CUDA-specific numerical kernels.
"""

from ryd_gate.backends.tn_common.compiler import (
    TNCompiler,
    TNEvolutionIR,
    tn_lattice_spec_from_hamiltonian_ir,
    tn_lattice_spec_from_system,
)

from .backend import GPUTNDependencyError, GPUTNTDVPBackend, gputn_available
from .options import GPUTNOptions
from .rydberg_engine import CuTensorNetRydbergEngine, GPUTNKernelError

__all__ = [
    "CuTensorNetRydbergEngine",
    "GPUTNDependencyError",
    "GPUTNKernelError",
    "GPUTNOptions",
    "GPUTNTDVPBackend",
    "TNCompiler",
    "TNEvolutionIR",
    "gputn_available",
    "tn_lattice_spec_from_system",
    "tn_lattice_spec_from_hamiltonian_ir",
]
