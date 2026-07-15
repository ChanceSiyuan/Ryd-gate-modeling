"""Shared tensor-network IR, lattice specs, and dispatch helpers.

Concrete numerical kernels live in sibling packages: ``tenpy_mps`` and
``peps2d``.
"""

from .compiler import (
    TNEvolutionIR,
    tn_lattice_spec_from_hamiltonian_ir,
    tn_lattice_spec_from_system,
)
from .lattice_spec import (
    TNLatticeSpec,
    create_tn_lattice_spec,
    diagonal_order_mapping,
    ordering_mapping,
    snake_order_mapping,
)
from .sites import local_levels, resolve_level_structure

__all__ = [
    "TNEvolutionIR",
    "TNLatticeSpec",
    "create_tn_lattice_spec",
    "diagonal_order_mapping",
    "local_levels",
    "ordering_mapping",
    "resolve_level_structure",
    "snake_order_mapping",
    "tn_lattice_spec_from_system",
    "tn_lattice_spec_from_hamiltonian_ir",
]
