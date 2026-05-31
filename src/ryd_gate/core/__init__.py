"""Core primitives for the Rydberg lattice model.

Contents
--------
- ``rydberg_system`` — RydbergSystem (universal model: lattice + level
  structure + protocol → blocks + observables + time-dependent Hamiltonian)
- ``system_model``   — SystemModel ABC consumed by solvers
- ``basis``          — BasisSpec: site/level labels and Hilbert dimensions
- ``blocks``         — BlockRegistry: named Hamiltonian operator blocks
- ``observables``    — ObservableRegistry: named measurement operators
- ``operators``      — Low-level operator builders (Kronecker embed, projectors)
- ``interactions``   — vdw_couplings for pairwise Rydberg interactions
- ``states``         — Many-body state constructors (product, AF, domain)
"""

from ryd_gate.core.rydberg_system import (
    DEFAULT_C6,
    InteractionSpec,
    LevelStructureSpec,
    RydbergSystem,
    RydbergSystemModel,
    TransitionSpec,
    level_structure,
)

__all__ = [
    "RydbergSystem",
    "RydbergSystemModel",
    "LevelStructureSpec",
    "TransitionSpec",
    "InteractionSpec",
    "DEFAULT_C6",
    "level_structure",
]
