"""Core primitives for the Rydberg lattice model.

Contents
--------
- ``system``          — RydbergSystem (universal model: lattice + level
  structure + protocol → symbolic blocks + observables)
- ``level_structures``— level/transition/interaction specs + presets
- ``rb87_params``     — Rb87 seven-level physical parameter sets
- ``local_blocks``    — single-atom Hamiltonian matrix blocks
- ``factories``       — RydbergSystem.from_lattice construction
- ``system_model``   — SystemModel ABC consumed by solvers
- ``basis``          — BasisSpec: site/level labels and Hilbert dimensions
- ``blocks``         — BlockRegistry: named matrix or symbolic Hamiltonian blocks
- ``operator_spec``  — symbolic local/sum/pair operator descriptions
- ``observables``    — ObservableRegistry: named measurement matrices or specs
- ``operators``      — Low-level operator builders (Kronecker embed, projectors)
- ``interactions``   — vdw_couplings for pairwise Rydberg interactions
- ``states``         — Many-body state constructors (product, AF, domain)
"""

from ryd_gate.core.basis import BasisSpec
from ryd_gate.core.blocks import BlockRegistry
from ryd_gate.core.level_structures import (
    DEFAULT_C6,
    InteractionSpec,
    LevelStructureSpec,
    TransitionSpec,
    level_structure,
)
from ryd_gate.core.observables import Observable, ObservableRegistry
from ryd_gate.core.system import RydbergSystem

__all__ = [
    "RydbergSystem",
    "LevelStructureSpec",
    "TransitionSpec",
    "InteractionSpec",
    "DEFAULT_C6",
    "level_structure",
    "BasisSpec",
    "BlockRegistry",
    "Observable",
    "ObservableRegistry",
]
