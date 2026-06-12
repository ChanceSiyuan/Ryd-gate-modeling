"""Core primitives for the Rydberg lattice model.

Contents
--------
- ``system``           — RydbergSystem (universal model: lattice + level
  structure + protocol → symbolic blocks + observables) and the
  ``from_lattice`` construction logic
- ``model``            — BasisSpec, BlockRegistry, ObservableRegistry, and
  the SystemModel ABC consumed by solvers
- ``level_structures`` — level/transition/interaction specs + presets, plus
  protocol-channel lowering helpers
- ``operators``        — concrete operator builders (Kronecker embed,
  projectors) and symbolic local/sum/pair operator specs
- ``physical_models``  — vdw_couplings, Rb87 seven-level physical parameter
  sets, single-atom Hamiltonian matrix blocks
- ``serialization``    — ValidationIssue/raise_for_errors primitives and the
  schema-tag serialization contract
- ``states``           — many-body state constructors (product, AF, domain)
"""

from ryd_gate.core.level_structures import (
    DEFAULT_C6,
    InteractionSpec,
    LevelStructureSpec,
    TransitionSpec,
    level_structure,
)
from ryd_gate.core.model import (
    BasisSpec,
    BlockRegistry,
    Observable,
    ObservableRegistry,
)
from ryd_gate.core.serialization import ValidationIssue, raise_for_errors
from ryd_gate.core.system import RydbergSystem

__all__ = [
    "RydbergSystem",
    "LevelStructureSpec",
    "TransitionSpec",
    "InteractionSpec",
    "DEFAULT_C6",
    "level_structure",
    "ValidationIssue",
    "raise_for_errors",
    "BasisSpec",
    "BlockRegistry",
    "Observable",
    "ObservableRegistry",
]
