"""Core primitives for the Rydberg lattice model.

Contents
--------
- ``system``           — RydbergSystem (level structure + register + protocol)
- ``model``            — BasisSpec
- ``observables``      — ObservableExpr / ObservableFactory (``system.observables``)
- ``level_structures`` — level-structure presets (resolved, immutable)
- ``operators``        — operator builders and symbolic operator specs
- ``physical_models``  — ARC interaction/C6 and Rb87 physical parameter resolution
- ``lowering``         — the canonical protocol→backend drive-lowering seam (P17)
- ``states``           — initial-state normalization + label→state compiler
- ``effective_theory`` — Schrieffer-Wolff / resolvent reductions
"""

from ryd_gate.core.level_structures import level_structure
from ryd_gate.core.model import BasisSpec
from ryd_gate.core.observables import ObservableExpr, ObservableFactory
from ryd_gate.core.system import RydbergSystem

__all__ = [
    "RydbergSystem",
    "level_structure",
    "BasisSpec",
    "ObservableExpr",
    "ObservableFactory",
]
