"""Core primitives for the Rydberg lattice model.

Contents
--------
- ``system``           — RydbergSystem (universal model: level structure +
  register + protocol → operators/channels + observables)
- ``model``            — BasisSpec
- ``observables``      — ObservableExpr / ObservableFactory (scalar observable
  expressions; ``system.observables`` is the bound factory)
- ``level_structures`` — level-structure presets (resolved, immutable) +
  InteractionSpec + protocol-channel lowering helpers
- ``operators``        — concrete operator builders (Kronecker embed,
  projectors) and symbolic local/sum/pair operator specs
- ``physical_models``  — vdw_couplings, Rb87 physical parameter resolution,
  single-atom Hamiltonian matrix blocks
- ``states``           — many-body state constructors (product, AF, domain)
- ``effective_theory`` — Schrieffer-Wolff / resolvent reductions and the
  CZ -> effective {0,1,r} lowerers
"""

from ryd_gate.core.level_structures import (
    DEFAULT_C6,
    InteractionSpec,
    level_structure,
)
from ryd_gate.core.model import BasisSpec
from ryd_gate.core.observables import ObservableExpr, ObservableFactory
from ryd_gate.core.system import RydbergSystem

__all__ = [
    "RydbergSystem",
    "InteractionSpec",
    "DEFAULT_C6",
    "level_structure",
    "BasisSpec",
    "ObservableExpr",
    "ObservableFactory",
]
