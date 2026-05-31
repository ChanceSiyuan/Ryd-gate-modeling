"""Optional tensor-network path for large-scale lattice systems (requires tenpy).

Uses MPS (matrix product state) evolution via the TeNPy library to simulate
Rydberg lattices beyond the dense-matrix regime (N ≳ 20 atoms).

Install the optional dependency: ``pip install physics-tenpy``

Contents
--------
- ``lattice_spec`` — TNLatticeSpec: MPS lattice specification and snake ordering
- ``simulate``     — simulate_tn(): MPS time evolution entry point
- ``model``        — TeNPy Hamiltonian model builder
- ``state``        — product_state_mps, domain_state_mps constructors
- ``observables``  — MPS observable measurements
"""

from .lattice_spec import TNLatticeSpec, create_tn_lattice_spec, snake_order_mapping
from .simulate import simulate_tn

__all__ = [
    "TNLatticeSpec",
    "create_tn_lattice_spec",
    "snake_order_mapping",
    "simulate_tn",
]
