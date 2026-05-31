"""Pure-geometry helpers for N-atom Rydberg arrays.

Scope: lattice shapes, coordinates, sublattice signs, and basic helpers.
This package does **not** define energy levels, Hamiltonians, states,
observables, interactions, or evolution; for those see:

- ``ryd_gate.core.interactions``           — pairwise VdW coupling computation
- ``ryd_gate.core.rydberg_system``         — level structure, Hamiltonian blocks,
                                              and observables for lattice models
- ``ryd_gate.core.states``                 — product/AF/domain/ground state constructors
- ``ryd_gate.analysis.lattice_observables``— bit/trit-mask occupation measurement
- ``ryd_gate.solvers.dispatch.simulate``   — time evolution (drives in
                                              ``ryd_gate.protocols``)

Contents
--------
- ``geometry`` — LatticeGeometry dataclass; shape factories
  (``make_chain``, ``make_square_lattice``, ``make_triangular_lattice``,
  ``make_geometry_from_coords``); ``is_in_domain`` helper.
- ``plotting`` — ``plot_spatial_rydberg``, ``plot_population_evolution``
  (visualizations of physics quantities on lattice coordinates).
"""

from .geometry import (
    LatticeGeometry,
    is_in_domain,
    make_chain,
    make_geometry_from_coords,
    make_square_lattice,
    make_triangular_lattice,
)
from .plotting import plot_population_evolution, plot_spatial_rydberg

__all__ = [
    # Geometry
    "LatticeGeometry",
    "make_chain",
    "make_square_lattice",
    "make_triangular_lattice",
    "make_geometry_from_coords",
    "is_in_domain",
    # Plotting
    "plot_spatial_rydberg",
    "plot_population_evolution",
]
