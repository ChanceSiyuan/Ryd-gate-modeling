"""Pure-geometry helpers for N-atom Rydberg arrays.

Scope: the atom register (ids, coordinates, sublattice signs) and basic
helpers. This package does **not** define energy levels, Hamiltonians,
states, observables, interactions, or evolution; for those see:

- ``ryd_gate.core.interactions``           — pairwise VdW coupling computation
- ``ryd_gate.core.system``                 — level structure, Hamiltonian blocks,
                                              and observables for lattice models
- ``ryd_gate.core.states``                 — product/AF/domain/ground state constructors
- ``ryd_gate.analysis.lattice_observables``— bit/trit-mask occupation measurement
- ``exact.simulate``                    — time evolution (drives in
                                              ``ryd_gate.protocols``)

Contents
--------
- ``geometry`` — :class:`Register` (constructed via ``Register.chain`` /
  ``square`` / ``rectangle`` / ``triangular`` / ``from_coordinates``),
  :class:`RegisterLayout`, and the ``is_in_domain`` helper.
- ``plotting`` — ``plot_spatial_rydberg``, ``plot_population_evolution``
  (visualizations of physics quantities on lattice coordinates).
"""

from .geometry import Register, RegisterLayout, is_in_domain


def __getattr__(name: str):
    if name in {"plot_population_evolution", "plot_spatial_rydberg"}:
        from . import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module 'ryd_gate.lattice' has no attribute {name!r}")

__all__ = [
    # Geometry
    "Register",
    "RegisterLayout",
    "is_in_domain",
    # Plotting
    "plot_spatial_rydberg",
    "plot_population_evolution",
]
