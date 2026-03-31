"""Many-body simulation of 2D Rydberg atom arrays (2-level and 3-level)."""

from .evolution import (
    EvolutionResult,
    evolve_3level_sweep,
    evolve_constant_H,
    evolve_sweep,
)
from .geometry import (
    LatticeGeometry,
    SquareLattice,
    is_in_domain,
    make_3level_square_lattice,
    make_geometry_from_coords,
    make_square_lattice,
)
from .observables import (
    measure_from_states,
    measure_rydberg_occupation,
    precompute_bit_masks,
    precompute_trit_masks,
    staggered_magnetization,
)
from .operators import (
    ThreeLevelOps,
    build_3level_operators,
    build_hamiltonian,
    build_hamiltonian_base,
    build_operators,
)
from .plotting import plot_population_evolution, plot_spatial_rydberg
from .solver import solve_lattice
from .states import (
    af_config,
    checkerboard_rydberg,
    domain_config,
    ground_state,
    product_state,
    product_state_3level,
)

__all__ = [
    # 2-level
    "SquareLattice",
    "make_square_lattice",
    "build_operators",
    "build_hamiltonian",
    "build_hamiltonian_base",
    "evolve_constant_H",
    "evolve_sweep",
    "solve_lattice",
    "product_state",
    "af_config",
    "domain_config",
    "is_in_domain",
    "precompute_bit_masks",
    "measure_from_states",
    # 3-level
    "LatticeGeometry",
    "make_3level_square_lattice",
    "make_geometry_from_coords",
    "ThreeLevelOps",
    "build_3level_operators",
    "EvolutionResult",
    "evolve_3level_sweep",
    "product_state_3level",
    "ground_state",
    "checkerboard_rydberg",
    "precompute_trit_masks",
    "measure_rydberg_occupation",
    "staggered_magnetization",
    "plot_spatial_rydberg",
    "plot_population_evolution",
]
