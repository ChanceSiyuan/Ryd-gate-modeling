"""Post-processing and metrics for Rydberg simulation results.

Contents
--------
- ``gate_metrics``         — average_gate_infidelity, error_budget, sss_infidelity,
                             bell_infidelity, population_evolution, state_infidelity
- ``observable_metrics``   — measure_observables, measure_trajectory, state_overlap
- ``addressing_metrics``   — AddressingEvaluator: pinning error, crosstalk, leakage decomposition
- ``local_addressing``     — Default sweep parameters and evaluate_addressing() wrapper
- ``coarsening``           — Lattice domain identification, boundary masks, magnetization
- ``lattice_observables``  — Bit/trit-mask occupation, staggered magnetization for
                             many-body lattice states (2-level and 3-level)
- ``spin_observables``     — TFIM sigma_z / C_zz conversions and benchmark errors
- ``symmetry``             — D4 symmetry-error convergence diagnostics
"""

from .coarsening import (
    build_neighbor_lists,
    coarsegrained_boundary_mask,
    correct_single_spin_flips,
    domain_area_distribution,
    identify_domains,
    local_staggered_magnetization,
)
from .lattice_observables import (
    measure_from_states,
    measure_rydberg_occupation,
    precompute_bit_masks,
    precompute_trit_masks,
    staggered_magnetization,
)
from .spin_observables import (
    center_line_sites,
    center_reference_site,
    connected_zz_from_connected_nn,
    epsilon_z,
    epsilon_zz,
    line_pairs_from_reference,
    sigma_z_from_rydberg_occ,
)
from .symmetry import (
    SymmetryError,
    d4_permutations,
    d4_symmetry_error,
    first_unconverged_time,
)

__all__ = [
    # Coarsening / domain analysis
    "build_neighbor_lists",
    "correct_single_spin_flips",
    "coarsegrained_boundary_mask",
    "identify_domains",
    "domain_area_distribution",
    "local_staggered_magnetization",
    # Lattice observables (bit/trit-mask occupation measurement)
    "precompute_bit_masks",
    "precompute_trit_masks",
    "measure_from_states",
    "measure_rydberg_occupation",
    "staggered_magnetization",
    # TFIM / spin observables and paper-style benchmark metrics
    "sigma_z_from_rydberg_occ",
    "connected_zz_from_connected_nn",
    "center_line_sites",
    "center_reference_site",
    "line_pairs_from_reference",
    "epsilon_z",
    "epsilon_zz",
    # D4 symmetry diagnostics
    "SymmetryError",
    "d4_permutations",
    "d4_symmetry_error",
    "first_unconverged_time",
]
