"""Post-processing and metrics for Rydberg simulation results.

Contents
--------
- ``gate_metrics`` — average_gate_infidelity, error_budget, sss_infidelity,
  bell_infidelity, population_evolution, state_infidelity
- ``observables``  — bit/trit-mask occupation and staggered magnetization for
  many-body lattice states (2-level and 3-level), and observable-registry
  metrics (measure_observables, measure_trajectory, state_overlap)
- ``addressing``   — AddressingEvaluator (pinning error, crosstalk, leakage
  decomposition), default sweep parameters, and evaluate_addressing() wrapper
- ``coarsening``   — Lattice domain identification, boundary masks,
  magnetization
"""

from .coarsening import (
    build_neighbor_lists,
    coarsegrained_boundary_mask,
    correct_single_spin_flips,
    domain_area_distribution,
    identify_domains,
    local_staggered_magnetization,
)
from .observables import (
    measure_from_states,
    measure_observables,
    measure_rydberg_occupation,
    measure_trajectory,
    norm_squared,
    precompute_bit_masks,
    precompute_trit_masks,
    staggered_magnetization,
    state_overlap,
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
    # Observable-registry measurement + overlaps
    "measure_observables",
    "measure_trajectory",
    "state_overlap",
    "norm_squared",
]
