"""Analysis and post-processing tools for Rydberg simulation results."""

from .coarsening import (
    build_neighbor_lists,
    coarsegrained_boundary_mask,
    correct_single_spin_flips,
    domain_area_distribution,
    identify_domains,
    local_staggered_magnetization,
)

__all__ = [
    "build_neighbor_lists",
    "correct_single_spin_flips",
    "coarsegrained_boundary_mask",
    "identify_domains",
    "domain_area_distribution",
    "local_staggered_magnetization",
]
