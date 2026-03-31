#!/usr/bin/env python
"""Validate 3-level lattice protocol: adiabatic sweep → checkerboard on 3×3.

Starting from |ggg...g⟩, a slow sweep of the 420nm laser phase adiabatically
prepares the Rydberg checkerboard (antiferromagnetic) phase on a 3×3 square
lattice, demonstrating that the many-body Rydberg blockade produces spatial
ordering.
"""

import matplotlib.pyplot as plt
import numpy as np

from ryd_gate.lattice import (
    build_3level_operators,
    evolve_3level_sweep,
    ground_state,
    make_3level_square_lattice,
    measure_rydberg_occupation,
    plot_spatial_rydberg,
    precompute_trit_masks,
    staggered_magnetization,
)


def main():
    Lx, Ly = 3, 3
    geom = make_3level_square_lattice(Lx, Ly, spacing_um=5.0)
    ops = build_3level_operators(
        geom, Delta=2 * np.pi * 9.1e9,
        Omega_1013=2 * np.pi * 491e6, Omega_420=2 * np.pi * 491e6,
    )
    psi0 = ground_state(geom.N)

    print(f"Running 3-level lattice sweep on {Lx}x{Ly} lattice...")
    print(f"  Hilbert space dim = 3^{geom.N} = {3**geom.N}")

    result = evolve_3level_sweep(
        psi0, ops,
        delta_start=-2 * np.pi * 40e6, delta_end=2 * np.pi * 40e6,
        t_sweep=1.5e-6, n_steps=300,
        store_every=10, verbose=True,
    )

    masks = precompute_trit_masks(geom.N)
    occ_final = measure_rydberg_occupation(result.psi_final, masks)
    ms = staggered_magnetization(occ_final, geom.sublattice)

    print(f"\nFinal staggered magnetization: m_s = {ms:.3f}")
    print(f"Per-atom Rydberg populations:")
    for i, (x, y) in enumerate(geom.coords):
        sub = '+' if geom.sublattice[i] > 0 else '-'
        print(f"  Atom {i} ({x:.0f},{y:.0f}) [{sub}]: P_r = {occ_final[i]:.3f}")

    # Plot spatial distribution
    fig = plot_spatial_rydberg(
        geom.coords, occ_final, geom.sublattice,
        title=f"3×3 checkerboard after adiabatic sweep (m_s = {ms:.2f})",
    )
    fig.savefig("fig_3level_lattice_checkerboard.png", dpi=150, bbox_inches='tight')
    print("\nSaved fig_3level_lattice_checkerboard.png")

    # Plot evolution if stored
    if result.states is not None:
        from ryd_gate.lattice import plot_population_evolution
        occ_traj = measure_rydberg_occupation(result.states, masks)
        fig2 = plot_population_evolution(
            result.times, occ_traj, geom.sublattice,
        )
        fig2.suptitle("Rydberg population during adiabatic sweep")
        fig2.tight_layout()
        fig2.savefig("fig_3level_lattice_evolution.png", dpi=150)
        print("Saved fig_3level_lattice_evolution.png")

    plt.show()


if __name__ == "__main__":
    main()
