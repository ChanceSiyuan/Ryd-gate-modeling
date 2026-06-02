#!/usr/bin/env python
"""Validate a 3-level lattice protocol: detuning sweep -> checkerboard on 3x3.

Starting from |111...1>, a detuning sweep on the |1>-|r> transition prepares
a Rydberg checkerboard phase on a 3x3 square lattice.
"""

import matplotlib.pyplot as plt
import numpy as np

from ryd_gate import DigitalAnalogProtocol, RydbergSystem, Segment, simulate
from ryd_gate.analysis.lattice_observables import (
    measure_rydberg_occupation,
    precompute_trit_masks,
    staggered_magnetization,
)
from ryd_gate.model.system import InteractionSpec
from ryd_gate.lattice import make_square_lattice, plot_spatial_rydberg


def main():
    Lx, Ly = 3, 3
    geom = make_square_lattice(Lx, Ly, spacing_um=5.0)
    n_steps = 300
    delta_start = -2 * np.pi * 40e6
    delta_end = 2 * np.pi * 40e6
    t_sweep = 1.5e-6
    omega_R = 2 * np.pi * 5e6
    dt = t_sweep / n_steps
    segments = [
        Segment(
            duration=dt,
            omega_R=omega_R,
            delta_R=delta_start + (delta_end - delta_start) * (k + 0.5) / n_steps,
        )
        for k in range(n_steps)
    ]
    protocol = DigitalAnalogProtocol(segments, n_steps=n_steps)
    system = RydbergSystem.from_lattice(
        geom,
        "01r",
        interaction=InteractionSpec(mode="all"),
        protocol=protocol,
    )
    psi0 = system.product_state(["1"] * geom.N)

    print(f"Running 3-level 01r lattice sweep on {Lx}x{Ly} lattice...")
    print(f"  Hilbert space dim = 3^{geom.N} = {3**geom.N}")

    result = simulate(system, [], psi0)

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
