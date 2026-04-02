#!/usr/bin/env python3
"""Small-scale demo of local addressing experiments from Manovitz et al.

Simulates two key experiments from "Quantum coarsening and collective
dynamics on a programmable simulator" on a small (default 4x4) Rydberg
atom array:

  1. Domain shrinking (Fig 3): An AF2 domain inside AF1 bulk shrinks
     via curvature-driven coarsening after local pinning is released.

  2. Higgs mode (Fig 5): One sublattice is pinned then released,
     producing long-lived oscillations of the staggered magnetization.

Usage:
    uv run python scripts/demo_local_addressing.py
    uv run python scripts/demo_local_addressing.py --experiment domain
    uv run python scripts/demo_local_addressing.py --experiment higgs
    uv run python scripts/demo_local_addressing.py --Lx 3 --Ly 3
"""

import argparse
import os
import time as _time

import numpy as np
import matplotlib.pyplot as plt

from ryd_gate import create_lattice_system, SweepProtocol, simulate
from ryd_gate.analysis.coarsening import (
    build_neighbor_lists,
    coarsegrained_boundary_mask,
    correct_single_spin_flips,
    identify_domains,
    local_staggered_magnetization,
)
from ryd_gate.lattice import (
    domain_config,
    is_in_domain,
    measure_from_states,
    precompute_bit_masks,
    product_state,
)

# ---------------------------------------------------------------------------
# Physics constants (in units of Omega = 1)
# ---------------------------------------------------------------------------
V_NN = 24.0          # nearest-neighbor van der Waals interaction
DELTA_START = -3.0    # sweep start (deep in disordered phase)
DELTA_PIN = -4.0      # local pinning detuning strength
T_SWEEP = 55.0        # adiabatic sweep duration
OMEGA_RAMP_FRAC = 0.1 # fraction of sweep for Omega ramp-up


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def _setup_experiment(Lx, Ly):
    """Build lattice system and bit masks (shared by both experiments)."""
    system = create_lattice_system(Lx=Lx, Ly=Ly, V_nn=V_NN, Omega=1.0)
    bit_masks = precompute_bit_masks(system.N)
    print(f"  Built {Lx}x{Ly} lattice system ({system.N} atoms, "
          f"dim = {2**system.N})")
    return system, bit_masks


# ---------------------------------------------------------------------------
# Experiment 1: Domain shrinking
# ---------------------------------------------------------------------------

def run_domain_shrinking(Lx, Ly, n_steps, figdir, setup=None):
    """Prepare an AF2 domain inside AF1 bulk, release, and watch it shrink."""
    print("=" * 60)
    print("Experiment 1: Domain Shrinking (curvature-driven coarsening)")
    print("=" * 60)

    system, bit_masks = setup or _setup_experiment(Lx, Ly)
    N = system.N
    coords = system.coords
    sublattice = system.sublattice

    Delta_f = 2.5
    cx, cy = (Lx - 1) / 2.0, (Ly - 1) / 2.0
    domain_radius = 0.8

    # --- Phase 1: Adiabatic sweep with pinning ---
    print("\n  Phase 1: Adiabatic sweep with local pinning...")
    psi0 = product_state([0] * N, N)
    target = domain_config(coords, sublattice, (cx, cy), domain_radius)
    addressing = {i: DELTA_PIN for i in range(N) if target[i] == 0}

    sweep_proto = SweepProtocol(
        addressing=addressing,
        omega_ramp_frac=OMEGA_RAMP_FRAC,
        n_steps=min(n_steps, 40),
    )

    t0 = _time.time()
    sweep_result = simulate(
        system, sweep_proto,
        [DELTA_START, Delta_f, T_SWEEP],
        psi0,
    )
    psi_after_sweep = sweep_result.psi_final
    print(f"    Sweep done in {_time.time() - t0:.1f}s")

    ms_sw, n_sw, _ = measure_from_states(psi_after_sweep, bit_masks, sublattice)
    print(f"    m_s after sweep: {ms_sw:.4f}")
    print(f"    <n> after sweep: {n_sw:.4f}")

    # --- Phase 2: Free evolution (hold) ---
    print("  Phase 2: Free evolution (pinning off)...")
    t_hold = 6.0

    hold_proto = SweepProtocol(n_steps=n_steps)
    t0 = _time.time()
    hold_result = simulate(
        system, hold_proto,
        [Delta_f, Delta_f, t_hold],
        psi_after_sweep,
        t_eval=np.linspace(0, t_hold, n_steps),
    )
    hold_times = hold_result.times
    hold_states = hold_result.states
    print(f"    Hold evolution done in {_time.time() - t0:.1f}s")

    print("  Computing observables...")
    ms, n_mean, occ_all = measure_from_states(hold_states, bit_masks, sublattice)

    # Domain area via vectorized dot product
    domain_weight = np.zeros(N)
    for i, (ix, iy) in enumerate(coords):
        if is_in_domain(ix, iy, cx, cy, domain_radius):
            domain_weight[i] = 1.0 if sublattice[i] < 0 else -1.0
    domain_areas = occ_all @ domain_weight + np.sum(domain_weight < 0)

    # --- Post-processing: both methods ---
    print("  Post-processing (coarsening analysis)...")
    nn_lists, nnn_lists = build_neighbor_lists(coords)

    # Pick the final snapshot for the comparison figure
    snap_idx = len(hold_times) - 1
    occ_raw = occ_all[snap_idx]            # continuous expectation values
    occ_bin = (occ_raw > 0.5).astype(float)  # binary threshold

    # ms.tex pipeline: continuous m_i on raw data
    m_local = local_staggered_magnetization(occ_bin, sublattice, nn_lists)

    # coarsen.tex pipeline: spin-flip correction then convolution
    occ_corr = correct_single_spin_flips(occ_bin, sublattice, nn_lists, nnn_lists)
    flipped_mask = (occ_corr != occ_bin)
    C_vals, is_boundary = coarsegrained_boundary_mask(occ_corr, Lx, Ly)
    labels = identify_domains(occ_corr, sublattice, nn_lists)
    n_domains = len(np.unique(labels))
    print(f"    Flipped sites: {int(flipped_mask.sum())}, "
          f"Boundary sites: {int(is_boundary.sum())}, Domains: {n_domains}")

    # ------------------------------------------------------------------ #
    #  Figure 1: ms.tex vs coarsen.tex comparison (2 rows x 2 cols)      #
    # ------------------------------------------------------------------ #
    os.makedirs(figdir, exist_ok=True)

    def _draw_lattice(ax, coords, values, cmap, vmin, vmax, labels_arr=None,
                      fmt='.1f', title='', highlight=None):
        """Draw lattice sites as circles, colored by values, annotated."""
        xs = coords[:, 1]  # y-coord -> horizontal
        ys = coords[:, 0]  # x-coord -> vertical
        sc = ax.scatter(xs, ys, c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=700, edgecolors='k', linewidths=1.0, zorder=2)
        if highlight is not None:
            idx_h = np.where(highlight)[0]
            ax.scatter(xs[idx_h], ys[idx_h], s=700, facecolors='none',
                       edgecolors='magenta', linewidths=3, zorder=3)
        if labels_arr is not None:
            for i in range(len(coords)):
                val = labels_arr[i]
                txt = f'{val:{fmt}}' if isinstance(val, float) else str(val)
                ax.annotate(txt, (xs[i], ys[i]), ha='center', va='center',
                            fontsize=7, fontweight='bold', zorder=4)
        ax.set_xlim(-0.6, max(xs) + 0.6)
        ax.set_ylim(-0.6, max(ys) + 0.6)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        return sc

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # (0,0): ms.tex -- raw occupation n_i
    _draw_lattice(axes[0, 0], coords, occ_bin, 'coolwarm', 0, 1,
                  labels_arr=occ_bin.astype(float), fmt='.0f',
                  title=r'(a) Raw $n_i$ [ms.tex input]')

    # (1,0): ms.tex -- continuous m_i
    sc = _draw_lattice(axes[1, 0], coords, m_local, 'RdBu', -1, 1,
                       labels_arr=m_local, fmt='.2f',
                       title=r'(b) $m_i = (-1)^{x+y}(n_i - C_i/N_i)$ [ms.tex]')
    fig.colorbar(sc, ax=axes[1, 0], label=r'$m_i$', shrink=0.85)

    # (0,1): coarsen.tex -- corrected occupation (flipped sites highlighted)
    _draw_lattice(axes[0, 1], coords, occ_corr, 'coolwarm', 0, 1,
                  labels_arr=occ_corr.astype(float), fmt='.0f',
                  highlight=flipped_mask,
                  title=r'(c) After spin-flip correction [coarsen.tex]'
                  '\n(magenta = flipped)')

    # (1,1): coarsen.tex -- bulk vs boundary classification
    # Encode: 0 = AF1 bulk, 1 = AF2 bulk, 2 = boundary
    af_type = sublattice * (2 * occ_corr - 1)  # +1 = AF1, -1 = AF2
    class_map = np.where(is_boundary, 2, np.where(af_type > 0, 0, 1)).astype(float)
    import matplotlib.colors as mcolors
    cmap_class = mcolors.ListedColormap(['#2196F3', '#FF9800', '#E53935'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm_class = mcolors.BoundaryNorm(bounds, cmap_class.N)
    _draw_lattice(axes[1, 1], coords, class_map, cmap_class, -0.5, 2.5,
                  labels_arr=None,
                  title='(d) Bulk/boundary classification [coarsen.tex]')
    # Legend for classification
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor='#2196F3', label='AF1 bulk'),
                    Patch(facecolor='#FF9800', label='AF2 bulk'),
                    Patch(facecolor='#E53935', label='Boundary')]
    axes[1, 1].legend(handles=legend_elems, loc='upper right', fontsize=7,
                      framealpha=0.9)

    fig.suptitle(f'Post-processing comparison  ({Lx}x{Ly},  '
                 f't = {hold_times[snap_idx]:.1f}/$\\Omega$,  '
                 f'$\\Delta/\\Omega$ = {Delta_f:.1f})',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(figdir, 'demo_postprocessing_comparison.png')
    fig.savefig(path, dpi=150)
    print(f"\n  Comparison figure saved to {path}")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    #  Figure 2: Time-series observables                                  #
    # ------------------------------------------------------------------ #
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

    axes2[0].plot(hold_times, ms, 'b-', lw=1.5)
    axes2[0].set_xlabel('Hold time ($1/\\Omega$)')
    axes2[0].set_ylabel('$m_s$')
    axes2[0].set_title('Staggered magnetization')
    axes2[0].axhline(0, color='gray', ls='--', lw=0.5)

    axes2[1].plot(hold_times, domain_areas, 'r-', lw=1.5)
    axes2[1].set_xlabel('Hold time ($1/\\Omega$)')
    axes2[1].set_ylabel('Domain area (sites)')
    axes2[1].set_title('Central domain area')

    axes2[2].plot(hold_times, n_mean, 'g-', lw=1.5)
    axes2[2].set_xlabel('Hold time ($1/\\Omega$)')
    axes2[2].set_ylabel('$\\langle n \\rangle$')
    axes2[2].set_title('Mean Rydberg fraction')

    fig2.suptitle(f'Domain Shrinking ({Lx}x{Ly}, '
                  f'$\\Delta/\\Omega$ = {Delta_f:.1f})', fontsize=13)
    fig2.tight_layout()
    path2 = os.path.join(figdir, 'demo_domain_shrinking.png')
    fig2.savefig(path2, dpi=150)
    print(f"  Time-series figure saved to {path2}")
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Experiment 2: Higgs mode oscillations
# ---------------------------------------------------------------------------

def run_higgs_mode(Lx, Ly, n_steps, figdir, setup=None):
    """Pin one sublattice, release, and observe order parameter oscillations."""
    print("\n" + "=" * 60)
    print("Experiment 2: Higgs Mode Oscillations")
    print("=" * 60)

    system, bit_masks = setup or _setup_experiment(Lx, Ly)
    N = system.N
    sublattice = system.sublattice

    Delta_values = [0.0, 1.1, 2.5]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(Delta_values)))
    addressing = {i: DELTA_PIN for i, s in enumerate(sublattice) if s > 0}

    all_results = {}
    for Delta_f in Delta_values:
        print(f"\n  --- Delta/Omega = {Delta_f:.1f} ---")
        psi0 = product_state([0] * N, N)

        # Sweep phase with sublattice pinning
        sweep_proto = SweepProtocol(
            addressing=addressing,
            omega_ramp_frac=OMEGA_RAMP_FRAC,
            n_steps=min(n_steps // 2, 40),
        )
        t0 = _time.time()
        sweep_result = simulate(
            system, sweep_proto,
            [DELTA_START, Delta_f, T_SWEEP],
            psi0,
        )
        psi = sweep_result.psi_final
        ms_sw, _, _ = measure_from_states(psi, bit_masks, sublattice)
        print(f"    Sweep: {_time.time() - t0:.1f}s, m_s = {ms_sw:.4f}")

        # Hold phase (pinning off)
        hold_proto = SweepProtocol(n_steps=n_steps)
        t0 = _time.time()
        hold_result = simulate(
            system, hold_proto,
            [Delta_f, Delta_f, 10.0],
            psi,
            t_eval=np.linspace(0, 10.0, n_steps),
        )
        print(f"    Hold: {_time.time() - t0:.1f}s")

        ms, n_mean, _ = measure_from_states(
            hold_result.states, bit_masks, sublattice)
        all_results[Delta_f] = {
            'times': hold_result.times, 'ms': ms, 'n_mean': n_mean}

    # --- Plotting ---
    os.makedirs(figdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for Delta_f, color in zip(Delta_values, colors):
        r = all_results[Delta_f]
        ax.plot(r['times'], r['ms'], color=color, lw=1.2,
                label=f'$\\Delta/\\Omega$ = {Delta_f:.1f}')
    ax.set_xlabel('Hold time ($1/\\Omega$)')
    ax.set_ylabel('Staggered magnetization $m_s$')
    ax.set_title('Order parameter oscillations')
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', ls='--', lw=0.5)

    ax = axes[1]
    for Delta_f, color in zip(Delta_values, colors):
        r = all_results[Delta_f]
        ms_centered = r['ms'] - np.mean(r['ms'])
        dt = r['times'][1] - r['times'][0]
        freqs = np.fft.rfftfreq(len(ms_centered), d=dt)
        power = np.abs(np.fft.rfft(ms_centered * np.hanning(len(ms_centered)))) ** 2
        power[0] = 0
        pmax = power.max()
        ax.plot(freqs, power / pmax if pmax > 0 else power,
                color=color, lw=1.2,
                label=f'$\\Delta/\\Omega$ = {Delta_f:.1f}')
    ax.set_xlabel('Frequency ($\\Omega / 2\\pi$)')
    ax.set_ylabel('Spectral power (normalized)')
    ax.set_title('Oscillation spectrum')
    ax.set_xlim(0, 2.0)
    ax.legend(fontsize=8)

    fig.suptitle(f'Higgs Mode Demo ({Lx}x{Ly} lattice)', fontsize=14)
    fig.tight_layout()
    path = os.path.join(figdir, 'demo_higgs_mode.png')
    fig.savefig(path, dpi=150)
    print(f"\n  Figure saved to {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Demo of local addressing experiments on a Rydberg atom array.')
    parser.add_argument('--experiment', choices=['domain', 'higgs', 'both'],
                        default='both', help='Which experiment to run (default: both)')
    parser.add_argument('--Lx', type=int, default=4,
                        help='Lattice width (default: 4)')
    parser.add_argument('--Ly', type=int, default=4,
                        help='Lattice height (default: 4)')
    parser.add_argument('--n-steps', type=int, default=200,
                        help='Time steps for hold phase (default: 200)')
    parser.add_argument('--figdir', type=str, default='docs/figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    print(f"Rydberg Array Local Addressing Demo")
    print(f"Lattice: {args.Lx} x {args.Ly} ({args.Lx * args.Ly} atoms, "
          f"dim = {2**(args.Lx * args.Ly)})")
    print()

    run_both = args.experiment == 'both'
    setup = _setup_experiment(args.Lx, args.Ly) if run_both else None

    if args.experiment in ('domain', 'both'):
        run_domain_shrinking(args.Lx, args.Ly, args.n_steps, args.figdir, setup)
    if args.experiment in ('higgs', 'both'):
        run_higgs_mode(args.Lx, args.Ly, args.n_steps, args.figdir, setup)

    print("\nDone.")


if __name__ == '__main__':
    main()
