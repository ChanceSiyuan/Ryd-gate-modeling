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

from ryd_gate.lattice import (
    build_hamiltonian,
    build_operators,
    domain_config,
    evolve_constant_H,
    evolve_sweep,
    is_in_domain,
    make_square_lattice,
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
    """Build lattice, operators, and bit masks (shared by both experiments)."""
    lattice = make_square_lattice(Lx, Ly)
    ops = build_operators(lattice.N, lattice.vdw_pairs, V_NN, verbose=True)
    bit_masks = precompute_bit_masks(lattice.N)
    return lattice, ops, bit_masks


# ---------------------------------------------------------------------------
# Experiment 1: Domain shrinking
# ---------------------------------------------------------------------------

def run_domain_shrinking(Lx, Ly, n_steps, figdir, setup=None):
    """Prepare an AF2 domain inside AF1 bulk, release, and watch it shrink."""
    print("=" * 60)
    print("Experiment 1: Domain Shrinking (curvature-driven coarsening)")
    print("=" * 60)

    lattice, ops, bit_masks = setup or _setup_experiment(Lx, Ly)
    N, coords, sublattice = lattice.N, lattice.coords, lattice.sublattice

    Delta_f = 2.5
    cx, cy = (Lx - 1) / 2.0, (Ly - 1) / 2.0
    domain_radius = 0.8

    # --- Phase 1: Adiabatic sweep with pinning ---
    print("\n  Phase 1: Adiabatic sweep with local pinning...")
    psi0 = product_state([0] * N, N)
    target = domain_config(coords, sublattice, (cx, cy), domain_radius)
    pin_deltas = np.where(target == 0, DELTA_PIN, 0.0)

    t0 = _time.time()
    psi_after_sweep = evolve_sweep(
        psi0, DELTA_START, Delta_f, T_SWEEP,
        min(n_steps, 40), pin_deltas, ops, OMEGA_RAMP_FRAC)
    print(f"    Sweep done in {_time.time() - t0:.1f}s")

    ms_sw, n_sw, _ = measure_from_states(psi_after_sweep, bit_masks, sublattice)
    print(f"    m_s after sweep: {ms_sw:.4f}")
    print(f"    <n> after sweep: {n_sw:.4f}")

    # --- Phase 2: Free evolution (hold) ---
    print("  Phase 2: Free evolution (pinning off)...")
    t_hold = 6.0
    H_hold = build_hamiltonian(1.0, Delta_f, np.zeros(N), ops)

    t0 = _time.time()
    hold_times, hold_states = evolve_constant_H(
        psi_after_sweep, H_hold, t_hold, n_steps)
    print(f"    Hold evolution done in {_time.time() - t0:.1f}s")

    print("  Computing observables...")
    ms, n_mean, occ_all = measure_from_states(hold_states, bit_masks, sublattice)

    # Domain area via vectorized dot product
    domain_weight = np.zeros(N)
    for i, (ix, iy) in enumerate(coords):
        if is_in_domain(ix, iy, cx, cy, domain_radius):
            domain_weight[i] = 1.0 if sublattice[i] < 0 else -1.0
    domain_areas = occ_all @ domain_weight + np.sum(domain_weight < 0)

    # --- Plotting ---
    os.makedirs(figdir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    snap_indices = [0, len(hold_times) // 4, len(hold_times) - 1]
    for col, idx in enumerate(snap_indices):
        ax = axes[0, col]
        local_ms = sublattice * (2 * occ_all[idx] - 1)
        im = ax.imshow(local_ms.reshape(Lx, Ly), cmap='RdBu', vmin=-1, vmax=1,
                       origin='lower', interpolation='nearest')
        ax.set_title(f't = {hold_times[idx]:.1f} / $\\Omega$')
        ax.set_xlabel('y')
        ax.set_ylabel('x')
    fig.colorbar(im, ax=axes[0, :], label='Local staggered mag.',
                 shrink=0.6, pad=0.02)

    axes[1, 0].plot(hold_times, ms, 'b-', lw=1.5)
    axes[1, 0].set_xlabel('Hold time ($1/\\Omega$)')
    axes[1, 0].set_ylabel('$m_s$')
    axes[1, 0].set_title('Staggered magnetization')
    axes[1, 0].axhline(0, color='gray', ls='--', lw=0.5)

    axes[1, 1].plot(hold_times, domain_areas, 'r-', lw=1.5)
    axes[1, 1].set_xlabel('Hold time ($1/\\Omega$)')
    axes[1, 1].set_ylabel('Domain area (sites)')
    axes[1, 1].set_title('Central domain area')

    axes[1, 2].plot(hold_times, n_mean, 'g-', lw=1.5)
    axes[1, 2].set_xlabel('Hold time ($1/\\Omega$)')
    axes[1, 2].set_ylabel('$\\langle n \\rangle$')
    axes[1, 2].set_title('Mean Rydberg fraction')

    fig.suptitle(f'Domain Shrinking Demo ({Lx}x{Ly} lattice, '
                 f'$\\Delta/\\Omega$ = {Delta_f:.1f})', fontsize=14)
    fig.subplots_adjust(top=0.88, hspace=0.35, wspace=0.3)
    path = os.path.join(figdir, 'demo_domain_shrinking.png')
    fig.savefig(path, dpi=150)
    print(f"\n  Figure saved to {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2: Higgs mode oscillations
# ---------------------------------------------------------------------------

def run_higgs_mode(Lx, Ly, n_steps, figdir, setup=None):
    """Pin one sublattice, release, and observe order parameter oscillations."""
    print("\n" + "=" * 60)
    print("Experiment 2: Higgs Mode Oscillations")
    print("=" * 60)

    lattice, ops, bit_masks = setup or _setup_experiment(Lx, Ly)
    N, sublattice = lattice.N, lattice.sublattice

    Delta_values = [0.0, 1.1, 2.5]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(Delta_values)))
    pin_deltas = np.where(sublattice > 0, DELTA_PIN, 0.0)

    all_results = {}
    for Delta_f in Delta_values:
        print(f"\n  --- Delta/Omega = {Delta_f:.1f} ---")
        psi0 = product_state([0] * N, N)

        t0 = _time.time()
        psi = evolve_sweep(
            psi0, DELTA_START, Delta_f, T_SWEEP,
            min(n_steps // 2, 40), pin_deltas, ops, OMEGA_RAMP_FRAC)
        ms_sw, _, _ = measure_from_states(psi, bit_masks, sublattice)
        print(f"    Sweep: {_time.time() - t0:.1f}s, m_s = {ms_sw:.4f}")

        H_hold = build_hamiltonian(1.0, Delta_f, np.zeros(N), ops)
        t0 = _time.time()
        hold_times, hold_states = evolve_constant_H(psi, H_hold, 10.0, n_steps)
        print(f"    Hold: {_time.time() - t0:.1f}s")

        ms, n_mean, _ = measure_from_states(hold_states, bit_masks, sublattice)
        all_results[Delta_f] = {'times': hold_times, 'ms': ms, 'n_mean': n_mean}

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
