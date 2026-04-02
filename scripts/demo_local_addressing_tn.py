#!/usr/bin/env python3
"""Large-scale demo of local addressing experiments using TeNPy (MPS/TDVP).

Mirrors the experiments from demo_local_addressing.py but uses tensor
network methods (DMRG + TDVP) to support system sizes up to 16x16.

Requires: pip install physics-tenpy  (or pip install ryd-gate[tn])

Usage:
    uv run python scripts/demo_local_addressing_tn.py
    uv run python scripts/demo_local_addressing_tn.py --experiment domain
    uv run python scripts/demo_local_addressing_tn.py --Lx 16 --Ly 16
    uv run python scripts/demo_local_addressing_tn.py --chi-max 512 --dt 0.1
"""

import argparse
import os
import time as _time

import numpy as np
import matplotlib.pyplot as plt

from ryd_gate.analysis.coarsening import (
    build_neighbor_lists,
    coarsegrained_boundary_mask,
    correct_single_spin_flips,
    identify_domains,
)
from ryd_gate.lattice import domain_config, is_in_domain
from ryd_gate.protocols.sweep import SweepProtocol
from ryd_gate.tn import TNLatticeSpec, create_tn_lattice_spec, simulate_tn
from ryd_gate.tn.observables import (
    measure_site_occupations,
    measure_staggered_magnetization,
    measure_mean_rydberg,
)
from ryd_gate.tn.state import product_state_mps, domain_state_mps

# ---------------------------------------------------------------------------
# Physics constants (in units of Omega = 1)
# ---------------------------------------------------------------------------
V_NN = 24.0
DELTA_START = -3.0
DELTA_PIN = -4.0
T_SWEEP = 55.0
OMEGA_RAMP_FRAC = 0.1


# ---------------------------------------------------------------------------
# Experiment 1: Domain shrinking (TN)
# ---------------------------------------------------------------------------

def run_domain_shrinking_tn(spec, args):
    """Prepare AF2 domain inside AF1 bulk, release, watch it shrink."""
    print("=" * 60)
    print("Experiment 1: Domain Shrinking (TN, curvature-driven coarsening)")
    print("=" * 60)

    Lx, Ly = spec.Lx, spec.Ly
    Delta_f = 2.5
    cx, cy = (Lx - 1) / 2.0, (Ly - 1) / 2.0
    domain_radius = min(Lx, Ly) / 4.0

    # --- Phase 1: Adiabatic sweep with pinning ---
    print("\n  Phase 1: Adiabatic sweep with local pinning...")
    target = domain_config(spec.coords, spec.sublattice, (cx, cy), domain_radius)
    addressing = {i: DELTA_PIN for i in range(spec.N) if target[i] == 0}

    sweep_proto = SweepProtocol(
        addressing=addressing,
        omega_ramp_frac=OMEGA_RAMP_FRAC,
    )

    t0 = _time.time()
    sweep_result = simulate_tn(
        spec, sweep_proto,
        [DELTA_START, Delta_f, T_SWEEP],
        initial_state="all_ground",
        method="tdvp",
        backend_options={"chi_max": args.chi_max, "dt": args.dt},
    )
    psi_after_sweep = sweep_result.psi_final
    print(f"    Sweep done in {_time.time() - t0:.1f}s")

    ms_sw = measure_staggered_magnetization(psi_after_sweep, spec)
    n_sw = measure_mean_rydberg(psi_after_sweep, spec)
    print(f"    m_s after sweep: {ms_sw:.4f}")
    print(f"    <n> after sweep: {n_sw:.4f}")

    # --- Phase 2: Free evolution (hold) ---
    print("  Phase 2: Free evolution (pinning off)...")
    t_hold = 6.0
    hold_proto = SweepProtocol()
    n_eval = min(args.n_eval, 50)
    t_eval = np.linspace(0, t_hold, n_eval)

    t0 = _time.time()
    hold_result = simulate_tn(
        spec, hold_proto,
        [Delta_f, Delta_f, t_hold],
        initial_state=psi_after_sweep,
        method="tdvp",
        t_eval=t_eval,
        observables=["m_s", "n_mean"],
        backend_options={"chi_max": args.chi_max, "dt": args.dt},
    )
    print(f"    Hold evolution done in {_time.time() - t0:.1f}s")

    obs = hold_result.metadata.get("obs", {})
    hold_times = hold_result.times
    ms = obs.get("m_s", np.array([]))
    n_mean = obs.get("n_mean", np.array([]))

    # --- Post-processing on final state ---
    print("  Post-processing (coarsening analysis on final state)...")
    occ_final = measure_site_occupations(hold_result.psi_final, spec)
    occ_bin = (occ_final > 0.5).astype(float)
    nn_lists, nnn_lists = build_neighbor_lists(spec.coords)
    occ_corr = correct_single_spin_flips(occ_bin, spec.sublattice, nn_lists, nnn_lists)
    _, is_bnd = coarsegrained_boundary_mask(occ_corr, Lx, Ly)
    labels = identify_domains(occ_corr, spec.sublattice, nn_lists)
    n_domains = len(np.unique(labels))
    print(f"    Final state: {n_domains} domains")

    # --- Plotting ---
    os.makedirs(args.figdir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: final local staggered magnetization
    local_ms = spec.sublattice * (2 * occ_final - 1)
    im = axes[0, 0].imshow(local_ms.reshape(Lx, Ly), cmap='RdBu', vmin=-1, vmax=1,
                            origin='lower', interpolation='nearest')
    axes[0, 0].set_title('Final local staggered mag.')
    fig.colorbar(im, ax=axes[0, 0], shrink=0.8)

    # Top-right: domain boundaries
    bnd_img = is_bnd.astype(float).reshape(Lx, Ly)
    axes[0, 1].imshow(bnd_img, cmap='Reds', vmin=0, vmax=1,
                      origin='lower', interpolation='nearest')
    axes[0, 1].set_title(f'Boundaries ({n_domains} domains)')

    # Bottom-left: staggered magnetization over time
    if len(ms) > 0 and hold_times is not None:
        axes[1, 0].plot(hold_times, ms, 'b-', lw=1.5)
    axes[1, 0].set_xlabel('Hold time ($1/\\Omega$)')
    axes[1, 0].set_ylabel('$m_s$')
    axes[1, 0].set_title('Staggered magnetization')
    axes[1, 0].axhline(0, color='gray', ls='--', lw=0.5)

    # Bottom-right: mean Rydberg fraction over time
    if len(n_mean) > 0 and hold_times is not None:
        axes[1, 1].plot(hold_times, n_mean, 'g-', lw=1.5)
    axes[1, 1].set_xlabel('Hold time ($1/\\Omega$)')
    axes[1, 1].set_ylabel('$\\langle n \\rangle$')
    axes[1, 1].set_title('Mean Rydberg fraction')

    fig.suptitle(f'Domain Shrinking (TN, {Lx}x{Ly}, '
                 f'$\\chi$={args.chi_max}, $\\Delta/\\Omega$={Delta_f:.1f})',
                 fontsize=14)
    fig.tight_layout()
    path = os.path.join(args.figdir, 'demo_domain_shrinking_tn.png')
    fig.savefig(path, dpi=150)
    print(f"\n  Figure saved to {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2: Higgs mode oscillations (TN)
# ---------------------------------------------------------------------------

def run_higgs_mode_tn(spec, args):
    """Pin one sublattice, release, observe order parameter oscillations."""
    print("\n" + "=" * 60)
    print("Experiment 2: Higgs Mode Oscillations (TN)")
    print("=" * 60)

    Delta_values = [0.0, 1.1, 2.5]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(Delta_values)))
    addressing = {i: DELTA_PIN for i, s in enumerate(spec.sublattice) if s > 0}

    all_results = {}
    for Delta_f in Delta_values:
        print(f"\n  --- Delta/Omega = {Delta_f:.1f} ---")

        sweep_proto = SweepProtocol(
            addressing=addressing,
            omega_ramp_frac=OMEGA_RAMP_FRAC,
        )
        t0 = _time.time()
        sweep_result = simulate_tn(
            spec, sweep_proto,
            [DELTA_START, Delta_f, T_SWEEP],
            initial_state="all_ground",
            method="tdvp",
            backend_options={"chi_max": args.chi_max, "dt": args.dt},
        )
        psi = sweep_result.psi_final
        ms_sw = measure_staggered_magnetization(psi, spec)
        print(f"    Sweep: {_time.time() - t0:.1f}s, m_s = {ms_sw:.4f}")

        hold_proto = SweepProtocol()
        t_hold = 10.0
        n_eval = min(args.n_eval, 50)
        t_eval = np.linspace(0, t_hold, n_eval)

        t0 = _time.time()
        hold_result = simulate_tn(
            spec, hold_proto,
            [Delta_f, Delta_f, t_hold],
            initial_state=psi,
            method="tdvp",
            t_eval=t_eval,
            observables=["m_s", "n_mean"],
            backend_options={"chi_max": args.chi_max, "dt": args.dt},
        )
        print(f"    Hold: {_time.time() - t0:.1f}s")

        obs = hold_result.metadata.get("obs", {})
        all_results[Delta_f] = {
            "times": hold_result.times,
            "ms": obs.get("m_s", np.array([])),
            "n_mean": obs.get("n_mean", np.array([])),
        }

    # --- Plotting ---
    os.makedirs(args.figdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for Delta_f, color in zip(Delta_values, colors):
        r = all_results[Delta_f]
        if len(r["ms"]) > 0 and r["times"] is not None:
            ax.plot(r["times"], r["ms"], color=color, lw=1.2,
                    label=f'$\\Delta/\\Omega$ = {Delta_f:.1f}')
    ax.set_xlabel('Hold time ($1/\\Omega$)')
    ax.set_ylabel('Staggered magnetization $m_s$')
    ax.set_title('Order parameter oscillations')
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', ls='--', lw=0.5)

    ax = axes[1]
    for Delta_f, color in zip(Delta_values, colors):
        r = all_results[Delta_f]
        if len(r["ms"]) < 4 or r["times"] is None:
            continue
        ms_centered = r["ms"] - np.mean(r["ms"])
        dt = r["times"][1] - r["times"][0]
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

    fig.suptitle(f'Higgs Mode (TN, {spec.Lx}x{spec.Ly}, '
                 f'$\\chi$={args.chi_max})', fontsize=14)
    fig.tight_layout()
    path = os.path.join(args.figdir, 'demo_higgs_mode_tn.png')
    fig.savefig(path, dpi=150)
    print(f"\n  Figure saved to {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Large-scale local addressing demo using TeNPy (MPS/TDVP).')
    parser.add_argument('--experiment', choices=['domain', 'higgs', 'both'],
                        default='both')
    parser.add_argument('--Lx', type=int, default=10)
    parser.add_argument('--Ly', type=int, default=10)
    parser.add_argument('--chi-max', type=int, default=256,
                        help='Max MPS bond dimension (default: 256)')
    parser.add_argument('--dt', type=float, default=0.2,
                        help='TDVP time step (default: 0.2)')
    parser.add_argument('--n-eval', type=int, default=30,
                        help='Number of observable evaluation points (default: 30)')
    parser.add_argument('--figdir', type=str, default='docs/figures')
    args = parser.parse_args()

    print(f"Rydberg Array Local Addressing Demo (TN)")
    print(f"Lattice: {args.Lx} x {args.Ly} ({args.Lx * args.Ly} atoms)")
    print(f"Backend: TDVP, chi_max={args.chi_max}, dt={args.dt}")
    print()

    spec = create_tn_lattice_spec(
        Lx=args.Lx, Ly=args.Ly, V_nn=V_NN, Omega=1.0)

    if args.experiment in ('domain', 'both'):
        run_domain_shrinking_tn(spec, args)
    if args.experiment in ('higgs', 'both'):
        run_higgs_mode_tn(spec, args)

    print("\nDone.")


if __name__ == '__main__':
    main()
