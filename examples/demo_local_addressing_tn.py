#!/usr/bin/env python3
"""Large-scale demo of local addressing experiments using a tensor-network backend.

Mirrors the experiments from demo_local_addressing.py but routes the same
``ryd_gate.simulate`` call through a tensor-network backend
(``backend="mps"`` or ``"peps"``) to support system sizes far beyond exact
diagonalization (default 10x10).

Requires the tensor-network extras: ``pip install ryd-gate[tn]`` (MPS) and/or
``ryd-gate[tn-2d]`` (PEPS).

Usage:
    python examples/demo_local_addressing_tn.py
    python examples/demo_local_addressing_tn.py --experiment domain
    python examples/demo_local_addressing_tn.py --Lx 16 --Ly 16
    python examples/demo_local_addressing_tn.py --backend peps --chi-max 512 --dt 0.1
"""

import argparse
import os
import time as _time
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from _coarsening import (
    build_neighbor_lists,
    coarsegrained_boundary_mask,
    correct_single_spin_flips,
    identify_domains,
)

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.physics import arc_pair_c6_rad_s_um6
from ryd_gate.protocols import SweepProtocol

# ---------------------------------------------------------------------------
# Physics constants (in units of Omega = 1)
# ---------------------------------------------------------------------------
OMEGA = 1.0  # rad/s; sets the (arbitrary) physical unit — all else is Omega-scaled
RYD_LEVEL = 70
V_NN = 24.0
DELTA_START = -3.0
DELTA_PIN = -4.0
T_SWEEP = 55.0
OMEGA_RAMP_FRAC = 0.1


def _build_config(Lx, Ly):
    """Shared lattice + level-structure setup for both experiments.

    Fixes the nearest-neighbour van der Waals interaction to ``V_NN`` (in units
    of ``OMEGA``) by choosing the spacing so ``C6/spacing**6 == V_NN * OMEGA`` and
    a cutoff just past the NN distance, reproducing the old NN-only interaction.
    """
    c6 = arc_pair_c6_rad_s_um6(
        n1=RYD_LEVEL, l1=0, j1=0.5, mj1=-0.5, mj2=-0.5,
        theta=0.0, phi=0.0, degenerate=False,
    )
    spacing = (c6 / (V_NN * OMEGA)) ** (1.0 / 6.0)
    cfg = SimpleNamespace(
        ls=level_structure("1r", ryd_level=RYD_LEVEL),
        register=Register.rectangle(Lx, Ly, spacing_um=spacing),
        cutoff=spacing * 1.2,  # NN only (NNN at sqrt(2)*spacing is excluded)
        sublattice=np.array([(-1) ** (r + c) for r in range(Lx) for c in range(Ly)]),
        grid=np.array([(r, c) for r in range(Lx) for c in range(Ly)], dtype=float),
        Lx=Lx, Ly=Ly, N=Lx * Ly,
    )
    return cfg


def _build_system(cfg, proto):
    return RydbergSystem(
        level_structure=cfg.ls,
        register=cfg.register,
        protocol=proto,
        interaction_cutoff_um=cfg.cutoff,
    )


def _make_continuous_protocol(delta_start, delta_end, t_sweep, t_hold, addressing=None):
    """One continuous piecewise sweep+hold schedule (units of Omega).

    Sweep phase (``t <= t_sweep``): Omega ramps up over the first
    ``OMEGA_RAMP_FRAC`` of the sweep then holds at full; Delta chirps
    ``delta_start -> delta_end``; the local pinning (``addressing``) is on.
    Hold phase (``t > t_sweep``): Omega and Delta stay at their end-of-sweep
    values and the pinning is released — one continuous schedule, no seam.
    """
    addressing = addressing or {}
    ramp_time = OMEGA_RAMP_FRAC * t_sweep

    def omega_half(t):
        frac = 1.0 if ramp_time == 0 else min(1.0, max(0.0, t / ramp_time))
        return 0.5 * OMEGA * frac

    def detuning(t):
        if t <= ramp_time:
            d = delta_start
        elif t <= t_sweep:
            chirp = max(t_sweep - ramp_time, np.finfo(float).eps)
            frac = np.clip((t - ramp_time) / chirp, 0.0, 1.0)
            d = delta_start + (delta_end - delta_start) * frac
        else:
            d = delta_end
        return d * OMEGA

    def local_detuning(t, i):
        return addressing.get(i, 0.0) * OMEGA if t <= t_sweep else 0.0

    return SweepProtocol(
        t_gate_s=t_sweep + t_hold,
        omega_half_rad_s=omega_half,
        detuning_rad_s=detuning,
        local_detuning_rad_s=local_detuning if addressing else None,
    )


def _is_in_domain(ix, iy, cx, cy, radius):
    """Whether grid site ``(ix, iy)`` lies in a square domain around ``(cx, cy)``."""
    return abs(ix - cx) <= radius and abs(iy - cy) <= radius


def _domain_config(grid, sublattice, center, radius):
    """AF1 bulk with an AF2 domain in a square region around ``center``."""
    config = (sublattice > 0).astype(int)  # AF1: sublattice +1 sites excited
    cx, cy = center
    for i, (ix, iy) in enumerate(grid):
        if _is_in_domain(ix, iy, cx, cy, radius):
            config[i] = 1 if sublattice[i] < 0 else 0
    return config


def _site_occupations(result, N):
    """(n_times, N) per-site ``<n_r_i>`` from the requested expectations."""
    return np.column_stack([result.expectation(f"n_r_{i}") for i in range(N)])


def _run(cfg, proto, t_eval, backend, backend_options):
    """One continuous evolution on the chosen TN backend; returns per-site occ."""
    system = _build_system(cfg, proto)
    site_obs = {f"n_r_{i}": system.observables.n("r", i) for i in range(cfg.N)}
    result = simulate(
        system, ["1"] * cfg.N, backend=backend, t_eval=t_eval,
        observables=site_obs, backend_options=backend_options,
    )
    return _site_occupations(result, cfg.N)


# ---------------------------------------------------------------------------
# Experiment 1: Domain shrinking (TN)
# ---------------------------------------------------------------------------


def run_domain_shrinking_tn(cfg, args, backend_options):
    """Prepare AF2 domain inside AF1 bulk, release, watch it shrink."""
    print("=" * 60)
    print("Experiment 1: Domain Shrinking (TN, curvature-driven coarsening)")
    print("=" * 60)

    N, grid, sublattice = cfg.N, cfg.grid, cfg.sublattice
    Lx, Ly = cfg.Lx, cfg.Ly

    Delta_f = 2.5
    cx, cy = (Lx - 1) / 2.0, (Ly - 1) / 2.0
    domain_radius = min(Lx, Ly) / 4.0
    t_hold = 6.0

    # Pin the bulk (target-ground) sites during the sweep so the AF2 domain forms;
    # the pinning is released for the hold phase (all within one protocol).
    print("\n  One continuous sweep (with pinning) + hold (pinning off)...")
    target = _domain_config(grid, sublattice, (cx, cy), domain_radius)
    addressing = {i: DELTA_PIN for i in range(N) if target[i] == 0}

    proto = _make_continuous_protocol(DELTA_START, Delta_f, T_SWEEP, t_hold, addressing)
    n_eval = min(args.n_eval, 50)
    t_eval = np.linspace(T_SWEEP, T_SWEEP + t_hold, n_eval)

    t0 = _time.time()
    occ_all = _run(cfg, proto, t_eval, args.backend, backend_options)
    print(f"    Sweep+hold done in {_time.time() - t0:.1f}s")

    hold_times = t_eval - T_SWEEP  # hold clock: 0 at end of sweep
    ms_sw = (occ_all[0] * 2 - 1) @ sublattice / N
    n_sw = occ_all[0].mean()
    print(f"    m_s after sweep: {ms_sw:.4f}")
    print(f"    <n> after sweep: {n_sw:.4f}")

    ms = (occ_all * 2 - 1) @ sublattice / N
    n_mean = occ_all.mean(axis=1)

    # --- Post-processing on final snapshot ---
    print("  Post-processing (coarsening analysis on final state)...")
    occ_final = occ_all[-1]
    occ_bin = (occ_final > 0.5).astype(float)
    nn_lists, nnn_lists = build_neighbor_lists(grid)
    occ_corr = correct_single_spin_flips(occ_bin, sublattice, nn_lists, nnn_lists)
    _, is_bnd = coarsegrained_boundary_mask(occ_corr, Lx, Ly)
    labels = identify_domains(occ_corr, sublattice, nn_lists)
    n_domains = len(np.unique(labels))
    print(f"    Final state: {n_domains} domains")

    # --- Plotting ---
    os.makedirs(args.figdir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: final local staggered magnetization
    local_ms = sublattice * (2 * occ_final - 1)
    im = axes[0, 0].imshow(
        local_ms.reshape(Lx, Ly), cmap="RdBu", vmin=-1, vmax=1, origin="lower", interpolation="nearest"
    )
    axes[0, 0].set_title("Final local staggered mag.")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.8)

    # Top-right: domain boundaries
    bnd_img = is_bnd.astype(float).reshape(Lx, Ly)
    axes[0, 1].imshow(bnd_img, cmap="Reds", vmin=0, vmax=1, origin="lower", interpolation="nearest")
    axes[0, 1].set_title(f"Boundaries ({n_domains} domains)")

    # Bottom-left: staggered magnetization over time
    axes[1, 0].plot(hold_times, ms, "b-", lw=1.5)
    axes[1, 0].set_xlabel("Hold time ($1/\\Omega$)")
    axes[1, 0].set_ylabel("$m_s$")
    axes[1, 0].set_title("Staggered magnetization")
    axes[1, 0].axhline(0, color="gray", ls="--", lw=0.5)

    # Bottom-right: mean Rydberg fraction over time
    axes[1, 1].plot(hold_times, n_mean, "g-", lw=1.5)
    axes[1, 1].set_xlabel("Hold time ($1/\\Omega$)")
    axes[1, 1].set_ylabel("$\\langle n \\rangle$")
    axes[1, 1].set_title("Mean Rydberg fraction")

    fig.suptitle(
        f"Domain Shrinking ({args.backend.upper()}, {Lx}x{Ly}, "
        f"$\\chi$={args.chi_max}, $\\Delta/\\Omega$={Delta_f:.1f})", fontsize=14
    )
    fig.tight_layout()
    path = os.path.join(args.figdir, "demo_domain_shrinking_tn.png")
    fig.savefig(path, dpi=150)
    print(f"\n  Figure saved to {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2: Higgs mode oscillations (TN)
# ---------------------------------------------------------------------------


def run_higgs_mode_tn(cfg, args, backend_options):
    """Pin one sublattice, release, observe order parameter oscillations."""
    print("\n" + "=" * 60)
    print("Experiment 2: Higgs Mode Oscillations (TN)")
    print("=" * 60)

    N, sublattice = cfg.N, cfg.sublattice

    Delta_values = [0.0, 1.1, 2.5]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(Delta_values)))
    addressing = {i: DELTA_PIN for i, s in enumerate(sublattice) if s > 0}
    t_hold = 10.0
    n_eval = min(args.n_eval, 50)

    all_results = {}
    for Delta_f in Delta_values:
        print(f"\n  --- Delta/Omega = {Delta_f:.1f} ---")

        # One continuous protocol: sweep with sublattice pinning, then hold with
        # the pinning released — no continuation seam, one simulate() call.
        proto = _make_continuous_protocol(DELTA_START, Delta_f, T_SWEEP, t_hold, addressing)
        t_eval = np.linspace(T_SWEEP, T_SWEEP + t_hold, n_eval)

        t0 = _time.time()
        occ = _run(cfg, proto, t_eval, args.backend, backend_options)
        hold_times = t_eval - T_SWEEP
        ms_sw = (occ[0] * 2 - 1) @ sublattice / N
        print(f"    Sweep+hold: {_time.time() - t0:.1f}s, m_s(sweep) = {ms_sw:.4f}")

        all_results[Delta_f] = {
            "times": hold_times,
            "ms": (occ * 2 - 1) @ sublattice / N,
            "n_mean": occ.mean(axis=1),
        }

    # --- Plotting ---
    os.makedirs(args.figdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for Delta_f, color in zip(Delta_values, colors):
        r = all_results[Delta_f]
        ax.plot(r["times"], r["ms"], color=color, lw=1.2, label=f"$\\Delta/\\Omega$ = {Delta_f:.1f}")
    ax.set_xlabel("Hold time ($1/\\Omega$)")
    ax.set_ylabel("Staggered magnetization $m_s$")
    ax.set_title("Order parameter oscillations")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", ls="--", lw=0.5)

    ax = axes[1]
    for Delta_f, color in zip(Delta_values, colors):
        r = all_results[Delta_f]
        if len(r["ms"]) < 4:
            continue
        ms_centered = r["ms"] - np.mean(r["ms"])
        dt = r["times"][1] - r["times"][0]
        freqs = np.fft.rfftfreq(len(ms_centered), d=dt)
        power = np.abs(np.fft.rfft(ms_centered * np.hanning(len(ms_centered)))) ** 2
        power[0] = 0
        pmax = power.max()
        ax.plot(
            freqs, power / pmax if pmax > 0 else power, color=color, lw=1.2, label=f"$\\Delta/\\Omega$ = {Delta_f:.1f}"
        )
    ax.set_xlabel("Frequency ($\\Omega / 2\\pi$)")
    ax.set_ylabel("Spectral power (normalized)")
    ax.set_title("Oscillation spectrum")
    ax.set_xlim(0, 2.0)
    ax.legend(fontsize=8)

    fig.suptitle(f"Higgs Mode ({args.backend.upper()}, {cfg.Lx}x{cfg.Ly}, $\\chi$={args.chi_max})", fontsize=14)
    fig.tight_layout()
    path = os.path.join(args.figdir, "demo_higgs_mode_tn.png")
    fig.savefig(path, dpi=150)
    print(f"\n  Figure saved to {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Large-scale local addressing demo on a tensor-network backend.")
    parser.add_argument("--experiment", choices=["domain", "higgs", "both"], default="both")
    parser.add_argument("--Lx", type=int, default=10)
    parser.add_argument("--Ly", type=int, default=10)
    parser.add_argument("--backend", choices=["mps", "peps"], default="mps",
                        help="Tensor-network backend passed to simulate() (default: mps)")
    parser.add_argument("--chi-max", type=int, default=256, help="Max bond dimension (default: 256)")
    parser.add_argument("--dt", type=float, default=0.2, help="TN time step (default: 0.2)")
    parser.add_argument("--discarded-weight-tol", type=float, default=1e-6,
                        help="Per-step discarded-weight tolerance, 0<tol<1 (default: 1e-6)")
    parser.add_argument("--n-eval", type=int, default=30, help="Number of observable evaluation points (default: 30)")
    parser.add_argument("--figdir", type=str,
                        default="results/lattice_dynamics/local_addressing/plots")
    args = parser.parse_args()

    print("Rydberg Array Local Addressing Demo (TN)")
    print(f"Lattice: {args.Lx} x {args.Ly} ({args.Lx * args.Ly} atoms)")
    print(f"Backend: {args.backend}, chi_max={args.chi_max}, dt={args.dt}")
    print()

    cfg = _build_config(args.Lx, args.Ly)
    # MPS and PEPS take different, non-overlapping option schemas. The register is a
    # Register.rectangle with an NN-only cutoff (see _build_config), so the same
    # physical system runs on either backend.
    if args.backend == "mps":
        # MPS TDVP options are exactly these three keys (E23).
        backend_options = {
            "time_step_s": args.dt,
            "bond_dimension": args.chi_max,
            "discarded_weight_tolerance": args.discarded_weight_tol,
        }
    else:
        # PEPS real-time options are exactly these ten keys; the NTU/environment
        # settings are report-only numerical evidence, not convergence gates.
        backend_options = {
            "time_step_s": args.dt,
            "bond_dimension": args.chi_max,
            "svd_tolerance": 1e-10,
            "ntu_max_iterations": 100,
            "ntu_iteration_tolerance": 1e-10,
            "measurement_method": "belief_propagation",  # one-site site occupations
            "environment_bond_dimension": args.chi_max,
            "environment_tolerance": 1e-8,
            "environment_max_iterations": 50,
            "device": "cpu",
        }

    if args.experiment in ("domain", "both"):
        run_domain_shrinking_tn(cfg, args, backend_options)
    if args.experiment in ("higgs", "both"):
        run_higgs_mode_tn(cfg, args, backend_options)

    print("\nDone.")


if __name__ == "__main__":
    main()
