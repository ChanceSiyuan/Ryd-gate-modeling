#!/usr/bin/env python3
"""Small-scale demo of local addressing experiments from Manovitz et al.

Simulates two key experiments from "Quantum coarsening and collective
dynamics on a programmable simulator" on a small (default 4x4) Rydberg
atom array:

  1. Domain shrinking (Fig 3): An AF2 domain inside AF1 bulk shrinks
     via curvature-driven coarsening after local pinning is released.

  2. Higgs mode (Fig 5): One sublattice is pinned then released,
     producing long-lived oscillations of the staggered magnetization.

The whole demo is dimensionless: energies and times are in units of the
Rabi frequency Omega (fixed to OMEGA = 1 rad/s below). The nearest-neighbour
van der Waals interaction is fixed to V_NN by choosing the lattice spacing so
that C6/spacing^6 = V_NN, with an interaction cutoff just past the NN distance
so the model stays nearest-neighbour only.

Usage:
    python examples/demo_local_addressing.py
    python examples/demo_local_addressing.py --experiment domain
    python examples/demo_local_addressing.py --experiment higgs
    python examples/demo_local_addressing.py --Lx 3 --Ly 3
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
    local_staggered_magnetization,
)

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.physics import arc_pair_c6_rad_s_um6
from ryd_gate.protocols import SweepProtocol

# ---------------------------------------------------------------------------
# Physics constants (in units of Omega = 1)
# ---------------------------------------------------------------------------
OMEGA = 1.0  # rad/s; sets the (arbitrary) physical unit — all else is Omega-scaled
RYD_LEVEL = 70  # nS Rydberg level backing the effective "1r" interaction
V_NN = 24.0  # nearest-neighbor van der Waals interaction
DELTA_START = -3.0  # sweep start (deep in disordered phase)
DELTA_PIN = -4.0  # local pinning detuning strength
T_SWEEP = 55.0  # adiabatic sweep duration
OMEGA_RAMP_FRAC = 0.1  # fraction of sweep for Omega ramp-up


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
    print(f"  Built {Lx}x{Ly} lattice system ({cfg.N} atoms, dim = {2 ** cfg.N})")
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


# ---------------------------------------------------------------------------
# Experiment 1: Domain shrinking
# ---------------------------------------------------------------------------


def run_domain_shrinking(cfg, n_steps, figdir):
    """Prepare an AF2 domain inside AF1 bulk, release, and watch it shrink."""
    print("=" * 60)
    print("Experiment 1: Domain Shrinking (curvature-driven coarsening)")
    print("=" * 60)

    N, grid, sublattice = cfg.N, cfg.grid, cfg.sublattice
    Lx, Ly = cfg.Lx, cfg.Ly

    Delta_f = 2.5
    cx, cy = (Lx - 1) / 2.0, (Ly - 1) / 2.0
    domain_radius = 0.8
    t_hold = 6.0

    # Pin the bulk (target-ground) sites during the sweep so the AF2 domain forms;
    # the pinning is released for the hold phase (all within one protocol).
    print("\n  One continuous sweep (with pinning) + hold (pinning off)...")
    target = _domain_config(grid, sublattice, (cx, cy), domain_radius)
    addressing = {i: DELTA_PIN for i in range(N) if target[i] == 0}

    proto = _make_continuous_protocol(DELTA_START, Delta_f, T_SWEEP, t_hold, addressing)
    system = _build_system(cfg, proto)
    site_obs = {f"n_r_{i}": system.observables.n("r", i) for i in range(N)}

    t_eval = np.linspace(T_SWEEP, T_SWEEP + t_hold, n_steps)
    t0 = _time.time()
    result = simulate(system, ["1"] * N, t_eval=t_eval, observables=site_obs)
    print(f"    Sweep+hold done in {_time.time() - t0:.1f}s")

    occ_all = _site_occupations(result, N)
    hold_times = t_eval - T_SWEEP  # hold clock: 0 at end of sweep

    # End-of-sweep profile (t == T_SWEEP, pinning still on).
    occ_sw = occ_all[0]
    ms_sw = (occ_sw * 2 - 1) @ sublattice / N
    n_sw = occ_sw.mean()
    print(f"    m_s after sweep: {ms_sw:.4f}")
    print(f"    <n> after sweep: {n_sw:.4f}")

    print("  Computing observables...")
    ms = (occ_all * 2 - 1) @ sublattice / N
    n_mean = occ_all.mean(axis=1)

    # Domain area via vectorized dot product
    domain_weight = np.zeros(N)
    for i, (ix, iy) in enumerate(grid):
        if _is_in_domain(ix, iy, cx, cy, domain_radius):
            domain_weight[i] = 1.0 if sublattice[i] < 0 else -1.0
    domain_areas = occ_all @ domain_weight + np.sum(domain_weight < 0)

    # --- Post-processing: both methods ---
    print("  Post-processing (coarsening analysis)...")
    nn_lists, nnn_lists = build_neighbor_lists(grid)

    # Pick the final snapshot for the comparison figure
    snap_idx = len(hold_times) - 1
    occ_raw = occ_all[snap_idx]  # continuous expectation values
    occ_bin = (occ_raw > 0.5).astype(float)  # binary threshold

    # ms.tex pipeline: continuous m_i on raw data
    m_local = local_staggered_magnetization(occ_bin, sublattice, nn_lists)

    # coarsen.tex pipeline: spin-flip correction then convolution
    occ_corr = correct_single_spin_flips(occ_bin, sublattice, nn_lists, nnn_lists)
    flipped_mask = occ_corr != occ_bin
    C_vals, is_boundary = coarsegrained_boundary_mask(occ_corr, Lx, Ly)
    labels = identify_domains(occ_corr, sublattice, nn_lists)
    n_domains = len(np.unique(labels))
    print(
        f"    Flipped sites: {int(flipped_mask.sum())}, Boundary sites: {int(is_boundary.sum())}, Domains: {n_domains}"
    )

    # ------------------------------------------------------------------ #
    #  Figure 1: ms.tex vs coarsen.tex comparison (2 rows x 2 cols)      #
    # ------------------------------------------------------------------ #
    os.makedirs(figdir, exist_ok=True)

    def _draw_lattice(ax, coords, values, cmap, vmin, vmax, labels_arr=None, fmt=".1f", title="", highlight=None):
        """Draw lattice sites as circles, colored by values, annotated."""
        xs = coords[:, 1]  # y-coord -> horizontal
        ys = coords[:, 0]  # x-coord -> vertical
        sc = ax.scatter(
            xs, ys, c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=700, edgecolors="k", linewidths=1.0, zorder=2
        )
        if highlight is not None:
            idx_h = np.where(highlight)[0]
            ax.scatter(xs[idx_h], ys[idx_h], s=700, facecolors="none", edgecolors="magenta", linewidths=3, zorder=3)
        if labels_arr is not None:
            for i in range(len(coords)):
                val = labels_arr[i]
                txt = f"{val:{fmt}}" if isinstance(val, float) else str(val)
                ax.annotate(txt, (xs[i], ys[i]), ha="center", va="center", fontsize=7, fontweight="bold", zorder=4)
        ax.set_xlim(-0.6, max(xs) + 0.6)
        ax.set_ylim(-0.6, max(ys) + 0.6)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        return sc

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # (0,0): ms.tex -- raw occupation n_i
    _draw_lattice(
        axes[0, 0],
        grid,
        occ_bin,
        "coolwarm",
        0,
        1,
        labels_arr=occ_bin.astype(float),
        fmt=".0f",
        title=r"(a) Raw $n_i$ [ms.tex input]",
    )

    # (1,0): ms.tex -- continuous m_i
    sc = _draw_lattice(
        axes[1, 0],
        grid,
        m_local,
        "RdBu",
        -1,
        1,
        labels_arr=m_local,
        fmt=".2f",
        title=r"(b) $m_i = (-1)^{x+y}(n_i - C_i/N_i)$ [ms.tex]",
    )
    fig.colorbar(sc, ax=axes[1, 0], label=r"$m_i$", shrink=0.85)

    # (0,1): coarsen.tex -- corrected occupation (flipped sites highlighted)
    _draw_lattice(
        axes[0, 1],
        grid,
        occ_corr,
        "coolwarm",
        0,
        1,
        labels_arr=occ_corr.astype(float),
        fmt=".0f",
        highlight=flipped_mask,
        title=r"(c) After spin-flip correction [coarsen.tex]"
        "\n(magenta = flipped)",
    )

    # (1,1): coarsen.tex -- bulk vs boundary classification
    # Encode: 0 = AF1 bulk, 1 = AF2 bulk, 2 = boundary
    af_type = sublattice * (2 * occ_corr - 1)  # +1 = AF1, -1 = AF2
    class_map = np.where(is_boundary, 2, np.where(af_type > 0, 0, 1)).astype(float)
    import matplotlib.colors as mcolors

    cmap_class = mcolors.ListedColormap(["#2196F3", "#FF9800", "#E53935"])
    _draw_lattice(
        axes[1, 1],
        grid,
        class_map,
        cmap_class,
        -0.5,
        2.5,
        labels_arr=None,
        title="(d) Bulk/boundary classification [coarsen.tex]",
    )
    # Legend for classification
    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor="#2196F3", label="AF1 bulk"),
        Patch(facecolor="#FF9800", label="AF2 bulk"),
        Patch(facecolor="#E53935", label="Boundary"),
    ]
    axes[1, 1].legend(handles=legend_elems, loc="upper right", fontsize=7, framealpha=0.9)

    fig.suptitle(
        f"Post-processing comparison  ({Lx}x{Ly},  "
        f"t = {hold_times[snap_idx]:.1f}/$\\Omega$,  "
        f"$\\Delta/\\Omega$ = {Delta_f:.1f})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(figdir, "demo_postprocessing_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"\n  Comparison figure saved to {path}")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    #  Figure 2: Time-series observables                                  #
    # ------------------------------------------------------------------ #
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

    axes2[0].plot(hold_times, ms, "b-", lw=1.5)
    axes2[0].set_xlabel("Hold time ($1/\\Omega$)")
    axes2[0].set_ylabel("$m_s$")
    axes2[0].set_title("Staggered magnetization")
    axes2[0].axhline(0, color="gray", ls="--", lw=0.5)

    axes2[1].plot(hold_times, domain_areas, "r-", lw=1.5)
    axes2[1].set_xlabel("Hold time ($1/\\Omega$)")
    axes2[1].set_ylabel("Domain area (sites)")
    axes2[1].set_title("Central domain area")

    axes2[2].plot(hold_times, n_mean, "g-", lw=1.5)
    axes2[2].set_xlabel("Hold time ($1/\\Omega$)")
    axes2[2].set_ylabel("$\\langle n \\rangle$")
    axes2[2].set_title("Mean Rydberg fraction")

    fig2.suptitle(f"Domain Shrinking ({Lx}x{Ly}, $\\Delta/\\Omega$ = {Delta_f:.1f})", fontsize=13)
    fig2.tight_layout()
    path2 = os.path.join(figdir, "demo_domain_shrinking.png")
    fig2.savefig(path2, dpi=150)
    print(f"  Time-series figure saved to {path2}")
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Experiment 2: Higgs mode oscillations
# ---------------------------------------------------------------------------


def run_higgs_mode(cfg, n_steps, figdir):
    """Pin one sublattice, release, and observe order parameter oscillations."""
    print("\n" + "=" * 60)
    print("Experiment 2: Higgs Mode Oscillations")
    print("=" * 60)

    N, sublattice = cfg.N, cfg.sublattice
    Lx, Ly = cfg.Lx, cfg.Ly

    Delta_values = [0.0, 1.1, 2.5]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(Delta_values)))
    addressing = {i: DELTA_PIN for i, s in enumerate(sublattice) if s > 0}
    t_hold = 10.0

    all_results = {}
    for Delta_f in Delta_values:
        print(f"\n  --- Delta/Omega = {Delta_f:.1f} ---")

        # One continuous protocol: sweep with sublattice pinning, then hold with
        # the pinning released — no continuation seam, one simulate() call.
        proto = _make_continuous_protocol(DELTA_START, Delta_f, T_SWEEP, t_hold, addressing)
        system = _build_system(cfg, proto)
        site_obs = {f"n_r_{i}": system.observables.n("r", i) for i in range(N)}

        t_eval = np.linspace(T_SWEEP, T_SWEEP + t_hold, n_steps)
        t0 = _time.time()
        result = simulate(system, ["1"] * N, t_eval=t_eval, observables=site_obs)

        occ = _site_occupations(result, N)
        hold_times = t_eval - T_SWEEP
        ms_sw = (occ[0] * 2 - 1) @ sublattice / N
        print(f"    Sweep+hold: {_time.time() - t0:.1f}s, m_s(sweep) = {ms_sw:.4f}")

        ms = (occ * 2 - 1) @ sublattice / N
        n_mean = occ.mean(axis=1)
        all_results[Delta_f] = {"times": hold_times, "ms": ms, "n_mean": n_mean}

    # --- Plotting ---
    os.makedirs(figdir, exist_ok=True)
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

    fig.suptitle(f"Higgs Mode Demo ({Lx}x{Ly} lattice)", fontsize=14)
    fig.tight_layout()
    path = os.path.join(figdir, "demo_higgs_mode.png")
    fig.savefig(path, dpi=150)
    print(f"\n  Figure saved to {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Demo of local addressing experiments on a Rydberg atom array.")
    parser.add_argument(
        "--experiment",
        choices=["domain", "higgs", "both"],
        default="both",
        help="Which experiment to run (default: both)",
    )
    parser.add_argument("--Lx", type=int, default=4, help="Lattice width (default: 4)")
    parser.add_argument("--Ly", type=int, default=4, help="Lattice height (default: 4)")
    parser.add_argument("--n-steps", type=int, default=200,
                        help="Hold-phase t_eval samples (default: 200); the exact solver itself is adaptive")
    parser.add_argument("--figdir", type=str,
                        default="results/lattice_dynamics/local_addressing/plots",
                        help="Output directory for figures")
    args = parser.parse_args()

    print("Rydberg Array Local Addressing Demo")
    print(f"Lattice: {args.Lx} x {args.Ly} ({args.Lx * args.Ly} atoms, dim = {2 ** (args.Lx * args.Ly)})")
    print()

    cfg = _build_config(args.Lx, args.Ly)

    if args.experiment in ("domain", "both"):
        run_domain_shrinking(cfg, args.n_steps, args.figdir)
    if args.experiment in ("higgs", "both"):
        run_higgs_mode(cfg, args.n_steps, args.figdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
