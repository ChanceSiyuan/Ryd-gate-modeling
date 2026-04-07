#!/usr/bin/env python3
"""Single-qubit local addressing analysis: wavelength x power optimization.

Parallel to scan_local_addressing.py but uses single-atom (N=1) simulations
instead of the 2-atom Rydberg blockade model. Two independent 1-atom
simulations per grid point:

  Sim A (pinned atom): delta_A applied, should stay in |g>.
  Sim B (virtual crosstalk atom at distance d): delta_B = eta * delta_A applied,
      should undergo adiabatic passage |g> -> |r>.

Subcommands:
    optimize       Run 2D wavelength x power scan + plot.
    optimize-plot  Re-plot from saved CSV (no simulation).

Usage:
    uv run python scripts/scan_local_addressing_singlequbit.py optimize
    uv run python scripts/scan_local_addressing_singlequbit.py optimize-plot
"""

import argparse
import os
import time as _time

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi

from ryd_gate.core.atomic_system import (
    POWER_REF_UW,
    compute_shift_scatter,
    create_analog_system,
)
from ryd_gate.protocols.sweep import SweepProtocol


# ═══════════════════════════════════════════════════════════════════════
# Data generation
# ═══════════════════════════════════════════════════════════════════════

def _run_optimize_scan(args):
    """Run the 2D grid scan with two 1-atom sims per point."""
    from ryd_gate import simulate

    print("=" * 60)
    print("  Single-Qubit Local Addressing: Wavelength x Power Scan")
    print("=" * 60)

    system = create_analog_system(
        detuning_sign=1, blackmanflag=True,
        n_atoms=1,
        Delta_Hz=2.4e9,
        rabi_420_Hz=135e6,
        rabi_1013_Hz=135e6,
    )
    psi0 = np.array([1.0, 0.0, 0.0], dtype=complex)  # |g>

    ac_stark_peak = system.rabi_420 ** 2 / (4 * abs(system.Delta))

    t_gate = args.t_gate_us * 1e-6
    x_sweep = [
        2 * pi * args.delta_start_mhz * 1e6 / system.rabi_eff,
        2 * pi * args.delta_end_mhz * 1e6 / system.rabi_eff,
        t_gate / system.time_scale,
    ]

    eta = np.exp(-2 * (args.distance_um / args.waist_um) ** 2)
    print(f"  Gaussian tail: eta = exp(-2*(a/w)^2) = {eta:.4e}")
    print(f"  Omega_eff/(2pi) = {system.rabi_eff/(2*pi)/1e6:.2f} MHz")
    print(f"  AC Stark peak = {ac_stark_peak/(2*pi)/1e6:.2f} MHz")
    print(f"  Sweep: [{args.delta_start_mhz}, {args.delta_end_mhz}] MHz, "
          f"T_gate = {args.t_gate_us} us")
    print(f"  N_atoms = 1 (no Rydberg interaction)")
    print()

    wls = np.linspace(args.wl_min, args.wl_max, args.n_wl)

    # Adiabatic threshold: |delta_A| > 4*Omega_eff
    adiabatic_ratio = 4.0
    adiabatic_power_min = {}
    print(f"  Adiabatic condition: |delta_A| > {adiabatic_ratio:.0f} Omega_eff "
          f"= {adiabatic_ratio * system.rabi_eff/(2*pi)/1e6:.1f} MHz")
    print(f"  {'WL (nm)':>10} {'|shift|/P (MHz/uW)':>20} {'P_min_adiab (uW)':>18}")
    for wl in wls:
        shift_ref, _ = compute_shift_scatter(wl)
        shift_per_uw = abs(float(shift_ref)) / POWER_REF_UW
        p_adiab = adiabatic_ratio * system.rabi_eff / (2 * pi) / shift_per_uw
        adiabatic_power_min[wl] = p_adiab
        print(f"  {wl:10.2f} {shift_per_uw/1e6:20.4f} {p_adiab:18.1f}")

    # Auto power range
    p_adiab_worst = max(adiabatic_power_min.values())
    if args.power_min_uw is None:
        args.power_min_uw = POWER_REF_UW
        print(f"\n  Auto power_min = {args.power_min_uw:.1f} uW")
    if args.power_max_uw is None:
        args.power_max_uw = max(2.0 * p_adiab_worst, 3.0 * args.power_min_uw)
        print(f"  Auto power_max = {args.power_max_uw:.1f} uW")
    print()

    powers = np.linspace(args.power_min_uw, args.power_max_uw, args.n_power)
    n_total = len(wls) * len(powers)

    dtype = [
        ("wl", float), ("power_uw", float),
        ("delta_A_mhz", float), ("delta_B_mhz", float),
        ("scatter_A_hz", float),
        # Pinned atom (A)
        ("P_g_A", float), ("P_e_A", float), ("P_r_A", float),
        # Crosstalk virtual atom (B)
        ("P_g_B", float), ("P_e_B", float), ("P_r_B", float),
        # Cost metrics
        ("not_pinned", float), ("leakage", float),
        ("crosstalk", float), ("scatter_penalty", float),
        ("total_cost", float),
        ("delta_A_over_Omega", float),
    ]
    grid = np.empty(n_total, dtype=dtype)

    print(f"  Scanning {args.n_wl} x {args.n_power} = {n_total} grid points "
          f"(2 sims each)...")
    t0 = _time.time()
    idx = 0
    for wl in wls:
        shift_ref, scatter_ref = compute_shift_scatter(wl)
        for power_uw in powers:
            scale = power_uw / POWER_REF_UW
            delta_A = 2 * pi * float(shift_ref) * scale
            scatter_A = float(scatter_ref) * scale
            delta_B = eta * delta_A

            # Sim A: pinned atom
            proto_A = SweepProtocol(
                addressing={0: delta_A}, ac_stark_shift=ac_stark_peak,
            )
            res_A = simulate(system, proto_A, x_sweep, psi0)
            psi_A = res_A.psi_final
            P_g_A = np.abs(psi_A[0])**2
            P_e_A = np.abs(psi_A[1])**2
            P_r_A = np.abs(psi_A[2])**2

            # Sim B: crosstalk virtual atom
            proto_B = SweepProtocol(
                addressing={0: delta_B}, ac_stark_shift=ac_stark_peak,
            )
            res_B = simulate(system, proto_B, x_sweep, psi0)
            psi_B = res_B.psi_final
            P_g_B = np.abs(psi_B[0])**2
            P_e_B = np.abs(psi_B[1])**2
            P_r_B = np.abs(psi_B[2])**2

            not_pinned = 1.0 - P_g_A
            leakage = P_e_A
            crosstalk = 1.0 - P_r_B
            scatter_pen = 1.0 - np.exp(-scatter_A * t_gate)
            total_cost = not_pinned + crosstalk + scatter_pen

            grid[idx] = (
                wl, power_uw,
                delta_A / (2 * pi * 1e6), delta_B / (2 * pi * 1e6),
                scatter_A,
                P_g_A, P_e_A, P_r_A,
                P_g_B, P_e_B, P_r_B,
                not_pinned, leakage, crosstalk, scatter_pen, total_cost,
                abs(delta_A) / system.rabi_eff,
            )
            idx += 1

    elapsed = _time.time() - t0
    print(f"  Grid scan done in {elapsed:.1f}s ({elapsed/n_total:.2f}s per point)")

    best_idx = np.argmin(grid["total_cost"])
    best = grid[best_idx]
    print(f"\n  Best point: lambda={best['wl']:.1f} nm, P={best['power_uw']:.0f} uW")
    print(f"    delta_A = {best['delta_A_mhz']:.2f} MHz "
          f"({best['delta_A_over_Omega']:.1f} Omega_eff)")
    print(f"    P_g_A={best['P_g_A']:.4f}, P_r_B={best['P_r_B']:.4f}")
    print(f"    not_pinned={best['not_pinned']:.4f}, "
          f"crosstalk={best['crosstalk']:.4f}, scatter={best['scatter_penalty']:.4f}")
    print(f"    TOTAL COST = {best['total_cost']:.6f}")

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "addressing_1q_grid.csv")
    np.savetxt(csv_path, grid, fmt="%.6e",
               header=",".join(grid.dtype.names), delimiter=",")
    print(f"\n  Grid saved to {csv_path}")

    meta = {
        "system": system, "eta": eta, "x_sweep": x_sweep,
        "ac_stark_peak": ac_stark_peak, "wls": wls, "powers": powers,
        "best": best, "best_idx": best_idx,
        "adiabatic_power_min": adiabatic_power_min,
    }
    return grid, meta


# ═══════════════════════════════════════════════════════════════════════
# Shared plot helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_adiabatic_boundary(meta):
    """Return callable that overlays the adiabatic boundary on an axis."""
    adiabatic_power_min = meta["adiabatic_power_min"]
    powers = meta["powers"]
    adiab_wls = np.array(sorted(adiabatic_power_min.keys()))
    adiab_pows = np.array([adiabatic_power_min[w] for w in adiab_wls])

    def _add(ax):
        p_lo, p_hi = powers[0], powers[-1]
        mask = (adiab_pows >= p_lo) & (adiab_pows <= p_hi)
        if mask.any():
            ax.plot(adiab_pows[mask], adiab_wls[mask], "r--", lw=2, alpha=0.9,
                    label=r"$|\delta_A| = 4\,\Omega_{\rm eff}$")
    return _add


def _compute_lz_params(meta):
    """Compute single-crossing Landau-Zener parameters."""
    system = meta["system"]
    x_sweep = meta["x_sweep"]
    Omega_eff = system.rabi_eff
    t_gate_s = float(x_sweep[2]) * system.time_scale
    delta_start_rad = float(x_sweep[0]) * Omega_eff
    delta_end_rad = float(x_sweep[1]) * Omega_eff
    alpha = abs(delta_end_rad - delta_start_rad) / t_gate_s

    # Single-crossing LZ: P_transition = 1 - exp(-pi Omega^2 / (2 alpha))
    P_LZ = np.exp(-pi * Omega_eff**2 / (2 * alpha))

    def _lz_not_pinned(delta_A_rad):
        """Probability of leaving |g> at single LZ crossing.

        For pinning to work, the resonance must be OUTSIDE the sweep range.
        If inside, P_transition ~ 1 - P_LZ ~ 1 (fully adiabatic).
        """
        sweep_half = abs(delta_end_rad - delta_start_rad) / 2
        inside = np.abs(delta_A_rad) < sweep_half
        # If crossing is inside sweep range, atom transitions adiabatically
        return np.where(inside, 1.0 - P_LZ, 0.0)

    return {
        "Omega_eff": Omega_eff,
        "alpha": alpha,
        "P_LZ": P_LZ,
        "sweep_half_rad": abs(delta_end_rad - delta_start_rad) / 2,
        "lz_not_pinned": _lz_not_pinned,
    }


# ═══════════════════════════════════════════════════════════════════════
# Individual plot functions
# ═══════════════════════════════════════════════════════════════════════

def _plot_total_cost(grid, meta, args):
    """Total cost heatmap -> addressing_1q_heatmap.png"""
    wls = meta["wls"]; powers = meta["powers"]; best = meta["best"]
    add_boundary = _make_adiabatic_boundary(meta)

    cost_2d = grid["total_cost"].reshape(args.n_wl, args.n_power)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(powers, wls, cost_2d, cmap="viridis_r", shading="auto")
    add_boundary(ax)
    ax.plot(best["power_uw"], best["wl"], "r*", ms=15, mew=2,
            label=f"Best: $E$={best['total_cost']:.4f}")
    ax.set_xlabel("Power ($\\mu$W)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title("Single-qubit total cost $E_{total}(\\lambda, P)$")
    fig.colorbar(im, ax=ax, label="$E_{total}$")
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_1q_heatmap.png")
    fig.savefig(path, dpi=150)
    print(f"  Heatmap saved to {path}")
    plt.close(fig)


def _plot_components(grid, meta, args):
    """2x2 component overview -> addressing_1q_components.png"""
    wls = meta["wls"]; powers = meta["powers"]; best = meta["best"]
    add_boundary = _make_adiabatic_boundary(meta)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (field, title, cmap) in zip(axes.flat, [
        ("not_pinned", "Not pinned  $1 - P_g^A$", "Reds"),
        ("crosstalk", "Crosstalk  $1 - P_r^B$", "Blues"),
        ("scatter_penalty", "Scatter $1-e^{-\\Gamma T}$", "Oranges"),
        ("total_cost", "Total cost $E_{total}$", "viridis_r"),
    ]):
        data = grid[field].reshape(args.n_wl, args.n_power)
        im = ax.pcolormesh(powers, wls, data, cmap=cmap, shading="auto")
        add_boundary(ax)
        ax.plot(best["power_uw"], best["wl"], "k*", ms=12)
        ax.set_xlabel("Power ($\\mu$W)")
        ax.set_ylabel("Wavelength (nm)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.suptitle("Single-qubit cost decomposition", fontsize=13)
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_1q_components.png")
    fig.savefig(path, dpi=150)
    print(f"  Components saved to {path}")
    plt.close(fig)


def _plot_not_pinned(grid, meta, args):
    """Not-pinned heatmap -> addressing_1q_not_pinned.png"""
    wls = meta["wls"]; powers = meta["powers"]; best = meta["best"]
    add_boundary = _make_adiabatic_boundary(meta)

    data = grid["not_pinned"].reshape(args.n_wl, args.n_power)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(powers, wls, data, cmap="Reds", shading="auto")
    add_boundary(ax)
    ax.plot(best["power_uw"], best["wl"], "k*", ms=15, mew=2, label="Best overall")
    ax.set_xlabel("Power ($\\mu$W)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title("Not pinned  $1 - P_g^A$  (pinned atom left $|g\\rangle$)")
    fig.colorbar(im, ax=ax, label="$1 - P_g^A$")
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_1q_not_pinned.png")
    fig.savefig(path, dpi=150)
    print(f"  Not-pinned heatmap saved to {path}")
    plt.close(fig)


def _plot_crosstalk(grid, meta, args):
    """Crosstalk heatmap -> addressing_1q_crosstalk.png"""
    wls = meta["wls"]; powers = meta["powers"]; best = meta["best"]
    add_boundary = _make_adiabatic_boundary(meta)

    data = grid["crosstalk"].reshape(args.n_wl, args.n_power)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(powers, wls, data, cmap="Blues", shading="auto")
    add_boundary(ax)
    ax.plot(best["power_uw"], best["wl"], "k*", ms=15, mew=2, label="Best overall")
    ax.set_xlabel("Power ($\\mu$W)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title("Crosstalk  $1 - P_r^B$  (virtual atom failed $|g\\rangle \\to |r\\rangle$)")
    fig.colorbar(im, ax=ax, label="$1 - P_r^B$")
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_1q_crosstalk.png")
    fig.savefig(path, dpi=150)
    print(f"  Crosstalk heatmap saved to {path}")
    plt.close(fig)


def _plot_scatter(grid, meta, args):
    """Scatter penalty heatmap -> addressing_1q_scatter.png"""
    wls = meta["wls"]; powers = meta["powers"]; best = meta["best"]
    add_boundary = _make_adiabatic_boundary(meta)

    data = grid["scatter_penalty"].reshape(args.n_wl, args.n_power)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(powers, wls, data, cmap="Oranges", shading="auto")
    add_boundary(ax)
    ax.plot(best["power_uw"], best["wl"], "k*", ms=15, mew=2, label="Best overall")
    ax.set_xlabel("Power ($\\mu$W)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title("Scattering penalty  $1 - e^{-\\Gamma_A T}$")
    fig.colorbar(im, ax=ax, label="scatter penalty")
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_1q_scatter.png")
    fig.savefig(path, dpi=150)
    print(f"  Scatter heatmap saved to {path}")
    plt.close(fig)


def _plot_smoothed(grid, meta, args):
    """Gaussian-smoothed cost landscape -> addressing_1q_smoothed.png"""
    from scipy.ndimage import gaussian_filter1d

    wls = meta["wls"]; powers = meta["powers"]; best = meta["best"]
    add_boundary = _make_adiabatic_boundary(meta)

    sigma_smooth = 3
    cost_2d_raw = grid["total_cost"].reshape(args.n_wl, args.n_power)
    cost_2d_smooth = gaussian_filter1d(cost_2d_raw, sigma=sigma_smooth, axis=1)

    smooth_flat_idx = np.argmin(cost_2d_smooth)
    smooth_wi, smooth_pi = np.unravel_index(smooth_flat_idx, cost_2d_smooth.shape)
    best_smooth_wl = wls[smooth_wi]
    best_smooth_pow = powers[smooth_pi]
    best_smooth_cost = cost_2d_smooth[smooth_wi, smooth_pi]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(powers, wls, cost_2d_smooth, cmap="viridis_r", shading="auto")
    add_boundary(ax)
    ax.plot(best["power_uw"], best["wl"], "r*", ms=14, mew=2, mfc="none",
            label=f"Raw best: $E$={best['total_cost']:.4f}")
    ax.plot(best_smooth_pow, best_smooth_wl, "r*", ms=14, mew=2,
            label=f"Smoothed best: $E$={best_smooth_cost:.4f}")
    ax.set_xlabel("Power ($\\mu$W)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title(r"Smoothed cost (Gaussian $\sigma$="
                 f"{sigma_smooth} pts)")
    fig.colorbar(im, ax=ax, label="$E_{total}$ (smoothed)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_1q_smoothed.png")
    fig.savefig(path, dpi=150)
    print(f"  Smoothed heatmap saved to {path}")
    print(f"    Smoothed best: lambda={best_smooth_wl:.2f} nm, "
          f"P={best_smooth_pow:.1f} uW, cost={best_smooth_cost:.5f}")
    plt.close(fig)


def _plot_lz_slices(grid, meta, args):
    """Landau-Zener 1D slices per wavelength -> addressing_1q_lz_slices.png

    Single-crossing LZ: inside the sweep range the crossing is adiabatic
    (atom transitions |g> -> |r>), so not_pinned ~ 1 - P_LZ ~ 1.
    Outside the sweep range, not_pinned ~ 0.
    """
    wls = meta["wls"]
    lz = _compute_lz_params(meta)
    Omega_eff = lz["Omega_eff"]
    P_LZ = lz["P_LZ"]
    alpha = lz["alpha"]
    _lz_not_pinned = lz["lz_not_pinned"]

    n_sel = min(6, args.n_wl)
    sel_indices = np.linspace(0, args.n_wl - 1, n_sel, dtype=int)

    fig, axes = plt.subplots(2, (n_sel + 1) // 2,
                              figsize=(5 * ((n_sel + 1) // 2), 8), squeeze=False)
    axes_flat = axes.flat

    for panel, wi in enumerate(sel_indices):
        ax = axes_flat[panel]
        wl = wls[wi]
        delta_A_arr = np.abs(grid["delta_A_mhz"].reshape(args.n_wl, args.n_power)[wi])
        delta_A_rad = delta_A_arr * 1e6 * 2 * pi
        ratio = delta_A_rad / Omega_eff

        sim_err = grid["not_pinned"].reshape(args.n_wl, args.n_power)[wi]

        ratio_dense = np.linspace(ratio.min(), ratio.max(), 500)
        lz_dense = _lz_not_pinned(ratio_dense * Omega_eff)

        ax.plot(ratio, sim_err * 100, "o", ms=3, color="#1f77b4", label="Simulation")
        ax.plot(ratio_dense, lz_dense * 100, "-", lw=1.2, color="#d62728",
                label="LZ theory", alpha=0.85)
        ax.axvline(4.0, color="gray", ls=":", lw=1, alpha=0.6)
        ax.set_title(f"$\\lambda$ = {wl:.1f} nm", fontsize=10)
        ax.set_xlabel(r"$|\delta_A| / \Omega_{\rm eff}$", fontsize=9)
        ax.set_ylabel("Not pinned (%)", fontsize=9)
        ax.tick_params(labelsize=8)
        if panel == 0:
            ax.legend(fontsize=7, loc="upper right")

    for j in range(len(sel_indices), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Single-atom LZ transition (adiabatic passage)\n"
        f"$P_{{LZ}}$ = {P_LZ:.2e},  "
        f"$\\alpha/(2\\pi)$ = {alpha/(2*pi)/1e12:.2f} THz/s",
        fontsize=11)
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_1q_lz_slices.png")
    fig.savefig(path, dpi=150)
    print(f"  LZ 1D slices saved to {path}")
    plt.close(fig)


def _plot_leakage(grid, meta, args):
    """Intermediate-state leakage heatmap -> addressing_1q_leakage.png"""
    wls = meta["wls"]; powers = meta["powers"]; best = meta["best"]
    add_boundary = _make_adiabatic_boundary(meta)

    data = grid["leakage"].reshape(args.n_wl, args.n_power)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(powers, wls, data, cmap="Purples", shading="auto")
    add_boundary(ax)
    ax.plot(best["power_uw"], best["wl"], "k*", ms=15, mew=2, label="Best overall")
    ax.set_xlabel("Power ($\\mu$W)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title("Intermediate-state leakage  $P_e^A$")
    fig.colorbar(im, ax=ax, label="$P_e^A$")
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_1q_leakage.png")
    fig.savefig(path, dpi=150)
    print(f"  Leakage heatmap saved to {path}")
    plt.close(fig)


def _plot_atom_populations(grid, meta, args):
    """Side-by-side atom A and B population heatmaps -> addressing_1q_populations.png"""
    wls = meta["wls"]; powers = meta["powers"]; best = meta["best"]
    add_boundary = _make_adiabatic_boundary(meta)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for col, (prefix, atom_label) in enumerate([("A", "Pinned atom A"), ("B", "Virtual atom B")]):
        for row, (level, cmap) in enumerate([("g", "Greens"), ("e", "Purples"), ("r", "Reds")]):
            if col == 0:
                ax = axes[0, row]
            else:
                ax = axes[1, row]
            field = f"P_{level}_{prefix}"
            data = grid[field].reshape(args.n_wl, args.n_power)
            im = ax.pcolormesh(powers, wls, data, cmap=cmap, shading="auto",
                               vmin=0, vmax=1)
            add_boundary(ax)
            ax.plot(best["power_uw"], best["wl"], "k*", ms=10)
            ax.set_xlabel("Power ($\\mu$W)")
            ax.set_ylabel("Wavelength (nm)")
            ax.set_title(f"{atom_label}: $P_{{{level}}}$")
            fig.colorbar(im, ax=ax)

    fig.suptitle("Single-qubit level populations", fontsize=13)
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_1q_populations.png")
    fig.savefig(path, dpi=150)
    print(f"  Population heatmaps saved to {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Subcommands
# ═══════════════════════════════════════════════════════════════════════

GRID_DTYPE = [
    ("wl", float), ("power_uw", float),
    ("delta_A_mhz", float), ("delta_B_mhz", float),
    ("scatter_A_hz", float),
    ("P_g_A", float), ("P_e_A", float), ("P_r_A", float),
    ("P_g_B", float), ("P_e_B", float), ("P_r_B", float),
    ("not_pinned", float), ("leakage", float),
    ("crosstalk", float), ("scatter_penalty", float),
    ("total_cost", float),
    ("delta_A_over_Omega", float),
]


def _all_plots(grid, meta, args):
    """Generate all plots with progress banners."""
    os.makedirs(args.outdir, exist_ok=True)

    print("\n[1/8] Total cost heatmap...")
    _plot_total_cost(grid, meta, args)

    print("\n[2/8] Cost component overview (2x2)...")
    _plot_components(grid, meta, args)

    print("\n[3/8] Not-pinned heatmap...")
    _plot_not_pinned(grid, meta, args)

    print("\n[4/8] Crosstalk heatmap...")
    _plot_crosstalk(grid, meta, args)

    print("\n[5/8] Scatter penalty heatmap...")
    _plot_scatter(grid, meta, args)

    print("\n[6/8] Smoothed cost landscape...")
    _plot_smoothed(grid, meta, args)

    print("\n[7/8] Landau-Zener 1D slices...")
    _plot_lz_slices(grid, meta, args)

    print("\n[8/8] Leakage + population heatmaps...")
    _plot_leakage(grid, meta, args)
    _plot_atom_populations(grid, meta, args)


def cmd_optimize(args):
    """Run 2D scan + plot."""
    grid, meta = _run_optimize_scan(args)
    _all_plots(grid, meta, args)


def cmd_optimize_plot(args):
    """Re-plot from saved CSV (no simulation)."""
    raw = np.loadtxt(args.csv, delimiter=",", dtype=float)
    grid = np.empty(raw.shape[0], dtype=GRID_DTYPE)
    for i, name in enumerate(grid.dtype.names):
        grid[name] = raw[:, i]
    print(f"  Loaded {len(grid)} points from {args.csv}")

    wls = np.unique(grid["wl"])
    powers = np.unique(grid["power_uw"])
    args.n_wl = len(wls)
    args.n_power = len(powers)

    system = create_analog_system(
        detuning_sign=1, blackmanflag=True, n_atoms=1,
        Delta_Hz=2.4e9, rabi_420_Hz=135e6, rabi_1013_Hz=135e6,
    )
    eta = np.exp(-2 * (args.distance_um / args.waist_um) ** 2)
    ac_stark_peak = system.rabi_420 ** 2 / (4 * abs(system.Delta))
    t_gate = args.t_gate_us * 1e-6
    x_sweep = [
        2 * pi * args.delta_start_mhz * 1e6 / system.rabi_eff,
        2 * pi * args.delta_end_mhz * 1e6 / system.rabi_eff,
        t_gate / system.time_scale,
    ]

    adiabatic_ratio = 4.0
    adiabatic_power_min = {}
    for wl in wls:
        shift_ref, _ = compute_shift_scatter(wl)
        shift_per_uw = abs(float(shift_ref)) / POWER_REF_UW
        adiabatic_power_min[wl] = adiabatic_ratio * system.rabi_eff / (2 * pi) / shift_per_uw

    best_idx = np.argmin(grid["total_cost"])
    best = grid[best_idx]

    meta = {
        "system": system, "eta": eta, "x_sweep": x_sweep,
        "ac_stark_peak": ac_stark_peak, "wls": wls, "powers": powers,
        "best": best, "best_idx": best_idx,
        "adiabatic_power_min": adiabatic_power_min,
    }
    _all_plots(grid, meta, args)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_opt = sub.add_parser("optimize",
                            help="2D wavelength x power scan (single qubit)")
    p_opt.add_argument("--wl-min", type=float, default=781.0)
    p_opt.add_argument("--wl-max", type=float, default=786.0)
    p_opt.add_argument("--n-wl", type=int, default=60)
    p_opt.add_argument("--power-min-uw", type=float, default=None)
    p_opt.add_argument("--power-max-uw", type=float, default=None)
    p_opt.add_argument("--n-power", type=int, default=40)
    p_opt.add_argument("--distance-um", type=float, default=4.0,
                        help="Crosstalk distance in um (default: 4.0)")
    p_opt.add_argument("--waist-um", type=float, default=1.0)
    p_opt.add_argument("--delta-start-mhz", type=float, default=-40.0)
    p_opt.add_argument("--delta-end-mhz", type=float, default=40.0)
    p_opt.add_argument("--t-gate-us", type=float, default=4.5)
    p_opt.add_argument("--outdir", type=str, default="results")

    p_plt = sub.add_parser("optimize-plot",
                            help="Re-plot from saved CSV (no simulation)")
    p_plt.add_argument("--csv", type=str, default="results/addressing_1q_grid.csv")
    p_plt.add_argument("--distance-um", type=float, default=4.0)
    p_plt.add_argument("--waist-um", type=float, default=1.0)
    p_plt.add_argument("--delta-start-mhz", type=float, default=-40.0)
    p_plt.add_argument("--delta-end-mhz", type=float, default=40.0)
    p_plt.add_argument("--t-gate-us", type=float, default=4.5)
    p_plt.add_argument("--outdir", type=str, default="results")

    args = parser.parse_args()

    if args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "optimize-plot":
        cmd_optimize_plot(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
