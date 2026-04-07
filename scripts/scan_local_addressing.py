#!/usr/bin/env python3
"""Unified local addressing analysis: wavelength optimization and noise sensitivity.

Subcommands:
    wavelength   Scan AC Stark shift, scattering, and FOM vs laser wavelength.
    noise        Sweep noise sources (detuning, RIN, amplitude) and measure addressing quality.

Usage:
    uv run python scripts/scan_local_addressing.py wavelength
    uv run python scripts/scan_local_addressing.py wavelength --with-sim
    uv run python scripts/scan_local_addressing.py noise
    uv run python scripts/scan_local_addressing.py noise --combined
"""

import argparse
import os
import time as _time

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi

from ryd_gate.analysis.local_addressing import (
    BASELINE_AMP,
    BASELINE_DETUNING_HZ,
    BASELINE_RIN,
    COMBINED_SCALE_MAX,
    DEFAULT_LOCAL_DETUNING,
    DEFAULT_LOCAL_SCATTER,
    default_sweep_x,
    evaluate_addressing,
)
from ryd_gate.core.atomic_system import (
    LAMBDA_D2,
    LAMBDA_PAPER,
    POWER_REF_UW,
    build_product_state_map,
    compute_shift_scatter,
    create_analog_system,
)
from ryd_gate.protocols.sweep import SweepProtocol

FIGDIR = "docs/figures"


# ═══════════════════════════════════════════════════════════════════════
# Subcommand: wavelength
# ═══════════════════════════════════════════════════════════════════════

def cmd_wavelength(args):
    """Scan AC Stark shift and scattering vs wavelength."""
    print("=" * 60)
    print("  Local Addressing Wavelength Scan (780.5 -- 786 nm)")
    print("=" * 60)

    wavelengths = np.linspace(780.5, 786.0, 200)
    shifts, scatters = compute_shift_scatter(wavelengths)

    shifts_MHz = shifts / 1e6
    fom = np.abs(shifts) / np.maximum(scatters, 1e-10)

    # Plot
    os.makedirs(FIGDIR, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(wavelengths, shifts_MHz, "b-", lw=2)
    axes[0].axvline(LAMBDA_D2, color="red", ls="--", alpha=0.5, label=f"D2 ({LAMBDA_D2:.1f} nm)")
    axes[0].axvline(LAMBDA_PAPER, color="green", ls=":", alpha=0.7, label=f"{LAMBDA_PAPER:.0f} nm (paper)")
    axes[0].set_ylabel("AC Stark Shift (MHz)")
    axes[0].set_title("Ground State AC Stark Shift (5S$_{1/2}$)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(wavelengths, scatters, "r-", lw=2)
    axes[1].axvline(LAMBDA_PAPER, color="green", ls=":", alpha=0.7)
    axes[1].set_ylabel("Scattering Rate (Hz)")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)

    axes[2].plot(wavelengths, fom / 1e6, "purple", lw=2)
    axes[2].axvline(LAMBDA_PAPER, color="green", ls=":", alpha=0.7, label=f"{LAMBDA_PAPER:.0f} nm (paper)")
    axes[2].set_xlabel("Wavelength (nm)")
    axes[2].set_ylabel("|Shift| / Scatter ($\\times 10^6$)")
    axes[2].set_title("Figure of Merit")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGDIR, "addressing_wavelength_scan.png")
    fig.savefig(path, dpi=150)
    print(f"\nFigure saved to {path}")
    plt.close(fig)

    # Table
    print(f"\n{'Wavelength':>12} {'Shift (MHz)':>12} {'Scatter (Hz)':>14} {'FOM (x1e6)':>12}")
    print("-" * 54)
    for lam in [781.0, 782.0, 783.0, 784.0, 785.0, 786.0]:
        s, sc = compute_shift_scatter(lam)
        f = abs(float(s)) / max(float(sc), 1e-10)
        print(f"{lam:12.1f} {float(s)/1e6:12.2f} {float(sc):14.1f} {f/1e6:12.2f}")

    # Optional MC sim
    if args.with_sim:
        print("\n" + "=" * 60)
        print("  Addressing Simulation at Sampled Wavelengths")
        print("=" * 60)

        system = create_analog_system(detuning_sign=1)
        initial_state = build_product_state_map(n_levels=3)["gg"]
        x = default_sweep_x(system)
        sample_lams = np.arange(781.0, 786.5, 0.5)
        sample_shifts, sample_scatters = compute_shift_scatter(sample_lams)

        pin_errs, xtalk_errs, leak_errs = [], [], []
        for i, lam in enumerate(sample_lams):
            protocol = SweepProtocol(
                addressing={0: 2 * pi * sample_shifts[i]},
                scatter_rate=sample_scatters[i],
            )
            pin, xtalk, leak = evaluate_addressing(
                system, initial_state, protocol, x,
                {"sigma_detuning": BASELINE_DETUNING_HZ}, args.n_mc, seed=42)
            pin_errs.append(pin)
            xtalk_errs.append(xtalk)
            leak_errs.append(leak)
            print(f"  {lam:.1f} nm: shift={sample_shifts[i]/1e6:.2f} MHz, "
                  f"scatter={sample_scatters[i]:.1f} Hz, pin_err={pin:.4f}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sample_lams, pin_errs, "o-", color="red", label="Pinning error")
        ax.plot(sample_lams, xtalk_errs, "s-", color="blue", label="Crosstalk error")
        ax.plot(sample_lams, leak_errs, "^-", color="orange", label="Leakage loss")
        ax.axvline(LAMBDA_PAPER, color="green", ls=":", alpha=0.7, label=f"{LAMBDA_PAPER:.0f} nm (paper)")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Error")
        ax.set_title("Addressing Quality vs Local Laser Wavelength")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_yscale("log")
        fig.tight_layout()
        path = os.path.join(FIGDIR, "addressing_wavelength_quality.png")
        fig.savefig(path, dpi=150)
        print(f"Figure saved to {path}")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Subcommand: noise
# ═══════════════════════════════════════════════════════════════════════

def cmd_noise(args):
    """Sweep noise parameters and measure addressing quality."""
    print("=" * 60)
    print("  Addressing Noise Sensitivity Scan")
    print("=" * 60)

    system = create_analog_system(detuning_sign=1)
    initial_state = build_product_state_map(n_levels=3)["gg"]
    protocol = SweepProtocol(
        addressing={0: DEFAULT_LOCAL_DETUNING},
        scatter_rate=DEFAULT_LOCAL_SCATTER,
    )
    x = default_sweep_x(system)

    scans = {
        "Global detuning": {
            "param": "sigma_detuning",
            "values": np.linspace(0, 500e3, args.n_points),
            "xlabel": "Detuning noise $\\sigma_\\Delta$ (kHz)",
            "xscale": 1e-3,
        },
        "Local RIN": {
            "param": "sigma_local_rin",
            "values": np.linspace(0, 0.05, args.n_points),
            "xlabel": "RIN $\\sigma_{\\mathrm{RIN}}$ (%)",
            "xscale": 100,
        },
        "Amplitude noise": {
            "param": "sigma_amplitude",
            "values": np.linspace(0, 0.05, args.n_points),
            "xlabel": "Amplitude noise $\\sigma_\\Omega$ (%)",
            "xscale": 100,
        },
    }

    results = {}
    for name, cfg in scans.items():
        print(f"\n  Scanning: {name}")
        t0 = _time.time()
        pin_errs, xtalk_errs, leak_errs = [], [], []
        for val in cfg["values"]:
            kwargs = {"sigma_detuning": 0.0, "sigma_local_rin": 0.0, "sigma_amplitude": 0.0}
            kwargs[cfg["param"]] = val
            pin, xtalk, leak = evaluate_addressing(
                system, initial_state, protocol, x, kwargs, args.n_mc)
            pin_errs.append(pin)
            xtalk_errs.append(xtalk)
            leak_errs.append(leak)
        print(f"    Done in {_time.time() - t0:.1f}s")
        results[name] = (np.array(pin_errs), np.array(xtalk_errs), np.array(leak_errs))

    # Optional combined sweep
    if args.combined:
        print("\n  Scanning: Combined (all noise sources)")
        scale_factors = np.linspace(0, COMBINED_SCALE_MAX, args.n_points)
        t0 = _time.time()
        pin_comb, xtalk_comb, leak_comb = [], [], []
        for scale in scale_factors:
            pin, xtalk, leak = evaluate_addressing(
                system, initial_state, protocol, x,
                {"sigma_detuning": BASELINE_DETUNING_HZ * scale,
                 "sigma_local_rin": BASELINE_RIN * scale,
                 "sigma_amplitude": BASELINE_AMP * scale},
                args.n_mc)
            pin_comb.append(pin)
            xtalk_comb.append(xtalk)
            leak_comb.append(leak)
        print(f"    Done in {_time.time() - t0:.1f}s")

    # Plot
    os.makedirs(FIGDIR, exist_ok=True)
    n_panels = 4 if args.combined else 3
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flat

    for ax, (name, cfg) in zip(axes_flat, scans.items()):
        pin, xtalk, leak = results[name]
        xvals = cfg["values"] * cfg["xscale"]
        ax.plot(xvals, pin, "o-", color="red", label="Pinning", markersize=5)
        ax.plot(xvals, xtalk, "s-", color="blue", label="Crosstalk", markersize=5)
        ax.plot(xvals, leak, "^-", color="orange", label="Leakage", markersize=5)
        ax.set_xlabel(cfg["xlabel"])
        ax.set_ylabel("Error")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    if args.combined:
        ax = axes_flat[3]
        ax.plot(scale_factors, pin_comb, "o-", color="red", label="Pinning", markersize=5)
        ax.plot(scale_factors, xtalk_comb, "s-", color="blue", label="Crosstalk", markersize=5)
        ax.plot(scale_factors, leak_comb, "^-", color="orange", label="Leakage", markersize=5)
        ax.axvline(1.0, color="green", ls=":", alpha=0.7, label="Baseline")
        ax.set_xlabel("Noise scale factor")
        ax.set_ylabel("Error")
        ax.set_title("Combined noise (all sources)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        axes_flat[3].set_visible(False)

    fig.suptitle("Addressing Noise Sensitivity Analysis", fontsize=14)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "addressing_noise_sensitivity.png")
    fig.savefig(path, dpi=150)
    print(f"\nFigure saved to {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Subcommand: optimize
# ═══════════════════════════════════════════════════════════════════════

def _run_optimize_scan(args):
    """Run the 2D grid scan and return (grid, metadata dict)."""
    from ryd_gate import simulate
    from ryd_gate.core.models.analog_3level import Analog3LevelModel

    print("=" * 60)
    print("  Local Addressing: Wavelength x Power Optimization")
    print("=" * 60)

    model = Analog3LevelModel.from_defaults(
        detuning_sign=1, blackmanflag=True,
        distance_um=args.distance_um,
        Delta_Hz=2.4e9,       # paper: ~2.4 GHz intermediate detuning
        rabi_420_Hz=135e6,    # paper: balanced Rabi, Omega_eff ≈ 3.8 MHz
        rabi_1013_Hz=135e6,
    )
    system = model.system
    psi0 = np.zeros(9, dtype=complex)
    psi0[0] = 1.0  # |gg>

    # AC Stark feed-forward: compensate Blackman-envelope light shift
    ac_stark_peak = system.rabi_420 ** 2 / (4 * abs(system.Delta))

    t_gate = args.t_gate_us * 1e-6
    x_sweep = [
        2 * pi * args.delta_start_mhz * 1e6 / system.rabi_eff,
        2 * pi * args.delta_end_mhz * 1e6 / system.rabi_eff,
        t_gate / system.time_scale,
    ]

    # Gaussian tail factor
    eta = np.exp(-2 * (args.distance_um / args.waist_um) ** 2)
    print(f"  Gaussian tail: eta = exp(-2*(a/w)^2) = {eta:.4e}")
    print(f"  Omega_eff/(2pi) = {system.rabi_eff/(2*pi)/1e6:.2f} MHz")
    print(f"  AC Stark peak = {ac_stark_peak/(2*pi)/1e6:.2f} MHz")
    print(f"  V_ryd/(2pi) = {system.v_ryd/(2*pi)/1e6:.1f} MHz")
    print(f"  Sweep: [{args.delta_start_mhz}, {args.delta_end_mhz}] MHz, "
          f"T_gate = {args.t_gate_us} us")
    print()

    wls = np.linspace(args.wl_min, args.wl_max, args.n_wl)

    # Adiabatic threshold: |delta_A| > 4*Omega_eff per wavelength.
    adiabatic_ratio = 4.0
    adiabatic_power_min = {}  # wl -> min power (uW) for adiabatic condition
    print(f"  Adiabatic condition: |delta_A| > {adiabatic_ratio:.0f} Omega_eff "
          f"= {adiabatic_ratio * system.rabi_eff/(2*pi)/1e6:.1f} MHz")
    print(f"  {'WL (nm)':>10} {'|shift|/P (MHz/uW)':>20} {'P_min_adiab (uW)':>18}")
    for wl in wls:
        shift_ref, _ = compute_shift_scatter(wl)
        shift_per_uw = abs(float(shift_ref)) / POWER_REF_UW  # Hz per uW
        p_adiab = adiabatic_ratio * system.rabi_eff / (2 * pi) / shift_per_uw
        adiabatic_power_min[wl] = p_adiab
        print(f"  {wl:10.2f} {shift_per_uw/1e6*1e0:20.4f} {p_adiab:18.1f}")

    # Auto-set power range if not specified by user
    p_adiab_worst = max(adiabatic_power_min.values())
    if args.power_min_uw is None:
        args.power_min_uw = POWER_REF_UW
        print(f"\n  Auto power_min = {args.power_min_uw:.1f} uW (reference power)")
    if args.power_max_uw is None:
        args.power_max_uw = max(2.0 * p_adiab_worst, 3.0 * args.power_min_uw)
        print(f"  Auto power_max = {args.power_max_uw:.1f} uW")
    print()

    powers = np.linspace(args.power_min_uw, args.power_max_uw, args.n_power)
    n_total = len(wls) * len(powers)

    # Result arrays
    dtype = [
        ("wl", float), ("power_uw", float),
        ("delta_A_mhz", float), ("delta_B_mhz", float),
        ("scatter_A_hz", float),
        ("P_gg", float), ("P_gr", float), ("P_rg", float), ("P_rr", float),
        ("pinning_leak", float), ("crosstalk", float),
        ("scatter_penalty", float), ("total_cost", float),
        ("P_target", float), ("other_residual", float),
        ("delta_A_over_Omega", float), ("delta_A_over_V", float),
    ]
    grid = np.empty(n_total, dtype=dtype)

    print(f"  Scanning {args.n_wl} x {args.n_power} = {n_total} grid points...")
    t0 = _time.time()
    idx = 0
    for wl in wls:
        shift_ref, scatter_ref = compute_shift_scatter(wl)  # Hz at ref power
        for power_uw in powers:
            scale = power_uw / POWER_REF_UW
            delta_A = 2 * pi * float(shift_ref) * scale  # rad/s
            scatter_A = float(scatter_ref) * scale  # Hz
            delta_B = eta * delta_A

            protocol = SweepProtocol(addressing={0: delta_A, 1: delta_B},
                                     ac_stark_shift=ac_stark_peak)
            result = simulate(system, protocol, x_sweep, psi0)
            psi_f = result.psi_final

            P_gg = model.observables.measure("pop_gg", psi_f)
            P_gr = model.observables.measure("pop_gr", psi_f)
            P_rg = model.observables.measure("pop_rg", psi_f)
            P_rr = model.observables.measure("pop_rr", psi_f)
            print(f"Keeping rate P_gr: {P_gr} for wavelength {wl} nm and power {power_uw} uW")
            pinning_leak = P_rg + P_rr
            crosstalk = P_gg
            scatter_pen = 1.0 - np.exp(-scatter_A * t_gate)
            total_cost = pinning_leak + crosstalk + scatter_pen
            print(f"pinning_leak: {pinning_leak}, crosstalk: {crosstalk}, scatter_pen: {scatter_pen}, total_cost: {total_cost}")
            grid[idx] = (
                wl, power_uw,
                delta_A / (2 * pi * 1e6), delta_B / (2 * pi * 1e6),
                scatter_A,
                P_gg, P_gr, P_rg, P_rr,
                pinning_leak, crosstalk, scatter_pen, total_cost,
                P_gr, 1.0 - (P_gg + P_gr + P_rg + P_rr),
                abs(delta_A) / system.rabi_eff,
                abs(delta_A) / system.v_ryd,
            )
            idx += 1

    elapsed = _time.time() - t0
    print(f"  Grid scan done in {elapsed:.1f}s ({elapsed/n_total:.2f}s per point)")

    # Find best
    best_idx = np.argmin(grid["total_cost"])
    best = grid[best_idx]
    print(f"\n  Best point: lambda={best['wl']:.1f} nm, P={best['power_uw']:.0f} uW")
    print(f"    delta_A = {best['delta_A_mhz']:.2f} MHz "
          f"({best['delta_A_over_Omega']:.1f} Omega_eff, "
          f"{best['delta_A_over_V']:.3f} V_ryd)")
    print(f"    P_gg={best['P_gg']:.4f}, P_gr={best['P_gr']:.4f}, "
          f"P_rg={best['P_rg']:.4f}, P_rr={best['P_rr']:.6f}")
    print(f"    pinning_leak={best['pinning_leak']:.4f}, "
          f"crosstalk={best['crosstalk']:.4f}, scatter={best['scatter_penalty']:.4f}")
    print(f"    TOTAL COST = {best['total_cost']:.6f}")

    # Save results
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "addressing_opt_grid.csv")
    np.savetxt(csv_path, grid, fmt="%.6e",
               header=",".join(grid.dtype.names), delimiter=",")
    print(f"\n  Grid saved to {csv_path}")

    # --- Top-K candidates ---
    top_k = min(args.top_k, n_total)
    ranking = np.argsort(grid["total_cost"])[:top_k]
    print(f"\n  Top-{top_k} candidates:")
    print(f"  {'Rank':>4} {'WL(nm)':>8} {'P(uW)':>8} {'dA(MHz)':>9} "
          f"{'P_gr':>7} {'Leak':>7} {'Xtalk':>7} {'Scat':>7} {'Total':>8}")
    for rank, ri in enumerate(ranking):
        r = grid[ri]
        print(f"  {rank+1:4d} {r['wl']:8.1f} {r['power_uw']:8.0f} "
              f"{r['delta_A_mhz']:9.2f} {r['P_gr']:7.4f} "
              f"{r['pinning_leak']:7.4f} {r['crosstalk']:7.4f} "
              f"{r['scatter_penalty']:7.4f} {r['total_cost']:8.5f}")

    # --- Hybrid validation of top-K with scatter in Hamiltonian ---
    if top_k > 0:
        print(f"\n  Hybrid validation (scatter in Hamiltonian)...")
        for rank, ri in enumerate(ranking):
            r = grid[ri]
            delta_A = r["delta_A_mhz"] * 2 * pi * 1e6
            delta_B = eta * delta_A
            scatter_A = r["scatter_A_hz"]
            scatter_B = eta * scatter_A

            proto_scat = SweepProtocol(
                addressing={0: delta_A, 1: delta_B},
                scatter_rates={0: scatter_A, 1: scatter_B},
                ac_stark_shift=ac_stark_peak,
            )
            res = simulate(system, proto_scat, x_sweep, psi0)
            P_gr_dyn = model.observables.measure("pop_gr", res.psi_final)
            norm = float(np.real(np.vdot(res.psi_final, res.psi_final)))
            print(f"    #{rank+1}: P_gr={P_gr_dyn:.4f}, norm={norm:.6f}, "
                  f"norm_loss={1-norm:.4e}")

    meta = {
        "system": system, "model": model, "eta": eta, "x_sweep": x_sweep,
        "ac_stark_peak": ac_stark_peak, "wls": wls, "powers": powers,
        "best": best, "best_idx": best_idx,
        "adiabatic_power_min": adiabatic_power_min,
    }
    return grid, meta


# ── shared helpers ────────────────────────────────────────────────────

def _make_adiabatic_boundary(meta):
    """Return a callable that overlays the adiabatic condition boundary on an axis."""
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


def _compute_lzs_params(meta):
    """Compute LZS transfer-matrix scalars from meta.

    Returns a dict with keys: Omega_eff, alpha, P_LZ, envelope, phi_stokes,
    lzs_error (callable: delta_A_rad -> error probability).
    """
    system = meta["system"]
    x_sweep = meta["x_sweep"]
    Omega_eff = system.rabi_eff
    t_gate_s = float(x_sweep[2]) * system.time_scale
    delta_start_rad = float(x_sweep[0]) * Omega_eff
    delta_end_rad = float(x_sweep[1]) * Omega_eff
    alpha = abs(delta_end_rad - delta_start_rad) / t_gate_s
    P_LZ = np.exp(-pi * Omega_eff**2 / (2 * alpha))
    envelope = 4 * P_LZ * (1 - P_LZ)
    phi_stokes = pi / 4

    def _lzs_error(delta_A_rad):
        Phi_dyn = delta_A_rad**2 / (2 * alpha)
        return envelope * np.cos(Phi_dyn / 2 + phi_stokes)**2

    return {
        "Omega_eff": Omega_eff,
        "alpha": alpha,
        "P_LZ": P_LZ,
        "envelope": envelope,
        "phi_stokes": phi_stokes,
        "lzs_error": _lzs_error,
    }


# ── individual plot functions ──────────────────────────────────────────

def _plot_total_cost(grid, meta, args):
    """Total cost heatmap → addressing_opt_heatmap.png"""
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
    ax.set_title("Total addressing cost $E_{total}(\\lambda, P)$")
    fig.colorbar(im, ax=ax, label="$E_{total}$")
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_opt_heatmap.png")
    fig.savefig(path, dpi=150)
    print(f"  Heatmap saved to {path}")
    plt.close(fig)


def _plot_components(grid, meta, args):
    """2×2 cost component overview → addressing_opt_components.png"""
    wls = meta["wls"]; powers = meta["powers"]; best = meta["best"]
    add_boundary = _make_adiabatic_boundary(meta)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (field, title, cmap) in zip(axes.flat, [
        ("pinning_leak", "Pinning leak $P_{rg}+P_{rr}$", "Reds"),
        ("crosstalk", "Crosstalk $P_{gg}$", "Blues"),
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
    fig.suptitle("Cost decomposition", fontsize=13)
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_opt_components.png")
    fig.savefig(path, dpi=150)
    print(f"  Components saved to {path}")
    plt.close(fig)


def _plot_pinning_leak(grid, meta, args):
    """Pinning leak heatmap → addressing_opt_pinning_leak.png"""
    wls = meta["wls"]; powers = meta["powers"]; best = meta["best"]
    add_boundary = _make_adiabatic_boundary(meta)

    data = grid["pinning_leak"].reshape(args.n_wl, args.n_power)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(powers, wls, data, cmap="Reds", shading="auto")
    add_boundary(ax)
    ax.plot(best["power_uw"], best["wl"], "k*", ms=15, mew=2, label="Best overall")
    ax.set_xlabel("Power ($\\mu$W)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title("Pinning leak  $P_{rg} + P_{rr}$")
    fig.colorbar(im, ax=ax, label="pinning leak")
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_opt_pinning_leak.png")
    fig.savefig(path, dpi=150)
    print(f"  Pinning leak heatmap saved to {path}")
    plt.close(fig)


def _plot_crosstalk(grid, meta, args):
    """Crosstalk heatmap → addressing_opt_crosstalk.png"""
    wls = meta["wls"]; powers = meta["powers"]; best = meta["best"]
    add_boundary = _make_adiabatic_boundary(meta)

    data = grid["crosstalk"].reshape(args.n_wl, args.n_power)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(powers, wls, data, cmap="Blues", shading="auto")
    add_boundary(ax)
    ax.plot(best["power_uw"], best["wl"], "k*", ms=15, mew=2, label="Best overall")
    ax.set_xlabel("Power ($\\mu$W)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title("Crosstalk  $P_{gg}$")
    fig.colorbar(im, ax=ax, label="crosstalk")
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_opt_crosstalk.png")
    fig.savefig(path, dpi=150)
    print(f"  Crosstalk heatmap saved to {path}")
    plt.close(fig)


def _plot_scatter(grid, meta, args):
    """Scatter penalty heatmap → addressing_opt_scatter.png"""
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
    path = os.path.join(args.outdir, "addressing_opt_scatter.png")
    fig.savefig(path, dpi=150)
    print(f"  Scatter heatmap saved to {path}")
    plt.close(fig)


def _plot_smoothed(grid, meta, args):
    """Gaussian-smoothed cost landscape → addressing_opt_smoothed.png"""
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
                 f"{sigma_smooth} pts, washes out LZS fringes)")
    fig.colorbar(im, ax=ax, label="$E_{total}$ (smoothed)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_opt_smoothed.png")
    fig.savefig(path, dpi=150)
    print(f"  Smoothed heatmap saved to {path}")
    print(f"    Smoothed best: lambda={best_smooth_wl:.2f} nm, "
          f"P={best_smooth_pow:.1f} uW, cost={best_smooth_cost:.5f}")
    plt.close(fig)


def _plot_stuckelberg(grid, meta, args):
    """LZS 1D slices per wavelength with theory overlay → addressing_opt_stuckelberg.png

    Transfer-matrix result: P = 4 P_LZ (1-P_LZ) cos^2(Phi_dyn/2 + phi_S)
    """
    wls = meta["wls"]
    lzs = _compute_lzs_params(meta)
    Omega_eff = lzs["Omega_eff"]
    P_LZ = lzs["P_LZ"]
    envelope = lzs["envelope"]
    alpha = lzs["alpha"]
    _lzs_error = lzs["lzs_error"]

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

        sim_error = (grid["pinning_leak"].reshape(args.n_wl, args.n_power)[wi]
                     + grid["crosstalk"].reshape(args.n_wl, args.n_power)[wi])

        ratio_dense = np.linspace(ratio.min(), ratio.max(), 500)
        lzs_dense = _lzs_error(ratio_dense * Omega_eff)

        ax.plot(ratio, sim_error * 100, "o", ms=3, color="#1f77b4", label="Simulation")
        ax.plot(ratio_dense, lzs_dense * 100, "-", lw=1.2, color="#d62728",
                label="LZS theory", alpha=0.85)
        ax.axvline(4.0, color="gray", ls=":", lw=1, alpha=0.6)
        ax.set_title(f"$\\lambda$ = {wl:.1f} nm", fontsize=10)
        ax.set_xlabel(r"$|\delta_A| / \Omega_{\rm eff}$", fontsize=9)
        ax.set_ylabel("Error (%)", fontsize=9)
        ax.tick_params(labelsize=8)
        if panel == 0:
            ax.legend(fontsize=7, loc="upper right")

    for j in range(len(sel_indices), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        r"LZS interference (transfer-matrix):  "
        r"$P = 4P_{\rm LZ}(1-P_{\rm LZ})\cos^2(\Phi_{\rm dyn}/2 + \phi_S)$"
        f"\n$P_{{LZ}}$ = {P_LZ:.2e},  envelope = {envelope*100:.3f}%,  "
        f"$\\alpha/(2\\pi)$ = {alpha/(2*pi)/1e12:.2f} THz/s",
        fontsize=11)
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_opt_stuckelberg.png")
    fig.savefig(path, dpi=150)
    print(f"  Stückelberg 1D overlay saved to {path}")
    plt.close(fig)


def _plot_lzs_2d(grid, meta, args):
    """LZS fringe contours on 2D coherent-error heatmap → addressing_opt_lzs_fringes_2d.png"""
    wls = meta["wls"]; powers = meta["powers"]; best = meta["best"]
    add_boundary = _make_adiabatic_boundary(meta)
    lzs = _compute_lzs_params(meta)
    alpha = lzs["alpha"]
    phi_stokes = lzs["phi_stokes"]

    delta_A_2d = np.abs(grid["delta_A_mhz"].reshape(args.n_wl, args.n_power))
    delta_A_2d_rad = delta_A_2d * 1e6 * 2 * pi
    Phi_St_2d = delta_A_2d_rad**2 / (2 * alpha) / 2 + phi_stokes
    cos2_2d = np.cos(Phi_St_2d)**2

    leak_xtalk_2d = (grid["pinning_leak"] + grid["crosstalk"]).reshape(
        args.n_wl, args.n_power)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(powers, wls, leak_xtalk_2d, cmap="RdYlGn_r", shading="auto")
    add_boundary(ax)
    ax.contour(powers, wls, cos2_2d, levels=[0.05], colors="cyan",
               linewidths=1.2, linestyles="-", alpha=0.9)
    ax.contour(powers, wls, cos2_2d, levels=[0.95], colors="magenta",
               linewidths=1.2, linestyles="--", alpha=0.9)
    ax.plot(best["power_uw"], best["wl"], "k*", ms=14, mew=2)
    ax.set_xlabel("Power ($\\mu$W)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title(
        "Coherent error (leak + xtalk) with LZS fringe contours\n"
        r"cyan = destructive ($\cos^2\Phi_{\rm St}=0$),  "
        r"magenta = constructive ($\cos^2\Phi_{\rm St}=1$)")
    fig.colorbar(im, ax=ax, label="pinning leak + crosstalk")
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_opt_lzs_fringes_2d.png")
    fig.savefig(path, dpi=150)
    print(f"  LZS 2D fringe map saved to {path}")
    plt.close(fig)


def _plot_spectrum(grid, meta, args):
    """Instantaneous eigenenergy spectrum for best candidate → addressing_opt_spectrum_best.png"""
    from ryd_gate.compilers.dense_atomic import DenseAtomicCompiler

    best = meta["best"]
    system = meta["system"]
    eta = meta["eta"]
    x_sweep = meta["x_sweep"]
    ac_stark_peak = meta["ac_stark_peak"]

    print(f"\n  Anti-blockade spectrum for best candidate...")
    delta_A_best = best["delta_A_mhz"] * 2 * pi * 1e6
    delta_B_best = eta * delta_A_best
    n_spec = 200
    Delta_scan = np.linspace(
        args.delta_start_mhz * 2 * pi * 1e6,
        args.delta_end_mhz * 2 * pi * 1e6, n_spec)

    compiler = DenseAtomicCompiler()
    proto_best = SweepProtocol(addressing={0: delta_A_best, 1: delta_B_best},
                               ac_stark_shift=ac_stark_peak)
    params_best = proto_best.unpack_params(x_sweep, system)
    ir = compiler.compile(system, proto_best, params_best)

    H_static = np.zeros((9, 9), dtype=complex)
    for term in ir.static_terms:
        c = term.coefficient(0) if callable(term.coefficient) else term.coefficient
        H_static += c * np.asarray(term.operator)

    eigvals_all = np.empty((n_spec, 9))
    for i, Delta in enumerate(Delta_scan):
        t_at_delta = (Delta - params_best["delta_start"]) / (
            params_best["delta_end"] - params_best["delta_start"]) * params_best["t_gate"]
        t_at_delta = np.clip(t_at_delta, 0, params_best["t_gate"])
        H = H_static.copy()
        for term in ir.drive_terms:
            coeff = term.coefficient(t_at_delta) if callable(term.coefficient) else term.coefficient
            H += coeff * np.asarray(term.operator)
        eigvals_all[i] = np.sort(np.linalg.eigvalsh(H.real + H.real.T) / 2)

    fig, ax = plt.subplots(figsize=(10, 6))
    Delta_mhz = Delta_scan / (2 * pi * 1e6)
    for j in range(9):
        ax.plot(Delta_mhz, eigvals_all[:, j] / (2 * pi * 1e6), lw=0.8)
    ax.axvline(0, color="gray", ls=":", lw=0.5, label=r"$\Delta=0$")
    dA_mhz = abs(delta_A_best) / (2 * pi * 1e6)
    ax.axhline(-dA_mhz, color="red", ls="--", lw=1,
               label=f"$|\\delta_A|={dA_mhz:.1f}$ MHz")
    v_mhz = system.v_ryd / (2 * pi * 1e6)
    ax.axhline(v_mhz, color="purple", ls="--", lw=1,
               label=f"$V_{{ryd}}={v_mhz:.0f}$ MHz")
    ax.set_xlabel(r"Two-photon detuning $\Delta/(2\pi)$ (MHz)")
    ax.set_ylabel("Eigenenergy / $(2\\pi)$ (MHz)")
    ax.set_title(f"Instantaneous spectrum (best: $\\lambda$={best['wl']:.1f} nm, "
                 f"$\\delta_A$={best['delta_A_mhz']:.1f} MHz)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(args.outdir, "addressing_opt_spectrum_best.png")
    fig.savefig(path, dpi=150)
    print(f"  Spectrum saved to {path}")
    plt.close(fig)

    if abs(delta_A_best) > 0.5 * system.v_ryd:
        print(f"  WARNING: |delta_A| = {dA_mhz:.1f} MHz approaches "
              f"V_ryd = {v_mhz:.0f} MHz -- check spectrum for resonances!")


def _plot_optimize(grid, meta, args):
    """Generate all plots from the optimization grid data."""
    os.makedirs(args.outdir, exist_ok=True)

    print("\n[1/9] Total cost heatmap...")
    _plot_total_cost(grid, meta, args)

    print("\n[2/9] Cost component overview (2×2)...")
    _plot_components(grid, meta, args)

    print("\n[3/9] Pinning leak heatmap...")
    _plot_pinning_leak(grid, meta, args)

    print("\n[4/9] Crosstalk heatmap...")
    _plot_crosstalk(grid, meta, args)

    print("\n[5/9] Scatter penalty heatmap...")
    _plot_scatter(grid, meta, args)

    print("\n[6/9] Smoothed cost landscape...")
    _plot_smoothed(grid, meta, args)

    print("\n[7/9] Stückelberg 1D slices...")
    _plot_stuckelberg(grid, meta, args)

    print("\n[8/9] LZS 2D fringe map...")
    _plot_lzs_2d(grid, meta, args)

    print("\n[9/9] Anti-blockade spectrum...")
    _plot_spectrum(grid, meta, args)


def cmd_optimize(args):
    """2D (wavelength x power) optimization of local addressing parameters."""
    grid, meta = _run_optimize_scan(args)
    _plot_optimize(grid, meta, args)


def cmd_optimize_plot(args):
    """Re-plot from a previously saved CSV grid (no simulation)."""
    from ryd_gate.core.models.analog_3level import Analog3LevelModel

    # Load grid
    dtype = [
        ("wl", float), ("power_uw", float),
        ("delta_A_mhz", float), ("delta_B_mhz", float),
        ("scatter_A_hz", float),
        ("P_gg", float), ("P_gr", float), ("P_rg", float), ("P_rr", float),
        ("pinning_leak", float), ("crosstalk", float),
        ("scatter_penalty", float), ("total_cost", float),
        ("P_target", float), ("other_residual", float),
        ("delta_A_over_Omega", float), ("delta_A_over_V", float),
    ]
    raw = np.loadtxt(args.csv, delimiter=",", dtype=float)
    grid = np.empty(raw.shape[0], dtype=dtype)
    for i, name in enumerate(grid.dtype.names):
        grid[name] = raw[:, i]
    print(f"  Loaded {len(grid)} points from {args.csv}")

    # Reconstruct grid shape from unique values
    wls = np.unique(grid["wl"])
    powers = np.unique(grid["power_uw"])
    args.n_wl = len(wls)
    args.n_power = len(powers)

    # Rebuild model for spectrum plot and adiabatic boundary
    model = Analog3LevelModel.from_defaults(
        detuning_sign=1, blackmanflag=True,
        distance_um=args.distance_um,
        Delta_Hz=2.4e9,
        rabi_420_Hz=135e6,
        rabi_1013_Hz=135e6,
    )
    system = model.system
    eta = np.exp(-2 * (args.distance_um / args.waist_um) ** 2)
    ac_stark_peak = system.rabi_420 ** 2 / (4 * abs(system.Delta))

    t_gate = args.t_gate_us * 1e-6
    x_sweep = [
        2 * pi * args.delta_start_mhz * 1e6 / system.rabi_eff,
        2 * pi * args.delta_end_mhz * 1e6 / system.rabi_eff,
        t_gate / system.time_scale,
    ]

    # Adiabatic boundary
    adiabatic_ratio = 4.0
    adiabatic_power_min = {}
    for wl in wls:
        shift_ref, _ = compute_shift_scatter(wl)
        shift_per_uw = abs(float(shift_ref)) / POWER_REF_UW
        adiabatic_power_min[wl] = adiabatic_ratio * system.rabi_eff / (2 * pi) / shift_per_uw

    best_idx = np.argmin(grid["total_cost"])
    best = grid[best_idx]

    meta = {
        "system": system, "model": model, "eta": eta, "x_sweep": x_sweep,
        "ac_stark_peak": ac_stark_peak, "wls": wls, "powers": powers,
        "best": best, "best_idx": best_idx,
        "adiabatic_power_min": adiabatic_power_min,
    }
    os.makedirs(args.outdir, exist_ok=True)

    print("\n[1/9] Total cost heatmap...")
    _plot_total_cost(grid, meta, args)

    print("\n[2/9] Cost component overview (2×2)...")
    _plot_components(grid, meta, args)

    print("\n[3/9] Pinning leak heatmap...")
    _plot_pinning_leak(grid, meta, args)

    print("\n[4/9] Crosstalk heatmap...")
    _plot_crosstalk(grid, meta, args)

    print("\n[5/9] Scatter penalty heatmap...")
    _plot_scatter(grid, meta, args)

    print("\n[6/9] Smoothed cost landscape...")
    _plot_smoothed(grid, meta, args)

    print("\n[7/9] Stückelberg 1D slices...")
    _plot_stuckelberg(grid, meta, args)

    print("\n[8/9] LZS 2D fringe map...")
    _plot_lzs_2d(grid, meta, args)

    print("\n[9/9] Anti-blockade spectrum...")
    _plot_spectrum(grid, meta, args)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_wl = sub.add_parser("wavelength", help="Scan laser wavelength (fast physics)")
    p_wl.add_argument("--with-sim", action="store_true",
                       help="Also run MC addressing sim at sampled wavelengths")
    p_wl.add_argument("--n-mc", type=int, default=50,
                       help="MC shots per wavelength for --with-sim (default: 50)")

    p_ns = sub.add_parser("noise", help="Scan noise parameters")
    p_ns.add_argument("--n-mc", type=int, default=50,
                       help="MC shots per point (default: 50)")
    p_ns.add_argument("--n-points", type=int, default=8,
                       help="Scan points per noise source (default: 8)")
    p_ns.add_argument("--combined", action="store_true",
                       help="Also run combined noise sweep")

    p_opt = sub.add_parser("optimize",
                            help="2D wavelength x power optimization")
    p_opt.add_argument("--wl-min", type=float, default=781.0)
    p_opt.add_argument("--wl-max", type=float, default=786.0)
    p_opt.add_argument("--n-wl", type=int, default=60)
    p_opt.add_argument("--power-min-uw", type=float, default=140,
                        help="Min power in uW (default: auto from adiabatic guarantee)")
    p_opt.add_argument("--power-max-uw", type=float, default=500,
                        help="Max power in uW (default: 2x adiabatic min)")
    p_opt.add_argument("--n-power", type=int, default=40)
    p_opt.add_argument("--distance-um", type=float, default=4.0,
                        help="Atom separation in um (default: 4.0, "
                             "gives V_ryd >> delta_A to avoid anti-blockade)")
    p_opt.add_argument("--waist-um", type=float, default=1.0,
                        help="Addressing beam waist in um (default: 1.0)")
    p_opt.add_argument("--delta-start-mhz", type=float, default=-40.0)
    p_opt.add_argument("--delta-end-mhz", type=float, default=40.0)
    p_opt.add_argument("--t-gate-us", type=float, default=9)
    p_opt.add_argument("--top-k", type=int, default=5)
    p_opt.add_argument("--outdir", type=str, default="results")

    p_plt = sub.add_parser("optimize-plot",
                            help="Re-plot from saved CSV (no simulation)")
    p_plt.add_argument("--csv", type=str, default="results/addressing_opt_grid.csv",
                        help="Path to grid CSV from optimize run")
    p_plt.add_argument("--distance-um", type=float, default=4.0)
    p_plt.add_argument("--waist-um", type=float, default=2.0)
    p_plt.add_argument("--delta-start-mhz", type=float, default=-40.0)
    p_plt.add_argument("--delta-end-mhz", type=float, default=40.0)
    p_plt.add_argument("--t-gate-us", type=float, default=4.5)
    p_plt.add_argument("--outdir", type=str, default="results")

    args = parser.parse_args()

    if args.command == "wavelength":
        cmd_wavelength(args)
    elif args.command == "noise":
        cmd_noise(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "optimize-plot":
        cmd_optimize_plot(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
