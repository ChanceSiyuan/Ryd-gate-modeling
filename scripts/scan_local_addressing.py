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
    build_sss_state_map,
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
        initial_state = build_sss_state_map(n_levels=3)["00"]
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
    initial_state = build_sss_state_map(n_levels=3)["00"]
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

    args = parser.parse_args()

    if args.command == "wavelength":
        cmd_wavelength(args)
    elif args.command == "noise":
        cmd_noise(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
