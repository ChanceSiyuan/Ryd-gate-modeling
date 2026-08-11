#!/usr/bin/env python3
"""Analyze which laser-noise frequencies drive the optimal 297 nm gate error."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(
    os.environ.get("RYD_GATE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, os.fspath(ROOT / "src"))

from ryd_gate.phase_noise import PhaseNoisePSD  # noqa: E402


RESULT_ROOT = ROOT / "results" / "297_laser_noise"
FILTER_ROOT = ROOT / "results" / "max_leakage_297" / "a3.0" / "filter"
OMEGA_HZ = 14.25e6
LOGICAL_INPUTS = ("00", "01", "10", "11")
COLORS = {"ECDL": "#c75b32", "seed": "#087f76"}


def load_optimal_kernel() -> tuple[np.ndarray, np.ndarray]:
    """Load the four-input kernel at n=73, T=1 us, Omega=14.25 MHz."""
    for path in sorted(FILTER_ROOT.glob("filter_*.npz")):
        with np.load(path) as data:
            mask = (
                (np.asarray(data["n_idx"]) == 7)
                & (np.asarray(data["t_idx"]) == 0)
                & np.isclose(np.asarray(data["omega297_mhz"]), 14.25)
                & np.isclose(np.asarray(data["dsweep_mhz"]), 17.5)
            )
            rows = np.flatnonzero(mask)
            if not rows.size:
                continue
            row = int(rows[0])
            f_bins = np.asarray(data["f_bins"])
            if f_bins.ndim > 1:
                f_bins = f_bins[row]
            return f_bins.copy(), np.asarray(data["kernel"])[row].copy()
    raise RuntimeError("optimal-point filter kernel not found")


def phase_contribution(laser: str, f_bins: np.ndarray,
                       kernel: np.ndarray) -> tuple[PhaseNoisePSD, np.ndarray]:
    psd = PhaseNoisePSD.from_csv(
        RESULT_ROOT / f"psd_{laser}.csv",
        harmonic=4,
        extrapolation="power",
    )
    contribution = 2.0 * np.pi**2 * kernel * psd.s_dnu(f_bins)[None, :]
    return psd, contribution


def print_summary(f_bins: np.ndarray, contributions: dict[str, np.ndarray]) -> None:
    near = (f_bins >= OMEGA_HZ / 2.0) & (f_bins <= 2.0 * OMEGA_HZ)
    high = f_bins > 1e6
    print("laser f_min_Hz worst_input eps_phase gt1MHz/total "
          "Omega_octave/gt1MHz high_peak_MHz")
    for laser, contribution in contributions.items():
        high_state = int(np.argmax(contribution[:, high].sum(axis=1)))
        high_peak = f_bins[high][np.argmax(contribution[high_state, high])]
        for f_min in (1.0, 10.0):
            included = f_bins >= f_min
            totals = contribution[:, included].sum(axis=1)
            state = int(np.argmax(totals))
            total = float(totals[state])
            high_error = float(contribution[state, high].sum())
            near_error = float(contribution[state, near].sum())
            print(
                f"{laser:4s} {f_min:8.1f} {LOGICAL_INPUTS[state]:>11s} "
                f"{total:.9g} {high_error / total:.6f} "
                f"{near_error / high_error:.6f} {high_peak / 1e6:.6f}"
            )


def render(f_bins: np.ndarray, psds: dict[str, PhaseNoisePSD],
           contributions: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    fig.patch.set_facecolor("#f3efe4")
    f_plot = np.logspace(5, np.log10(f_bins[-1]), 1000)
    near_lo, near_hi = OMEGA_HZ / 2.0, 2.0 * OMEGA_HZ

    ax = axes[0]
    ax.set_facecolor("#fffdf7")
    for laser, psd in psds.items():
        ax.loglog(f_plot, psd.s_dnu(f_plot), color=COLORS[laser],
                  linewidth=2.2, label=laser)
    ax.axvline(1e6, color="#4e4a42", linestyle="--", linewidth=1.2,
               label="measurement edge")
    ax.axvspan(near_lo, near_hi, color="#d8c9a7", alpha=0.35, linewidth=0)
    ax.axvline(OMEGA_HZ, color="#8c7a52", linestyle=":", linewidth=1.5,
               label=r"$\Omega/2\pi$")
    ax.set_xlabel("Offset frequency (Hz)")
    ax.set_ylabel(r"$S_{\delta\nu}^{297}$ (Hz$^2$/Hz)")
    ax.set_title("Measured spectrum + power-law tail", weight="bold")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    ax.set_facecolor("#fffdf7")
    high = f_bins > 1e6
    for laser, contribution in contributions.items():
        state = int(np.argmax(contribution[:, high].sum(axis=1)))
        share = 100.0 * contribution[state, high] / contribution[state, high].sum()
        ax.semilogx(f_bins[high], share, color=COLORS[laser], linewidth=2.2,
                    label=f"{laser}, input {LOGICAL_INPUTS[state]}")
    ax.axvspan(near_lo, near_hi, color="#d8c9a7", alpha=0.35, linewidth=0,
               label=r"$[\Omega/2,2\Omega]/2\pi$")
    ax.axvline(OMEGA_HZ, color="#8c7a52", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Offset frequency (Hz)")
    ax.set_ylabel("Share of >1 MHz phase error per log bin (%)")
    ax.set_title("Filter-weighted fidelity-loss contribution", weight="bold")
    ax.legend(frameon=False, fontsize=8.5)

    for ax in axes:
        ax.grid(which="major", color="#d8d1c3", linewidth=0.8, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Why the high-frequency error is controlled near the Rabi frequency",
                 fontsize=14.5, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "noise_extrapolation_filter_overlap.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    f_bins, kernel = load_optimal_kernel()
    psds: dict[str, PhaseNoisePSD] = {}
    contributions: dict[str, np.ndarray] = {}
    for laser in ("ECDL", "seed"):
        psds[laser], contributions[laser] = phase_contribution(
            laser, f_bins, kernel
        )
    print_summary(f_bins, contributions)
    render(f_bins, psds, contributions)


if __name__ == "__main__":
    main()
