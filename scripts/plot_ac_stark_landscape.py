#!/usr/bin/env python3
"""Plot AC Stark shift and scattering rate as a function of (power, wavelength).

Uses the calibrated ``compute_shift_scatter`` from ``ryd_gate.physics.ac_stark``,
which returns values at ``POWER_REF_UW`` (160 μW). Both quantities scale
linearly in power, so we just build a 1D profile in wavelength and multiply by
``P / POWER_REF_UW`` to get the 2D grid.

Outputs (under ``--outdir``, default ``results/``):
    [<prefix>_]ac_stark_landscape.png      — 1×2 heatmap (shift | scatter)
    [<prefix>_]ac_stark_profiles.png       — 1D profiles at a few fixed powers
    [<prefix>_]ac_stark_landscape.csv      — raw grid data

Usage
-----
    uv run --active python scripts/plot_ac_stark_landscape.py
    uv run --active python scripts/plot_ac_stark_landscape.py \
        --wl-min 780.5 --wl-max 794.5 --n-wl 400 \
        --power-min 1 --power-max 500 --n-power 200 \
        --prefix wide
"""
from __future__ import annotations

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm

# Use matplotlib's built-in mathtext engine (no external LaTeX install needed).
# All label strings below use raw r"$...$" with LaTeX-style commands.
mpl.rcParams.update({
    "text.usetex": False,          # keep dependency-free; mathtext handles `$...$`
    "mathtext.fontset": "cm",       # Computer Modern look-alike
    "mathtext.default": "regular",
    "font.family": "serif",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# Common slim-colorbar kwargs used by both heatmap panels. `fraction=0.032`
# makes the colorbar noticeably narrower than matplotlib's default (0.15).
_CBAR_KW = dict(fraction=0.032, pad=0.02, aspect=28)

from ryd_gate.physics.ac_stark import (
    G_F_5S12_F2,
    LAMBDA_D1,
    LAMBDA_D2,
    LAMBDA_PAPER,
    POWER_REF_UW,
    compute_shift_scatter,
)


def _prefixed(filename, prefix):
    if not prefix:
        return filename
    return f"{prefix}_{filename}"


def build_grid(wl_nm, power_uw):
    """Return (shift_MHz, scatter_Hz) on the 2D (wl, power) grid.

    Shape of each output: (len(wl_nm), len(power_uw)). Both quantities scale
    linearly in power, so this is just an outer product of the 1D
    compute_shift_scatter profile with P/POWER_REF_UW.
    """
    shift_ref_hz, scatter_ref_hz = compute_shift_scatter(wl_nm)  # shape (n_wl,)
    scale = power_uw / POWER_REF_UW                              # shape (n_power,)
    shift_hz  = shift_ref_hz[:, None]   * scale[None, :]
    scatter_hz = scatter_ref_hz[:, None] * scale[None, :]
    return shift_hz / 1e6, scatter_hz  # shift in MHz, scatter in Hz


def plot_landscape(wl_nm, power_uw, shift_mhz, scatter_hz, args):
    """2×1 heatmap figure — shift (symlog) on the left, scatter (log) on the right."""
    os.makedirs(args.outdir, exist_ok=True)
    fig, (ax_s, ax_sc) = plt.subplots(1, 2, figsize=(14, 6.5))

    # ── Light-shift heatmap (signed, so use SymLogNorm) ──────────────────
    lin_thr = 0.5  # MHz: below this magnitude show linearly
    vmax = float(np.nanmax(np.abs(shift_mhz)))
    norm_shift = SymLogNorm(linthresh=lin_thr, linscale=0.5,
                            vmin=-vmax, vmax=vmax, base=10)
    im_s = ax_s.pcolormesh(power_uw, wl_nm, shift_mhz,
                           cmap="seismic", norm=norm_shift, shading="auto")
    # Zero-shift contour so the "magic wavelength" where shift crosses 0 is clear.
    try:
        cs = ax_s.contour(power_uw, wl_nm, shift_mhz,
                          levels=[0.0], colors="k", linewidths=1.2, linestyles="--")
        ax_s.clabel(cs, inline=True, fontsize=8,
                    fmt=r"$\Delta_{\mathrm{LS}} = 0$")
    except (ValueError, IndexError):
        pass  # no zero crossing in the plotted range
    # Reference lines: D1, D2, calibration wavelength.
    for wl, label, color in [
        (LAMBDA_D1,
         rf"$\mathrm{{D_1}} = {LAMBDA_D1:.2f}\,\mathrm{{nm}}$", "0.3"),
        (LAMBDA_D2,
         rf"$\mathrm{{D_2}} = {LAMBDA_D2:.2f}\,\mathrm{{nm}}$", "0.3"),
        (LAMBDA_PAPER,
         rf"$\lambda_{{\mathrm{{cal}}}} = {LAMBDA_PAPER:.0f}\,\mathrm{{nm}}$",
         "gold"),
    ]:
        if wl_nm[0] <= wl <= wl_nm[-1]:
            ax_s.axhline(wl, color=color, lw=1.0, ls=":", alpha=0.8)
            ax_s.text(power_uw[-1], wl, "  " + label,
                      color=color, fontsize=7, va="center", ha="left")
    # Calibration point marker
    if (wl_nm[0] <= LAMBDA_PAPER <= wl_nm[-1]
            and power_uw[0] <= POWER_REF_UW <= power_uw[-1]):
        ax_s.plot(POWER_REF_UW, LAMBDA_PAPER, "k*", ms=10, mec="gold", mew=1.0,
                  label=(rf"cal $({POWER_REF_UW:.0f}\,\mu\mathrm{{W}},\ "
                         rf"{LAMBDA_PAPER:.0f}\,\mathrm{{nm}})$"))
        ax_s.legend(fontsize=8, loc="lower right")
    ax_s.set_xlabel(r"Laser power $P$ ($\mu$W)")
    ax_s.set_ylabel(r"Wavelength $\lambda$ (nm)")
    ax_s.set_title(
        r"Ground-state AC Stark shift $\Delta_{\mathrm{LS}}$ (MHz)")
    cb_s = fig.colorbar(im_s, ax=ax_s, extend="both", **_CBAR_KW)
    cb_s.set_label(r"$\Delta_{\mathrm{LS}}$ (MHz)")

    # ── Scattering-rate heatmap (positive-definite, pure log) ─────────────
    # Floor tiny values to avoid log(0).
    sc_floor = max(1e-3, float(np.nanmin(scatter_hz[scatter_hz > 0])) if
                   np.any(scatter_hz > 0) else 1e-3)
    sc_ceil = float(np.nanmax(scatter_hz))
    norm_sc = LogNorm(vmin=sc_floor, vmax=sc_ceil)
    im_sc = ax_sc.pcolormesh(power_uw, wl_nm,
                             np.clip(scatter_hz, sc_floor, None),
                             cmap="magma_r", norm=norm_sc, shading="auto")
    for wl, label, color in [
        (LAMBDA_D1,
         rf"$\mathrm{{D_1}} = {LAMBDA_D1:.2f}\,\mathrm{{nm}}$", "0.3"),
        (LAMBDA_D2,
         rf"$\mathrm{{D_2}} = {LAMBDA_D2:.2f}\,\mathrm{{nm}}$", "0.3"),
        (LAMBDA_PAPER,
         rf"$\lambda_{{\mathrm{{cal}}}} = {LAMBDA_PAPER:.0f}\,\mathrm{{nm}}$",
         "gold"),
    ]:
        if wl_nm[0] <= wl <= wl_nm[-1]:
            ax_sc.axhline(wl, color=color, lw=1.0, ls=":", alpha=0.8)
            ax_sc.text(power_uw[-1], wl, "  " + label,
                       color=color, fontsize=7, va="center", ha="left")
    if (wl_nm[0] <= LAMBDA_PAPER <= wl_nm[-1]
            and power_uw[0] <= POWER_REF_UW <= power_uw[-1]):
        ax_sc.plot(POWER_REF_UW, LAMBDA_PAPER, "w*", ms=10, mec="k", mew=1.0)
    ax_sc.set_xlabel(r"Laser power $P$ ($\mu$W)")
    ax_sc.set_ylabel(r"Wavelength $\lambda$ (nm)")
    ax_sc.set_title(r"Scattering rate $\Gamma_{\mathrm{sc}}$ (Hz)")
    cb_sc = fig.colorbar(im_sc, ax=ax_sc, **_CBAR_KW)
    cb_sc.set_label(r"$\Gamma_{\mathrm{sc}}$ (Hz)")

    fig.suptitle(
        r"AC Stark landscape — calibrated at "
        rf"$\lambda = {LAMBDA_PAPER:.0f}\,\mathrm{{nm}}$, "
        rf"$P = {POWER_REF_UW:.0f}\,\mu\mathrm{{W}}$ "
        r"(Manovitz et al.)",
        fontsize=13)
    fig.tight_layout()
    path = os.path.join(args.outdir,
                        _prefixed("ac_stark_landscape.png", args.prefix))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_profiles(wl_nm, power_uw, shift_mhz, scatter_hz, args):
    """1D slices: shift & scatter vs wavelength at a few representative powers."""
    os.makedirs(args.outdir, exist_ok=True)
    # Pick up to 5 representative powers (or all if n_power ≤ 5).
    if len(power_uw) <= 5:
        idxs = np.arange(len(power_uw))
    else:
        idxs = np.linspace(0, len(power_uw) - 1, 5, dtype=int)

    fig, (ax_s, ax_sc) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)

    # Plot |Δ_LS| on a pure log axis. The sign (red-detuned vs blue-detuned)
    # is encoded in the linestyle: solid for Δ_LS < 0 (red-detuned, shift
    # pushes |g⟩ down, typical for λ > λ_{tune-out}) and dashed for
    # Δ_LS > 0 (blue-detuned). Avoids losing information to log-scale
    # positivity while still making the dynamic range readable.
    cmap = plt.get_cmap("viridis")
    for k, ip in enumerate(idxs):
        P = power_uw[ip]
        color = cmap((k + 0.5) / len(idxs))
        s = shift_mhz[:, ip]
        mask_neg = s < 0
        mask_pos = s > 0
        ax_s.plot(wl_nm[mask_neg], np.abs(s[mask_neg]),
                  lw=1.6, color=color, ls="-",
                  label=rf"$P = {P:.0f}\,\mu\mathrm{{W}}$")
        ax_s.plot(wl_nm[mask_pos], np.abs(s[mask_pos]),
                  lw=1.6, color=color, ls="--")
        ax_sc.plot(wl_nm, scatter_hz[:, ip], lw=1.6, color=color,
                   label=rf"$P = {P:.0f}\,\mu\mathrm{{W}}$")

    # Tune-out wavelength: find where |Δ_LS| at the largest tabulated power
    # is minimum. That's the (calibration-derived) magic wavelength.
    iP_ref = idxs[-1]
    i_tune = int(np.argmin(np.abs(shift_mhz[:, iP_ref])))
    lambda_tune = wl_nm[i_tune]

    # Reference vertical lines (D1, D2, calibration, tune-out)
    for ax in (ax_s, ax_sc):
        for wl, color, label in [
            (LAMBDA_D1, "0.35",
             rf"$\mathrm{{D_1}}$ ({LAMBDA_D1:.2f} nm)"),
            (LAMBDA_D2, "0.35",
             rf"$\mathrm{{D_2}}$ ({LAMBDA_D2:.2f} nm)"),
            (LAMBDA_PAPER, "gold",
             rf"$\lambda_{{\mathrm{{cal}}}}$"),
            (lambda_tune, "red",
             rf"$\lambda_{{\mathrm{{tune-out}}}}\!\approx\!{lambda_tune:.2f}$ nm"),
        ]:
            if wl_nm[0] <= wl <= wl_nm[-1]:
                ax.axvline(wl, color=color, lw=1.0, ls=":", alpha=0.85)
        ax.grid(alpha=0.25, which="both")

    ax_s.set_ylabel(r"$|\Delta_{\mathrm{LS}}|$  (MHz)")
    ax_s.set_title(
        r"Ground-state AC Stark shift magnitude "
        r"$|\Delta_{\mathrm{LS}}(\lambda)|$"
        "\n"
        r"(solid: $\Delta_{\mathrm{LS}}<0$ red-detuned,  "
        r"dashed: $\Delta_{\mathrm{LS}}>0$ blue-detuned)",
        fontsize=11)
    ax_s.set_yscale("log")
    # Clamp the lower y limit so the dip at the tune-out doesn't stretch the
    # axis to 1e-8 MHz. 10 kHz = 1e-2 MHz is already below any physics relevant
    # at these powers.
    ax_s.set_ylim(1e-2, None)

    ax_sc.set_ylabel(r"$\Gamma_{\mathrm{sc}}$  (Hz)")
    ax_sc.set_xlabel(r"Wavelength  $\lambda$  (nm)")
    ax_sc.set_title(r"Photon scattering rate $\Gamma_{\mathrm{sc}}(\lambda)$")
    ax_sc.set_yscale("log")

    # Annotate the tune-out on the shift panel. Must read ylim *after* the
    # set_ylim() clamp above so the arrow lands inside the visible range.
    lo_s, _ = ax_s.get_ylim()
    ax_s.annotate(
        rf"tune-out $\approx {lambda_tune:.2f}$ nm",
        xy=(lambda_tune, lo_s * 1.5),
        xytext=(lambda_tune + 0.4, lo_s * 20),
        color="red", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    # One shared legend at the top (powers) to avoid duplication.
    handles, labels = ax_s.get_legend_handles_labels()
    ax_s.legend(handles, labels, fontsize=9, loc="upper left", ncol=1,
                framealpha=0.9, title=r"$P_{\mathrm{laser}}$")
    ax_sc.legend(fontsize=9, loc="lower right", framealpha=0.9,
                 title=r"$P_{\mathrm{laser}}$")

    fig.tight_layout()
    path = os.path.join(args.outdir,
                        _prefixed("ac_stark_profiles.png", args.prefix))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_scatter_per_shift(wl_nm, args):
    """Two power-independent figures of merit vs wavelength.

    Top:    Γ_sc / |Δ_LS|   (Hz of ground-state scattering per Hz of light shift).
    Bottom: |Δ_LS| required to make the ground-state photon scattering equal
            the Rydberg-state decay rate Γ_r = 1/151.55 μs ≈ 6.6 kHz.
            Equivalently: |Δ_LS|_match = Γ_r / (Γ_sc/|Δ_LS|).
    Both ratios are power-independent because Γ_sc and Δ_LS both scale
    linearly in laser power.
    """
    os.makedirs(args.outdir, exist_ok=True)
    shift_ref_hz, scatter_ref_hz = compute_shift_scatter(wl_nm)
    ratio = scatter_ref_hz / np.abs(shift_ref_hz)        # Hz / Hz
    shift_match_mhz = (args.gamma_ryd_hz / ratio) / 1e6  # MHz

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # ── Panel 1: Γ_sc / |Δ_LS| ───────────────────────────────────────────
    ax.plot(wl_nm, ratio, lw=1.8, color="C0")
    ax.set_yscale("log")
    ax.set_ylabel(
        r"$\Gamma_{\mathrm{sc}}\,/\,|\Delta_{\mathrm{LS}}|$"
        "\n(Hz scattering per Hz light shift)")
    ax.set_title(
        r"Scattering rate per unit light shift "
        r"$\Gamma_{\mathrm{sc}}(\lambda)\,/\,|\Delta_{\mathrm{LS}}(\lambda)|$"
        "   (power-independent)")
    ax.grid(alpha=0.3, which="both")

    ax_r = ax.twinx()
    ax_r.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax_r.set_ylim(lo * 1e6, hi * 1e6)
    ax_r.set_ylabel(r"(Hz per MHz light shift)")

    # ── Panel 2: required |Δ_LS| to match Γ_r ───────────────────────────
    ax2.plot(wl_nm, shift_match_mhz, lw=1.8, color="C3")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Wavelength  $\lambda$  (nm)")
    ax2.set_ylabel(
        r"$|\Delta_{\mathrm{LS}}|$ to reach $\Gamma_{\mathrm{sc}}=\Gamma_r$"
        "\n(MHz)")
    ax2.set_title(
        r"Light shift needed for ground-state $\Gamma_{\mathrm{sc}}$ "
        rf"to equal $\Gamma_r = {args.gamma_ryd_hz:.0f}\,\mathrm{{Hz}}$ "
        rf"($\tau_r = {1e6/args.gamma_ryd_hz:.1f}\,\mu\mathrm{{s}}$)")
    ax2.grid(alpha=0.3, which="both")

    # Tune-out marker (where |Δ_LS| → 0, ratio → ∞, required shift → 0)
    i_tune = int(np.argmin(np.abs(shift_ref_hz)))
    lambda_tune = wl_nm[i_tune]

    refs = [
        (LAMBDA_D1, "0.35", rf"$\mathrm{{D_1}}$ ({LAMBDA_D1:.2f} nm)"),
        (LAMBDA_D2, "0.35", rf"$\mathrm{{D_2}}$ ({LAMBDA_D2:.2f} nm)"),
        (LAMBDA_PAPER, "gold", rf"$\lambda_{{\mathrm{{cal}}}}$"),
        (lambda_tune, "red",
         rf"$\lambda_{{\mathrm{{tune-out}}}}\!\approx\!{lambda_tune:.2f}$ nm"),
    ]
    for axx in (ax, ax2):
        for wl, color, label in refs:
            if wl_nm[0] <= wl <= wl_nm[-1]:
                axx.axvline(wl, color=color, lw=1.0, ls=":", alpha=0.85)
        for wl, color, label in refs:
            if wl_nm[0] <= wl <= wl_nm[-1]:
                axx.text(wl, axx.get_ylim()[1], "  " + label, color=color,
                         fontsize=8, va="top", ha="left", rotation=90)

    # Print value at calibration wavelength for sanity.
    if wl_nm[0] <= LAMBDA_PAPER <= wl_nm[-1]:
        i = int(np.argmin(np.abs(wl_nm - LAMBDA_PAPER)))
        print(f"  @ λ_cal={wl_nm[i]:.2f} nm: "
              f"Γ_sc/|Δ_LS| = {ratio[i]:.3e} Hz/Hz, "
              f"|Δ_LS| to match Γ_r = {shift_match_mhz[i]:.3f} MHz")

    fig.tight_layout()
    path = os.path.join(args.outdir,
                        _prefixed("ac_stark_scatter_per_shift.png", args.prefix))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_polarization_sensitivity(wl_nm, args):
    """Sweep polarization purity to test the 784–786 nm sweet spot.

    Compares the scalar-only (linear-pol) model against several levels of
    circular contamination on a |F=2, m_F=-2⟩ atom (g_F = +1/2). The
    polarization parameter passed to ``compute_shift_scatter`` is
    ``pol = P · g_F · m_F``; for m_F=-2 this is just ``-P/1``, so a
    helicity P ∈ [-1, 1] maps directly to pol ∈ [1, -1].

    Plots, for each polarization scenario:
      (a) |Δ_LS|(λ)            — vertical position of the tune-out
      (b) Γ_sc / |Δ_LS|         — scattering per shift figure of merit
      (c) |Δ_LS| to match Γ_r   — required shift for ground scattering
                                  to equal Rydberg decay
    """
    os.makedirs(args.outdir, exist_ok=True)
    m_F = -2
    helicities = [0.0, 0.01, 0.05, 0.10, 0.30]  # |P| values, σ⁺ direction
    cmap = plt.get_cmap("viridis")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    tune_out_table = []
    for k, P in enumerate(helicities):
        pol = P * G_F_5S12_F2 * m_F     # = -P  for m_F=-2
        shift_hz, scatter_hz = compute_shift_scatter(wl_nm, pol=pol)
        ratio = scatter_hz / np.abs(shift_hz)
        shift_match_mhz = (args.gamma_ryd_hz / ratio) / 1e6

        i_tune = int(np.argmin(np.abs(shift_hz)))
        lambda_tune = wl_nm[i_tune]
        tune_out_table.append((P, lambda_tune))

        color = cmap(0.15 + 0.75 * k / max(1, len(helicities) - 1))
        label = (rf"$P={P:.2f}$ (linear)" if P == 0
                 else rf"$P={P:.2f}\ \sigma^+$ contamination")

        ax1.plot(wl_nm, np.abs(shift_hz) / 1e6, lw=1.6, color=color, label=label)
        ax2.plot(wl_nm, ratio, lw=1.6, color=color, label=label)
        ax3.plot(wl_nm, shift_match_mhz, lw=1.6, color=color, label=label)
        ax1.axvline(lambda_tune, color=color, lw=0.8, ls=":", alpha=0.6)
        ax3.axvline(lambda_tune, color=color, lw=0.8, ls=":", alpha=0.6)

    for axx in (ax1, ax2, ax3):
        for wl, color in [(LAMBDA_D1, "0.35"), (LAMBDA_D2, "0.35"),
                          (LAMBDA_PAPER, "gold")]:
            if wl_nm[0] <= wl <= wl_nm[-1]:
                axx.axvline(wl, color=color, lw=1.0, ls="--", alpha=0.6)
        axx.grid(alpha=0.3, which="both")
        axx.set_yscale("log")

    ax1.set_ylabel(r"$|\Delta_{\mathrm{LS}}|$  (MHz)")
    ax1.set_ylim(1e-2, None)
    ax1.set_title(
        r"Light shift magnitude vs polarization purity "
        rf"($|F=2, m_F={m_F}\rangle$, $g_F={G_F_5S12_F2}$)")
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.9)

    ax2.set_ylabel(
        r"$\Gamma_{\mathrm{sc}}/|\Delta_{\mathrm{LS}}|$" "\n(Hz / Hz)")
    ax2.set_title("Scattering per unit light shift")

    ax3.set_xlabel(r"Wavelength  $\lambda$  (nm)")
    ax3.set_ylabel(
        r"$|\Delta_{\mathrm{LS}}|$ to reach $\Gamma_r$" "\n(MHz)")
    ax3.set_title(
        rf"Light shift required so $\Gamma_{{\mathrm{{sc}}}}=\Gamma_r"
        rf"={args.gamma_ryd_hz:.0f}\,\mathrm{{Hz}}$")

    fig.tight_layout()
    path = os.path.join(args.outdir,
                        _prefixed("ac_stark_pol_sensitivity.png", args.prefix))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # Console summary at the 784–786 sweet-spot wavelengths.
    print("  Polarization sensitivity at sweet-spot wavelengths "
          "(|F=2, m_F=-2⟩):")
    print(f"    {'P':>6}  {'λ_tune (nm)':>12}  "
          f"{'|ΔLS|@Γr (785 nm) [MHz]':>26}")
    for (P, lam_tune) in tune_out_table:
        pol = P * G_F_5S12_F2 * m_F
        s, sc = compute_shift_scatter(np.array([785.0]), pol=pol)
        match_mhz = (args.gamma_ryd_hz / (sc[0] / abs(s[0]))) / 1e6
        print(f"    {P:>6.2f}  {lam_tune:>12.3f}  {match_mhz:>26.2f}")


def plot_vector_optimization(wl_nm, args):
    """Joint scalar+vector Stark figure of merit, restricted to a sweet-spot window.

    Implements the "minimize vector sensitivity per unit usable scalar shift"
    principle: rather than chasing the tune-out (where the vector contamination
    looks small only because Δ_scalar → 0), we minimize

        FOM(λ) ≡ |∂Δ_LS / ∂pol|_{pol=0} / |Δ_LS(λ, pol=0)|

    which is dimensionless and equals the fractional vector-shift error per
    unit polarization impurity. Smaller is better. We then restrict the search
    to ``[--vec-window-min, --vec-window-max]`` (default 784–786 nm) — the
    region you've already identified as having usable scalar shift and low
    scattering — and report the best point inside that window.
    """
    os.makedirs(args.outdir, exist_ok=True)

    # Δ_LS(pol=0) — scalar baseline.
    shift0_hz, _ = compute_shift_scatter(wl_nm, pol=0.0)
    abs_scalar_mhz = np.abs(shift0_hz) / 1e6

    # Vector sensitivity ∂Δ/∂pol at pol=0 by central finite difference.
    # The model is linear in pol, so any small step is exact up to roundoff.
    eps = 1e-3
    shift_p, _ = compute_shift_scatter(wl_nm, pol=+eps)
    shift_m, _ = compute_shift_scatter(wl_nm, pol=-eps)
    dshift_dpol_mhz = (shift_p - shift_m) / (2 * eps) / 1e6

    fom = np.abs(dshift_dpol_mhz) / abs_scalar_mhz  # dimensionless

    # Restrict to the user-specified sweet-spot window.
    in_win = (wl_nm >= args.vec_window_min) & (wl_nm <= args.vec_window_max)
    if not np.any(in_win):
        print(f"  [vector-opt] window [{args.vec_window_min}, "
              f"{args.vec_window_max}] nm not in plot range; skipping.")
        return
    fom_win = np.where(in_win, fom, np.inf)
    i_opt = int(np.argmin(fom_win))
    lambda_opt = wl_nm[i_opt]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    ax_a.plot(wl_nm, abs_scalar_mhz, lw=1.6, color="C0",
              label=r"$|\Delta_{\mathrm{LS}}(\lambda,\,P=0)|$")
    ax_a.set_yscale("log")
    ax_a.set_ylabel(r"$|\Delta_{\mathrm{LS}}|$  (MHz)")
    ax_a.set_title("Usable scalar light shift (linear pol, calibration cell)")
    ax_a.set_ylim(1e-2, None)

    ax_b.plot(wl_nm, np.abs(dshift_dpol_mhz), lw=1.6, color="C2",
              label=r"$|\partial\Delta_{\mathrm{LS}}/\partial\mathrm{pol}|$")
    ax_b.set_yscale("log")
    ax_b.set_ylabel(
        r"$|\partial\Delta_{\mathrm{LS}}/\partial\mathrm{pol}|$" "\n(MHz / unit pol)")
    ax_b.set_title("Vector-shift sensitivity (slope at pol = 0)")

    ax_c.plot(wl_nm, fom, lw=1.6, color="C3", label="FOM (all λ)")
    ax_c.plot(wl_nm[in_win], fom[in_win], lw=2.4, color="C3",
              label=rf"window [{args.vec_window_min:.1f}, "
                    rf"{args.vec_window_max:.1f}] nm")
    ax_c.axvspan(args.vec_window_min, args.vec_window_max,
                 color="C3", alpha=0.08)
    ax_c.axvline(lambda_opt, color="k", lw=1.0, ls="-")
    ax_c.scatter([lambda_opt], [fom[i_opt]], color="k", zorder=5,
                 label=rf"$\lambda^\star = {lambda_opt:.3f}$ nm,  "
                       rf"FOM $={fom[i_opt]:.4f}$")
    ax_c.set_yscale("log")
    ax_c.set_ylabel(
        r"FOM $=|\partial\Delta/\partial\mathrm{pol}|/|\Delta|$"
        "\n(unitless; rel. error per unit pol)")
    ax_c.set_xlabel(r"Wavelength  $\lambda$  (nm)")
    ax_c.set_title("Joint scalar+vector figure of merit (smaller = better)")
    ax_c.legend(fontsize=8, loc="upper left", framealpha=0.9)

    for axx in (ax_a, ax_b, ax_c):
        for wl, color in [(LAMBDA_D1, "0.35"), (LAMBDA_D2, "0.35"),
                          (LAMBDA_PAPER, "gold")]:
            if wl_nm[0] <= wl <= wl_nm[-1]:
                axx.axvline(wl, color=color, lw=1.0, ls="--", alpha=0.6)
        axx.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(args.outdir,
                        _prefixed("ac_stark_vector_opt.png", args.prefix))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # Console report: best point in window, plus a few comparison rows.
    print("  Vector-Stark joint-optimization report "
          f"(window {args.vec_window_min:.2f}–{args.vec_window_max:.2f} nm):")
    print(f"    {'λ (nm)':>10}  {'|Δ_scalar| (MHz)':>18}  "
          f"{'|∂Δ/∂pol| (MHz)':>18}  {'FOM':>10}")
    sample_wl = sorted(set(
        [float(args.vec_window_min), float(args.vec_window_max),
         float(lambda_opt), 784.0, 785.0, 786.0]
    ))
    for swl in sample_wl:
        if not (wl_nm[0] <= swl <= wl_nm[-1]):
            continue
        i = int(np.argmin(np.abs(wl_nm - swl)))
        marker = " ←★" if i == i_opt else ""
        print(f"    {wl_nm[i]:>10.3f}  {abs_scalar_mhz[i]:>18.3f}  "
              f"{abs(dshift_dpol_mhz[i]):>18.4f}  {fom[i]:>10.5f}{marker}")
    print(f"    Optimum in window: λ* = {lambda_opt:.3f} nm, "
          f"FOM = {fom[i_opt]:.5f}")
    print(f"    Interpretation: a 1% polarization impurity (P·g_F·m_F = 0.01)")
    print(f"    contaminates the scalar shift by "
          f"≈ {fom[i_opt]*0.01*100:.3f}% at λ*.")


def save_csv(wl_nm, power_uw, shift_mhz, scatter_hz, args):
    """Flatten the grid to a long CSV with one row per (wl, power) cell."""
    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for i, wl in enumerate(wl_nm):
        for j, P in enumerate(power_uw):
            rows.append((wl, P, shift_mhz[i, j], scatter_hz[i, j]))
    arr = np.array(rows)
    path = os.path.join(args.outdir,
                        _prefixed("ac_stark_landscape.csv", args.prefix))
    np.savetxt(path, arr, fmt="%.6e", delimiter=",",
               header=f"wavelength_nm,power_uw,shift_mhz,scatter_hz\n"
                      f"# calibrated at lambda={LAMBDA_PAPER} nm, "
                      f"P={POWER_REF_UW} uW",
               comments="")
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wl-min", type=float, default=780.5,
                   help="Minimum wavelength [nm] (default: just above D2)")
    p.add_argument("--wl-max", type=float, default=794.5,
                   help="Maximum wavelength [nm] (default: just below D1)")
    p.add_argument("--n-wl", type=int, default=400,
                   help="Number of wavelength samples")
    p.add_argument("--power-min", type=float, default=1.0,
                   help="Minimum laser power [μW]")
    p.add_argument("--power-max", type=float, default=500.0,
                   help="Maximum laser power [μW]")
    p.add_argument("--n-power", type=int, default=200,
                   help="Number of power samples")
    p.add_argument("--outdir", type=str, default="results",
                   help="Output directory")
    p.add_argument("--prefix", type=str, default="",
                   help="Filename prefix for outputs")
    p.add_argument("--vec-window-min", type=float, default=784.0,
                   help="Lower edge of sweet-spot window for vector-opt FOM")
    p.add_argument("--vec-window-max", type=float, default=786.0,
                   help="Upper edge of sweet-spot window for vector-opt FOM")
    p.add_argument("--gamma-ryd-hz", type=float, default=1.0 / 151.55e-6,
                   help="Rydberg-state decay rate Γ_r in Hz "
                        "(default: 1/151.55 μs ≈ 6598 Hz, n=70 Rb 'our' system)")
    p.add_argument("--no-csv", action="store_true",
                   help="Skip writing the flattened CSV")
    args = p.parse_args()

    if args.wl_min >= args.wl_max:
        raise ValueError("--wl-min must be less than --wl-max")
    if args.power_min >= args.power_max:
        raise ValueError("--power-min must be less than --power-max")

    # Avoid evaluating exactly on the D1/D2 singularities.
    def _nudge_away(wl):
        for line in (LAMBDA_D1, LAMBDA_D2):
            if abs(wl - line) < 1e-4:
                return wl + 1e-3
        return wl

    wl_nm = np.linspace(args.wl_min, args.wl_max, args.n_wl)
    wl_nm = np.array([_nudge_away(w) for w in wl_nm])
    power_uw = np.linspace(args.power_min, args.power_max, args.n_power)

    print(f"Grid: {args.n_wl} wavelengths × {args.n_power} powers "
          f"= {args.n_wl * args.n_power} cells")
    print(f"  λ:  {wl_nm[0]:.3f} .. {wl_nm[-1]:.3f} nm   "
          f"(D1={LAMBDA_D1:.3f}, D2={LAMBDA_D2:.3f})")
    print(f"  P:  {power_uw[0]:.1f} .. {power_uw[-1]:.1f} μW   "
          f"(calib {POWER_REF_UW:.0f} μW)")
    print()

    shift_mhz, scatter_hz = build_grid(wl_nm, power_uw)

    print(f"  Δ_LS range:    [{shift_mhz.min():.3e}, {shift_mhz.max():.3e}] MHz")
    print(f"  Γ_sc range:    [{scatter_hz.min():.3e}, {scatter_hz.max():.3e}] Hz")
    # Report the values at the calibration cell for sanity.
    if (wl_nm[0] <= LAMBDA_PAPER <= wl_nm[-1]
            and power_uw[0] <= POWER_REF_UW <= power_uw[-1]):
        iwl = int(np.argmin(np.abs(wl_nm - LAMBDA_PAPER)))
        ip = int(np.argmin(np.abs(power_uw - POWER_REF_UW)))
        print(f"  @ calib cell   ({wl_nm[iwl]:.2f} nm, {power_uw[ip]:.1f} μW): "
              f"Δ_LS={shift_mhz[iwl, ip]:.4f} MHz, "
              f"Γ_sc={scatter_hz[iwl, ip]:.2f} Hz")
    print()

    plot_landscape(wl_nm, power_uw, shift_mhz, scatter_hz, args)
    plot_profiles(wl_nm, power_uw, shift_mhz, scatter_hz, args)
    plot_scatter_per_shift(wl_nm, args)
    plot_polarization_sensitivity(wl_nm, args)
    plot_vector_optimization(wl_nm, args)
    if not args.no_csv:
        save_csv(wl_nm, power_uw, shift_mhz, scatter_hz, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
