#!/usr/bin/env python3
"""Plot AC Stark shift and scattering rate as a function of (power, wavelength).

Uses the calibrated ``compute_shift_scatter`` from ``ryd_gate.core.ac_stark``,
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

from ryd_gate.core.ac_stark import (
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
    if not args.no_csv:
        save_csv(wl_nm, power_uw, shift_mhz, scatter_hz, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
