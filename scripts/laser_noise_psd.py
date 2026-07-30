#!/usr/bin/env python
"""Digitize and model the two 297 nm seed-laser frequency-noise spectra.

``results/297_laser_noise/`` holds two measured frequency-noise amplitude spectral
density (ASD) plots, ``sqrt(S_dnu)`` in Hz/sqrt(Hz) over 1 Hz - 1 MHz, taken on the
**1180/1187 nm fundamental**:

  * ``ECDL_phasenoise.png``   the ECDL (higher phase noise, full power)
  * ``seed_lasernoise.png``   the low-phase-noise seed (roughly half the power)

297 nm is the fourth harmonic of the fundamental, so the optical phase is multiplied
by ``HARMONIC = 4`` and the frequency-noise density by ``HARMONIC**2 = 16``.

This module produces, from the PNGs alone:

  * ``psd_<laser>.csv``   digitized (f, ASD_mean, ASD_lo, ASD_hi) at the fundamental,
    where lo/hi are the envelope of the individual (blue) traces around the (red) mean
  * ``psd_model.png/pdf`` the model figure: measurement, the 297 nm conversion, the
    two extrapolations above the 1 MHz measurement edge, and the cumulative sigma_nu

Conventions
-----------
Stored densities are **one-sided** (the instrument convention): ``sigma_nu**2 =
int_0^inf S_dnu(f) df``.  PRA 107, 042611 uses two-sided densities, a factor 2
smaller; the conversion happens where that formalism is used, not here.

Above the 1 MHz measurement edge nothing is measured, but ``f ~ Omega/2pi =
9-18 MHz`` is exactly where the gate is most sensitive.  Two extrapolations bracket
it, both assuming **no servo bump**:

  * ``flat``  hold S_dnu at its 1 MHz value (conservative white-noise floor)
  * ``power`` continue the power law fitted to the last measured decade (optimistic)

Usage
-----
    python scripts/laser_noise_psd.py            # digitize + render the model figure
"""
from __future__ import annotations

import os

import numpy as np

NOISE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "297_laser_noise")

HARMONIC = 4                     # 1188 nm -> 297 nm; S_dnu scales as HARMONIC**2
OMEGA_BAND_MHZ = (9.0, 18.0)     # the sweep's Omega_297/2pi axis
FIT_DECADE_HZ = (1e5, 1e6)       # last measured decade, used for the power-law fit

# Axis calibration of each PNG, read off the plot frame:
# (x_left_px, x_right_px, log10 f_left, log10 f_right,
#  y_top_px, y_bot_px, log10 ASD_top, log10 ASD_bot)
AXES = {
    "ECDL": ("ECDL_phasenoise", 268.5, 1854.5, 0.0, 6.0, 256.5, 1300.5, 5.0, 0.0),
    "seed": ("seed_lasernoise", 85.5, 1246.0, 0.0, 6.0, 60.5, 823.5, 6.0, 0.0),
}

LABEL = {"ECDL": "ECDL (full power)", "seed": "seed (low noise, ~half power)"}
COLOR = {"ECDL": "#D55E00", "seed": "#0072B2"}


# ── digitization ─────────────────────────────────────────────────────────────


def digitize(laser: str) -> np.ndarray:
    """(n, 4) array of (f_Hz, ASD_mean, ASD_lo, ASD_hi) at the fundamental.

    The red trace is the mean of the sweeps; the blue traces are the individual
    sweeps, kept as a lo/hi envelope.  One sample per pixel column.
    """
    from PIL import Image

    name, x0, x1, lf0, lf1, y0, y1, la0, la1 = AXES[laser]
    im = np.asarray(Image.open(os.path.join(NOISE_DIR, name + ".png")).convert("RGB"))
    im = im.astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    red = (r > 130) & (g < 110) & (b < 110)
    blue = (b > 110) & (r < 110) & (g < 110)

    top = int(np.ceil(y0))
    rows = slice(top, int(np.floor(y1)) + 1)

    def asd(px):
        return 10.0 ** (la0 + (px - y0) / (y1 - y0) * (la1 - la0))

    out = []
    for c in range(int(np.ceil(x0)), int(np.floor(x1)) + 1):
        ri = np.nonzero(red[rows, c])[0]
        if ri.size == 0:
            continue
        bi = np.nonzero(blue[rows, c])[0]
        f = 10.0 ** (lf0 + (c - x0) / (x1 - x0) * (lf1 - lf0))
        lo, hi = (asd(bi.max() + top), asd(bi.min() + top)) if bi.size else (np.nan, np.nan)
        out.append((f, asd(np.median(ri) + top), lo, hi))
    return np.asarray(out)


def write_csv(laser: str, data: np.ndarray) -> str:
    path = os.path.join(NOISE_DIR, f"psd_{laser}.csv")
    header = ("f_Hz,asd_mean_Hz_per_rtHz,asd_lo_Hz_per_rtHz,asd_hi_Hz_per_rtHz\n"
              "# digitized from " + AXES[laser][0] + ".png; fundamental (1180/1187 nm);\n"
              "# one-sided frequency-noise ASD sqrt(S_dnu)\n")
    with open(path, "w") as fh:
        fh.write(header)
        for f, m, lo, hi in data:
            fh.write(f"{f:.6g},{m:.6g},{lo:.6g},{hi:.6g}\n")
    return path


# ── PSD model ────────────────────────────────────────────────────────────────


def power_law_exponent(data: np.ndarray) -> float:
    """Least-squares slope p of ASD ~ f**(-p) over the last measured decade."""
    f, a = data[:, 0], data[:, 1]
    m = (f >= FIT_DECADE_HZ[0]) & (f <= FIT_DECADE_HZ[1])
    return -float(np.polyfit(np.log10(f[m]), np.log10(a[m]), 1)[0])


def s_dnu_297(laser: str, data: np.ndarray, f: np.ndarray, mode: str) -> np.ndarray:
    """One-sided S_dnu (Hz^2/Hz) at 297 nm, log-log interpolated and extrapolated.

    ``mode`` is ``"flat"`` (hold the 1 MHz value) or ``"power"`` (continue the
    fitted power law) above the measurement edge.  Below the 1 Hz edge the lowest
    measured slope is continued; those frequencies are frozen over a microsecond
    gate and only enter the quasi-static offset.
    """
    fm, am = data[:, 0], data[:, 1]
    f = np.asarray(f, dtype=float)
    asd = 10.0 ** np.interp(np.log10(f), np.log10(fm), np.log10(am))

    hi = f > fm[-1]
    if mode == "flat":
        asd[hi] = am[-1]
    elif mode == "power":
        asd[hi] = am[-1] * (f[hi] / fm[-1]) ** (-power_law_exponent(data))
    else:
        raise ValueError(f"mode must be 'flat' or 'power'; got {mode!r}")
    return HARMONIC ** 2 * asd ** 2


def sigma_nu_cumulative(laser: str, data: np.ndarray, f: np.ndarray) -> np.ndarray:
    """sqrt of the running integral of S_dnu at 297 nm, from f[0] up to each f."""
    s = s_dnu_297(laser, data, f, "flat")
    return np.sqrt(np.concatenate([[0.0], np.cumsum(np.diff(f) * 0.5 * (s[1:] + s[:-1]))]))


# ── figure ───────────────────────────────────────────────────────────────────


def render(datasets: dict[str, np.ndarray]) -> tuple[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    f_meas = np.logspace(0, 6, 900)
    f_ext = np.logspace(0, np.log10(5e7), 1400)

    ax = axes[0]
    for laser, d in datasets.items():
        ax.fill_between(d[:, 0], d[:, 2], d[:, 3], color=COLOR[laser], alpha=0.18, lw=0)
        ax.plot(d[:, 0], d[:, 1], color=COLOR[laser], lw=2.0, label=LABEL[laser])
    ax.set_title("digitized measurement (fundamental 1180/1187 nm)", fontsize=10)
    ax.set_ylabel(r"$\sqrt{S_{\delta\nu}}$  (Hz/$\sqrt{\mathrm{Hz}}$)")
    ax.set_ylim(1, 3e5)
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    ax = axes[1]
    ax.axvspan(OMEGA_BAND_MHZ[0] * 1e6, OMEGA_BAND_MHZ[1] * 1e6,
               color="0.85", zorder=0)
    ax.text(np.sqrt(np.prod(OMEGA_BAND_MHZ)) * 1e6, 3e10,
            r"$\Omega_{297}/2\pi$" + "\nswept", ha="center", va="top",
            fontsize=8, color="0.35")
    for laser, d in datasets.items():
        ax.plot(f_meas, s_dnu_297(laser, d, f_meas, "flat"),
                color=COLOR[laser], lw=2.0, label=LABEL[laser])
        ext = f_ext[f_ext > d[-1, 0]]
        ax.plot(ext, s_dnu_297(laser, d, ext, "flat"),
                color=COLOR[laser], lw=1.6, ls="--")
        ax.plot(ext, s_dnu_297(laser, d, ext, "power"),
                color=COLOR[laser], lw=1.6, ls=":")
    ax.axvline(1e6, color="0.55", lw=1.0)
    ax.text(8e5, 3e0, "measurement edge", fontsize=8, color="0.4",
            ha="right", va="bottom")
    ax.set_title(r"converted to 297 nm ($\times$16), no servo bump assumed", fontsize=10)
    ax.set_ylabel(r"$S_{\delta\nu}$  (Hz$^2$/Hz, one-sided)")
    ax.set_ylim(1e0, 1e12)
    ax.plot([], [], color="0.4", lw=1.6, ls="--", label="flat extrapolation")
    ax.plot([], [], color="0.4", lw=1.6, ls=":", label="power-law extrapolation")
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    ax = axes[2]
    for laser, d in datasets.items():
        ax.plot(f_meas, sigma_nu_cumulative(laser, d, f_meas) / 1e3,
                color=COLOR[laser], lw=2.0, label=LABEL[laser])
    ax.set_title(r"cumulative $\sigma_\nu(<f)$ at 297 nm", fontsize=10)
    ax.set_ylabel(r"$\sigma_\nu$  (kHz)")
    ax.set_yscale("linear")
    ax.legend(fontsize=8, loc="lower right", frameon=False)

    for ax in axes:
        ax.set_xscale("log")
        if ax is not axes[2]:
            ax.set_yscale("log")
        ax.set_xlabel("frequency  (Hz)")
        ax.grid(True, which="major", color="0.9", lw=0.6)
        ax.grid(True, which="minor", color="0.95", lw=0.4)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle("297 nm drive laser frequency noise: measurement, 4th-harmonic "
                 "conversion, and the unmeasured band that sets the gate error",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = os.path.join(NOISE_DIR, "psd_model.png")
    pdf = os.path.join(NOISE_DIR, "psd_model.pdf")
    fig.savefig(png, dpi=160)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    datasets = {}
    for laser in AXES:
        d = digitize(laser)
        datasets[laser] = d
        print(f"[{laser}] {len(d)} samples, f = {d[0, 0]:.3g}..{d[-1, 0]:.4g} Hz -> "
              f"{write_csv(laser, d)}")

    import json
    model = {
        laser: {
            "csv": f"psd_{laser}.csv",
            "harmonic": HARMONIC,
            "power_law_exponent": power_law_exponent(d),
            "s_dnu_edge_297": HARMONIC ** 2 * float(d[-1, 1]) ** 2,
            "f_edge_hz": float(d[-1, 0]),
        }
        for laser, d in datasets.items()
    }
    path = os.path.join(NOISE_DIR, "psd_model.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(model, fh, indent=2, sort_keys=True)
    os.replace(path + ".tmp", path)
    print(f"wrote {path}")

    print(f"\n{'laser':6s} {'p (ASD~f^-p)':>13s} {'S_dnu(1MHz,297)':>17s} "
          f"{'S_dnu(13.5MHz) flat':>20s} {'power':>10s} {'sigma_nu(<1MHz)':>17s}")
    f13 = np.asarray([13.5e6])
    for laser, d in datasets.items():
        p = power_law_exponent(d)
        s1 = HARMONIC ** 2 * d[-1, 1] ** 2
        sf = s_dnu_297(laser, d, f13, "flat")[0]
        sp = s_dnu_297(laser, d, f13, "power")[0]
        sig = sigma_nu_cumulative(laser, d, np.logspace(0, 6, 4000))[-1]
        print(f"{laser:6s} {p:13.3f} {s1:17.1f} {sf:20.1f} {sp:10.1f} "
              f"{sig / 1e3:14.1f} kHz")

    png, pdf = render(datasets)
    print(f"\nwrote {png}\n      {pdf}")


if __name__ == "__main__":
    main()
