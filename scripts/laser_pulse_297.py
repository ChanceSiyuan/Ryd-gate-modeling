#!/usr/bin/env python3
"""Render the intensity, Rabi envelope, phase, and chirp of the best 297 nm pulse."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(
    os.environ.get("RYD_GATE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
RESULT_ROOT = ROOT / "results" / "297_laser_noise"

RYD_N = 73
T_GATE_US = 1.0
RAMP_FRACTION = 0.15
OMEGA_PEAK_MHZ = 14.25
DSWEEP_MHZ = 17.5
BEAM_AREA_UM2 = 420.0
LOSS_FRACTION = 0.8


def quintic(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def power_envelope(s: np.ndarray) -> np.ndarray:
    return np.where(
        s < RAMP_FRACTION,
        quintic(s / RAMP_FRACTION),
        np.where(
            s > 1.0 - RAMP_FRACTION,
            quintic((1.0 - s) / RAMP_FRACTION),
            1.0,
        ),
    )


def pulse_values() -> dict[str, np.ndarray | float]:
    cache = np.load(RESULT_ROOT / "omega_per_watt.npz")
    omega_at_1w = {
        int(n): float(omega)
        for n, omega in zip(cache["ryd_n"], cache["omega_mhz_at_1w"], strict=True)
    }
    nominal_peak_w = (
        (OMEGA_PEAK_MHZ / omega_at_1w[RYD_N]) ** 2 / (1.0 - LOSS_FRACTION)
    )
    atom_peak_w = (1.0 - LOSS_FRACTION) * nominal_peak_w
    intensity_peak_w_m2 = atom_peak_w / (BEAM_AREA_UM2 * 1e-12)

    s = np.linspace(0.0, 1.0, 1201)
    t_us = s * T_GATE_US
    envelope = power_envelope(s)
    phase_rad = -DSWEEP_MHZ * T_GATE_US * np.sin(2.0 * np.pi * s)
    detuning_mhz = -DSWEEP_MHZ * np.cos(2.0 * np.pi * s)
    return {
        "t_us": t_us,
        "envelope": envelope,
        "rabi_mhz": OMEGA_PEAK_MHZ * np.sqrt(envelope),
        "intensity_w_m2": intensity_peak_w_m2 * envelope,
        "phase_rad": phase_rad,
        "detuning_mhz": detuning_mhz,
        "nominal_peak_w": nominal_peak_w,
        "atom_peak_w": atom_peak_w,
        "intensity_peak_w_m2": intensity_peak_w_m2,
        "omega_at_1w_mhz": omega_at_1w[RYD_N],
    }


def render(values: dict[str, np.ndarray | float]) -> None:
    colors = {
        "intensity": "#c75b32",
        "rabi": "#087f76",
        "phase": "#315b7d",
        "detuning": "#b07824",
    }
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 7.0), sharex=True)
    fig.patch.set_facecolor("#f3efe4")
    t_us = np.asarray(values["t_us"])

    ax = axes[0]
    ax.set_facecolor("#fffdf7")
    intensity_kw_cm2 = np.asarray(values["intensity_w_m2"]) / 1e7
    line_i, = ax.plot(t_us, intensity_kw_cm2, color=colors["intensity"],
                      linewidth=2.5, label="intensity at atoms")
    ax.set_ylabel(r"Intensity (kW/cm$^2$)", color=colors["intensity"])
    ax.tick_params(axis="y", colors=colors["intensity"])
    twin = ax.twinx()
    line_o, = twin.plot(t_us, values["rabi_mhz"], color=colors["rabi"],
                        linewidth=2.2, linestyle="--", label=r"$\Omega(t)/2\pi$")
    twin.set_ylabel(r"Target Rabi frequency $\Omega/2\pi$ (MHz)",
                    color=colors["rabi"])
    twin.tick_params(axis="y", colors=colors["rabi"])
    ax.legend([line_i, line_o], [line_i.get_label(), line_o.get_label()],
              frameon=False, loc="center")
    ax.set_title("Power envelope and target-transition Rabi amplitude", weight="bold")

    ax = axes[1]
    ax.set_facecolor("#fffdf7")
    line_p, = ax.plot(t_us, values["phase_rad"], color=colors["phase"],
                      linewidth=2.5, label=r"control phase $\varphi_c$")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Unwrapped control phase (rad)", color=colors["phase"])
    ax.tick_params(axis="y", colors=colors["phase"])
    twin = ax.twinx()
    line_d, = twin.plot(t_us, values["detuning_mhz"], color=colors["detuning"],
                        linewidth=2.2, linestyle="--",
                        label=r"$\dot\varphi_c/2\pi$")
    twin.set_ylabel("Instantaneous frequency offset (MHz)",
                    color=colors["detuning"])
    twin.tick_params(axis="y", colors=colors["detuning"])
    ax.legend([line_p, line_d], [line_p.get_label(), line_d.get_label()],
              frameon=False, loc="upper right")
    ax.set_title("Optical control phase and its instantaneous chirp", weight="bold")

    for ax in axes:
        ax.axvline(RAMP_FRACTION * T_GATE_US, color="#9b9383", linewidth=1.0,
                   linestyle=":")
        ax.axvline((1.0 - RAMP_FRACTION) * T_GATE_US, color="#9b9383",
                   linewidth=1.0, linestyle=":")
        ax.grid(axis="x", color="#d8d1c3", linewidth=0.8, alpha=0.8)
        ax.spines[["top"]].set_visible(False)

    fig.suptitle(
        r"Best evaluated 297 nm pulse: $n=73$, $T=1\,\mu$s, "
        r"$\Omega_{\rm pk}/2\pi=14.25$ MHz, $D_{\rm sw}/2\pi=17.5$ MHz",
        fontsize=13.5,
        weight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "optimal_pulse_time_dependence.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    values = pulse_values()
    print(f"omega_at_1W_n73_MHz={values['omega_at_1w_mhz']:.9g}")
    print(f"nominal_peak_power_W={values['nominal_peak_w']:.9g}")
    print(f"atom_peak_power_W={values['atom_peak_w']:.9g}")
    print(f"peak_intensity_W_m2={values['intensity_peak_w_m2']:.9g}")
    render(values)


if __name__ == "__main__":
    main()
