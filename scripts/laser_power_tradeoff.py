#!/usr/bin/env python3
"""Plot the ECDL/seed fidelity trade-off when seed power is half of ECDL."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(
    os.environ.get("RYD_GATE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, os.fspath(ROOT / "scripts"))

import max_leakage_297_sweep as mls  # noqa: E402
from sweeplib.plotting import plot_metric_values  # noqa: E402


STORE_ROOT = ROOT / "results" / "max_leakage_297" / "a3.0"
NOISE_ROOT = ROOT / "results" / "297_laser_noise"
ECDL_POWER_W = np.array([2.0, 4.0, 6.0, 8.0])
SEED_POWER_RATIO = 0.5
LOSS_FRACTION = 0.8
F_MIN_HZ = (1.0, 10.0)


def nominal_power_w(key, manifest: dict, omega_at_1w: dict[int, float]) -> float:
    """Nominal input power for a scan point, including the 80% optical loss."""
    ryd_n = int(manifest["axes"]["ryd_n"][key.n_idx])
    omega_mhz = float(key.omega_mhz())
    return (omega_mhz / omega_at_1w[ryd_n]) ** 2 / (1.0 - LOSS_FRACTION)


def best_under_power(total: dict, phase: dict, manifest: dict,
                     omega_at_1w: dict[int, float], power_cap_w: float):
    hw_limit = float(manifest["axes"]["dsweep_hw_limit_mhz"])
    candidates = [
        key for key in total
        if float(key.dsweep_mhz()) <= hw_limit + 1e-9
        and float(np.max(phase[key])) <= mls.EPS_PHASE_REGIME_MAX
        and nominal_power_w(key, manifest, omega_at_1w) <= power_cap_w + 1e-9
    ]
    if not candidates:
        raise RuntimeError(f"no valid point below {power_cap_w:g} W")
    key = min(candidates, key=lambda candidate: total[candidate])
    return 100.0 * (1.0 - float(total[key])), key


def calculate() -> dict[float, dict[str, list[tuple[float, object]]]]:
    store = mls.Store(os.fspath(STORE_ROOT))
    manifest = store.load_manifest()
    if manifest is None:
        raise RuntimeError(f"missing manifest under {STORE_ROOT}")
    records = store.load_records(manifest, include_states=False)

    cache = np.load(NOISE_ROOT / "omega_per_watt.npz")
    omega_at_1w = {
        int(n): float(omega)
        for n, omega in zip(cache["ryd_n"], cache["omega_mhz_at_1w"], strict=True)
    }

    results: dict[float, dict[str, list[tuple[float, object]]]] = {}
    for f_min in F_MIN_HZ:
        results[f_min] = {}
        for laser in ("ECDL", "seed"):
            phase = mls.phase_noise_values(
                store, manifest, laser, "power", f_min
            )
            total, *_ = plot_metric_values(
                store,
                manifest,
                records,
                "total_error_phase",
                scatter_channels=mls.SCATTER_CHANNELS,
                extra_values=phase,
            )
            ratio = 1.0 if laser == "ECDL" else SEED_POWER_RATIO
            results[f_min][laser] = [
                best_under_power(total, phase, manifest, omega_at_1w, ratio * power)
                for power in ECDL_POWER_W
            ]
    return results


def render(results: dict[float, dict[str, list[tuple[float, object]]]]) -> None:
    colors = {"ECDL": "#c75b32", "seed": "#087f76"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    fig.patch.set_facecolor("#f3efe4")

    for ax, f_min in zip(axes, F_MIN_HZ, strict=True):
        ax.set_facecolor("#fffdf7")
        for laser in ("ECDL", "seed"):
            fidelity = [entry[0] for entry in results[f_min][laser]]
            ax.plot(
                ECDL_POWER_W,
                fidelity,
                color=colors[laser],
                marker="o" if laser == "ECDL" else "s",
                linewidth=2.4,
                markersize=6.5,
                label=laser,
            )

        crossover = (6.0, 8.0) if f_min == 1.0 else (4.0, 6.0)
        ax.axvspan(*crossover, color="#d8c9a7", alpha=0.35, linewidth=0)
        ax.text(
            sum(crossover) / 2,
            99.365,
            "crossover\nbracket",
            ha="center",
            va="bottom",
            color="#655b49",
            fontsize=8.5,
        )
        ax.set_title(rf"$f_{{\min}}={f_min:g}$ Hz", fontsize=13, weight="bold")
        ax.set_xlabel("ECDL power cap (W)\nseed cap = 1/2 ECDL cap")
        ax.set_xticks(ECDL_POWER_W)
        ax.set_ylim(99.35, 99.78)
        ax.grid(axis="y", color="#d8d1c3", linewidth=0.8, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Best power-constrained fidelity (%)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "297 nm laser choice with seed intensity limited to one half",
        fontsize=15,
        weight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.02,
        r"Each point is re-optimized with $D_{\rm sw}\leq20$ MHz and "
        r"$\max_s\epsilon_{\rm phase}^s\leq0.1$.",
        ha="center",
        color="#514b40",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(NOISE_ROOT / "power_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_tables(results: dict[float, dict[str, list[tuple[float, object]]]]) -> None:
    for f_min in F_MIN_HZ:
        print(f"f_min = {f_min:g} Hz")
        print("ECDL_W seed_W ECDL_fidelity seed_fidelity winner")
        for idx, ecdl_power in enumerate(ECDL_POWER_W):
            ecdl = results[f_min]["ECDL"][idx][0]
            seed = results[f_min]["seed"][idx][0]
            winner = "seed" if seed > ecdl else "ECDL"
            print(
                f"{ecdl_power:6.1f} {SEED_POWER_RATIO * ecdl_power:6.1f} "
                f"{ecdl:13.4f}% {seed:13.4f}% {winner}"
            )
        print()


def main() -> None:
    results = calculate()
    print_tables(results)
    render(results)


if __name__ == "__main__":
    main()
