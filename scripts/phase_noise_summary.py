#!/usr/bin/env python3
"""Recompute the phase-noise headline table from the stored 297 nm sweep.

Reuses `max_leakage_297_sweep.phase_noise_values` and
`sweeplib.plotting.plot_metric_values` rather than reimplementing the budget, so
this table cannot disagree with the rendered maps: the same per-logical-input sum
(coherent leakage + scattering + eps_phase) is formed and only then maximised.

Read-only.  Prints a Markdown table; writes nothing.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import max_leakage_297_sweep as mls  # noqa: E402
from sweeplib.plotting import plot_metric_values  # noqa: E402


def _implementable(keys, hw_limit_mhz: float) -> set:
    """Keys whose sweep amplitude is within the detuning hardware limit."""
    return {k for k in keys if float(k.dsweep_mhz()) <= hw_limit_mhz + 1e-9}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=os.path.join("results", "max_leakage_297", "a3.0"))
    ap.add_argument("--lasers", default="ECDL,seed")
    ap.add_argument("--extrapolation", default="power")
    ap.add_argument("--f-min", type=float, default=mls.KERNEL_F_MIN_HZ)
    args = ap.parse_args()

    store = mls.Store(args.output)
    manifest = store.load_manifest()
    if manifest is None:
        raise SystemExit(f"no manifest under {store.root}")
    records = store.load_records(manifest, include_states=False)
    hw = float(manifest["axes"]["dsweep_hw_limit_mhz"])
    # A key carries axis *indices*; the physical n and T live on the scan config.
    ryd_n = tuple(manifest["axes"]["ryd_n"])
    t_gate_us = tuple(manifest["axes"]["t_gate_us"])

    base, *_ = plot_metric_values(store, manifest, records, "total_error",
                                  scatter_channels=mls.SCATTER_CHANNELS)

    print(f"store            : {args.output}")
    print(f"extrapolation    : {args.extrapolation}   f_min = {args.f_min:g} Hz")
    print(f"D_sw hw limit    : {hw:g} MHz")
    print(f"points with a noise-free budget: {len(base)}")
    print()

    rows = []
    for laser in args.lasers.split(","):
        extra = mls.phase_noise_values(store, manifest, laser,
                                       args.extrapolation, args.f_min)
        tot, *_ = plot_metric_values(store, manifest, records,
                                     "total_error_phase",
                                     scatter_channels=mls.SCATTER_CHANNELS,
                                     extra_values=extra)
        eps = {k: float(np.max(v)) for k, v in extra.items()}
        ok = _implementable(set(tot) & set(base), hw)
        if not ok:
            raise SystemExit(f"no implementable points for {laser}")
        n_bad = sum(1 for k in ok if eps[k] > mls.EPS_PHASE_REGIME_MAX)
        inside = {k for k in ok if eps[k] <= mls.EPS_PHASE_REGIME_MAX}
        best = min(inside, key=lambda k: tot[k])
        rows.append((laser, best, base[best], tot[best], eps[best],
                     len(ok), n_bad))

    print(f"| 激光器 | 试点最优门点 $(n,T,\\Omega/2\\pi,D_{{\\rm sw}})$ | "
          f"$F_0$ | $F_{{\\rm noise}}$ | $\\Delta F$ |")
    print("|---|---|---:|---:|---:|")
    for laser, k, b, t, _e, _n, _nb in rows:
        pt = (f"$({ryd_n[k.n_idx]},{t_gate_us[k.t_idx]:.1f}\\,\\mu\\mathrm{{s}},"
              f"{float(k.omega_mhz()):g}\\,\\mathrm{{MHz}},"
              f"{float(k.dsweep_mhz()):g}\\,\\mathrm{{MHz}})$")
        print(f"| {laser} | {pt} | {100*(1-b):.4f}% | {100*(1-t):.4f}% | "
              f"{100*(t-b):.4f} 百分点 |")

    print()
    for laser, k, b, t, e, n, nb in rows:
        print(f"{laser:>5}: best {k.id()}  implementable {n}  "
              f"out-of-regime {nb} ({100*nb/n:.1f}%)  "
              f"eps_phase@best {e:.3e}  noise-free budget {b:.3e}")


if __name__ == "__main__":
    main()
