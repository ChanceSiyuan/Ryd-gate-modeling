"""Shared 8x9 map-family rendering for the two-atom CZ max-leakage sweeps.

Rasters are *visualization only*: log-linear (log10) interpolation of the scattered
exact nodes onto a fine mesh, an axis-holdout leave-one-out (LOO) credibility veil,
the audit-derived credibility floor, and the 8x9 panel-grid renderer with family
edge-labeling (top row carries the gate-time titles; the left column carries the
row-variable ylabel).  Each script supplies its scatter-channel table, the manifest
axis key + labeller for the row variable, the x-axis label and the system
description via ``PlotSpec``; everything else is byte-identical between the scripts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .store import Store, audit_pairs, best_records

PLOT_LOO_MASK_DEX = 0.2
PLOT_RASTER_N = 81


@dataclass(frozen=True)
class PlotSpec:
    """Per-script plot parameterization for :func:`render_panel_grid`.

    ``scatter_channels`` is the ordered channel table summed into the scattering
    budget (ODE: ``p_mid``/``p_ryd``/``p_r_garb``; 297: ``p_ryd``/``p_r_garb``);
    ``row_axis_key`` is the ``manifest["axes"]`` key holding the panel-row values
    and ``row_label`` maps one row value to its ylabel head (e.g. ``$\\Delta_e/2\\pi$
    = 20 GHz`` or ``$n$ = 70``); ``xlabel`` and ``system_desc`` are the fixed
    x-axis label and the suptitle system string.  ``hw_limit_mhz`` is the
    horizontal reference line drawn in every panel.
    """

    scatter_channels: tuple[str, ...]
    row_axis_key: str
    row_label: Callable[[float], str]
    xlabel: str
    system_desc: str
    hw_limit_mhz: float = 20.0


def credibility_floor(records, floor_min: float = 1e-12) -> tuple[float, dict]:
    """vmin = max(1e-12, 10 * P95(|L_prod - L_audit|)); documented fallback."""
    pairs = audit_pairs(records)
    if len(pairs) >= 8:
        diffs = np.abs([p - a for _, p, a in pairs])
        vmin = float(max(floor_min, 10.0 * np.percentile(diffs, 95)))
        info = {"rule": "10*P95(|L_prod - L_audit|)", "n_pairs": len(pairs),
                "p95_abs_diff": float(np.percentile(diffs, 95)), "vmin": vmin,
                "fallback": False}
    else:
        vmin = max(floor_min, 1e-11)
        info = {"rule": "fallback 1e-11 (fewer than 8 audit pairs)",
                "n_pairs": len(pairs), "vmin": vmin, "fallback": True}
    return vmin, info


def _panel_plot_data(values: dict, panel: tuple[int, int], vmin: float):
    """(x_mhz, y_mhz, z_log10) arrays of one panel's exact nodes, or None."""
    pts = [(float(k.omega_mhz()), float(k.dsweep_mhz()), v)
           for k, v in values.items() if k.panel == panel]
    if not pts:
        return None
    pts.sort(key=lambda t: (t[0], t[1]))
    x = np.asarray([p[0] for p in pts])
    y = np.asarray([p[1] for p in pts])
    z = np.log10(np.maximum([p[2] for p in pts], vmin / 10.0))
    return x, y, z


def plot_metric_values(store: Store, manifest: dict, records, metric: str,
                       *, scatter_channels: tuple[str, ...]):
    """(values, vmin, vmax, colorbar_label) for a plot metric.

    ``max_leakage`` reads the coherent-leakage records (audit-derived floor);
    the ``p_*`` metrics read the supplemental scatter series. ``total_error``
    adds coherent leakage and every scattering contribution per logical input
    before selecting the worst input.  The scattering budget sums exactly the
    per-script ``scatter_channels``.
    """
    if metric == "max_leakage":
        best = best_records(records)
        values = {k: r.max_leakage for k, r in best.items()}
        vmin, floor_info = credibility_floor(records)
        label = ("terminal max leakage  "
                 f"(floor {vmin:.1e}: "
                 f"{'audit-derived' if not floor_info['fallback'] else 'fallback'};"
                 " values at floor are below the numerical credibility floor)")
        return values, vmin, 1.0, label
    coherent = best_records(records) if metric == "total_error" else {}
    rows = [r for r in store.load_scatter_records(manifest) if r["status"] == "ok"]
    if not rows:
        raise SystemExit(f"no scatter records for --metric {metric}; "
                         "run the `scatter` subcommand first")
    per_key: dict = {}
    for r in rows:
        scattering = sum(r[ch] for ch in scatter_channels)
        if metric == "total_error":
            if r["key"] not in coherent:
                continue
            v = float(np.max(coherent[r["key"]].leakage + scattering))
        elif metric == "p_loss_total":
            v = float(np.max(scattering))
        else:
            v = float(np.max(r[metric]))
        cur = per_key.get(r["key"])
        if cur is None or r["rtol"] < cur[1]:
            per_key[r["key"]] = (v, r["rtol"])
    values = {k: v for k, (v, _) in per_key.items()}
    if not values:
        raise SystemExit(f"no overlapping coherent and scatter records for --metric {metric}")
    pos = [v for v in values.values() if v > 0]
    vmin = max(1e-12, min(pos)) if pos else 1e-12
    vmax = max(max(values.values()), vmin * 10)
    if metric == "total_error":
        label = ("worst-input total error budget (terminal coherent leakage + "
                 "first-order scattering)")
    else:
        label = (f"worst-input {metric} (scattering-rate integral, "
                 "trapezoid over 301 samples)")
    return values, vmin, vmax, label


def holdout_residuals(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Axis-neighbor leave-one-out residual (dex) for EVERY node.

    Each node is re-estimated by linear interpolation between its nearest present
    neighbors along each grid axis (holding it out); the residual is the worse of
    the two axes.  O(n log n), so no node is ever skipped — a node lacking both
    neighbors on both axes (panel edges) gets 0.
    """
    n = x.size
    resid = np.zeros(n)

    def _along(primary: np.ndarray, secondary: np.ndarray) -> None:
        for line in np.unique(secondary):
            idx = np.where(secondary == line)[0]
            if idx.size < 3:
                continue
            order = idx[np.argsort(primary[idx])]
            p, v = primary[order], z[order]
            est = v[:-2] + (v[2:] - v[:-2]) * (p[1:-1] - p[:-2]) / (p[2:] - p[:-2])
            np.maximum.at(resid, order[1:-1], np.abs(v[1:-1] - est))

    _along(x, y)
    _along(y, x)
    return resid


def _draw_panel(ax, x, y, z, vmin, vmax, cmap, hw_limit_mhz, veil: bool = True):
    """One panel: interpolated raster + uncertainty veil + nodes + hardware line.

    Regions whose nearest node has an axis-holdout LOO residual above
    ``PLOT_LOO_MASK_DEX`` are masked with a translucent white veil (the spec's
    "hatch or mask" rule) — a wash reads cleanly at any node density, where
    per-node hatched markers drown the map once grids reach 13x13/25x25.
    ``veil=False`` omits the overlay (raster is then pure interpolation —
    remember it is visualization only).
    """
    from matplotlib.colors import ListedColormap, LogNorm
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    xg = np.linspace(x.min(), x.max(), PLOT_RASTER_N)
    yg = np.linspace(y.min(), y.max(), PLOT_RASTER_N)
    XX, YY = np.meshgrid(xg, yg)
    resid = holdout_residuals(x, y, z)
    bad = (resid > PLOT_LOO_MASK_DEX) if veil else np.zeros(x.size, dtype=bool)
    if x.size >= 4 and np.unique(x).size > 1 and np.unique(y).size > 1:
        interp = LinearNDInterpolator(np.column_stack([x, y]), z)
        ZZ = interp(XX, YY)                     # NaN outside the convex hull
        # rasterized: in vector (PDF) output each mesh quad is otherwise a
        # separate filled path, and viewers antialias the quad boundaries into
        # hairline white seams; rasterizing embeds the color field as one image
        # (axes/markers/text stay vector) and shrinks the file dramatically.
        mesh = ax.pcolormesh(XX, YY, np.ma.masked_invalid(10.0 ** ZZ),
                             cmap=cmap, norm=norm, shading="nearest",
                             rasterized=True)
        if np.any(bad):
            near_bad = NearestNDInterpolator(
                np.column_stack([x, y]), bad.astype(float))(XX, YY)
            veil = np.ma.masked_where(
                (near_bad < 0.5) | ~np.isfinite(ZZ), np.ones_like(near_bad))
            ax.pcolormesh(XX, YY, veil, cmap=ListedColormap([(1, 1, 1, 0.45)]),
                          vmin=0, vmax=1, shading="nearest", rasterized=True)
    else:
        mesh = ax.scatter(x, y, c=np.maximum(10.0 ** z, vmin), cmap=cmap,
                          norm=norm, s=14)
        if np.any(bad):
            ax.scatter(x[bad], y[bad], marker="s", s=40, facecolors="none",
                       edgecolors="w", linewidths=0.7)
    ax.plot(x, y, ".", color="k", ms=1.2, alpha=0.4)
    ax.axhline(hw_limit_mhz, color="c", ls="--", lw=1.0, alpha=0.9)
    return mesh


def render_panel_grid(store: Store, manifest: dict, records, metric: str,
                      spec: PlotSpec, *, veil: bool = True,
                      dpi: int = 170) -> tuple[str, str]:
    """Render the 8x9 map family for ``metric`` into plots/; return (png, pdf).

    ``store`` must already resolve its manifest; ``records`` are the coherent
    records (states-skipped is fine).  No per-panel PNGs are emitted.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    store.ensure_dirs()
    values, vmin, vmax, cb_label = plot_metric_values(
        store, manifest, records, metric, scatter_channels=spec.scatter_channels)
    if not values:
        raise SystemExit("no successful records to plot")
    cmap = "magma_r"
    row_axis = manifest["axes"][spec.row_axis_key]
    tg = manifest["axes"]["t_gate_us"]

    n_rows, n_cols = len(row_axis), len(tg)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.1 * n_cols + 1.6, 1.9 * n_rows + 1.2),
                             sharex=True, sharey=True, constrained_layout=True)
    mesh = None
    for ri in range(n_rows):
        for ti in range(n_cols):
            ax = axes[ri][ti]
            data = _panel_plot_data(values, (ri, ti), vmin)
            if data is None:
                ax.set_facecolor("0.92")
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="0.4")
            else:
                mesh = _draw_panel(ax, *data, vmin, vmax, cmap,
                                   spec.hw_limit_mhz, veil=veil) or mesh
            if ri == 0:
                ax.set_title(f"T = {tg[ti]:g} us", fontsize=9)
            if ri == n_rows - 1:
                ax.set_xlabel(spec.xlabel, fontsize=8)
            if ti == 0:
                ax.set_ylabel(f"{spec.row_label(row_axis[ri])}\n"
                              r"$D_{\rm sweep}/2\pi$ (MHz)", fontsize=8)
            ax.tick_params(labelsize=7)
    if mesh is not None:
        cb = fig.colorbar(mesh, ax=axes, shrink=0.5, pad=0.01)
        cb.solids.set_rasterized(True)  # same PDF hairline-seam fix as the panels
        cb.set_label(cb_label, fontsize=9)
    if metric == "max_leakage":
        metric_title = "Coherent terminal leakage"
    elif metric == "total_error":
        metric_title = "Total first-order error budget (worst input)"
    else:
        metric_title = f"Scattering budget: {metric} (worst input)"
    dynamics_note = ("closed-dynamics trajectory + first-order scattering"
                     if metric == "total_error" else "closed dynamics")
    fig.suptitle(
        f"{metric_title}, {spec.system_desc} ({dynamics_note}, "
        "original-frame DOP853; rasters are log-linear interpolation between "
        "exact nodes — dots"
        + ("; white veil: interpolation untrusted, LOO residual > "
           f"{PLOT_LOO_MASK_DEX} dex)" if veil else
           "; NO uncertainty veil — raster is visualization only)"), fontsize=11)

    png = os.path.join(store.plots_dir, f"{metric}_8x9.png")
    pdf = os.path.join(store.plots_dir, f"{metric}_8x9.pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf, dpi=dpi)  # dpi applies to the rasterized mesh layers
    plt.close(fig)
    return png, pdf
