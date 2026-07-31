"""Shared 8x9 map-family rendering for the two-atom CZ max-leakage sweeps.

Rasters are *visualization only*: log-linear (log10) interpolation of the scattered
exact nodes onto a fine mesh, an axis-holdout leave-one-out (LOO) credibility veil,
the audit-derived credibility floor, and the 8x9 panel-grid renderer with family
edge-labeling (top row carries the gate-time titles; the left column carries the
row-variable ylabel).  Each script supplies its scatter-channel table, the manifest
axis key + labeller for the row variable, the x-axis label and the system
description via ``PlotSpec``; everything else is byte-identical between the scripts.

Metrics that are not derivable from the store alone -- the laser-phase-noise
``eps_phase``, and ``total_error_phase`` which adds it to the coherent and
scattering budgets -- arrive through ``extra_values``: one ``(4,)`` array of
per-logical-input error per point, computed by the caller from whatever noise model
it holds.  Such a render also carries a ``suffix``/``subdir`` so a model-dependent
figure can never land on the model-free one it re-renders.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .store import Store, audit_pairs, best_records

PLOT_LOO_MASK_DEX = 0.2
PLOT_RASTER_N = 81
PLOT_TABLE_STRIP_FRAC = 0.12    # figure height reserved for spec.table, if given


@dataclass(frozen=True)
class PlotSpec:
    """Per-script plot parameterization for :func:`render_panel_grid`.

    ``scatter_channels`` is the ordered channel table summed into the scattering
    budget (ODE: ``p_mid``/``p_ryd``/``p_r_garb``; 297: ``p_ryd``/``p_r_garb``);
    ``row_axis_key`` is the ``manifest["axes"]`` key holding the panel-row values
    and ``row_label`` maps one row value to its ylabel head (e.g. ``$\\Delta_e/2\\pi$
    = 20 GHz`` or ``$n$ = 70``); ``xlabel`` and ``system_desc`` are the fixed
    x-axis label and the suptitle system string.  ``hw_limit_mhz`` is the
    horizontal reference line drawn in every panel.  ``table``, when given, is a
    ``(col_labels, row_labels, cells, caption)`` bundle drawn as a strip under the
    grid — the power<->Rabi conversion the phase-noise figures are read against.
    """

    scatter_channels: tuple[str, ...]
    row_axis_key: str
    row_label: Callable[[float], str]
    xlabel: str
    system_desc: str
    hw_limit_mhz: float = 20.0
    table: tuple | None = None


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


def _positive_range(values: dict) -> tuple[float, float]:
    """(vmin, vmax) for a LogNorm over ``values``, ignoring exact zeros."""
    pos = [v for v in values.values() if v > 0]
    vmin = max(1e-12, min(pos)) if pos else 1e-12
    return vmin, max(max(values.values()), vmin * 10)


def plot_metric_values(store: Store, manifest: dict, records, metric: str,
                       *, scatter_channels: tuple[str, ...],
                       extra_values: dict | None = None):
    """(values, vmin, vmax, colorbar_label) for a plot metric.

    ``max_leakage`` reads the coherent-leakage records (audit-derived floor);
    the ``p_*`` metrics read the supplemental scatter series. ``total_error``
    adds coherent leakage and every scattering contribution per logical input
    before selecting the worst input.  The scattering budget sums exactly the
    per-script ``scatter_channels``.

    ``eps_phase`` is not in the store at all: it is the caller's per-point ``(4,)``
    ``extra_values`` array of noise-induced fidelity loss, reported as its worst
    logical input.  ``total_error_phase`` is ``total_error`` with that array added
    to the same per-input sum, so all three budgets are combined input by input and
    only then maximized — the three worst inputs are generally different points of
    the map and pairing their maxima would over-count.
    """
    if metric == "eps_phase":
        if not extra_values:
            raise SystemExit("no filter records for --metric eps_phase; "
                             "run the `filter` subcommand first")
        values = {k: float(np.max(v)) for k, v in extra_values.items()}
        vmin, vmax = _positive_range(values)
        return values, vmin, vmax, (
            "worst-input eps_phase  (laser-phase-noise fidelity loss, "
            "filter function on the stored kernels)")
    if metric == "max_leakage":
        best = best_records(records)
        values = {k: r.max_leakage for k, r in best.items()}
        vmin, floor_info = credibility_floor(records)
        label = ("terminal max leakage  "
                 f"(floor {vmin:.1e}: "
                 f"{'audit-derived' if not floor_info['fallback'] else 'fallback'};"
                 " values at floor are below the numerical credibility floor)")
        return values, vmin, 1.0, label
    totals = metric in ("total_error", "total_error_phase")
    if metric == "total_error_phase" and not extra_values:
        raise SystemExit("no filter records for --metric total_error_phase; "
                         "run the `filter` subcommand first")
    coherent = best_records(records) if totals else {}
    rows = [r for r in store.load_scatter_records(manifest) if r["status"] == "ok"]
    if not rows:
        raise SystemExit(f"no scatter records for --metric {metric}; "
                         "run the `scatter` subcommand first")
    per_key: dict = {}
    for r in rows:
        scattering = sum(r[ch] for ch in scatter_channels)
        if totals:
            if r["key"] not in coherent:
                continue
            budget = coherent[r["key"]].leakage + scattering
            if metric == "total_error_phase":
                if r["key"] not in extra_values:
                    continue
                budget = budget + extra_values[r["key"]]
            v = float(np.max(budget))
        elif metric == "p_loss_total":
            v = float(np.max(scattering))
        else:
            v = float(np.max(r[metric]))
        cur = per_key.get(r["key"])
        if cur is None or r["rtol"] < cur[1]:
            per_key[r["key"]] = (v, r["rtol"])
    values = {k: v for k, (v, _) in per_key.items()}
    if not values:
        raise SystemExit(f"no overlapping records for --metric {metric}")
    vmin, vmax = _positive_range(values)
    if metric == "total_error":
        label = ("worst-input total error budget (terminal coherent leakage + "
                 "first-order scattering)")
    elif metric == "total_error_phase":
        label = ("worst-input total error budget (terminal coherent leakage + "
                 "first-order scattering + laser phase noise)")
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


def _draw_table_strip(subfig, table: tuple) -> None:
    """The (col_labels, row_labels, cells, caption) strip under the panel grid."""
    col_labels, row_labels, cells, caption = table
    ax = subfig.subplots()
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=col_labels, rowLabels=row_labels,
                   cellLoc="center", rowLoc="center", bbox=[0.06, 0.0, 0.9, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    subfig.supxlabel(caption, fontsize=7)


def render_panel_grid(store: Store, manifest: dict, records, metric: str,
                      spec: PlotSpec, *, veil: bool = True, dpi: int = 170,
                      extra_values: dict | None = None, suffix: str = "",
                      subdir: str = "", title_note: str = "") -> tuple[str, str]:
    """Render the 8x9 map family for ``metric`` into plots/; return (png, pdf).

    ``store`` must already resolve its manifest; ``records`` are the coherent
    records (states-skipped is fine).  No per-panel PNGs are emitted.

    ``extra_values`` supplies the metrics the store cannot derive on its own (see
    the module docstring).  ``subdir``/``suffix`` place such a render at
    ``plots/<subdir>/<metric>_8x9_<suffix>.png``, which is what keeps a
    noise-model-dependent figure off the model-free ``plots/<metric>_8x9.png``, and
    ``title_note`` becomes a second suptitle line naming that model — the same
    figure rendered under two noise models is otherwise indistinguishable once a
    page is detached from its filename.  It is a second line, not a longer first
    one, because the suptitle already fills the figure width.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    store.ensure_dirs()
    values, vmin, vmax, cb_label = plot_metric_values(
        store, manifest, records, metric, scatter_channels=spec.scatter_channels,
        extra_values=extra_values)
    if not values:
        raise SystemExit("no successful records to plot")
    cmap = "magma_r"
    row_axis = manifest["axes"][spec.row_axis_key]
    tg = manifest["axes"]["t_gate_us"]

    n_rows, n_cols = len(row_axis), len(tg)
    grid_h = 1.9 * n_rows + 1.2
    strip_h = grid_h * PLOT_TABLE_STRIP_FRAC / (1.0 - PLOT_TABLE_STRIP_FRAC)
    has_table = spec.table is not None
    fig = plt.figure(figsize=(2.1 * n_cols + 1.6,
                              grid_h + (strip_h if has_table else 0.0)),
                     constrained_layout=True)
    if has_table:
        grid_fig, strip_fig = fig.subfigures(2, 1, height_ratios=[grid_h, strip_h])
    else:
        grid_fig, strip_fig = fig, None
    axes = grid_fig.subplots(n_rows, n_cols, sharex=True, sharey=True)
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
        cb = grid_fig.colorbar(mesh, ax=axes, shrink=0.5, pad=0.01)
        cb.solids.set_rasterized(True)  # same PDF hairline-seam fix as the panels
        cb.set_label(cb_label, fontsize=9)
    if strip_fig is not None:
        _draw_table_strip(strip_fig, spec.table)
    if metric == "max_leakage":
        metric_title = "Coherent terminal leakage"
    elif metric == "eps_phase":
        metric_title = "Laser-phase-noise fidelity loss (worst input)"
    elif metric in ("total_error", "total_error_phase"):
        metric_title = "Total first-order error budget (worst input)"
    else:
        metric_title = f"Scattering budget: {metric} (worst input)"
    if metric == "eps_phase":
        dynamics_note = "closed-dynamics trajectory + first-order phase noise"
    elif metric == "total_error":
        dynamics_note = "closed-dynamics trajectory + first-order scattering"
    elif metric == "total_error_phase":
        dynamics_note = ("closed-dynamics trajectory + first-order scattering "
                         "and phase noise")
    else:
        dynamics_note = "closed dynamics"
    fig.suptitle(
        f"{metric_title}, {spec.system_desc} ({dynamics_note}, "
        "original-frame DOP853; rasters are log-linear interpolation between "
        "exact nodes — dots"
        + ("; white veil: interpolation untrusted, LOO residual > "
           f"{PLOT_LOO_MASK_DEX} dex)" if veil else
           "; NO uncertainty veil — raster is visualization only)")
        + (f"\n{title_note}" if title_note else ""), fontsize=11)

    outdir = os.path.join(store.plots_dir, subdir) if subdir else store.plots_dir
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"{metric}_8x9" + (f"_{suffix}" if suffix else ""))
    png, pdf = f"{stem}.png", f"{stem}.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf, dpi=dpi)  # dpi applies to the rasterized mesh layers
    plt.close(fig)
    return png, pdf
