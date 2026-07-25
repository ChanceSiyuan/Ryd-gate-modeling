"""Shared machinery for the two-atom CZ max-leakage sweep scripts.

``axes`` holds the nested rational-grid math and the per-script canonical
point-key factory; ``solver`` holds the quintic envelope, the block-max DOP853
error norm, and the injected-RHS batched integration kernel; ``store`` holds the
append-only chunk/scatter store + provenance gates; ``runner`` holds the fork-pool
worker entry, cost model and budget runner; ``plotting`` holds the log-linear
interpolation, LOO veil, credibility floor and 8x9 grid renderer; ``cli`` holds
the shared parser scaffold and derived-output resolution.
"""
from __future__ import annotations

from . import cli, plotting

from .axes import (
    LEVEL_DENS,
    LEVEL_SIZES,
    LEVEL_FROM_SIZE,
    canon_coord,
    coord_value_mhz,
    axis_coords,
    axis_values_mhz,
    make_pointkey_type,
)
from .solver import (
    LOGICAL_INPUTS,
    quintic,
    quintic_antideriv,
    envelope,
    envelope_integral,
    BlockMaxDOP853,
    make_block_solver_class,
    verify_scipy_error_norm,
    BatchResult,
    integrate_batch,
)
from .store import (
    Store,
    ProvenanceColumns,
    PointRecord,
    TIER_RANK,
    best_records,
    completed_keys,
    audit_pairs,
    _atomic_savez,
    _NO_STATES,
)
from .runner import (
    Batch,
    CostModel,
    Runner,
    group_batches,
    set_worker_context,
    _worker_run_batch,
    _worker_process_init,
)
from .plotting import (
    PlotSpec,
    credibility_floor,
    holdout_residuals,
    plot_metric_values,
    render_panel_grid,
    PLOT_LOO_MASK_DEX,
    PLOT_RASTER_N,
)
from .cli import (
    add_common_args,
    default_output,
    int_or_auto,
    resolve_output,
)

__all__ = [
    "LEVEL_DENS",
    "LEVEL_SIZES",
    "LEVEL_FROM_SIZE",
    "canon_coord",
    "coord_value_mhz",
    "axis_coords",
    "axis_values_mhz",
    "make_pointkey_type",
    "LOGICAL_INPUTS",
    "quintic",
    "quintic_antideriv",
    "envelope",
    "envelope_integral",
    "BlockMaxDOP853",
    "make_block_solver_class",
    "verify_scipy_error_norm",
    "BatchResult",
    "integrate_batch",
    "Store",
    "ProvenanceColumns",
    "PointRecord",
    "TIER_RANK",
    "best_records",
    "completed_keys",
    "audit_pairs",
    "Batch",
    "CostModel",
    "Runner",
    "group_batches",
    "set_worker_context",
    "cli",
    "plotting",
    "PlotSpec",
    "credibility_floor",
    "holdout_residuals",
    "plot_metric_values",
    "render_panel_grid",
    "PLOT_LOO_MASK_DEX",
    "PLOT_RASTER_N",
    "add_common_args",
    "default_output",
    "int_or_auto",
    "resolve_output",
]
