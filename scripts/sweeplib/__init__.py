"""Shared machinery for the two-atom CZ max-leakage sweep scripts.

``axes`` holds the nested rational-grid math and the per-script canonical
point-key factory; ``solver`` holds the quintic envelope, the block-max DOP853
error norm, and the injected-RHS batched integration kernel.
"""
from __future__ import annotations

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
]
