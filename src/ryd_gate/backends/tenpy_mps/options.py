"""Typed options for the TeNPy MPS backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TenpyOptions:
    """Options for ``backend="tenpy"`` TDVP/DMRG evolution.

    ``None`` means "use the backend default". ``dt`` and ``chi_max`` apply to
    TDVP; ``n_sweeps`` and ``mixer`` apply to DMRG ground-state search.
    """

    chi_max: int | None = None
    dt: float | None = None
    svd_min: float | None = None
    n_sweeps: int | None = None
    mixer: bool | None = None
