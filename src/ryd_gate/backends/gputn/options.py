"""Typed options for the CUDA tensor-network backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUTNOptions:
    """Options for ``backend="gputn"`` TDVP evolution.

    ``None`` means "use the backend default".
    """

    chi_max: int | None = None
    dt: float | None = None
    svd_min: float | None = None
    device_id: int | None = None
    require_gpu: bool | None = None
    kernel: str | None = None
    trotter_order: int | None = None
    statevector_max_sites: int | None = None
    return_state_vector: bool | None = None
