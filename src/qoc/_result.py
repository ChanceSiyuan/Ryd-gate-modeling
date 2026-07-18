"""QOC-owned, backend-independent optimization result (ADR-0019)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OptimizationResult:
    """Best candidate and method-independent solver status of one single-start solve.

    ``success`` means numerical solver termination only and carries no physical
    acceptance claim.
    """

    best_parameters: dict[str, float | np.ndarray]
    best_loss: float
    success: bool
    message: str
    method: str
    n_iterations: int
    n_evaluations: int
