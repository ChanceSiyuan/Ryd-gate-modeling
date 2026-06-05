"""Algorithm-agnostic evolution result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EvolutionResult:
    """Unified result object returned by simulation algorithm packages."""

    psi_final: Any
    times: np.ndarray | None = None
    states: Any | None = None
    metadata: dict = field(default_factory=dict)
