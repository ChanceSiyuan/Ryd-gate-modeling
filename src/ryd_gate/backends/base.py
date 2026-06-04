"""Abstract backend interface and unified evolution result."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ryd_gate.ir.matrix import HamiltonianIR


@dataclass
class EvolutionResult:
    """Unified result from any simulation backend."""

    psi_final: Any
    times: np.ndarray | None = None
    states: Any | None = None
    metadata: dict = field(default_factory=dict)


class SolverBackend(ABC):
    """Abstract simulation backend."""

    @abstractmethod
    def evolve(
        self,
        ir: HamiltonianIR,
        psi0: Any,
        t_gate: float,
        t_eval: np.ndarray | bool | None = None,
    ) -> EvolutionResult:
        """Evolve initial state under a compiled IR."""
        ...
