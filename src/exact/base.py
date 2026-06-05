"""Exact state-vector backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from ryd_gate.ir.evolution import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.ir import HamiltonianIR


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
