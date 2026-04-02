"""Abstract base class for solver backends and unified EvolutionResult."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ryd_gate.compilers.ir import HamiltonianIR


@dataclass
class EvolutionResult:
    """Unified result from any solver backend.

    Attributes
    ----------
    psi_final : Any
        Final state vector (ndarray or MPS).
    times : ndarray or None
        Time points at which states were stored.
    states : Any or None
        Intermediate states (ndarray or list of MPS).
    metadata : dict
        Extra info passed through from the HamiltonianIR
        (e.g. t_gate, amplitude_scale, param_set).
    """

    psi_final: Any  # ndarray or MPS
    times: np.ndarray | None = None
    states: Any | None = None  # ndarray or list of MPS
    metadata: dict = field(default_factory=dict)


class SolverBackend(ABC):
    """Abstract solver backend.

    Takes a compiled HamiltonianIR and evolves a quantum state.
    Concrete subclasses implement different numerical strategies
    (dense ODE, sparse matrix exponential, tensor network, etc.).
    """

    @abstractmethod
    def evolve(
        self,
        ir: HamiltonianIR,
        psi0: Any,
        t_gate: float,
        t_eval: np.ndarray | None = None,
    ) -> EvolutionResult:
        """Evolve initial state under the Hamiltonian described by ir.

        Parameters
        ----------
        ir : HamiltonianIR
            Compiled Hamiltonian intermediate representation.
        psi0 : Any
            Initial state vector (ndarray for dense, may differ for MPS).
        t_gate : float
            Total evolution time.
        t_eval : ndarray or None
            Time points at which to store intermediate states.
            If None, only the final state is returned.

        Returns
        -------
        EvolutionResult
            Contains psi_final and optionally times/states.
        """
        ...
