"""Abstract base classes for pulse protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ryd_gate.core.atomic_system import AtomicSystem


class Protocol(ABC):
    """Abstract base class for CZ gate pulse protocols.

    A protocol defines how pulse parameters map to physical quantities
    and provides the phase modulation function for the 420nm laser.
    """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of parameters in the parameter vector x."""
        ...

    @property
    @abstractmethod
    def theta_index(self) -> int:
        """Index of theta (single-qubit Z rotation) in x."""
        ...

    @property
    @abstractmethod
    def t_gate_index(self) -> int:
        """Index of scaled gate time in x."""
        ...

    @abstractmethod
    def validate_params(self, x: list[float]) -> None:
        """Raise ValueError if x has wrong length."""
        ...

    @abstractmethod
    def unpack_params(self, x: list[float], system: AtomicSystem) -> dict:
        """Unpack x into physical quantities for the solver.

        Returns
        -------
        dict
            Must contain at minimum 'theta' and 't_gate'.
        """
        ...

    @abstractmethod
    def phase_420(self, t: float, params: dict) -> complex:
        """Compute exp(-i * phi(t)) for 420nm laser phase modulation."""
        ...

    @abstractmethod
    def get_optimization_bounds(self) -> tuple:
        """Return bounds for Nelder-Mead optimization."""
        ...


class SweepProtocol(ABC):
    """Abstract base class for protocols with fully time-dependent Hamiltonians.

    Unlike :class:`Protocol` (which separates static Hamiltonian from phase
    modulation), sweep protocols construct the entire H(t) at each timestep.
    """

    @abstractmethod
    def get_hamiltonian(
        self, t: float, system: AtomicSystem,
    ) -> "NDArray[np.complexfloating]":
        """Return the full 49x49 Hamiltonian at time t."""
        ...

    @property
    @abstractmethod
    def t_gate(self) -> float:
        """Total protocol duration in seconds."""
        ...
