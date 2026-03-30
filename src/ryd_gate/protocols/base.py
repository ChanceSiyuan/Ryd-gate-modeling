"""Abstract base class for pulse protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ryd_gate.core.atomic_system import AtomicSystem


class Protocol(ABC):
    """Abstract base class for all 420nm laser pulse protocols.

    A protocol defines how pulse parameters map to physical quantities
    and provides the phase modulation function for the 420nm laser.

    Subclasses must implement:
    - ``n_params``, ``validate_params``, ``unpack_params``, ``phase_420``

    CZ gate protocols additionally override:
    - ``theta_index``, ``t_gate_index``, ``get_optimization_bounds``
    """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of parameters in the parameter vector x."""
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
            Must contain at minimum ``'t_gate'``.
        """
        ...

    @abstractmethod
    def phase_420(self, t: float, params: dict) -> complex:
        """Compute exp(-i * phi(t)) for 420nm laser phase modulation."""
        ...

    # -- Optional hooks (CZ gate protocols override these) ----------------

    @property
    def theta_index(self) -> int | None:
        """Index of theta (single-qubit Z rotation) in x, or None."""
        return None

    @property
    def t_gate_index(self) -> int | None:
        """Index of scaled gate time in x, or None."""
        return None

    def get_optimization_bounds(self) -> tuple | None:
        """Return bounds for optimisation, or None if not applicable."""
        return None
