"""Abstract base class for pulse protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ryd_gate.core.atomic_system import AtomicSystem


class Protocol(ABC):
    """Abstract base class for pulse protocols.

    A protocol defines how pulse parameters map to time-dependent
    drive coefficients on named channels.

    Subclasses must implement:
    - ``n_params``, ``validate_params``, ``unpack_params``, ``get_drive_coefficients``

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
    def unpack_params(self, x: list[float], system: "AtomicSystem") -> dict:
        """Unpack x into physical quantities for the solver.

        Returns
        -------
        dict
            Must contain at minimum ``'t_gate'``.
        """
        ...

    def phase_420(self, t: float, params: dict) -> complex:
        """Compute exp(-i * phi(t)) for 420nm laser phase modulation.

        .. deprecated::
            Override ``get_drive_coefficients()`` instead.
            This default implementation extracts the phase from
            ``get_drive_coefficients()`` for backward compatibility
            with the legacy ``solve_gate()`` code path.
        """
        coeffs = self.get_drive_coefficients(t, params)
        return coeffs.get("drive_420", 1.0 + 0j)

    # -- New generalized interface -----------------------------------------

    @property
    def required_channels(self) -> frozenset[str]:
        """Channel names this protocol drives.

        Used by the compiler to validate system-protocol compatibility.
        Default assumes two-photon Raman (420nm + lightshift).
        """
        return frozenset({"drive_420", "drive_420_dag", "lightshift_zero"})

    @abstractmethod
    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        """Return {channel_name: coefficient(t)} for each drive term."""
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

    def get_ham_const_additions(self) -> "NDArray[np.complexfloating] | None":
        """Extra time-independent Hamiltonian terms, or None."""
        return None
