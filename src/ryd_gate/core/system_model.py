"""Abstract base class for quantum system models."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ryd_gate.core.basis import BasisSpec
from ryd_gate.core.blocks import BlockRegistry
from ryd_gate.core.observables import ObservableRegistry


class SystemModel(ABC):
    """Abstract base for all quantum system models.

    A SystemModel provides:
    - A basis specification with symbolic level labels
    - A registry of Hamiltonian matrix blocks or symbolic block specs
    - A registry of observables or symbolic observable specs

    ``RydbergSystem`` is the canonical implementation.
    """

    @property
    @abstractmethod
    def basis(self) -> BasisSpec:
        """The Hilbert space structure."""
        ...

    @property
    @abstractmethod
    def blocks(self) -> BlockRegistry:
        """Registry of Hamiltonian matrix blocks or symbolic block specs."""
        ...

    @property
    @abstractmethod
    def observables(self) -> ObservableRegistry:
        """Registry of measurement observables or symbolic observable specs."""
        ...

    @property
    @abstractmethod
    def param_set(self) -> str:
        """System identifier string (e.g. 'our', 'lukin', 'analog', 'lattice')."""
        ...
