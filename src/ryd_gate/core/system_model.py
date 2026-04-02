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
    - A registry of Hamiltonian operator blocks
    - A registry of observables for measurement

    Concrete subclasses (Rb87TwoAtomModel, Lattice2LevelModel, etc.)
    populate these registries from physical parameters.
    """

    @property
    @abstractmethod
    def basis(self) -> BasisSpec:
        """The Hilbert space structure."""
        ...

    @property
    @abstractmethod
    def blocks(self) -> BlockRegistry:
        """Registry of Hamiltonian operator blocks."""
        ...

    @property
    @abstractmethod
    def observables(self) -> ObservableRegistry:
        """Registry of measurement observables."""
        ...

    @property
    @abstractmethod
    def param_set(self) -> str:
        """System identifier string (e.g. 'our', 'lukin', 'analog', 'lattice')."""
        ...
