"""Foundational data structures for quantum system models.

- :class:`BasisSpec` — Hilbert space structure with symbolic level labels
- :class:`ObservableRegistry` — named measurement operators or specs
- :class:`SystemModel` — abstract base class consumed by solvers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class BasisSpec:
    """Describes the Hilbert space structure of a multi-site quantum system.

    Parameters
    ----------
    site_labels : tuple of str
        Labels for each site/atom (e.g. ("A", "B") for two-atom).
    local_levels : tuple of str
        Labels for each single-site energy level (e.g. ("0", "1", "e1", "e2", "e3", "r", "r_garb")).
    local_dim : int
        Number of levels per site (must equal len(local_levels)).
    total_dim : int
        Full Hilbert space dimension (local_dim ** n_sites).
    """

    site_labels: tuple[str, ...]
    local_levels: tuple[str, ...]
    local_dim: int
    total_dim: int

    def __post_init__(self):
        if self.local_dim != len(self.local_levels):
            raise ValueError(
                f"local_dim={self.local_dim} != len(local_levels)={len(self.local_levels)}"
            )
        expected_dim = self.local_dim ** len(self.site_labels)
        if self.total_dim != expected_dim:
            raise ValueError(
                f"total_dim={self.total_dim} != local_dim^n_sites={expected_dim}"
            )

    @property
    def n_sites(self) -> int:
        return len(self.site_labels)

    def level_index(self, label: str) -> int:
        """Return the integer index for a level label. Raises ValueError if not found."""
        try:
            return self.local_levels.index(label)
        except ValueError:
            raise ValueError(f"Level '{label}' not in {self.local_levels}") from None

    def site_index(self, label: str) -> int:
        """Return the integer index for a site label."""
        try:
            return self.site_labels.index(label)
        except ValueError:
            raise ValueError(f"Site '{label}' not in {self.site_labels}") from None

    def projector(self, site: str, level: str) -> NDArray[np.complexfloating]:
        """Build ``|level><level|`` on the given site, tensored with identity on other sites.

        Returns a total_dim x total_dim dense matrix.
        """
        site_idx = self.site_index(site)
        level_idx = self.level_index(level)

        sq = np.zeros((self.local_dim, self.local_dim), dtype=np.complex128)
        sq[level_idx, level_idx] = 1.0

        # Build tensor product: I^{site_idx} x sq x I^{n_sites - site_idx - 1}
        result = sq
        for i in range(self.n_sites - 1, -1, -1):
            if i == site_idx:
                continue
            if i > site_idx:
                result = np.kron(result, np.eye(self.local_dim, dtype=np.complex128))
            else:
                result = np.kron(np.eye(self.local_dim, dtype=np.complex128), result)
        return result


@dataclass(frozen=True)
class Observable:
    """A named observable with its operator and metadata."""

    name: str
    operator: Any  # ndarray, sparse matrix, or symbolic operator spec
    description: str = ""
    per_site: bool = False


class ObservableRegistry:
    """Registry of named observables for a quantum system.

    Provides symbolic access to measurement definitions, replacing hardcoded
    matrix indices throughout the codebase. Exact state-vector lattice
    observables may store symbolic specs and are evaluated by ``RydbergSystem``.
    """

    def __init__(self) -> None:
        self._observables: dict[str, Observable] = {}

    def register(
        self,
        name: str,
        operator: Any,
        description: str = "",
        per_site: bool = False,
    ) -> None:
        """Register a named observable."""
        self._observables[name] = Observable(
            name=name,
            operator=operator,
            description=description,
            per_site=per_site,
        )

    def get(self, name: str) -> Observable:
        """Get an Observable by name. Raises KeyError if not found."""
        return self._observables[name]

    def has(self, name: str) -> bool:
        """Return whether an observable is registered."""
        return name in self._observables

    def list_names(self) -> list[str]:
        """Return names of all registered observables."""
        return list(self._observables.keys())

    def measure(self, name: str, psi: np.ndarray) -> float:
        """Compute <psi|O|psi> for the named observable."""
        obs = self._observables[name]
        return float(np.real(np.vdot(psi, obs.operator @ psi)))

    def measure_all(self, psi: np.ndarray) -> dict[str, float]:
        """Compute expectation values of all registered observables."""
        return {name: self.measure(name, psi) for name in self._observables}

    def __contains__(self, name: str) -> bool:
        return name in self._observables

    def __len__(self) -> int:
        return len(self._observables)


class SystemModel(ABC):
    """Abstract base for all quantum system models.

    A SystemModel provides:
    - A basis specification with symbolic level labels
    - A primitive operator factory (``E[ket,bra]``) generated from the basis
    - Driveable Hamiltonian channels and static Hamiltonian terms built from it
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
    def operators(self):
        """Primitive ``E[ket,bra]`` operator factory generated from the basis."""
        ...

    @property
    @abstractmethod
    def hamiltonian_channels(self) -> dict:
        """Driveable primitive operators keyed by ``E[ket,bra]`` channel name."""
        ...

    @property
    @abstractmethod
    def static_hamiltonian_terms(self) -> list:
        """Static (protocol-independent) Hamiltonian terms."""
        ...

    @property
    @abstractmethod
    def observables(self) -> ObservableRegistry:
        """Registry of measurement observables or symbolic observable specs."""
        ...

    @property
    @abstractmethod
    def model_tag(self) -> str:
        """Level-structure tag identifying the atom model (e.g. 'rb87_7_mp',
        'analog_3', '1r') — not a numeric-parameter selector."""
        ...
