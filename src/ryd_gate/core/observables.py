"""Observable registry for quantum measurement operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Observable:
    """A named observable with its operator and metadata."""

    name: str
    operator: Any  # ndarray or sparse matrix
    description: str = ""
    per_site: bool = False


class ObservableRegistry:
    """Registry of named observables for a quantum system.

    Provides symbolic access to measurement operators, replacing
    hardcoded matrix indices throughout the codebase.
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
