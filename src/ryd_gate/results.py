"""User-facing simulation results: lazy state queries over kernel outputs.

``EvolutionResult`` (ir/evolution.py) stays a pure algorithm-agnostic
container; this module adds the *new* behavior — bitstring sampling in
register-id order, per-level populations, cached expectations — while keeping
the raw kernel result reachable at ``result.raw``.

``SimulationResult`` methods are one-line delegations to the state handle
plus cache bookkeeping; any further logic belongs in the handle
(backend-native handles arrive in Stage 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from ryd_gate.core.level_structures import LevelStructureSpec
    from ryd_gate.core.system import RydbergSystem
    from ryd_gate.ir.evolution import EvolutionResult
    from ryd_gate.lattice.geometry import Register
    from ryd_gate.sequence import Sequence

__all__ = ["ExactStateHandle", "SimulationResult"]

_SAMPLE_BASES = ("full-level", "rydberg", "computational")


@dataclass
class ExactStateHandle:
    """Dense-statevector state queries for exact-backend results."""

    psi: np.ndarray
    system: "RydbergSystem"
    register: "Register"
    level_structure: "LevelStructureSpec"

    def statevector(self, *, max_dim: int | None = None, copy: bool = True) -> np.ndarray:
        """The dense state; ``max_dim`` guards against accidental huge returns."""
        if max_dim is not None and self.psi.size > max_dim:
            raise ValueError(
                f"statevector size {self.psi.size} exceeds max_dim={max_dim}."
            )
        return self.psi.copy() if copy else self.psi

    def expectation(self, observable: str) -> float:
        """Delegate to the kernel: ``system.expectation(observable, psi)``."""
        return float(np.real(self.system.expectation(observable, self.psi)))

    def populations(self, level: str) -> np.ndarray:
        """Per-site ``|level>`` populations, ordered by ``register.ids``."""
        return np.array([
            float(np.real(self.system.expectation(f"n_{level}_{i}", self.psi)))
            for i in range(self.register.n_atoms)
        ])

    def sample(
        self,
        n_shots: int,
        basis: Literal["full-level", "rydberg", "computational"] = "rydberg",
        seed: int | None = None,
    ) -> dict[str, int]:
        """Multinomial bitstring counts from ``|psi|^2`` (computed on demand, never cached)."""
        if not isinstance(n_shots, (int, np.integer)) or isinstance(n_shots, bool) or n_shots <= 0:
            raise ValueError(f"n_shots must be a positive integer, got {n_shots!r}.")
        if basis not in _SAMPLE_BASES:
            raise ValueError(f"basis must be one of {_SAMPLE_BASES}, got {basis!r}.")

        probs = np.abs(np.asarray(self.psi).ravel()) ** 2
        total = float(probs.sum())
        if total <= 0:
            raise ValueError("result.zero_norm: cannot sample from a zero state.")
        probs = probs / total

        levels = self.system.basis.local_levels
        d = self.system.basis.local_dim
        n_sites = self.system.basis.n_sites

        def site_labels(index: int) -> tuple[str, ...]:
            return tuple(
                levels[(index // d ** (n_sites - 1 - site)) % d] for site in range(n_sites)
            )

        if basis == "computational":
            comp_prob = sum(
                p for idx, p in enumerate(probs)
                if p > 0 and all(label in ("0", "1") for label in site_labels(idx))
            )
            if comp_prob < 1.0 - 1e-10:
                raise ValueError(
                    "result.noncomputational_population: population outside |0>/|1> "
                    f"(computational weight {comp_prob:.12f}); sample with basis='rydberg' "
                    "or 'full-level' instead."
                )

        rydberg_levels = set(self.level_structure.rydberg_levels)
        multi_char = any(len(label) > 1 for label in levels)

        def key_for(index: int) -> str:
            labels = site_labels(index)
            if basis == "rydberg":
                return "".join("1" if label in rydberg_levels else "0" for label in labels)
            if basis == "computational":
                return "".join(labels)
            return (" " if multi_char else "").join(labels)  # full-level

        rng = np.random.default_rng(seed)
        counts_vector = rng.multinomial(int(n_shots), probs)
        counts: dict[str, int] = {}
        for index in np.nonzero(counts_vector)[0]:
            key = key_for(int(index))
            counts[key] = counts.get(key, 0) + int(counts_vector[index])
        return counts


@dataclass
class SimulationResult:
    """Lazy, cached user-facing result; the kernel result stays at ``raw``."""

    raw: "EvolutionResult"
    state: ExactStateHandle
    backend: str
    sequence: "Sequence"
    metadata: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    def statevector(self, *, max_dim: int | None = None, copy: bool = True) -> np.ndarray:
        return self.state.statevector(max_dim=max_dim, copy=copy)

    def expectation(self, observable: str, *, cache: bool = True) -> float:
        key = ("expectation", observable)
        if cache and key in self._cache:
            return self._cache[key]
        value = self.state.expectation(observable)
        if cache:
            self._cache[key] = value
        return value

    def populations(self, level: str, *, cache: bool = True) -> np.ndarray:
        key = ("populations", level)
        if cache and key in self._cache:
            return self._cache[key]
        value = self.state.populations(level)
        if cache:
            self._cache[key] = value
        return value

    def sample(
        self,
        n_shots: int,
        basis: Literal["full-level", "rydberg", "computational"] = "rydberg",
        seed: int | None = None,
    ) -> dict[str, int]:
        return self.state.sample(n_shots, basis=basis, seed=seed)

    def clear_cache(self) -> None:
        self._cache.clear()
