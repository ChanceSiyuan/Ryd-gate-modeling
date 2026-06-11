"""User-facing simulation results: lazy state queries over kernel outputs.

``EvolutionResult`` (ir/evolution.py) stays a pure algorithm-agnostic
container; this module adds the *new* behavior — bitstring sampling in
register-id order, per-level populations, cached expectations — while keeping
the raw kernel result reachable at ``result.raw``.

``SimulationResult`` methods are one-line delegations to the state handle
plus cache bookkeeping; any further logic belongs in the handle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from typing import Protocol as TypingProtocol

import numpy as np

if TYPE_CHECKING:
    from ryd_gate.backends.tn_common.lattice_spec import TNLatticeSpec
    from ryd_gate.core.level_structures import LevelStructureSpec
    from ryd_gate.core.system import RydbergSystem
    from ryd_gate.ir.evolution import EvolutionResult
    from ryd_gate.lattice.geometry import Register
    from ryd_gate.sequence import Sequence

__all__ = [
    "ExactStateHandle",
    "MPSStateHandle",
    "QuantumStateHandle",
    "SimulationResult",
    "StateMaterializationError",
    "UnsupportedResultQuery",
    "UnsupportedStateHandle",
]

_SAMPLE_BASES = ("full-level", "rydberg", "computational")
_CAP_EXPECTATION = "expectation"
_CAP_SAMPLING = "sampling"
_CAP_STATEVECTOR = "statevector"
_CAP_STATEVECTOR_MATERIALIZATION = "statevector_materialization"


class UnsupportedResultQuery(RuntimeError):
    """Raised when a backend-native result cannot answer a requested query."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class StateMaterializationError(RuntimeError):
    """Raised when converting a native backend state to a dense vector is unsafe."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class QuantumStateHandle(TypingProtocol):
    """Structural interface implemented by backend-native state handles."""

    kind: str
    n_atoms: int
    local_levels: tuple[str, ...]
    atom_ids: tuple[str, ...]

    @property
    def capabilities(self) -> frozenset[str]: ...

    def expectation(self, observable: str) -> float: ...

    def populations(self, level: str) -> np.ndarray: ...

    def sample(
        self,
        n_shots: int,
        basis: Literal["full-level", "rydberg", "computational"] = "rydberg",
        seed: int | None = None,
    ) -> dict[str, int]: ...

    def statevector(self, *, max_dim: int | None = None, copy: bool = True) -> np.ndarray: ...


@dataclass
class ExactStateHandle:
    """Dense-statevector state queries for exact-backend results."""

    kind: ClassVar[str] = "statevector"

    psi: np.ndarray
    system: "RydbergSystem"
    register: "Register"
    level_structure: "LevelStructureSpec"

    @property
    def n_atoms(self) -> int:
        return self.register.n_atoms

    @property
    def local_levels(self) -> tuple[str, ...]:
        return tuple(self.level_structure.levels)

    @property
    def atom_ids(self) -> tuple[str, ...]:
        assert self.register.ids is not None  # normalized in Register.__post_init__
        return tuple(self.register.ids)

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({_CAP_EXPECTATION, _CAP_SAMPLING, _CAP_STATEVECTOR})

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
class MPSStateHandle:
    """TeNPy MPS-native state queries; dense conversion is explicit and guarded."""

    kind: ClassVar[str] = "mps"

    mps: Any
    spec: "TNLatticeSpec"
    register_ids: tuple[str, ...]
    metadata: dict = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return int(self.spec.N)

    @property
    def local_levels(self) -> tuple[str, ...]:
        return tuple(self.spec.level_spec.levels)

    @property
    def atom_ids(self) -> tuple[str, ...]:
        return tuple(self.register_ids)

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({_CAP_EXPECTATION, _CAP_STATEVECTOR_MATERIALIZATION})

    def expectation(self, observable: str) -> float:
        resolved = self._resolve_atom_observable(observable)
        if not (
            resolved == "sum_nr"
            or resolved.startswith("sum_n_")
            or self._is_site_occupation(resolved)
        ):
            raise UnsupportedResultQuery(f"mps.observable_unsupported: {observable}")
        try:
            from ryd_gate.backends.tenpy_mps.backends import measure_mps_observable

            value = measure_mps_observable(self.mps, self.spec, resolved)
        except ValueError as exc:
            raise UnsupportedResultQuery(str(exc)) from exc
        arr = np.asarray(value)
        if arr.ndim != 0:
            raise UnsupportedResultQuery(f"mps.observable_unsupported: {observable}")
        return float(np.real(arr.item()))

    def populations(self, level: str) -> np.ndarray:
        if level not in self.local_levels:
            raise UnsupportedResultQuery(f"mps.observable_unsupported: n_{level}")
        try:
            from ryd_gate.backends.tenpy_mps.observables import measure_level_occupations

            return np.asarray(measure_level_occupations(self.mps, self.spec, level), dtype=float)
        except ValueError as exc:
            raise UnsupportedResultQuery(str(exc)) from exc

    def sample(
        self,
        n_shots: int,
        basis: Literal["full-level", "rydberg", "computational"] = "rydberg",
        seed: int | None = None,
    ) -> dict[str, int]:
        return self._unsupported_sample(n_shots, basis, seed)

    def statevector(self, *, max_dim: int | None = None, copy: bool = True) -> np.ndarray:
        dim = int(self.spec.level_spec.local_dim) ** int(self.spec.N)
        if max_dim is None:
            raise StateMaterializationError("mps.statevector_requires_max_dim")
        if dim > max_dim:
            raise StateMaterializationError("mps.statevector_too_large")
        vec = _mps_dense_statevector(self.mps, self.spec)
        return vec.copy() if copy else vec

    def _resolve_atom_observable(self, observable: str) -> str:
        if not observable.startswith("n_"):
            return observable
        try:
            head, target = observable.rsplit("_", 1)
        except ValueError:
            return observable
        if target in self.register_ids:
            return f"{head}_{self.register_ids.index(target)}"
        return observable

    def _is_site_occupation(self, observable: str) -> bool:
        if not observable.startswith("n_"):
            return False
        try:
            level, target = observable[2:].rsplit("_", 1)
        except ValueError:
            return False
        return level in self.local_levels and target.isdecimal()

    def _unsupported_sample(self, n_shots: int, basis: str, seed: int | None) -> dict[str, int]:
        del n_shots, basis, seed
        raise UnsupportedResultQuery("mps.sampling_not_implemented")


@dataclass
class UnsupportedStateHandle:
    """Backend result placeholder for native states that Stage 3 cannot query."""

    backend: str
    reason_code: str
    n_atoms: int = 0
    local_levels: tuple[str, ...] = ()
    atom_ids: tuple[str, ...] = ()
    kind: str = field(default="unsupported", init=False)

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset()

    def expectation(self, observable: str) -> float:
        del observable
        raise UnsupportedResultQuery(self.reason_code)

    def populations(self, level: str) -> np.ndarray:
        del level
        raise UnsupportedResultQuery(self.reason_code)

    def sample(
        self,
        n_shots: int,
        basis: Literal["full-level", "rydberg", "computational"] = "rydberg",
        seed: int | None = None,
    ) -> dict[str, int]:
        del n_shots, basis, seed
        raise UnsupportedResultQuery(self.reason_code)

    def statevector(self, *, max_dim: int | None = None, copy: bool = True) -> np.ndarray:
        del max_dim, copy
        raise UnsupportedResultQuery(self.reason_code)


@dataclass
class SimulationResult:
    """Lazy, cached user-facing result; the kernel result stays at ``raw``."""

    raw: "EvolutionResult"
    state: QuantumStateHandle
    backend: str
    sequence: "Sequence"
    metadata: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def capabilities(self) -> frozenset[str]:
        return self.state.capabilities

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


def _mps_dense_statevector(mps: Any, spec: "TNLatticeSpec") -> np.ndarray:
    """Contract a tiny TeNPy MPS into exact basis order."""
    theta = mps.get_theta(0, int(spec.N)).to_ndarray()
    phys = np.asarray(theta).reshape((int(spec.level_spec.local_dim),) * int(spec.N))
    if spec.level_structure == "1r":
        for axis in range(int(spec.N)):
            phys = np.take(phys, [1, 0], axis=axis)
    axes = [int(spec.inv_snake[i]) for i in range(int(spec.N))]
    if axes != list(range(int(spec.N))):
        phys = np.transpose(phys, axes)
    return np.asarray(phys, dtype=complex).reshape(-1)
