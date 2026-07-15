"""``ryd_gate.results`` — the single entry point for the three result types (API06).

All three are immutable and expose only physical readouts. The backend-native
final state, lazy evaluators and contraction settings stay private; there is no
``Result`` base class, factory, conversion or analysis helper, and none of these
are exported from the top-level ``ryd_gate`` namespace.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

__all__ = [
    "EvolutionResult",
    "GroundStateResult",
    "EnsembleResult",
    "PEPSAmplitudeEvidence",
    "PEPSSampleEvidence",
    "PEPSEvidence",
]


# ── PEPS evidence records (report-only provenance; PEPS02/23-26) ─────────────
#
# These are plain immutable value objects: no convergence boolean, no I/O, no
# ``from_dict``/``save``. They validate their own inputs so a caller-constructed
# record is as deeply immutable as a ledger snapshot, and mutating any source
# container afterwards cannot leak into the record.


def _pev_finite_nonneg(value, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not a bool; got {value!r}.")
    v = float(value)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be finite and non-negative; got {value!r}.")
    return v


def _pev_finite_positive(value, name: str) -> float:
    v = _pev_finite_nonneg(value, name)
    if v == 0.0:
        raise ValueError(f"{name} must be strictly positive; got {value!r}.")
    return v


def _pev_nonneg_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}.")
    return int(value)


def _pev_deep_freeze(value):
    """Recursively copy ``value`` into immutable containers with built-in scalars."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _pev_deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_pev_deep_freeze(v) for v in value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"evidence parameters may only hold mappings, sequences and scalars; got {type(value).__name__}.")


def _pev_to_plain(value):
    """Recursively convert frozen evidence containers to JSON-compatible dict/list."""
    if isinstance(value, Mapping):
        return {str(k): _pev_to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pev_to_plain(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class PEPSAmplitudeEvidence:
    """One distinct amplitude coefficient contraction (PEPS24)."""

    labels: tuple[str, ...]
    contraction_error: float

    def __post_init__(self) -> None:
        labels = tuple(str(x) for x in self.labels)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "contraction_error", _pev_finite_nonneg(self.contraction_error, "contraction_error"))

    def to_dict(self) -> dict[str, object]:
        return {"labels": list(self.labels), "contraction_error": self.contraction_error}


@dataclass(frozen=True, slots=True)
class PEPSSampleEvidence:
    """One successful ``sample()`` call (PEPS25)."""

    shots: int
    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.shots, bool) or not isinstance(self.shots, (int, np.integer)) or int(self.shots) < 1:
            raise ValueError(f"shots must be a positive integer; got {self.shots!r}.")
        object.__setattr__(self, "shots", int(self.shots))
        object.__setattr__(self, "seed", _pev_nonneg_int(self.seed, "seed"))

    def to_dict(self) -> dict[str, object]:
        return {"shots": self.shots, "seed": self.seed}


@dataclass(frozen=True, slots=True)
class PEPSEvidence:
    """Report-only PEPS run provenance (PEPS23/26). ``to_dict()`` is the only serializer."""

    parameters: Mapping[str, object]
    hamiltonian_scale_rad_s: float | None
    max_ntu_truncation_error: float
    cumulative_ntu_truncation_error: float | None
    environment_residual: float | None
    environment_iterations: int | None
    norm_contraction_error: float | None
    amplitudes: tuple[PEPSAmplitudeEvidence, ...]
    samples: tuple[PEPSSampleEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _pev_deep_freeze(dict(self.parameters)))
        object.__setattr__(
            self,
            "hamiltonian_scale_rad_s",
            None if self.hamiltonian_scale_rad_s is None
            else _pev_finite_positive(self.hamiltonian_scale_rad_s, "hamiltonian_scale_rad_s"),
        )
        object.__setattr__(
            self, "max_ntu_truncation_error", _pev_finite_nonneg(self.max_ntu_truncation_error, "max_ntu_truncation_error")
        )
        object.__setattr__(
            self,
            "cumulative_ntu_truncation_error",
            None if self.cumulative_ntu_truncation_error is None
            else _pev_finite_nonneg(self.cumulative_ntu_truncation_error, "cumulative_ntu_truncation_error"),
        )
        object.__setattr__(
            self,
            "environment_residual",
            None if self.environment_residual is None
            else _pev_finite_nonneg(self.environment_residual, "environment_residual"),
        )
        object.__setattr__(
            self,
            "environment_iterations",
            None if self.environment_iterations is None
            else _pev_nonneg_int(self.environment_iterations, "environment_iterations"),
        )
        object.__setattr__(
            self,
            "norm_contraction_error",
            None if self.norm_contraction_error is None
            else _pev_finite_nonneg(self.norm_contraction_error, "norm_contraction_error"),
        )
        amps = tuple(self.amplitudes)
        samples = tuple(self.samples)
        if not all(isinstance(a, PEPSAmplitudeEvidence) for a in amps):
            raise TypeError("amplitudes must all be PEPSAmplitudeEvidence.")
        if not all(isinstance(s, PEPSSampleEvidence) for s in samples):
            raise TypeError("samples must all be PEPSSampleEvidence.")
        object.__setattr__(self, "amplitudes", amps)
        object.__setattr__(self, "samples", samples)

    def to_dict(self) -> dict[str, object]:
        """Fresh JSON-compatible deep copy (the only serializer; PEPS26)."""
        return {
            "parameters": _pev_to_plain(self.parameters),
            "hamiltonian_scale_rad_s": self.hamiltonian_scale_rad_s,
            "max_ntu_truncation_error": self.max_ntu_truncation_error,
            "cumulative_ntu_truncation_error": self.cumulative_ntu_truncation_error,
            "environment_residual": self.environment_residual,
            "environment_iterations": self.environment_iterations,
            "norm_contraction_error": self.norm_contraction_error,
            "amplitudes": [a.to_dict() for a in self.amplitudes],
            "samples": [s.to_dict() for s in self.samples],
        }


def _check_shots_seed(shots, seed) -> tuple[int, int]:
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or shots < 1:
        raise ValueError(f"shots must be a positive integer; got {shots!r}.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer; got {seed!r}.")
    if seed < 0:
        raise ValueError(f"seed must be a non-negative integer; got {seed!r}.")
    return int(shots), int(seed)


def _readonly(arr) -> np.ndarray:
    out = np.array(arr, dtype=float, copy=True)
    out.setflags(write=False)
    return out


class EvolutionResult:
    """Result of one time evolution (ER01-ER09).

    Public surface: ``times`` and the three physical readouts
    ``expectation(name)``, ``amplitude(level_labels)`` and
    ``sample(shots=..., seed=...)``. The final backend state is private.
    """

    __slots__ = ("_times", "_expectations", "_reader")

    def __init__(self, *, times, expectations, reader) -> None:
        self._times = _readonly(times)
        self._expectations = {k: _readonly(v) for k, v in dict(expectations).items()}
        self._reader = reader  # private state reader: .amplitude(labels), .sample(shots, seed)

    @property
    def times(self) -> np.ndarray:
        """Measurement times (read-only 1-D array); ``[t_gate]`` when ``t_eval=None``."""
        return self._times

    def expectation(self, name: str) -> np.ndarray:
        """Real ``float64`` expectation array for ``name`` (shape ``times.shape``)."""
        try:
            return self._expectations[name]
        except KeyError:
            raise KeyError(
                f"no expectation {name!r} on this result; request it at simulate time "
                f"via observables={{{name!r}: <ObservableExpr>}}. "
                f"Stored: {sorted(self._expectations)}."
            ) from None

    def amplitude(self, level_labels) -> complex:
        """Complex basis amplitude ``<labels | psi(t_gate)>`` for a flat label list."""
        return self._reader.amplitude(list(level_labels))

    def sample(self, *, shots: int, seed: int) -> Counter:
        """Lazily sample computational-basis outcomes as ``Counter[tuple[str, ...]]``."""
        s, sd = _check_shots_seed(shots, seed)
        return self._reader.sample(s, sd)

    @property
    def peps_evidence(self) -> "PEPSEvidence | None":
        """PEPS run provenance (``None`` for non-PEPS backends); building it runs nothing."""
        snapshot = getattr(self._reader, "_peps_evidence_snapshot", None)
        return snapshot() if snapshot is not None else None


class GroundStateResult:
    """Result of a ground-state search (E15/E16/E21).

    ``expectation(name)`` returns a real scalar (``"energy"`` reserved and always
    present); ``amplitude`` needs an explicit ``phase_reference`` because the
    ground-state global phase is gauge.
    """

    __slots__ = ("_expectations", "_reader")

    def __init__(self, *, expectations, reader) -> None:
        self._expectations = {k: float(v) for k, v in dict(expectations).items()}
        self._reader = reader  # .amplitude(labels, phase_reference), .sample(shots, seed)

    def expectation(self, name: str) -> float:
        """Real scalar expectation for ``name`` (``"energy"`` always available)."""
        try:
            return self._expectations[name]
        except KeyError:
            raise KeyError(
                f"no expectation {name!r} on this ground state; request it via "
                f"observables={{{name!r}: <ObservableExpr>}}. "
                f"Stored: {sorted(self._expectations)}."
            ) from None

    def amplitude(self, level_labels, *, phase_reference) -> complex:
        """Complex basis amplitude with the global phase fixed by ``phase_reference``."""
        return self._reader.amplitude(list(level_labels), list(phase_reference))

    def sample(self, *, shots: int, seed: int) -> Counter:
        s, sd = _check_shots_seed(shots, seed)
        return self._reader.sample(s, sd)

    @property
    def peps_evidence(self) -> "PEPSEvidence | None":
        """PEPS ground-state run provenance (``None`` for non-PEPS backends)."""
        snapshot = getattr(self._reader, "_peps_evidence_snapshot", None)
        return snapshot() if snapshot is not None else None


class EnsembleResult:
    """Raw shot-major container of noisy realizations (N11/N16/N18)."""

    __slots__ = ("results", "realizations", "seed")

    def __init__(self, *, results, realizations, seed: int) -> None:
        object.__setattr__(self, "results", tuple(results))
        object.__setattr__(self, "realizations", tuple(realizations))
        object.__setattr__(self, "seed", int(seed))

    def __setattr__(self, name, value):
        raise AttributeError("EnsembleResult is immutable.")
