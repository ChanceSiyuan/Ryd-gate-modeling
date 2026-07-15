"""Exact PEPS option schemas → private frozen records (PEPS21/22/35). No YASTN import.

Real-time has exactly ten mandatory keys; ground state has exactly nine. Missing,
unknown, aliased or old keys are rejected. No hidden clamp or default rewrites a
valid user input.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

_MEASUREMENT_METHODS = ("belief_propagation", "ctm")
_DEVICES = ("cpu", "cuda")

_REAL_TIME_KEYS = (
    "time_step_s", "bond_dimension", "svd_tolerance", "ntu_max_iterations",
    "ntu_iteration_tolerance", "measurement_method", "environment_bond_dimension",
    "environment_tolerance", "environment_max_iterations", "device",
)
_GROUND_KEYS = (
    "bond_dimension", "svd_tolerance", "ntu_max_iterations", "ntu_iteration_tolerance",
    "environment_bond_dimension", "environment_tolerance", "environment_max_iterations",
    "imaginary_time_schedule", "device",
)


@dataclass(frozen=True, slots=True)
class _RealTimeOptions:
    time_step_s: float
    bond_dimension: int
    svd_tolerance: float
    ntu_max_iterations: int
    ntu_iteration_tolerance: float
    measurement_method: str
    environment_bond_dimension: int
    environment_tolerance: float
    environment_max_iterations: int
    device: str

    def parameters(self) -> Mapping[str, object]:
        return MappingProxyType({k: getattr(self, k) for k in _REAL_TIME_KEYS})


@dataclass(frozen=True, slots=True)
class _GroundOptions:
    bond_dimension: int
    svd_tolerance: float
    ntu_max_iterations: int
    ntu_iteration_tolerance: float
    environment_bond_dimension: int
    environment_tolerance: float
    environment_max_iterations: int
    imaginary_time_schedule: tuple[tuple[float, int], ...]
    device: str

    def parameters(self) -> Mapping[str, object]:
        return MappingProxyType({k: getattr(self, k) for k in _GROUND_KEYS})


def _canonical(options, schema: tuple[str, ...], what: str) -> dict:
    if not isinstance(options, Mapping):
        raise ValueError(f"{what} must be a mapping with exactly {list(schema)}; got {type(options).__name__}.")
    have = set(options)
    want = set(schema)
    if have != want:
        raise ValueError(
            f"{what} must be exactly {list(schema)}; missing={sorted(want - have)}, "
            f"unknown={sorted(have - want)}."
        )
    return dict(options)


def _pos_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return int(value)


def _real(value, name: str) -> float:
    if isinstance(value, bool) or isinstance(value, str) or isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number; got {value!r}.")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number; got {type(value).__name__}.")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return v


def _tol_open_unit(value, name: str) -> float:
    v = _real(value, name)
    if not (0.0 < v < 1.0):
        raise ValueError(f"{name} must satisfy 0 < {name} < 1; got {value!r}.")
    return v


def _positive_float(value, name: str) -> float:
    v = _real(value, name)
    if not v > 0.0:
        raise ValueError(f"{name} must be strictly positive; got {value!r}.")
    return v


def _measurement_method(value) -> str:
    if value not in _MEASUREMENT_METHODS:
        raise ValueError(f"measurement_method must be one of {_MEASUREMENT_METHODS}; got {value!r}.")
    return value


def _device(value) -> str:
    if value not in _DEVICES:
        raise ValueError(f"device must be 'cpu' or 'cuda'; got {value!r}.")
    return value


def validate_real_time_options(options) -> _RealTimeOptions:
    """Validate the exactly-ten real-time PEPS keys (PEPS21/35)."""
    o = _canonical(options, _REAL_TIME_KEYS, "backend='peps' backend_options")
    return _RealTimeOptions(
        time_step_s=_positive_float(o["time_step_s"], "time_step_s"),
        bond_dimension=_pos_int(o["bond_dimension"], "bond_dimension"),
        svd_tolerance=_tol_open_unit(o["svd_tolerance"], "svd_tolerance"),
        ntu_max_iterations=_pos_int(o["ntu_max_iterations"], "ntu_max_iterations"),
        ntu_iteration_tolerance=_positive_float(o["ntu_iteration_tolerance"], "ntu_iteration_tolerance"),
        measurement_method=_measurement_method(o["measurement_method"]),
        environment_bond_dimension=_pos_int(o["environment_bond_dimension"], "environment_bond_dimension"),
        environment_tolerance=_tol_open_unit(o["environment_tolerance"], "environment_tolerance"),
        environment_max_iterations=_pos_int(o["environment_max_iterations"], "environment_max_iterations"),
        device=_device(o["device"]),
    )


def validate_ground_options(options) -> _GroundOptions:
    """Validate the exactly-nine ground-state PEPS keys (PEPS22/35)."""
    o = _canonical(options, _GROUND_KEYS, "method='peps_imaginary_time' method_options")
    return _GroundOptions(
        bond_dimension=_pos_int(o["bond_dimension"], "bond_dimension"),
        svd_tolerance=_tol_open_unit(o["svd_tolerance"], "svd_tolerance"),
        ntu_max_iterations=_pos_int(o["ntu_max_iterations"], "ntu_max_iterations"),
        ntu_iteration_tolerance=_positive_float(o["ntu_iteration_tolerance"], "ntu_iteration_tolerance"),
        environment_bond_dimension=_pos_int(o["environment_bond_dimension"], "environment_bond_dimension"),
        environment_tolerance=_tol_open_unit(o["environment_tolerance"], "environment_tolerance"),
        environment_max_iterations=_pos_int(o["environment_max_iterations"], "environment_max_iterations"),
        imaginary_time_schedule=_validate_schedule(o["imaginary_time_schedule"]),
        device=_device(o["device"]),
    )


def _validate_schedule(schedule) -> tuple[tuple[float, int], ...]:
    if not isinstance(schedule, tuple):
        raise ValueError("imaginary_time_schedule must be a tuple of (dtau, max_steps) tuples, not a list.")
    if len(schedule) == 0:
        raise ValueError("imaginary_time_schedule must be a nonempty tuple.")
    parsed: list[tuple[float, int]] = []
    prev_dtau = None
    for entry in schedule:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError(f"each imaginary_time_schedule entry must be a 2-tuple (dtau, max_steps); got {entry!r}.")
        dtau = _positive_float(entry[0], "imaginary_time_schedule dtau")
        if prev_dtau is not None and not dtau < prev_dtau:
            raise ValueError("imaginary_time_schedule dtau values must be strictly decreasing (PEPS22).")
        prev_dtau = dtau
        parsed.append((dtau, _pos_int(entry[1], "imaginary_time_schedule max_steps")))
    return tuple(parsed)
