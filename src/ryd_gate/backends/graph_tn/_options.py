"""Exact graph-PEPS option schemas → private frozen records. No quimb import.

Real-time has exactly six mandatory keys; ground state has exactly six. Missing,
unknown or aliased keys are rejected. No hidden clamp or default rewrites a valid
user input. ``device='cuda'`` is a valid schema value but is rejected by the
engine at load time (there is no silent CPU fallback).
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

_MEASUREMENT_METHODS = ("exact", "cluster", "bp")
_DEVICES = ("cpu", "cuda")

_REAL_TIME_KEYS = (
    "time_step_s", "bond_dimension", "svd_cutoff",
    "measurement_method", "cluster_max_distance", "device",
)
_GROUND_KEYS = (
    "bond_dimension", "svd_cutoff", "imaginary_time_schedule",
    "measurement_method", "cluster_max_distance", "device",
)


class GraphTNError(ValueError):
    """A backend='graph_peps' geometry/option/topology capability or validity error."""


@dataclass(frozen=True, slots=True)
class _RealTimeOptions:
    time_step_s: float
    bond_dimension: int
    svd_cutoff: float
    measurement_method: str
    cluster_max_distance: int
    device: str

    def parameters(self) -> Mapping[str, object]:
        return MappingProxyType({k: getattr(self, k) for k in _REAL_TIME_KEYS})


@dataclass(frozen=True, slots=True)
class _GroundOptions:
    bond_dimension: int
    svd_cutoff: float
    imaginary_time_schedule: tuple[tuple[float, int], ...]
    measurement_method: str
    cluster_max_distance: int
    device: str

    def parameters(self) -> Mapping[str, object]:
        return MappingProxyType({k: getattr(self, k) for k in _GROUND_KEYS})


def _canonical(options, schema: tuple[str, ...], what: str) -> dict:
    if not isinstance(options, Mapping):
        raise GraphTNError(
            f"{what} must be a mapping with exactly {list(schema)}; got {type(options).__name__}."
        )
    have = set(options)
    want = set(schema)
    if have != want:
        raise GraphTNError(
            f"{what} must be exactly {list(schema)}; missing={sorted(want - have)}, "
            f"unknown={sorted(have - want)}."
        )
    return dict(options)


def _pos_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise GraphTNError(f"{name} must be a positive integer; got {value!r}.")
    return int(value)


def _real(value, name: str) -> float:
    if isinstance(value, bool) or isinstance(value, str) or (
        isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real)
    ):
        raise GraphTNError(f"{name} must be a real number; got {value!r}.")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise GraphTNError(f"{name} must be a real number; got {type(value).__name__}.")
    v = float(value)
    if not np.isfinite(v):
        raise GraphTNError(f"{name} must be finite; got {value!r}.")
    return v


def _positive_float(value, name: str) -> float:
    v = _real(value, name)
    if not v > 0.0:
        raise GraphTNError(f"{name} must be strictly positive; got {value!r}.")
    return v


def _cutoff(value, name: str) -> float:
    v = _real(value, name)
    if not (0.0 < v < 1.0):
        raise GraphTNError(f"{name} must satisfy 0 < {name} < 1; got {value!r}.")
    return v


def _measurement_method(value) -> str:
    if value not in _MEASUREMENT_METHODS:
        raise GraphTNError(f"measurement_method must be one of {_MEASUREMENT_METHODS}; got {value!r}.")
    return value


def _device(value) -> str:
    if value not in _DEVICES:
        raise GraphTNError(f"device must be 'cpu' or 'cuda'; got {value!r}.")
    return value


def _validate_schedule(schedule) -> tuple[tuple[float, int], ...]:
    if not isinstance(schedule, tuple):
        raise GraphTNError("imaginary_time_schedule must be a tuple of (tau, steps) tuples, not a list.")
    if len(schedule) == 0:
        raise GraphTNError("imaginary_time_schedule must be a nonempty tuple.")
    parsed: list[tuple[float, int]] = []
    prev_tau = None
    for entry in schedule:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise GraphTNError(f"each imaginary_time_schedule entry must be a 2-tuple (tau, steps); got {entry!r}.")
        tau = _positive_float(entry[0], "imaginary_time_schedule tau")
        if prev_tau is not None and not tau < prev_tau:
            raise GraphTNError("imaginary_time_schedule tau values must be strictly decreasing.")
        prev_tau = tau
        parsed.append((tau, _pos_int(entry[1], "imaginary_time_schedule steps")))
    return tuple(parsed)


def validate_real_time_options(options) -> _RealTimeOptions:
    """Validate the exactly-six real-time graph-PEPS keys."""
    o = _canonical(options, _REAL_TIME_KEYS, "backend='graph_peps' backend_options")
    return _RealTimeOptions(
        time_step_s=_positive_float(o["time_step_s"], "time_step_s"),
        bond_dimension=_pos_int(o["bond_dimension"], "bond_dimension"),
        svd_cutoff=_cutoff(o["svd_cutoff"], "svd_cutoff"),
        measurement_method=_measurement_method(o["measurement_method"]),
        cluster_max_distance=_pos_int(o["cluster_max_distance"], "cluster_max_distance"),
        device=_device(o["device"]),
    )


def validate_ground_options(options) -> _GroundOptions:
    """Validate the exactly-six ground-state graph-PEPS keys."""
    o = _canonical(options, _GROUND_KEYS, "method='graph_peps_imaginary_time' method_options")
    return _GroundOptions(
        bond_dimension=_pos_int(o["bond_dimension"], "bond_dimension"),
        svd_cutoff=_cutoff(o["svd_cutoff"], "svd_cutoff"),
        imaginary_time_schedule=_validate_schedule(o["imaginary_time_schedule"]),
        measurement_method=_measurement_method(o["measurement_method"]),
        cluster_max_distance=_pos_int(o["cluster_max_distance"], "cluster_max_distance"),
        device=_device(o["device"]),
    )
