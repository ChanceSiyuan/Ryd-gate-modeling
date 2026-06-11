"""Control channels: typed channel specs and compiler channel-id constants.

Two layers live here:

- :class:`ChannelSpec` — the product-facing description of a physical control
  channel and its constraints (used by ``DeviceSpec`` and pulse validation).
  Its ``amplitude_channels`` / ``detuning_channels`` maps carry the lowering
  contract from product channel to the internal compiler channel ids below.
- String constants — the canonical compiler channel ids used by
  ``get_drive_coefficients()`` to label time-dependent Hamiltonian terms.
  They are implementation constants, not product API.

Compiler channel conventions:

Two-atom 7-level CZ gates (TO/AR protocols):
    "drive_420"         -> H_420 coupling operator
    "drive_420_dag"     -> H_420 hermitian conjugate
    "lightshift_zero"   -> dark-state light-shift operator

Two-atom 3-level sweep (analog system):
    "drive_420"         -> H_420 coupling
    "drive_420_dag"     -> H_420 hermitian conjugate
    "lightshift_zero"   -> (zero matrix, included for interface uniformity)

N-atom 2-level lattice sweep:
    "global_X"          -> sum_i sigma^x_i (transverse field)
    "global_n"          -> sum_i n_i (detuning / longitudinal field)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from ryd_gate.core.serialization import check_schema, schema_tag

# Channel name constants for type safety
DRIVE_420 = "drive_420"
DRIVE_420_DAG = "drive_420_dag"
LIGHTSHIFT_ZERO = "lightshift_zero"
GLOBAL_X = "global_X"
GLOBAL_N = "global_n"

_CHANNEL_KINDS = ("rydberg", "raman", "microwave", "dmm", "custom")
_ADDRESSING = ("global", "local")


@dataclass(frozen=True)
class ChannelSpec:
    """A physical control channel and its hardware constraints.

    Does not compile pulses or sample waveforms; it is the constraint object
    that ``DeviceSpec.validate_pulse`` / ``Pulse.validate`` check against.
    """

    channel_id: str
    kind: Literal["rydberg", "raman", "microwave", "dmm", "custom"]
    transition: str
    addressing: Literal["global", "local"]
    amplitude_channels: Mapping[str, str] = field(default_factory=dict)
    detuning_channels: Mapping[str, str] = field(default_factory=dict)
    max_abs_amplitude_rad_per_us: float | None = None
    max_abs_detuning_rad_per_us: float | None = None
    min_duration_ns: int = 0
    max_duration_ns: int | None = None
    clock_period_ns: int = 1
    max_targets: int | None = None
    retarget_time_ns: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.channel_id, str) or not self.channel_id:
            raise ValueError("channel_id must be a non-empty string.")
        if self.kind not in _CHANNEL_KINDS:
            raise ValueError(f"kind must be one of {_CHANNEL_KINDS}, got {self.kind!r}.")
        if not isinstance(self.transition, str) or not self.transition:
            raise ValueError("transition must be a non-empty string.")
        if self.addressing not in _ADDRESSING:
            raise ValueError(f"addressing must be 'global' or 'local', got {self.addressing!r}.")
        if not isinstance(self.min_duration_ns, int) or isinstance(self.min_duration_ns, bool) or self.min_duration_ns < 0:
            raise ValueError(
                f"min_duration_ns must be a nonnegative integer, got {self.min_duration_ns!r}."
            )
        if self.max_duration_ns is not None and (
            not isinstance(self.max_duration_ns, int)
            or self.max_duration_ns < self.min_duration_ns
        ):
            raise ValueError(
                f"max_duration_ns must be None or an integer >= min_duration_ns, got {self.max_duration_ns!r}."
            )
        if not isinstance(self.clock_period_ns, int) or isinstance(self.clock_period_ns, bool) or self.clock_period_ns <= 0:
            raise ValueError(
                f"clock_period_ns must be a positive integer, got {self.clock_period_ns!r}."
            )
        for name in ("max_abs_amplitude_rad_per_us", "max_abs_detuning_rad_per_us"):
            limit = getattr(self, name)
            if limit is not None and not (float(limit) > 0):
                raise ValueError(f"{name} must be None or positive, got {limit!r}.")

    def to_dict(self) -> dict:
        return {
            "schema": schema_tag("channel"),
            "channel_id": self.channel_id,
            "kind": self.kind,
            "transition": self.transition,
            "addressing": self.addressing,
            "amplitude_channels": dict(self.amplitude_channels),
            "detuning_channels": dict(self.detuning_channels),
            "max_abs_amplitude_rad_per_us": self.max_abs_amplitude_rad_per_us,
            "max_abs_detuning_rad_per_us": self.max_abs_detuning_rad_per_us,
            "min_duration_ns": self.min_duration_ns,
            "max_duration_ns": self.max_duration_ns,
            "clock_period_ns": self.clock_period_ns,
            "max_targets": self.max_targets,
            "retarget_time_ns": self.retarget_time_ns,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChannelSpec":
        check_schema(data, "channel")
        return cls(
            channel_id=data["channel_id"],
            kind=data["kind"],
            transition=data["transition"],
            addressing=data["addressing"],
            amplitude_channels=dict(data.get("amplitude_channels", {})),
            detuning_channels=dict(data.get("detuning_channels", {})),
            max_abs_amplitude_rad_per_us=data.get("max_abs_amplitude_rad_per_us"),
            max_abs_detuning_rad_per_us=data.get("max_abs_detuning_rad_per_us"),
            min_duration_ns=data.get("min_duration_ns", 0),
            max_duration_ns=data.get("max_duration_ns"),
            clock_period_ns=data.get("clock_period_ns", 1),
            max_targets=data.get("max_targets"),
            retarget_time_ns=data.get("retarget_time_ns"),
        )
