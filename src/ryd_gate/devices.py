"""Device specifications: hardware constraints as frozen, validating data.

A :class:`DeviceSpec` *is* the validator — geometry limits, allowed atom
models, channel constraints — and never holds backend or job state. The
default permissive device for this repo's Rb87 models is
``DeviceSpec.virtual_rb87()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np

from ryd_gate.core.level_structures import DEFAULT_C6, LevelStructureSpec, level_structure
from ryd_gate.core.serialization import (
    ValidationIssue,
    check_schema,
    json_ready,
    schema_tag,
)
from ryd_gate.lattice import Register
from ryd_gate.protocols.channels import ChannelSpec
from ryd_gate.pulse import Pulse, _channel_limit_issues

__all__ = ["DeviceSpec"]


@dataclass(frozen=True)
class DeviceSpec:
    """Hardware and physical constraints for a neutral-atom device-like target."""

    name: str
    dimensions: Literal[2, 3]
    atom_species: str
    allowed_level_structures: tuple[str, ...]
    default_level_structure: str
    min_atom_distance_um: float
    max_atom_num: int | None = None
    max_radial_distance_um: float | None = None
    interaction_coeffs: Mapping[str, float] = field(default_factory=dict)
    channels: Mapping[str, ChannelSpec] = field(default_factory=dict)
    supports_slm_mask: bool = False
    max_sequence_duration_ns: int | None = None
    max_runs: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("DeviceSpec.name must be a non-empty string.")
        if self.dimensions not in (2, 3):
            raise ValueError(f"dimensions must be 2 or 3, got {self.dimensions!r}.")
        if self.default_level_structure not in self.allowed_level_structures:
            raise ValueError(
                f"default_level_structure {self.default_level_structure!r} is not in "
                f"allowed_level_structures {self.allowed_level_structures}."
            )
        if not (float(self.min_atom_distance_um) >= 0):
            raise ValueError(
                f"min_atom_distance_um must be nonnegative, got {self.min_atom_distance_um!r}."
            )

    # ── Built-in devices ────────────────────────────────────────────────

    @classmethod
    def virtual_rb87(cls) -> "DeviceSpec":
        """Default permissive Rb87 virtual device (no atom-count/radius caps).

        Channel maps carry the lowering contract from product channels to the
        internal compiler channel ids (consumed by Stage 2's sequence path).
        """
        channels = {
            "rydberg_global": ChannelSpec(
                channel_id="rydberg_global",
                kind="rydberg",
                transition="1_r",
                addressing="global",
                amplitude_channels={"1r": "global_X", "01r": "drive_R"},
                detuning_channels={"1r": "global_n", "01r": "delta_R"},
            ),
            "rydberg_local": ChannelSpec(
                channel_id="rydberg_local",
                kind="rydberg",
                transition="1_r",
                addressing="local",
                amplitude_channels={"1r": "global_X", "01r": "drive_R"},
                detuning_channels={"1r": "global_n", "01r": "delta_R"},
            ),
            "hyperfine_global": ChannelSpec(
                channel_id="hyperfine_global",
                kind="microwave",
                transition="0_1",
                addressing="global",
                amplitude_channels={"01r": "drive_hf"},
                detuning_channels={"01r": "delta_hf"},
            ),
        }
        return cls(
            name="virtual_rb87",
            dimensions=2,
            atom_species="Rb87",
            allowed_level_structures=("01", "1r", "01r", "ger", "analog_3", "rb87_7"),
            default_level_structure="01r",
            min_atom_distance_um=2.0,
            interaction_coeffs={"C6_rad_s_um6": DEFAULT_C6},
            channels=channels,
        )

    # ── Validation (returns issues; never raises) ───────────────────────

    def validate_register(self, register: Register) -> list[ValidationIssue]:
        """Geometry vs device: dimensions, atom count, pair distances, radius."""
        issues: list[ValidationIssue] = []
        if register.dimensions != self.dimensions:
            issues.append(ValidationIssue(
                "error", "register.dimensions",
                f"register is {register.dimensions}D but device {self.name} is "
                f"{self.dimensions}D.",
                ("register", "coords"),
            ))
        if self.max_atom_num is not None and register.n_atoms > self.max_atom_num:
            issues.append(ValidationIssue(
                "error", "register.max_atom_num",
                f"register has {register.n_atoms} atoms; device {self.name} allows "
                f"at most {self.max_atom_num}.",
                ("register", "ids"),
            ))
        if register.n_atoms >= 2:
            min_distance = min(d for _, _, d in register.distance_pairs())
            if min_distance < self.min_atom_distance_um:
                issues.append(ValidationIssue(
                    "error", "register.min_distance",
                    f"minimum pair distance {min_distance:.6g} um is below the device "
                    f"minimum {self.min_atom_distance_um:.6g} um.",
                    ("register", "coords"),
                ))
        if self.max_radial_distance_um is not None:
            radial = float(np.max(np.linalg.norm(register.coords, axis=1)))
            if radial > self.max_radial_distance_um:
                issues.append(ValidationIssue(
                    "error", "register.max_radial_distance",
                    f"maximum radial distance {radial:.6g} um exceeds the device "
                    f"limit {self.max_radial_distance_um:.6g} um.",
                    ("register", "coords"),
                ))
        return issues

    def validate_level_structure(
        self, spec: LevelStructureSpec | str
    ) -> list[ValidationIssue]:
        """Model vs device: allowed-list membership and species match."""
        if isinstance(spec, LevelStructureSpec):
            name, resolved = spec.name, spec
        else:
            name, resolved = spec, None
        if name not in self.allowed_level_structures:
            return [ValidationIssue(
                "error", "level_structure.unsupported",
                f"level structure {name!r} is not supported by device {self.name} "
                f"(allowed: {', '.join(self.allowed_level_structures)}).",
                ("level_structure", "name"),
            )]
        if resolved is None:
            try:
                resolved = level_structure(name)
            except ValueError:
                return []  # allowed by name but not a known preset: nothing more to check
        if resolved.species != self.atom_species:
            return [ValidationIssue(
                "error", "level_structure.species",
                f"level structure {name!r} is for species {resolved.species!r} but "
                f"device {self.name} hosts {self.atom_species!r}.",
                ("level_structure", "species"),
            )]
        return []

    def validate_pulse(self, pulse: Pulse, channel_id: str) -> list[ValidationIssue]:
        """Pulse vs one named channel's limits (shared rule set with Pulse.validate)."""
        channel = self.channels.get(channel_id)
        if channel is None:
            return [ValidationIssue(
                "error", "channel.unknown",
                f"unknown channel {channel_id!r} on device {self.name} "
                f"(available: {', '.join(self.channels) or 'none'}).",
                ("device", "channels"),
            )]
        return _channel_limit_issues(pulse, channel)

    # ── Physics helpers ─────────────────────────────────────────────────

    def rydberg_blockade_radius_um(self, rabi_rad_per_us: float) -> float:
        """Blockade radius ``(C6 / Omega)^(1/6)`` in micrometers."""
        c6 = self.interaction_coeffs.get("C6_rad_s_um6")
        if c6 is None:
            raise ValueError(
                f"device {self.name} has no 'C6_rad_s_um6' interaction coefficient."
            )
        rabi = float(rabi_rad_per_us)
        if not np.isfinite(rabi) or rabi <= 0:
            raise ValueError(f"rabi_rad_per_us must be positive, got {rabi_rad_per_us!r}.")
        return float(((c6 * 1e-6) / rabi) ** (1.0 / 6.0))

    def describe(self) -> str:
        """Human-readable constraint sheet (never raises)."""
        lines = [
            f"Device: {self.name}",
            f"  dimensions: {self.dimensions}",
            f"  species: {self.atom_species}",
            f"  level structures: {', '.join(self.allowed_level_structures)} "
            f"(default: {self.default_level_structure})",
            f"  min atom distance: {self.min_atom_distance_um} um",
        ]
        if self.max_atom_num is not None:
            lines.append(f"  max atoms: {self.max_atom_num}")
        if self.max_radial_distance_um is not None:
            lines.append(f"  max radial distance: {self.max_radial_distance_um} um")
        if self.max_sequence_duration_ns is not None:
            lines.append(f"  max sequence duration: {self.max_sequence_duration_ns} ns")
        if self.max_runs is not None:
            lines.append(f"  max runs: {self.max_runs}")
        for key, value in self.interaction_coeffs.items():
            lines.append(f"  {key}: {value:.6g}")
        lines.append(f"  SLM mask: {'yes' if self.supports_slm_mask else 'no'}")
        lines.append("  channels:")
        for channel_id, channel in self.channels.items():
            lines.append(
                f"    {channel_id}: kind={channel.kind}, transition={channel.transition}, "
                f"addressing={channel.addressing}"
            )
        if not self.channels:
            lines.append("    (none)")
        return "\n".join(lines)

    # ── Serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "schema": schema_tag("device"),
            "name": self.name,
            "dimensions": self.dimensions,
            "atom_species": self.atom_species,
            "allowed_level_structures": list(self.allowed_level_structures),
            "default_level_structure": self.default_level_structure,
            "min_atom_distance_um": float(self.min_atom_distance_um),
            "max_atom_num": self.max_atom_num,
            "max_radial_distance_um": self.max_radial_distance_um,
            "interaction_coeffs": {k: float(v) for k, v in self.interaction_coeffs.items()},
            "channels": {cid: ch.to_dict() for cid, ch in self.channels.items()},
            "supports_slm_mask": self.supports_slm_mask,
            "max_sequence_duration_ns": self.max_sequence_duration_ns,
            "max_runs": self.max_runs,
            "metadata": json_ready(dict(self.metadata), "device.metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeviceSpec":
        check_schema(data, "device")
        return cls(
            name=data["name"],
            dimensions=data["dimensions"],
            atom_species=data["atom_species"],
            allowed_level_structures=tuple(data["allowed_level_structures"]),
            default_level_structure=data["default_level_structure"],
            min_atom_distance_um=data["min_atom_distance_um"],
            max_atom_num=data.get("max_atom_num"),
            max_radial_distance_um=data.get("max_radial_distance_um"),
            interaction_coeffs=dict(data.get("interaction_coeffs", {})),
            channels={
                cid: ChannelSpec.from_dict(ch)
                for cid, ch in data.get("channels", {}).items()
            },
            supports_slm_mask=data.get("supports_slm_mask", False),
            max_sequence_duration_ns=data.get("max_sequence_duration_ns"),
            max_runs=data.get("max_runs"),
            metadata=dict(data.get("metadata", {})),
        )
