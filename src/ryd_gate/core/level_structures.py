"""Level-structure and interaction specifications for Rydberg systems.

Defines the small dataclasses that describe a site's local energy levels and
pairwise interactions, plus the built-in ``level_structure`` presets
(``01`` / ``1r`` / ``01r`` / ``analog_3`` / ``rb87_7``).

``LevelStructureSpec`` is both the compiler-facing level spec and the
user-facing atom model (there is no separate ``AtomModel`` class). Preset
*names* encode Hamiltonian construction semantics — ``analog_3`` mounts the
physical Rb87 analog blocks — whereas ``param_set`` tags (``rb87_7``:
``our``/``lukin``) only switch numerical sets within identical semantics.
Fully custom (symbolic) models are hand-built ``LevelStructureSpec``
instances passed straight to ``RydbergSystem.set_atom_level``.

Also hosts the shared lowering helpers for protocol channels: the exact
sparse compiler and TN backends both consume protocol coefficients, but they
materialize them into different representations.  Keeping the channel naming
rules in one place keeps per-site addressing, custom level specs, and
Hermitian-conjugate handling consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np

from ryd_gate.core.serialization import (
    ValidationIssue,
    check_schema,
    json_ready,
    schema_tag,
)

DEFAULT_C6 = 2 * np.pi * 874e9

_INTERACTION_KINDS = ("none", "ising_c6", "xy_c3", "custom")

# Backend support matrix (preset name -> capable backends).
_TN_CAPABLE = frozenset({"1r", "01r"})
# analog_3 (physical g/e/r ladder) is lowered to TN by both the YASTN PEPS engine
# (backend="peps") and the TeNPy MPS path (backend="mps").
_TN_ANALOG = _TN_CAPABLE | {"analog_3"}
_BACKEND_SUPPORT = {
    "exact": frozenset({"01", "1r", "01r", "analog_3", "rb87_7"}),
    "mps": _TN_ANALOG,
    "peps": _TN_ANALOG,
    "stabilizer": frozenset({"01"}),
}


@dataclass(frozen=True)
class TransitionSpec:
    """Single-site transition block definition.

    ``operator`` is ``|upper><lower|``.  Protocols decide whether to add
    the Hermitian conjugate through the compiler channel map.
    """

    name: str
    lower: str
    upper: str
    channel: str


@dataclass(frozen=True)
class LevelStructureSpec:
    """Local energy-level structure (the atom model) for every site."""

    name: str
    levels: tuple[str, ...]
    rydberg_levels: tuple[str, ...]
    transitions: tuple[TransitionSpec, ...] = ()
    detuning_levels: dict[str, str] = field(default_factory=dict)
    initial_level: str | None = None
    species: str = "Rb87"
    interaction_kind: Literal["none", "ising_c6", "xy_c3", "custom"] = "ising_c6"
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("LevelStructureSpec.name must be a non-empty string.")

    @property
    def local_dim(self) -> int:
        return len(self.levels)

    def index(self, level: str) -> int:
        try:
            return self.levels.index(level)
        except ValueError:
            raise ValueError(f"Unknown level '{level}' for {self.name}: {self.levels}") from None

    def initial_level_or_default(self) -> str:
        """The level each atom starts in by default: ``initial_level`` or ``levels[0]``."""
        return self.initial_level if self.initial_level is not None else self.levels[0]

    def physical_kwargs(self) -> dict[str, Any]:
        """Keyword arguments the system factory needs for this model.

        ``analog_3`` and ``rb87_7`` carry a ``param_set``; other presets need
        nothing. Extra entries in ``params`` pass through unchanged.
        """
        extra = {k: v for k, v in self.params.items() if k != "param_set"}
        if self.name == "analog_3":
            return {"param_set": "analog_3", **extra}
        if self.name == "rb87_7":
            return {"param_set": self.params.get("param_set", "our"), **extra}
        return {}

    def supports_backend(self, backend: str) -> bool:
        """Whether *backend* can simulate this model (Stage 1 truth table)."""
        allowed = _BACKEND_SUPPORT.get(backend)
        if allowed is None:
            return False
        return self.name in allowed

    def validate(self) -> list[ValidationIssue]:
        """Semantic self-check; structural problems raise at construction instead."""
        issues: list[ValidationIssue] = []
        if not self.levels:
            issues.append(ValidationIssue(
                "error", "level_structure.empty_levels",
                "levels must not be empty.", ("level_structure", "levels"),
            ))
        if len(set(self.levels)) != len(self.levels):
            issues.append(ValidationIssue(
                "error", "level_structure.duplicate_levels",
                f"levels contain duplicates: {self.levels}.", ("level_structure", "levels"),
            ))
        if self.initial_level is not None and self.initial_level not in self.levels:
            issues.append(ValidationIssue(
                "error", "level_structure.initial_level",
                f"initial_level {self.initial_level!r} is not in levels {self.levels}.",
                ("level_structure", "initial_level"),
            ))
        for level in self.rydberg_levels:
            if level not in self.levels:
                issues.append(ValidationIssue(
                    "error", "level_structure.rydberg_levels",
                    f"rydberg level {level!r} is not in levels {self.levels}.",
                    ("level_structure", "rydberg_levels"),
                ))
        for channel, level in self.detuning_levels.items():
            if level not in self.levels:
                issues.append(ValidationIssue(
                    "error", "level_structure.detuning_levels",
                    f"detuning channel {channel!r} targets unknown level {level!r}.",
                    ("level_structure", "detuning_levels"),
                ))
        if self.interaction_kind not in _INTERACTION_KINDS:
            issues.append(ValidationIssue(
                "error", "level_structure.interaction_kind",
                f"interaction_kind must be one of {_INTERACTION_KINDS}, got {self.interaction_kind!r}.",
                ("level_structure", "interaction_kind"),
            ))
        return issues

    def to_dict(self) -> dict:
        return {
            "schema": schema_tag("level-structure"),
            "name": self.name,
            "levels": list(self.levels),
            "rydberg_levels": list(self.rydberg_levels),
            "transitions": [
                {"name": t.name, "lower": t.lower, "upper": t.upper, "channel": t.channel}
                for t in self.transitions
            ],
            "detuning_levels": dict(self.detuning_levels),
            "initial_level": self.initial_level,
            "species": self.species,
            "interaction_kind": self.interaction_kind,
            "params": json_ready(dict(self.params), "level_structure.params"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LevelStructureSpec":
        check_schema(data, "level-structure")
        return cls(
            name=data["name"],
            levels=tuple(data["levels"]),
            rydberg_levels=tuple(data["rydberg_levels"]),
            transitions=tuple(
                TransitionSpec(t["name"], t["lower"], t["upper"], t["channel"])
                for t in data.get("transitions", [])
            ),
            detuning_levels=dict(data.get("detuning_levels", {})),
            initial_level=data.get("initial_level"),
            species=data.get("species", "Rb87"),
            interaction_kind=data.get("interaction_kind", "ising_c6"),
            params=dict(data.get("params", {})),
        )


@dataclass(frozen=True)
class InteractionSpec:
    """Pairwise Rydberg interaction construction."""

    C6: float = DEFAULT_C6
    max_range_um: float | None = None
    mode: Literal["all", "nn", "nnn"] = "all"


def level_structure(name: str) -> LevelStructureSpec:
    """Return a built-in level-structure preset."""
    presets = {
        "01": LevelStructureSpec(
            name="01",
            levels=("0", "1"),
            rydberg_levels=(),
            transitions=(),
            detuning_levels={},
            initial_level="0",
            interaction_kind="none",
        ),
        "1r": LevelStructureSpec(
            name="1r",
            levels=("1", "r"),
            rydberg_levels=("r",),
            transitions=(TransitionSpec("1_r", "1", "r", "global_X"),),
            detuning_levels={"global_n": "r"},
            initial_level="1",
        ),
        "01r": LevelStructureSpec(
            name="01r",
            levels=("0", "1", "r"),
            rydberg_levels=("r",),
            transitions=(
                TransitionSpec("R", "1", "r", "drive_R"),
                TransitionSpec("hf", "0", "1", "drive_hf"),
                # |0>-|r> (K0r) leg of the full effective CZ Hamiltonian.  Exact
                # backend only; the TN 01r lowering rejects a nonzero drive_0r.
                TransitionSpec("0r", "0", "r", "drive_0r"),
            ),
            detuning_levels={"delta_R": "r", "delta_hf": "1"},
            initial_level="1",
        ),
        # Physical Rb87 three-level ladder: analog blocks with static H_1013.
        # The *name* mounts the physics (stageplans/README D11/D13); symbolic
        # three-level models are hand-built LevelStructureSpec instances.
        "analog_3": LevelStructureSpec(
            name="analog_3",
            levels=("g", "e", "r"),
            rydberg_levels=("r",),
            transitions=(
                TransitionSpec("420", "g", "e", "drive_420"),
                TransitionSpec("1013", "e", "r", "H_1013"),
            ),
            detuning_levels={"delta_e": "e", "delta_R": "r"},
            initial_level="g",
            params={"param_set": "analog_3"},
        ),
        "rb87_7": LevelStructureSpec(
            name="rb87_7",
            levels=("0", "1", "e1", "e2", "e3", "r", "r_garb"),
            rydberg_levels=("r", "r_garb"),
            initial_level="0",
            params={"param_set": "our"},
        ),
    }
    try:
        return presets[name]
    except KeyError:
        raise ValueError(f"Unknown level-structure preset '{name}'.") from None


# ── Protocol-channel lowering helpers ────────────────────────────────────────

_HC_CHANNELS = frozenset({"global_X", "drive_R", "drive_hf", "drive_0r"})
_HC_CHANNEL_PREFIXES = ("global_X_", "drive_R_", "drive_hf_", "drive_0r_")


def split_site_channel(channel: str) -> tuple[str, int | None]:
    """Return ``(base_channel, site_index)`` for ``channel_i`` names."""
    base, sep, suffix = channel.rpartition("_")
    if sep and suffix.isdigit():
        return base, int(suffix)
    return channel, None


def declared_channels(level_spec: LevelStructureSpec) -> set[str]:
    """Return channels declared by a central level structure."""
    channels = {transition.channel for transition in level_spec.transitions}
    channels.update(level_spec.detuning_levels)
    return channels


def transition_channels(level_spec: LevelStructureSpec | None) -> set[str]:
    """Return non-Hermitian transition channels declared by ``level_spec``."""
    if level_spec is None:
        return set()
    return {transition.channel for transition in level_spec.transitions}


def validate_coeff_channels(
    coeffs: dict[str, complex],
    level_spec: LevelStructureSpec,
    *,
    level_structure_name: str | None = None,
) -> None:
    """Reject coefficients whose base channel is not in ``level_spec``."""
    declared = declared_channels(level_spec)
    unknown = []
    for channel in coeffs:
        base, _ = split_site_channel(channel)
        if base not in declared:
            unknown.append(channel)
    if unknown:
        name = level_structure_name or level_spec.name
        raise ValueError(
            f"Protocol emitted channel(s) not declared by level_structure "
            f"{name!r}: {sorted(unknown)}."
        )


def transition_channel(level_spec: LevelStructureSpec, lower: str, upper: str) -> str | None:
    """Find the channel for a declared ``|lower> <-> |upper>`` transition."""
    for transition in level_spec.transitions:
        if transition.lower == lower and transition.upper == upper:
            return transition.channel
    return None


def detuning_channel(level_spec: LevelStructureSpec, level: str) -> str | None:
    """Find the detuning channel for a declared level projector."""
    for channel, detuned_level in level_spec.detuning_levels.items():
        if detuned_level == level:
            return channel
    return None


def channel_needs_hermitian_conjugate(
    channel: str,
    level_spec: LevelStructureSpec | None = None,
) -> bool:
    """Return True when ``channel`` lowers to a non-Hermitian transition."""
    if channel in _HC_CHANNELS:
        return True
    if any(channel.startswith(prefix) for prefix in _HC_CHANNEL_PREFIXES):
        return True
    base, _ = split_site_channel(channel)
    return base in transition_channels(level_spec)


def block_name_for_drive_channel(system: Any, channel: str) -> str | None:
    """Map a protocol channel to a registered exact-compiler block."""
    if system.blocks.has(channel):
        return channel

    base, site = split_site_channel(channel)
    if site is None:
        return None

    level_spec = system.meta("level_spec", None) if hasattr(system, "meta") else None
    if isinstance(level_spec, LevelStructureSpec) and base in level_spec.detuning_levels:
        level = level_spec.detuning_levels[base]
        block = f"n_{level}_{site}"
        return block if system.blocks.has(block) else None

    return None


def site_profile_from_coeffs(
    coeffs: dict[str, complex],
    channel: str,
    n_sites: int,
    *,
    scale: float,
) -> np.ndarray | None:
    """Return a per-site profile if ``coeffs`` contains site-specific keys."""
    keys = [f"{channel}_{i}" for i in range(n_sites)]
    if not any(key in coeffs for key in keys):
        return None
    return np.array([scale * float(np.real(coeffs.get(key, 0.0))) for key in keys])


def channel_profile_from_coeffs(
    coeffs: dict[str, complex],
    channel: str,
    n_sites: int,
    *,
    scale: float,
) -> np.ndarray:
    """Return a per-site profile from either global or site-specific coeffs."""
    site_profile = site_profile_from_coeffs(coeffs, channel, n_sites, scale=scale)
    global_value = scale * float(np.real(coeffs.get(channel, 0.0)))
    if site_profile is not None:
        return global_value + site_profile
    return np.full(n_sites, global_value)


def profile_for_optional_channel(
    coeffs: dict[str, complex],
    channel: str | None,
    n_sites: int,
    *,
    scale: float,
) -> np.ndarray:
    """Return zeros when an optional channel is absent from a level spec."""
    if channel is None:
        return np.zeros(n_sites)
    return channel_profile_from_coeffs(coeffs, channel, n_sites, scale=scale)


def three_level_profiles_from_coeffs(
    coeffs: dict[str, complex],
    spec: Any,
) -> dict[str, np.ndarray]:
    """Map protocol coefficients to explicit 01r-style per-site profiles."""
    level_spec = spec.level_spec
    validate_coeff_channels(
        coeffs,
        level_spec,
        level_structure_name=getattr(spec, "level_structure", level_spec.name),
    )
    drive_0r = transition_channel(level_spec, "0", "r")
    if drive_0r is not None and any(
        abs(complex(v)) > 0
        for k, v in coeffs.items()
        if split_site_channel(k)[0] == drive_0r
    ):
        raise ValueError(
            "TN backends do not support the |0>-|r> (K0r) coupling; "
            "use the exact backend (e.g. EffectiveCZProtocol from "
            "lower_cz_to_effective_01r runs exact-only)."
        )
    drive_r = transition_channel(level_spec, "1", "r")
    drive_hf = transition_channel(level_spec, "0", "1")
    delta_r = detuning_channel(level_spec, "r")
    delta_hf = detuning_channel(level_spec, "1")

    return {
        "omega_R": profile_for_optional_channel(coeffs, drive_r, spec.N, scale=2.0),
        "omega_hf": profile_for_optional_channel(coeffs, drive_hf, spec.N, scale=2.0),
        "delta_R": profile_for_optional_channel(coeffs, delta_r, spec.N, scale=-1.0),
        "delta_hf": profile_for_optional_channel(coeffs, delta_hf, spec.N, scale=-1.0),
    }


def split_uniform_profile(profile: np.ndarray) -> tuple[float, np.ndarray | None]:
    """Return ``(uniform_value, None)`` or ``(mean, nonuniform_offsets)``."""
    if np.allclose(profile, profile[0]):
        return float(profile[0]), None
    mean = float(np.mean(profile))
    return mean, profile - mean


def two_level_drive_and_detuning_from_coeffs(
    coeffs: dict[str, complex],
    spec: Any,
) -> tuple[float | np.ndarray, float, np.ndarray | None]:
    """Map protocol channels onto the effective TN 1/r Hamiltonian."""
    profiles = three_level_profiles_from_coeffs(coeffs, spec)

    if np.any(np.abs(profiles["omega_hf"]) > 0):
        raise ValueError(
            "TN TDVP supports the |1>-|r> two-level subspace only; "
            "DigitalAnalogProtocol segments with omega_hf != 0 are not supported."
        )

    omega_profile = profiles["omega_R"]
    if np.allclose(omega_profile, omega_profile[0]):
        Omega_t: float | np.ndarray = float(omega_profile[0])
    else:
        Omega_t = omega_profile

    # 01r -> effective 1r mapping:
    # -Delta_R n_r - Delta_hf n_1 = const - (Delta_R - Delta_hf) n_r.
    Delta_t, delta_profile = split_uniform_profile(
        profiles["delta_R"] - profiles["delta_hf"]
    )
    return Omega_t, Delta_t, delta_profile
