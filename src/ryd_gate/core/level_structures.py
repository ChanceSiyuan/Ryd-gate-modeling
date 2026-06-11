"""Level-structure and interaction specifications for Rydberg systems.

Defines the small dataclasses that describe a site's local energy levels and
pairwise interactions, plus the built-in ``level_structure`` presets
(``01`` / ``1r`` / ``01r`` / ``ger`` / ``analog_3`` / ``rb87_7``).

``LevelStructureSpec`` is both the compiler-facing level spec and the
user-facing atom model (there is no separate ``AtomModel`` class). Preset
*names* encode Hamiltonian construction semantics — ``ger`` is the symbolic
protocol-driven ladder while ``analog_3`` mounts the physical Rb87 analog
blocks — whereas ``param_set`` tags (``rb87_7``: ``our``/``lukin``) only
switch numerical sets within identical semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np

from ryd_gate.core.serialization import check_schema, json_ready, schema_tag
from ryd_gate.core.validation import ValidationIssue

DEFAULT_C6 = 2 * np.pi * 874e9

_INTERACTION_KINDS = ("none", "ising_c6", "xy_c3", "custom")

# Stage 1 backend support matrix (preset name -> capable backends).
_TN_CAPABLE = frozenset({"1r", "01r"})
_BACKEND_SUPPORT = {
    "exact": frozenset({"01", "1r", "01r", "ger", "analog_3", "rb87_7"}),
    "mps": _TN_CAPABLE,
    "gputn": _TN_CAPABLE,
    "peps": _TN_CAPABLE,
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
            ),
            detuning_levels={"delta_R": "r", "delta_hf": "1"},
            initial_level="1",
        ),
        "ger": LevelStructureSpec(
            name="ger",
            levels=("g", "e", "r"),
            rydberg_levels=("r",),
            transitions=(
                TransitionSpec("420", "g", "e", "drive_420"),
                TransitionSpec("1013", "e", "r", "H_1013"),
            ),
            detuning_levels={"delta_e": "e", "delta_R": "r"},
            initial_level="g",
        ),
        # Physical Rb87 ladder: same topology/channels as `ger`, different
        # Hamiltonian construction (analog blocks with static H_1013). A
        # separate *name*, not a param_set tag — see stageplans/README D11.
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
