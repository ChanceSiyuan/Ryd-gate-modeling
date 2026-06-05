"""Level-structure and interaction specifications for Rydberg systems.

Defines the small dataclasses that describe a site's local energy levels and
pairwise interactions, plus the built-in ``level_structure`` presets
(``1r`` / ``01r`` / ``ger`` / ``rb87_7``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

DEFAULT_C6 = 2 * np.pi * 874e9


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
    """Local energy-level structure for every site."""

    name: str
    levels: tuple[str, ...]
    rydberg_levels: tuple[str, ...]
    transitions: tuple[TransitionSpec, ...] = ()
    detuning_levels: dict[str, str] = field(default_factory=dict)

    @property
    def local_dim(self) -> int:
        return len(self.levels)

    def index(self, level: str) -> int:
        try:
            return self.levels.index(level)
        except ValueError:
            raise ValueError(f"Unknown level '{level}' for {self.name}: {self.levels}") from None


@dataclass(frozen=True)
class InteractionSpec:
    """Pairwise Rydberg interaction construction."""

    C6: float = DEFAULT_C6
    max_range_um: float | None = None
    mode: Literal["all", "nn", "nnn"] = "all"


def level_structure(name: str) -> LevelStructureSpec:
    """Return a built-in level-structure preset."""
    presets = {
        "1r": LevelStructureSpec(
            name="1r",
            levels=("1", "r"),
            rydberg_levels=("r",),
            transitions=(TransitionSpec("1_r", "1", "r", "global_X"),),
            detuning_levels={"global_n": "r"},
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
        ),
        "rb87_7": LevelStructureSpec(
            name="rb87_7",
            levels=("0", "1", "e1", "e2", "e3", "r", "r_garb"),
            rydberg_levels=("r", "r_garb"),
        ),
    }
    try:
        return presets[name]
    except KeyError:
        raise ValueError(f"Unknown level-structure preset '{name}'.") from None
