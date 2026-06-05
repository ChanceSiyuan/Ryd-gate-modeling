"""Tensor-network level-structure normalization shared by TN algorithms."""

from __future__ import annotations

from ryd_gate.core.level_structures import LevelStructureSpec, TransitionSpec
from ryd_gate.core.level_structures import level_structure as core_level_structure


def resolve_level_structure(level_structure: str | LevelStructureSpec) -> LevelStructureSpec:
    """Resolve and validate a level-structure for current TN lowerings."""
    if isinstance(level_structure, LevelStructureSpec):
        spec = level_structure
    else:
        try:
            spec = core_level_structure(level_structure)
        except ValueError as exc:
            raise ValueError(
                f"Unknown TN level_structure {level_structure!r}; "
                "use a registered ryd_gate.core.level_structures.level_structure preset."
            ) from exc
    validate_tn_level_structure(spec)
    return spec


def validate_tn_level_structure(spec: LevelStructureSpec) -> None:
    """Validate that a central level spec can currently be lowered to TN IR."""
    if spec.name == "1r" and spec.levels == ("1", "r") and spec.rydberg_levels == ("r",):
        return
    if spec.name == "01r" and spec.levels == ("0", "1", "r") and spec.rydberg_levels == ("r",):
        require_transition(spec, "1", "r")
        require_transition(spec, "0", "1")
        return
    raise ValueError(
        f"TN level_structure {spec.name!r} is registered but not supported by the "
        "current TN lowering. Supported presets are '1r' and '01r'."
    )


def local_levels(level_structure: str | LevelStructureSpec) -> tuple[str, ...]:
    """Return local level labels from the shared central level-structure registry."""
    return resolve_level_structure(level_structure).levels


def require_transition(spec: LevelStructureSpec, lower: str, upper: str) -> TransitionSpec:
    """Return a transition spec or raise with a TN-specific message."""
    for transition in spec.transitions:
        if transition.lower == lower and transition.upper == upper:
            return transition
    raise ValueError(
        f"TN level_structure {spec.name!r} needs transition |{lower}> <-> |{upper}>."
    )
