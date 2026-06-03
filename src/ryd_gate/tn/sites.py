"""Local TeNPy site builders for TN simulations."""

from __future__ import annotations

import numpy as np

from ryd_gate.core.rydberg_system import LevelStructureSpec, TransitionSpec
from ryd_gate.core.rydberg_system import level_structure as core_level_structure


def _require_tenpy():
    try:
        import tenpy

        return tenpy
    except ImportError as exc:
        raise ImportError(
            "TeNPy is required for tensor network simulations. "
            "Install via: pip install physics-tenpy  "
            "or: pip install ryd-gate[tn]"
        ) from exc


def resolve_level_structure(level_structure: str | LevelStructureSpec) -> LevelStructureSpec:
    """Resolve a level-structure name through the exact-side central registry."""
    if isinstance(level_structure, LevelStructureSpec):
        spec = level_structure
    else:
        try:
            spec = core_level_structure(level_structure)
        except ValueError as exc:
            raise ValueError(
                f"Unknown TN level_structure {level_structure!r}; "
                "use a registered ryd_gate.core.rydberg_system.level_structure preset."
            ) from exc
    validate_tn_level_structure(spec)
    return spec


def validate_tn_level_structure(spec: LevelStructureSpec) -> None:
    """Validate that a central level spec can currently be lowered to the TN path."""
    if spec.name == "1r" and spec.levels == ("1", "r") and spec.rydberg_levels == ("r",):
        return
    if spec.name == "01r" and spec.levels == ("0", "1", "r") and spec.rydberg_levels == ("r",):
        _require_transition(spec, "1", "r")
        _require_transition(spec, "0", "1")
        return
    raise ValueError(
        f"TN level_structure {spec.name!r} is registered but not supported by the "
        "current TN MPO lowering. Supported presets are '1r' and '01r'."
    )


def local_levels(level_structure: str | LevelStructureSpec) -> tuple[str, ...]:
    """Return local level labels from the shared central level-structure registry."""
    return resolve_level_structure(level_structure).levels


def build_tenpy_site(level_structure: str | LevelStructureSpec) -> object:
    """Build the TeNPy local site for a shared :class:`LevelStructureSpec`."""
    _require_tenpy()
    spec = resolve_level_structure(level_structure)
    if spec.name == "1r":
        from tenpy.networks.site import SpinHalfSite

        return SpinHalfSite(conserve=None)
    return _build_explicit_site(spec)


def transition_x_op_name(spec: LevelStructureSpec, lower: str, upper: str) -> str:
    """Return the Hermitian transition operator name for a central transition."""
    transition = _require_transition(spec, lower, upper)
    return f"X_{transition.name}"


def _require_transition(spec: LevelStructureSpec, lower: str, upper: str) -> TransitionSpec:
    for transition in spec.transitions:
        if transition.lower == lower and transition.upper == upper:
            return transition
    raise ValueError(
        f"TN level_structure {spec.name!r} needs transition |{lower}> <-> |{upper}>."
    )


def _build_explicit_site(spec: LevelStructureSpec) -> object:
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.site import Site

    dim = spec.local_dim
    leg = npc.LegCharge.from_trivial(dim)
    ops = {}

    for idx, level in enumerate(spec.levels):
        projector = np.zeros((dim, dim), dtype=float)
        projector[idx, idx] = 1.0
        ops[f"n_{level}"] = projector

    for transition in spec.transitions:
        lower = spec.index(transition.lower)
        upper = spec.index(transition.upper)
        x_op = np.zeros((dim, dim), dtype=complex)
        x_op[upper, lower] = 1.0
        x_op[lower, upper] = 1.0
        ops[f"X_{transition.name}"] = x_op

    return Site(leg, state_labels=list(spec.levels), sort_charge=False, **ops)
