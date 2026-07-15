"""Level-structure presets: the only way to obtain a :class:`LevelStructure`.

:func:`level_structure` resolves a *preset name* plus its physical keyword
arguments (S07) into an immutable, fully specified :class:`LevelStructure`. Its
public surface is exactly the seven read-only characterization fields of S17;
everything the compiler needs (Rydberg-level set, atomic diagonal, laser legs,
pair-C6 providers, intermediate levels) is private implementation.

Presets: ``1r`` / ``01r`` / ``rb87_7_mp`` / ``rb87_7_pm`` / ``rb87_297_clock_4``.
There is no public DSL for arbitrary level structures; new physical models are
implemented and validated as presets in ``src``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

# Backend capability table (private; consumed by the docs capability matrix).
_BACKEND_SUPPORT = {
    "exact_ode": frozenset({"1r", "01r", "rb87_7_mp", "rb87_7_pm", "rb87_297_clock_4"}),
    "mps": frozenset({"1r", "01r"}),
    "peps": frozenset({"1r", "01r"}),
}


def _deep_freeze(obj):
    """Recursively wrap dicts in MappingProxyType and lists in tuples."""
    if isinstance(obj, Mapping):
        return MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_deep_freeze(v) for v in obj)
    return obj


@dataclass(frozen=True)
class _LaserLeg:
    """One physical-laser → primitive-transition mapping (private preset data).

    ``factor`` is the CG/dipole ratio the compiler multiplies onto the laser's
    complex drive amplitude; it already contains the ``1/2`` (P10), so the
    compiler adds only the Hermitian conjugate.
    """

    group: str        # "420" | "1013" | "297"
    channel: str      # "E[e1,0]", "E[r,e2]", "E[r_garb,1]", ...
    factor: complex


@dataclass(frozen=True, eq=False)
class LevelStructure:
    """Immutable resolved level-structure preset.

    Public (S17): ``name``, ``levels``, ``ryd_level``, ``magnetic_field_G``,
    ``quantization_axis``, ``decay_rates_per_s``, ``branching_ratios``. All other
    fields are private compiler data.
    """

    # -- public S17 surface --------------------------------------------------
    name: str
    levels: tuple[str, ...]
    ryd_level: int
    magnetic_field_G: float | None
    quantization_axis: tuple[float, float, float] | None
    decay_rates_per_s: Mapping[str, Mapping[str, float]]
    branching_ratios: Mapping[str, Mapping[str, float]]

    # -- private compiler data (leading underscore; not user API) ------------
    _rydberg_levels: tuple[str, ...] = ()
    _intermediate_levels: tuple[str, ...] = ()
    # real atomic diagonal energies (rad/s), only levels with a nonzero offset;
    # the protocol adds the intermediate detuning on top (P19); |0>-gauge zero.
    _static_diag: Mapping[str, float] = field(default_factory=dict)
    _laser_legs: tuple[_LaserLeg, ...] = ()
    # isotropic S-state pair C6 (rad/s·μm⁶) shared by all Rydberg-pair channels (I05).
    _pair_c6_isotropic: float | None = None
    # channel-resolved P-state pair C6 (I06): ((level_a, level_b), (theta,phi)->C6).
    _pair_c6_channels: tuple[tuple[tuple[str, str], Any], ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decay_rates_per_s", _deep_freeze(dict(self.decay_rates_per_s)))
        object.__setattr__(self, "branching_ratios", _deep_freeze(dict(self.branching_ratios)))
        object.__setattr__(self, "_static_diag", MappingProxyType(dict(self._static_diag)))


# Structural skeletons: level ordering + Rydberg/intermediate classification.
# Physical fields (diagonal, laser legs, interactions, decay) are resolved per
# call in level_structure() via core.physical_models.
_STRUCTURAL_PRESETS: dict[str, dict[str, Any]] = {
    "1r": dict(levels=("1", "r"), _rydberg_levels=("r",)),
    "01r": dict(levels=("0", "1", "r"), _rydberg_levels=("r",)),
    "rb87_7_mp": dict(
        levels=("0", "1", "e1", "e2", "e3", "r", "r_garb"),
        _rydberg_levels=("r", "r_garb"),
        _intermediate_levels=("e1", "e2", "e3"),
    ),
    "rb87_7_pm": dict(
        levels=("0", "1", "e1", "e2", "e3", "r", "r_garb"),
        _rydberg_levels=("r", "r_garb"),
        _intermediate_levels=("e1", "e2", "e3"),
    ),
    "rb87_297_clock_4": dict(
        levels=("0", "1", "r", "r_garb"),
        _rydberg_levels=("r", "r_garb"),
    ),
}

# The exact set of physical keyword arguments each preset accepts (S07).
_PRESET_KWARGS: dict[str, frozenset[str]] = {
    "1r": frozenset({"ryd_level"}),
    "01r": frozenset({"ryd_level"}),
    "rb87_7_mp": frozenset({"ryd_level", "magnetic_field_G"}),
    "rb87_7_pm": frozenset({"ryd_level", "magnetic_field_G"}),
    "rb87_297_clock_4": frozenset({"ryd_level", "magnetic_field_G", "quantization_axis"}),
}


def level_structure(name: str, **physical_kwargs) -> LevelStructure:
    """Resolve a built-in level-structure preset with its physical kwargs (S07).

    Returns the immutable :class:`LevelStructure`. Unknown preset names and
    unknown physical kwargs raise. There is no way to bypass this factory.
    """
    structural = _STRUCTURAL_PRESETS.get(name)
    if structural is None:
        raise ValueError(
            f"Unknown level-structure preset {name!r}; available: "
            f"{sorted(_STRUCTURAL_PRESETS)}."
        )

    allowed = _PRESET_KWARGS[name]
    unknown = set(physical_kwargs) - allowed
    if unknown:
        allowed_names = ", ".join(sorted(allowed)) or "none"
        raise TypeError(
            f"{name} does not accept physical parameter(s): "
            f"{', '.join(sorted(unknown))}. Allowed parameters: {allowed_names}."
        )

    from ryd_gate.core import physical_models

    if name in ("1r", "01r"):
        fields = physical_models.effective_1r_level_fields(**physical_kwargs)
    elif name in ("rb87_7_mp", "rb87_7_pm"):
        fields = physical_models.rb87_7_level_fields(name.removeprefix("rb87_7_"), **physical_kwargs)
    else:  # rb87_297_clock_4
        fields = physical_models.rb87_297_level_fields(**physical_kwargs)

    return LevelStructure(name=name, **structural, **fields)
