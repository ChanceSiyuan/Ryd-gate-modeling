"""Level-structure presets and interaction specs for Rydberg systems.

:func:`level_structure` is the only way to obtain a level structure: it
resolves a *preset name* plus its physical keyword arguments into an
immutable, fully specified :class:`LevelStructure` — structural level/
transition data plus the real physical quantities of that atom model
(detunings, decay rates, branching ratios, Rydberg level, laser channel
ratios, physical scales) as typed read-only fields.  There is no public
DSL for assembling arbitrary transition/detuning channels; new physical
level structures are implemented and tested as presets in ``src``.

Presets: ``1r`` / ``01r`` / ``analog_3`` / ``rb87_7_mp`` / ``rb87_7_pm`` /
``rb87_297_clock_4``.

Also hosts the shared lowering helpers for protocol channels: the exact
sparse compiler and TN backends both consume protocol coefficients, but they
materialize them into different representations.  Keeping the channel naming
rules in one place keeps per-site addressing and Hermitian-conjugate handling
consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np

DEFAULT_C6 = 2 * np.pi * 874e9

# Backend support matrix (preset name -> capable backends).
_TN_CAPABLE = frozenset({"1r", "01r"})
# analog_3 (physical g/e/r ladder) is lowered to TN by both the YASTN PEPS engine
# (backend="peps") and the TeNPy MPS path (backend="mps").
_TN_ANALOG = _TN_CAPABLE | {"analog_3"}
_BACKEND_SUPPORT = {
    "exact_ode": frozenset(
        {"1r", "01r", "analog_3", "rb87_7_mp", "rb87_7_pm", "rb87_297_clock_4"}
    ),
    "mps": _TN_ANALOG,
    "peps": _TN_ANALOG,
}


@dataclass(frozen=True)
class Transition:
    """Single-site transition block definition (internal preset plumbing).

    ``operator`` is ``|upper><lower|``.  Protocols decide whether to add
    the Hermitian conjugate through the compiler channel map.
    """

    name: str
    lower: str
    upper: str
    channel: str


@dataclass(frozen=True, eq=False)
class LevelStructure:
    """Immutable resolved level-structure preset (internal type).

    Obtained only through :func:`level_structure`; carries the structural
    level/transition data plus the preset's real physical quantities as typed
    read-only fields.  Fields that do not apply to a preset are ``None`` /
    empty.  All rates and energies are angular frequencies (rad/s); times are
    seconds.
    """

    # -- structural ---------------------------------------------------------
    name: str
    levels: tuple[str, ...]
    rydberg_levels: tuple[str, ...]
    transitions: tuple[Transition, ...] = ()
    detuning_levels: Mapping[str, str] = field(default_factory=dict)
    initial_level: str | None = None

    # -- single-atom static Hamiltonian --------------------------------------
    # d x d complex matrix of static level energies (may carry -i*gamma/2 decay
    # terms on the diagonal); None for purely symbolic presets.
    local_static: np.ndarray | None = None
    # Static off-diagonal couplings as (channel, coefficient) entries (the
    # compiler adds each Hermitian conjugate), e.g. analog_3's static 1013 leg.
    static_couplings: tuple[tuple[str, complex], ...] = ()

    # -- laser structure ------------------------------------------------------
    # Per-laser-group CG/dipole channel ratios the protocols multiply their
    # laser coefficients onto, e.g. {"420": {...}, "1013": {...}} / {"297": {...}}.
    laser_channel_ratios: Mapping[str, Mapping[str, complex]] = field(default_factory=dict)

    # -- physical data (None where not applicable) ----------------------------
    physical_model: str | None = None
    Delta: float | None = None                   # intermediate-state detuning
    t_rise: float | None = None                  # envelope rise time (s)
    rabi_eff: float | None = None                # effective two-photon Rabi
    time_scale: float | None = None              # 2*pi / rabi_eff
    rabi_420: float | None = None
    rabi_1013: float | None = None
    ryd_level: int | None = None
    ryd_state_decay_rate: float | None = None
    ryd_RD_rate: float | None = None
    ryd_BBR_rate: float | None = None
    ryd_garb_decay_rate: float | None = None
    mid_state_decay_rate: float | None = None
    ryd_branch: Mapping[str, float] | None = None
    mid_branch: Mapping[int, Mapping[str, float]] | None = None
    rydberg_indices: tuple[int, ...] | None = None
    magnetic_field_G: float | None = None
    ryd_zeeman_shift: float | None = None
    d_garb_ratio: float | None = None            # 297 garbage/target dipole ratio
    garb_zeeman_detuning: float | None = None    # 297 garbage-branch Zeeman detuning
    enable_rydberg_decay: bool = False
    enable_intermediate_decay: bool = False

    # -- interaction defaults --------------------------------------------------
    default_c6: float | None = None              # isotropic C6 when no InteractionSpec given
    pair_c6_fn: Any | None = None                # (theta, phi) -> C6 (297 anisotropic; lazy ARC)

    # -- interface -------------------------------------------------------------

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

    def supports_backend(self, backend: str) -> bool:
        """Whether *backend* can simulate this model."""
        allowed = _BACKEND_SUPPORT.get(backend)
        if allowed is None:
            return False
        return self.name in allowed


@dataclass(frozen=True)
class InteractionSpec:
    """Pairwise Rydberg interaction construction."""

    C6: float = DEFAULT_C6
    max_range_um: float | None = None
    mode: Literal["all", "nn", "nnn"] = "all"


# The 12 primitive driven transitions of the seven-level gate model (shared by
# both manifolds; the manifold differs only in the per-channel CG ratio carried
# by the protocol coefficient, not in which transitions exist).  420 nm drives
# |0>,|1> -> |e_F>; 1013 nm drives |e_F> -> |r>,|r_garb>.
_RB87_7_TRANSITIONS = (
    Transition("e1_1", "1", "e1", "E[e1,1]"),
    Transition("e2_1", "1", "e2", "E[e2,1]"),
    Transition("e3_1", "1", "e3", "E[e3,1]"),
    Transition("e1_0", "0", "e1", "E[e1,0]"),
    Transition("e2_0", "0", "e2", "E[e2,0]"),
    Transition("e3_0", "0", "e3", "E[e3,0]"),
    Transition("r_e1", "e1", "r", "E[r,e1]"),
    Transition("r_e2", "e2", "r", "E[r,e2]"),
    Transition("r_e3", "e3", "r", "E[r,e3]"),
    Transition("rg_e1", "e1", "r_garb", "E[r_garb,e1]"),
    Transition("rg_e2", "e2", "r_garb", "E[r_garb,e2]"),
    Transition("rg_e3", "e3", "r_garb", "E[r_garb,e3]"),
)

# Structural preset skeletons; physical fields are resolved per call in
# level_structure() (physical kwargs -> physical_models resolution).
_STRUCTURAL_PRESETS: dict[str, dict[str, Any]] = {
    "1r": dict(
        levels=("1", "r"),
        rydberg_levels=("r",),
        transitions=(Transition("1_r", "1", "r", "E[r,1]"),),
        detuning_levels={"E[r,r]": "r"},
        initial_level="1",
    ),
    "01r": dict(
        levels=("0", "1", "r"),
        rydberg_levels=("r",),
        transitions=(
            Transition("R", "1", "r", "E[r,1]"),
            Transition("hf", "0", "1", "E[1,0]"),
            # |0>-|r> (K0r) leg of the full effective CZ Hamiltonian.
            Transition("0r", "0", "r", "E[r,0]"),
        ),
        detuning_levels={"E[r,r]": "r", "E[1,1]": "1"},
        initial_level="1",
    ),
    # Physical Rb87 three-level ladder: analog blocks with static e-r coupling.
    "analog_3": dict(
        levels=("g", "e", "r"),
        rydberg_levels=("r",),
        transitions=(
            Transition("420", "g", "e", "E[e,g]"),
            Transition("1013", "e", "r", "E[r,e]"),
        ),
        detuning_levels={"E[e,e]": "e", "E[r,r]": "r"},
        initial_level="g",
    ),
    # Physical Rb87 seven-level gate model, split by manifold/polarization
    # convention (the suffix encodes the 420/1013 polarizations):
    #   rb87_7_mp  ->  sigma-(420) / sigma+(1013)   (was param_set="our")
    #   rb87_7_pm  ->  sigma+(420) / sigma-(1013)   (was param_set="lukin")
    "rb87_7_mp": dict(
        levels=("0", "1", "e1", "e2", "e3", "r", "r_garb"),
        rydberg_levels=("r", "r_garb"),
        transitions=_RB87_7_TRANSITIONS,
        initial_level="0",
    ),
    "rb87_7_pm": dict(
        levels=("0", "1", "e1", "e2", "e3", "r", "r_garb"),
        rydberg_levels=("r", "r_garb"),
        transitions=_RB87_7_TRANSITIONS,
        initial_level="0",
    ),
    # Physical Rb87 297 nm single-photon excitation from the clock-like
    # |F=2, mF=0> ground state |1> to nP3/2 (default n=53): a σ⁻ beam drives
    # the target (mJ=-1/2 -> mJ=-3/2) and garbage (mJ=+1/2 -> mJ=-1/2) Zeeman
    # branches, resolved by the bias-field Zeeman detuning of |r_garb>.  The
    # logical |0> (|F=1, mF=0>) is a dark spectator: no 297 transition
    # touches it.
    "rb87_297_clock_4": dict(
        levels=("0", "1", "r", "r_garb"),
        rydberg_levels=("r", "r_garb"),
        transitions=(
            Transition("297", "1", "r", "E[r,1]"),
            Transition("297_garb", "1", "r_garb", "E[r_garb,1]"),
        ),
        initial_level="0",
    ),
}

# The exact set of physical keyword arguments each preset accepts.
_PRESET_KWARGS: dict[str, frozenset[str]] = {
    "1r": frozenset(),
    "01r": frozenset(),
    "analog_3": frozenset({
        "detuning_sign", "Delta_Hz", "rabi_420_Hz", "rabi_1013_Hz",
        "enable_rydberg_decay", "enable_intermediate_decay",
    }),
    "rb87_7_mp": frozenset({
        "detuning_sign", "Delta_Hz", "ryd_level", "C6_rad_s_um6", "t_rise",
        "enable_rydberg_decay", "enable_intermediate_decay", "magnetic_field_G",
    }),
    "rb87_7_pm": frozenset({
        "detuning_sign", "Delta_Hz", "ryd_level", "C6_rad_s_um6", "t_rise",
        "enable_rydberg_decay", "enable_intermediate_decay", "magnetic_field_G",
    }),
    "rb87_297_clock_4": frozenset({
        "enable_rydberg_decay", "magnetic_field_G", "ryd_level",
    }),
}


def level_structure(name: str, **physical_kwargs) -> LevelStructure:
    """Resolve a built-in level-structure preset with its physical kwargs.

    Returns the immutable, fully specified :class:`LevelStructure` — the only
    supported way to obtain a level structure.  Unknown preset names and
    unknown physical kwargs raise.
    """
    if name == "rb87_7":
        raise ValueError(
            "'rb87_7' has been split by manifold/polarization convention: use "
            "'rb87_7_mp' (sigma-/sigma+, was param_set='our') or 'rb87_7_pm' "
            "(sigma+/sigma-, was param_set='lukin'), with explicit static kwargs "
            "(Delta_Hz, ryd_level, C6_rad_s_um6, t_rise)."
        )
    if name == "01":
        raise ValueError(
            "the '01' preset has been removed (no production calls; its "
            "Hamiltonian was the identity)."
        )
    structural = _STRUCTURAL_PRESETS.get(name)
    if structural is None:
        raise ValueError(f"Unknown level-structure preset '{name}'.")

    allowed = _PRESET_KWARGS[name]
    unknown = set(physical_kwargs) - allowed
    if unknown:
        allowed_names = ", ".join(sorted(allowed)) or "none"
        raise TypeError(
            f"{name} does not accept physical parameter(s): "
            f"{', '.join(sorted(unknown))}. Allowed parameters: {allowed_names}."
        )

    from ryd_gate.core import physical_models

    if name == "analog_3":
        fields = physical_models.analog_3_level_fields(**physical_kwargs)
    elif name in ("rb87_7_mp", "rb87_7_pm"):
        fields = physical_models.rb87_7_level_fields(
            name.removeprefix("rb87_7_"), **physical_kwargs
        )
    elif name == "rb87_297_clock_4":
        fields = physical_models.rb87_297_level_fields(**physical_kwargs)
    else:
        fields = {}

    return LevelStructure(name=name, **structural, **fields)


# ── Protocol-channel lowering helpers ────────────────────────────────────────


def split_site_channel(channel: str) -> tuple[str, int | None]:
    """Return ``(base_channel, site_index)`` for ``channel_i`` names."""
    base, sep, suffix = channel.rpartition("_")
    if sep and suffix.isdigit():
        return base, int(suffix)
    return channel, None


def declared_channels(level_spec: LevelStructure) -> set[str]:
    """Return channels declared by a central level structure."""
    channels = {transition.channel for transition in level_spec.transitions}
    channels.update(level_spec.detuning_levels)
    return channels


def validate_coeff_channels(
    coeffs: dict[str, complex],
    level_spec: LevelStructure,
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


def transition_channel(level_spec: LevelStructure, lower: str, upper: str) -> str | None:
    """Find the channel for a declared ``|lower> <-> |upper>`` transition."""
    for transition in level_spec.transitions:
        if transition.lower == lower and transition.upper == upper:
            return transition.channel
    return None


def detuning_channel(level_spec: LevelStructure, level: str) -> str | None:
    """Find the detuning channel for a declared level projector."""
    for channel, detuned_level in level_spec.detuning_levels.items():
        if detuned_level == level:
            return channel
    return None


def site_profile_from_coeffs(
    coeffs: dict[str, complex],
    channel: str,
    n_sites: int,
) -> np.ndarray | None:
    """Return a complex per-site profile if ``coeffs`` contains site-specific keys."""
    keys = [f"{channel}_{i}" for i in range(n_sites)]
    if not any(key in coeffs for key in keys):
        return None
    return np.array([complex(coeffs.get(key, 0.0)) for key in keys])


def channel_profile_from_coeffs(
    coeffs: dict[str, complex],
    channel: str,
    n_sites: int,
) -> np.ndarray:
    """Return a complex per-site profile from either global or site-specific coeffs."""
    site_profile = site_profile_from_coeffs(coeffs, channel, n_sites)
    global_value = complex(coeffs.get(channel, 0.0))
    if site_profile is not None:
        return global_value + site_profile
    return np.full(n_sites, global_value)


def profile_for_optional_channel(
    coeffs: dict[str, complex],
    channel: str | None,
    n_sites: int,
) -> np.ndarray:
    """Return zeros when an optional channel is absent from a level spec."""
    if channel is None:
        return np.zeros(n_sites, dtype=complex)
    return channel_profile_from_coeffs(coeffs, channel, n_sites)


def three_level_profiles_from_coeffs(
    coeffs: dict[str, complex],
    spec: Any,
) -> dict[str, np.ndarray]:
    """Map protocol coefficients to 01r per-site matrix-element profiles.

    Direct matrix-element convention: each coupling profile is the (complex)
    Hamiltonian entry ``H[upper, lower]`` on that channel — the backend adds
    only the Hermitian conjugate — and each energy profile is the real diagonal
    entry ``H[l, l]`` itself (an energy, not a negated detuning).
    """
    level_spec = spec.level_spec
    validate_coeff_channels(
        coeffs,
        level_spec,
        level_structure_name=getattr(spec, "level_structure", level_spec.name),
    )
    drive_r1 = transition_channel(level_spec, "1", "r")
    drive_10 = transition_channel(level_spec, "0", "1")
    drive_r0 = transition_channel(level_spec, "0", "r")
    energy_r = detuning_channel(level_spec, "r")
    energy_1 = detuning_channel(level_spec, "1")

    return {
        "coupling_r1": profile_for_optional_channel(coeffs, drive_r1, spec.N),
        "coupling_10": profile_for_optional_channel(coeffs, drive_10, spec.N),
        "coupling_r0": profile_for_optional_channel(coeffs, drive_r0, spec.N),
        "energy_1": np.real(profile_for_optional_channel(coeffs, energy_1, spec.N)),
        "energy_r": np.real(profile_for_optional_channel(coeffs, energy_r, spec.N)),
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

    for coupling_name in ("coupling_10", "coupling_r0"):
        if np.any(np.abs(profiles[coupling_name]) > 0):
            raise ValueError(
                "TN TDVP supports the |1>-|r> two-level subspace only; "
                f"segments with {coupling_name} != 0 are not supported (no |0> level on 1r)."
            )

    omega_profile = 2.0 * np.real(profiles["coupling_r1"])
    if np.allclose(omega_profile, omega_profile[0]):
        Omega_t: float | np.ndarray = float(omega_profile[0])
    else:
        Omega_t = omega_profile

    # Energies -> effective 1r detuning on the {|1>, |r>} subspace:
    # energy_1 n_1 + energy_r n_r = const - (energy_1 - energy_r) n_r.
    Delta_t, delta_profile = split_uniform_profile(
        profiles["energy_1"] - profiles["energy_r"]
    )
    return Omega_t, Delta_t, delta_profile
