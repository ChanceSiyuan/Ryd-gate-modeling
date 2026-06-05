"""Shared lowering helpers for protocol channels.

The exact sparse compiler and TN backends both consume protocol coefficients,
but they materialize them into different representations.  This module keeps
the channel naming rules in one place so per-site addressing, custom level
specs, and Hermitian-conjugate handling stay consistent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ryd_gate.core.level_structures import LevelStructureSpec

_HC_CHANNELS = frozenset({"global_X", "drive_R", "drive_hf"})
_HC_CHANNEL_PREFIXES = ("global_X_", "drive_R_", "drive_hf_")


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
    if site_profile is not None:
        return site_profile
    return np.full(n_sites, scale * float(np.real(coeffs.get(channel, 0.0))))


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
    """Return ``(uniform_value, None)`` or ``(0, nonuniform_profile)``."""
    if np.allclose(profile, profile[0]):
        return float(profile[0]), None
    return 0.0, profile


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
