"""The one private protocol→backend lowering seam (P17).

``lower_drives(system, realization=None)`` calls ``protocol._resolve(system)``
once, expands physical ``_LaserDrive`` groups through the preset's ``_LaserLeg``
table (injecting per-group amplitude/frequency noise when a realization is
given, N02), passes ``_ChannelDrive`` primitives through untouched, and returns
the gate duration plus one immutable coefficient callable per primitive
``E[ket,bra]`` channel. Exact, MPS, PEPS and the noise ensemble all consume this
single representation — there is no other seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ryd_gate.protocols._resolved import _ChannelDrive, _LaserDrive


@dataclass(frozen=True)
class _LoweredChannel:
    """One primitive channel with its combined time-dependent coefficient."""

    channel: str  # "E[ket,bra]"
    coefficient: Callable[[float], "complex | np.ndarray"]


def lower_drives(system, *, realization=None) -> tuple[float, tuple[_LoweredChannel, ...]]:
    """Return ``(t_gate, channels)`` — the canonical lowered drive representation."""
    resolved = system.protocol._resolve(system)
    legs_by_group: dict[str, list] = {}
    for leg in system.level_structure._laser_legs:
        legs_by_group.setdefault(leg.group, []).append(leg)

    acc: dict[str, list[Callable]] = {}
    for drive in resolved.drives:
        if isinstance(drive, _LaserDrive):
            coeff = _noisy_laser_coefficient(drive, realization)
            group_legs = legs_by_group.get(drive.group)
            if not group_legs:
                raise ValueError(
                    f"protocol drives laser group {drive.group!r}, but level "
                    f"structure {system.level_structure.name!r} defines no legs for it."
                )
            for leg in group_legs:
                acc.setdefault(leg.channel, []).append(
                    lambda t, c=coeff, f=leg.factor: f * c(t)
                )
        elif isinstance(drive, _ChannelDrive):
            acc.setdefault(drive.channel, []).append(drive.coefficient)
        else:  # pragma: no cover - _ResolvedProtocol only holds these two types
            raise TypeError(f"unexpected drive type {type(drive).__name__}.")

    channels = []
    for channel in sorted(acc):
        fns = acc[channel]
        coeff = fns[0] if len(fns) == 1 else _sum_coefficients(tuple(fns))
        channels.append(_LoweredChannel(channel, coeff))
    return float(resolved.t_gate), tuple(channels)


def _noisy_laser_coefficient(drive: _LaserDrive, realization):
    """Apply a shot's per-group amplitude/frequency noise to a laser coefficient."""
    base = drive.coefficient
    if realization is None:
        return base
    eps = realization["laser_amplitude_scales"].get(drive.group, 1.0)
    delta = realization["laser_frequency_offsets_rad_s"].get(drive.group, 0.0)
    if eps == 1.0 and delta == 0.0:
        return base
    return lambda t, base=base, eps=eps, delta=delta: eps * base(t) * np.exp(-1j * delta * t)


def _sum_coefficients(fns: tuple[Callable, ...]) -> Callable:
    return lambda t, fns=fns: sum(f(t) for f in fns)
