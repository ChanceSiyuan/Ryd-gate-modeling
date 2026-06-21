"""Shared protocol-lowering helpers for tensor-network backends.

All TN backends unpack a :class:`~ryd_gate.protocols.base.Protocol` against a
:class:`~ryd_gate.backends.tn_common.lattice_spec.TNLatticeSpec` rather than a full
``RydbergSystem``. :class:`TNProtocolContext` is the minimal system-like adapter that
``protocol.unpack_params(x, context)`` needs (``N``, ``basis.n_sites``, ``meta``), and
``pin_deltas_from_params``/``merge_pin_deltas`` turn unpacked params into per-site
local-detuning profiles. Keeping them here avoids each backend (TeNPy MPS, YASTN
PEPS) carrying its own copy.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ryd_gate.backends.tn_common.lattice_spec import TNLatticeSpec


class TNProtocolContext:
    """Minimal ``RydbergSystem``-like adapter over a :class:`TNLatticeSpec`.

    A ``Protocol`` unpacks its parameters against a spec instead of a full system.
    This exposes the system attributes ``unpack_params`` may consult — ``N`` /
    ``basis.n_sites`` (to size per-site profiles) — plus ``meta("n_sites")`` and
    ``meta("Omega")`` for protocols that look them up.
    """

    def __init__(self, spec: TNLatticeSpec) -> None:
        self._spec = spec
        self.basis = SimpleNamespace(n_sites=spec.N)

    @property
    def N(self) -> int:
        return self._spec.N

    def meta(self, name: str, default=None):
        if name == "Omega":
            return self._spec.Omega
        if name == "n_sites":
            return self._spec.N
        blocks = self._spec.local_blocks
        if blocks is not None:
            if name == "rabi_eff":
                return blocks.rabi_eff
            if name == "time_scale":
                return blocks.time_scale
        return default


def analog3_dt_guard(spec, dt) -> None:
    """Raise if an analog_3 time step is catastrophically coarse (wrong units).

    analog_3 runs in physical seconds with off-diagonal couplings ~2pi*491 MHz, so
    the natural-unit default ``dt`` (0.05/0.2) gives ~1 step over a ~0.5 us gate.
    """
    if getattr(spec, "level_structure", None) != "analog_3" or spec.local_blocks is None:
        return
    time_scale = spec.local_blocks.time_scale
    if float(dt) >= time_scale:
        raise ValueError(
            f"analog_3 needs a small TN step: dt={float(dt):.3g}s >= the effective Rabi "
            f"period time_scale={time_scale:.3g}s. Pass dt ~ time_scale/20-40 "
            f"(about {time_scale / 30:.2g}s); the default natural-unit dt is wrong for analog_3."
        )


def pin_deltas_from_params(params: dict, n_sites: int) -> np.ndarray | None:
    """Return a length-``n_sites`` local-detuning profile, or ``None`` if unset."""
    pin_map = params.get("pin_deltas") or {}
    if not pin_map:
        return None
    pin = np.zeros(n_sites)
    for idx, value in pin_map.items():
        if int(idx) < n_sites:
            pin[int(idx)] = float(value)
    return pin


def merge_pin_deltas(*profiles: np.ndarray | None, n_sites: int) -> np.ndarray | None:
    """Sum any number of optional per-site profiles, or ``None`` if all absent."""
    merged = np.zeros(n_sites)
    any_profile = False
    for profile in profiles:
        if profile is None:
            continue
        merged += profile
        any_profile = True
    return merged if any_profile else None
