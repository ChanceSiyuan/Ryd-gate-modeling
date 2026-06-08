"""Shared protocol-lowering helpers for tensor-network backends.

All TN backends unpack a :class:`~ryd_gate.protocols.base.Protocol` against a
:class:`~ryd_gate.backends.tn_common.lattice_spec.TNLatticeSpec` rather than a full
``RydbergSystem``. :class:`TNProtocolContext` is the minimal system-like adapter that
``protocol.unpack_params(x, context)`` needs (``N``, ``basis.n_sites``, ``meta``), and
``pin_deltas_from_params``/``merge_pin_deltas`` turn unpacked params into per-site
local-detuning profiles. Keeping them here avoids each backend (TeNPy, gputn, YASTN
PEPS, PEPSKit) carrying its own copy.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ryd_gate.backends.tn_common.lattice_spec import TNLatticeSpec


class TNProtocolContext:
    """Minimal ``RydbergSystem``-like adapter over a :class:`TNLatticeSpec`.

    Exposes only what ``Protocol.unpack_params`` reads: ``N``, ``basis.n_sites``,
    and ``meta("Omega"|"n_sites")``.
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
        return default


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
