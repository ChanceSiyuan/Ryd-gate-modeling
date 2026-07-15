"""SweepProtocol: a function-defined global Rydberg sweep on the |1>-|r> channels.

Produces only effective ``_ChannelDrive`` primitives (P17/P32), so it rejects
named physical-laser noise (N14). TFIM quench/anneal workflows are expressed by
callers as explicit ``SweepProtocol`` instances (P16); there is no dedicated
TFIM protocol class.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ryd_gate.protocols._resolved import _ChannelDrive, _ResolvedProtocol
from ryd_gate.protocols.base import Protocol


class SweepProtocol(Protocol):
    """Global |1>-|r> sweep::

        H[r,1](t)    = omega_half_rad_s(t)          # already Omega/2 (no extra 1/2)
        H[r,r](t)    = -detuning_rad_s(t)
        H_i[r,r](t) += -local_detuning_rad_s(t, i)  # optional per-site addressing
    """

    _compatible_presets = frozenset({"1r", "01r"})

    def __init__(
        self,
        *,
        t_gate_s: float,
        omega_half_rad_s: Callable[[float], float],
        detuning_rad_s: Callable[[float], float],
        local_detuning_rad_s: Callable[[float, int], float] | None = None,
    ) -> None:
        t = float(t_gate_s)
        if isinstance(t_gate_s, bool) or not np.isfinite(t) or t <= 0.0:
            raise ValueError(f"t_gate_s must be finite and positive; got {t_gate_s!r}.")
        if not callable(omega_half_rad_s):
            raise TypeError("omega_half_rad_s must be callable f(t).")
        if not callable(detuning_rad_s):
            raise TypeError("detuning_rad_s must be callable f(t).")
        if local_detuning_rad_s is not None and not callable(local_detuning_rad_s):
            raise TypeError("local_detuning_rad_s must be callable f(t, i) or None.")
        self._t_gate = t
        self._omega_half = omega_half_rad_s
        self._detuning = detuning_rad_s
        self._local = local_detuning_rad_s

    def _resolve(self, system) -> _ResolvedProtocol:
        n = system.N
        drives: list = [
            _ChannelDrive("E[r,1]", lambda t: complex(self._omega_half(t))),
            _ChannelDrive("E[r,r]", lambda t: -float(self._detuning(t))),
        ]
        if self._local is not None:
            drives.append(
                _ChannelDrive(
                    "E[r,r]",
                    lambda t, n=n: -np.array([float(self._local(t, i)) for i in range(n)]),
                )
            )
        return _ResolvedProtocol(t_gate=self._t_gate, drives=tuple(drives))
