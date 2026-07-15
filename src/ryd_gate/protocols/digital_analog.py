"""DigitalAnalogProtocol: the full local 3×3 ``{0,1,r}`` Hamiltonian (P09/P10).

Five mutually independent, optional control functions, each a direct Hamiltonian
matrix element ``H[ket,bra](t)`` (the compiler adds only the Hermitian
conjugate — no extra ``1/2``). Each returns a scalar (uniform over sites) or a
length-``N`` per-site profile. Only compatible with the ``01r`` preset; it is the
canonical replacement for the deleted ``analog_3`` model (S06).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ryd_gate.protocols._resolved import _ChannelDrive, _ResolvedProtocol
from ryd_gate.protocols.base import Protocol


def _real_channel(fn: Callable, name: str) -> Callable:
    def coeff(t, fn=fn, name=name):
        v = np.asarray(fn(t))
        if np.iscomplexobj(v) and np.any(np.abs(v.imag) > 0):
            raise ValueError(f"{name} must be real (diagonal energy); got complex value.")
        return v.real if v.ndim else float(np.real(v))

    return coeff


def _complex_channel(fn: Callable) -> Callable:
    def coeff(t, fn=fn):
        v = np.asarray(fn(t))
        return v.astype(complex) if v.ndim else complex(v)

    return coeff


class DigitalAnalogProtocol(Protocol):
    """Full ``01r`` direct-matrix-element protocol (five independent controls)."""

    _compatible_presets = frozenset({"01r"})

    def __init__(
        self,
        *,
        t_gate_s: float,
        coupling_10_rad_s: Callable | None = None,
        coupling_r0_rad_s: Callable | None = None,
        coupling_r1_rad_s: Callable | None = None,
        energy_1_rad_s: Callable | None = None,
        energy_r_rad_s: Callable | None = None,
    ) -> None:
        t = float(t_gate_s)
        if isinstance(t_gate_s, bool) or not np.isfinite(t) or t <= 0.0:
            raise ValueError(f"t_gate_s must be finite and positive; got {t_gate_s!r}.")
        controls = {
            "coupling_10_rad_s": coupling_10_rad_s,
            "coupling_r0_rad_s": coupling_r0_rad_s,
            "coupling_r1_rad_s": coupling_r1_rad_s,
            "energy_1_rad_s": energy_1_rad_s,
            "energy_r_rad_s": energy_r_rad_s,
        }
        for name, fn in controls.items():
            if fn is not None and not callable(fn):
                raise TypeError(f"{name} must be callable f(t) or None.")
        if all(fn is None for fn in controls.values()):
            raise ValueError("DigitalAnalogProtocol needs at least one control function.")
        self._t_gate = t
        self._c10 = coupling_10_rad_s
        self._cr0 = coupling_r0_rad_s
        self._cr1 = coupling_r1_rad_s
        self._e1 = energy_1_rad_s
        self._er = energy_r_rad_s

    def _resolve(self, system) -> _ResolvedProtocol:
        drives: list = []
        if self._c10 is not None:
            drives.append(_ChannelDrive("E[1,0]", _complex_channel(self._c10)))
        if self._cr0 is not None:
            drives.append(_ChannelDrive("E[r,0]", _complex_channel(self._cr0)))
        if self._cr1 is not None:
            drives.append(_ChannelDrive("E[r,1]", _complex_channel(self._cr1)))
        if self._e1 is not None:
            drives.append(_ChannelDrive("E[1,1]", _real_channel(self._e1, "energy_1_rad_s")))
        if self._er is not None:
            drives.append(_ChannelDrive("E[r,r]", _real_channel(self._er, "energy_r_rad_s")))
        return _ResolvedProtocol(t_gate=self._t_gate, drives=tuple(drives))
