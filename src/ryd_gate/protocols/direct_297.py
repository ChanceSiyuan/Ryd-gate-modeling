"""Direct 297 nm single-photon σ⁻ protocols (P14/P25/P27-P30).

Three user-selectable classes for the ``rb87_297_clock_4`` model — an
auto-calibrated π pulse, a generic physical-time CZ pulse, and a Time-Optimal
phase family — each taking an explicit target ``|1>-|r>`` Rabi frequency (the
power→Rabi conversion is done by ``ryd_gate.physics.rb87_297_clock_rabi_frequencies``
in the calling script). All produce a single ``297`` laser drive; the garbage
branch coupling comes from the preset's private leg ratio.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ryd_gate.protocols._resolved import _LaserDrive, _ResolvedProtocol
from ryd_gate.protocols.base import Protocol
from ryd_gate.protocols.pulses import blackman_pulse

_297 = frozenset({"rb87_297_clock_4"})


def _require_positive(name: str, value: float) -> float:
    v = float(value)
    if isinstance(value, bool) or not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be a finite positive number; got {value!r}.")
    return v


def _require_finite(name: str, value: float) -> float:
    v = float(value)
    if isinstance(value, bool) or not np.isfinite(v):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return v


def _laser_297(omega_max: float, envelope: Callable, phase: Callable) -> Callable:
    def coeff(t, omega_max=omega_max, envelope=envelope, phase=phase):
        a = float(envelope(t))
        p = float(phase(t))
        if not (np.isfinite(a) and np.isfinite(p)):
            raise ValueError("Direct-297 envelope/phase returned a non-finite value.")
        return omega_max * a * np.exp(-1j * p)

    return coeff


class Direct297PiProtocol(Protocol):
    """Auto-timed 297 nm σ⁻ π-pulse |1> → |r> (P27/P30)."""

    _compatible_presets = _297

    def __init__(self, *, omega_297_max_rad_s: float, rise_fraction: float = 0.15) -> None:
        self._omega = _require_positive("omega_297_max_rad_s", omega_297_max_rad_s)
        rf = _require_finite("rise_fraction", rise_fraction)
        if not (0.0 <= rf <= 0.5):
            raise ValueError(f"rise_fraction must satisfy 0 <= rise_fraction <= 0.5; got {rise_fraction!r}.")
        self._rise_fraction = rf
        # Normalized envelope A(s), s in [0, 1]; F = integral_0^1 A(s) ds.
        if rf == 0.0:
            self._area = 1.0
        else:
            s = np.linspace(0.0, 1.0, 20001)
            a = np.asarray(blackman_pulse(s, rf, 1.0), dtype=float)
            self._area = float(np.sum(0.5 * (a[:-1] + a[1:]) * np.diff(s)))
        self._t_gate = np.pi / (self._omega * self._area)

    def _envelope(self, t: float) -> float:
        if self._rise_fraction == 0.0:
            return 1.0
        return float(blackman_pulse(t / self._t_gate, self._rise_fraction, 1.0))

    def _resolve(self, system) -> _ResolvedProtocol:
        coeff = _laser_297(self._omega, self._envelope, lambda t: 0.0)
        return _ResolvedProtocol(t_gate=float(self._t_gate), drives=(_LaserDrive("297", coeff),))


class Direct297CZProtocol(Protocol):
    """Generic fixed-duration 297 nm CZ pulse with physical-time callables (P28)."""

    _compatible_presets = _297

    def __init__(
        self,
        *,
        t_gate_s: float,
        omega_297_max_rad_s: float,
        envelope_297: Callable,
        phase_297_rad: Callable | None = None,
    ) -> None:
        self._t_gate = _require_positive("t_gate_s", t_gate_s)
        self._omega = _require_positive("omega_297_max_rad_s", omega_297_max_rad_s)
        if not callable(envelope_297):
            raise TypeError("envelope_297 must be callable f(t).")
        self._env = envelope_297
        self._phase = phase_297_rad if phase_297_rad is not None else (lambda t: 0.0)
        if not callable(self._phase):
            raise TypeError("phase_297_rad must be callable f(t) or None.")

    def _resolve(self, system) -> _ResolvedProtocol:
        coeff = _laser_297(self._omega, self._env, self._phase)
        return _ResolvedProtocol(t_gate=float(self._t_gate), drives=(_LaserDrive("297", coeff),))


class Direct297TOProtocol(Protocol):
    """Time-Optimal 297 nm phase family with a Blackman envelope (P29)."""

    _compatible_presets = _297

    def __init__(
        self,
        *,
        omega_297_max_rad_s: float,
        rise_time_s: float,
        phase_amplitude_rad: float,
        modulation_frequency_ratio: float,
        phase_offset_rad: float,
        frequency_offset_ratio: float,
        duration_ratio: float,
    ) -> None:
        self._omega = _require_positive("omega_297_max_rad_s", omega_297_max_rad_s)
        self._rise = _require_positive("rise_time_s", rise_time_s)
        self._phase_amp = _require_finite("phase_amplitude_rad", phase_amplitude_rad)
        self._mod_ratio = _require_finite("modulation_frequency_ratio", modulation_frequency_ratio)
        self._phase_off = _require_finite("phase_offset_rad", phase_offset_rad)
        self._freq_off_ratio = _require_finite("frequency_offset_ratio", frequency_offset_ratio)
        self._dur_ratio = _require_positive("duration_ratio", duration_ratio)

    def _resolve(self, system) -> _ResolvedProtocol:
        t_gate = self._dur_ratio * 2 * np.pi / self._omega
        if 2 * self._rise > t_gate:
            raise ValueError(f"need 2*rise_time_s <= t_gate ({2*self._rise} > {t_gate}).")
        omega_mod = self._mod_ratio * self._omega
        delta = self._freq_off_ratio * self._omega
        env = lambda t, tg=t_gate: blackman_pulse(t, self._rise, tg)
        phase = lambda t: self._phase_amp * np.cos(omega_mod * t + self._phase_off) + delta * t
        coeff = _laser_297(self._omega, env, phase)
        return _ResolvedProtocol(t_gate=float(t_gate), drives=(_LaserDrive("297", coeff),))
