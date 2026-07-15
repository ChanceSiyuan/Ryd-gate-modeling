"""CZ-gate laser protocols: generic :class:`CZProtocol` plus the Time-Optimal
(:class:`TOProtocol`) and Amplitude-Robust (:class:`ARProtocol`) phase families.

All three drive the physical 420 nm and 1013 nm lasers of the ``rb87_7_mp`` /
``rb87_7_pm`` seven-level model. Each is fully specified at construction with
explicit peak Rabi amplitudes and a signed intermediate detuning (P19/P20); the
detuning is applied to the intermediate-state diagonals through the private
resolved-drive IR (P17), never baked into the preset.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ryd_gate.protocols._resolved import _ChannelDrive, _LaserDrive, _ResolvedProtocol
from ryd_gate.protocols.base import Protocol
from ryd_gate.protocols.pulses import blackman_pulse

_RB87_7 = frozenset({"rb87_7_mp", "rb87_7_pm"})


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


def _require_callable(name: str, fn) -> Callable:
    if not callable(fn):
        raise TypeError(f"{name} must be callable f(t)->value; got {type(fn).__name__}.")
    return fn


def _laser_coefficient(omega_max: float, envelope: Callable, phase: Callable) -> Callable:
    def coeff(t, omega_max=omega_max, envelope=envelope, phase=phase):
        a = float(envelope(t))
        p = float(phase(t))
        if not (np.isfinite(a) and np.isfinite(p)):
            raise ValueError("CZ envelope/phase returned a non-finite value.")
        return omega_max * a * np.exp(-1j * p)

    return coeff


class _LaserCZProtocol(Protocol):
    """Shared 420/1013 lowering for the CZ family (private)."""

    _compatible_presets = _RB87_7

    def _laser_resolve(
        self,
        system,
        *,
        t_gate: float,
        intermediate_detuning_rad_s: float,
        omega_420_max: float,
        omega_1013_max: float,
        envelope_420: Callable,
        phase_420: Callable,
        envelope_1013: Callable,
        phase_1013: Callable,
    ) -> _ResolvedProtocol:
        drives: list = [
            _LaserDrive("420", _laser_coefficient(omega_420_max, envelope_420, phase_420)),
            _LaserDrive("1013", _laser_coefficient(omega_1013_max, envelope_1013, phase_1013)),
        ]
        delta = float(intermediate_detuning_rad_s)
        if delta != 0.0:
            for level in system.level_structure._intermediate_levels:
                drives.append(_ChannelDrive(f"E[{level},{level}]", lambda t, d=delta: d))
        return _ResolvedProtocol(t_gate=float(t_gate), drives=tuple(drives))


def _effective_rabi(o420: float, o1013: float, delta: float) -> float:
    if delta == 0.0:
        raise ValueError("TO/AR require a nonzero intermediate_detuning_rad_s.")
    return o420 * o1013 / (2 * abs(delta))


class CZProtocol(_LaserCZProtocol):
    """Generic CZ pulse: explicit duration + four physical-time waveform callables (P24)."""

    def __init__(
        self,
        *,
        t_gate_s: float,
        intermediate_detuning_rad_s: float,
        omega_420_max_rad_s: float,
        omega_1013_max_rad_s: float,
        envelope_420: Callable,
        phase_420_rad: Callable,
        envelope_1013: Callable | None = None,
        phase_1013_rad: Callable | None = None,
    ) -> None:
        self._t_gate = _require_positive("t_gate_s", t_gate_s)
        self._delta = _require_finite("intermediate_detuning_rad_s", intermediate_detuning_rad_s)
        self._o420 = _require_positive("omega_420_max_rad_s", omega_420_max_rad_s)
        self._o1013 = _require_positive("omega_1013_max_rad_s", omega_1013_max_rad_s)
        self._env420 = _require_callable("envelope_420", envelope_420)
        self._phase420 = _require_callable("phase_420_rad", phase_420_rad)
        self._env1013 = _require_callable("envelope_1013", envelope_1013) if envelope_1013 is not None else (lambda t: 1.0)
        self._phase1013 = _require_callable("phase_1013_rad", phase_1013_rad) if phase_1013_rad is not None else (lambda t: 0.0)

    def _resolve(self, system) -> _ResolvedProtocol:
        return self._laser_resolve(
            system,
            t_gate=self._t_gate,
            intermediate_detuning_rad_s=self._delta,
            omega_420_max=self._o420,
            omega_1013_max=self._o1013,
            envelope_420=self._env420,
            phase_420=self._phase420,
            envelope_1013=self._env1013,
            phase_1013=self._phase1013,
        )


class TOProtocol(_LaserCZProtocol):
    """Time-Optimal CZ pulse: cosine 420 phase family with a Blackman envelope (P21-P23)."""

    def __init__(
        self,
        *,
        intermediate_detuning_rad_s: float,
        omega_420_max_rad_s: float,
        omega_1013_max_rad_s: float,
        rise_time_s: float,
        phase_amplitude_rad: float,
        modulation_frequency_ratio: float,
        phase_offset_rad: float,
        frequency_offset_ratio: float,
        duration_ratio: float,
    ) -> None:
        self._delta = _require_finite("intermediate_detuning_rad_s", intermediate_detuning_rad_s)
        if self._delta == 0.0:
            raise ValueError("TOProtocol requires a nonzero intermediate_detuning_rad_s.")
        self._o420 = _require_positive("omega_420_max_rad_s", omega_420_max_rad_s)
        self._o1013 = _require_positive("omega_1013_max_rad_s", omega_1013_max_rad_s)
        self._rise = _require_positive("rise_time_s", rise_time_s)
        self._phase_amp = _require_finite("phase_amplitude_rad", phase_amplitude_rad)
        self._mod_ratio = _require_finite("modulation_frequency_ratio", modulation_frequency_ratio)
        self._phase_off = _require_finite("phase_offset_rad", phase_offset_rad)
        self._freq_off_ratio = _require_finite("frequency_offset_ratio", frequency_offset_ratio)
        self._dur_ratio = _require_positive("duration_ratio", duration_ratio)

    def _resolve(self, system) -> _ResolvedProtocol:
        omega_eff = _effective_rabi(self._o420, self._o1013, self._delta)
        t_gate = self._dur_ratio * 2 * np.pi / omega_eff
        if 2 * self._rise > t_gate:
            raise ValueError(f"need 2*rise_time_s <= t_gate ({2*self._rise} > {t_gate}).")
        omega_mod = self._mod_ratio * omega_eff
        delta_phase = self._freq_off_ratio * omega_eff
        env = lambda t, tg=t_gate: blackman_pulse(t, self._rise, tg)
        phase = lambda t: self._phase_amp * np.cos(omega_mod * t + self._phase_off) + delta_phase * t
        return self._laser_resolve(
            system, t_gate=t_gate, intermediate_detuning_rad_s=self._delta,
            omega_420_max=self._o420, omega_1013_max=self._o1013,
            envelope_420=env, phase_420=phase,
            envelope_1013=lambda t: 1.0, phase_1013=lambda t: 0.0,
        )


class ARProtocol(_LaserCZProtocol):
    """Amplitude-Robust CZ pulse: dual-sine 420 phase family with a Blackman envelope."""

    def __init__(
        self,
        *,
        intermediate_detuning_rad_s: float,
        omega_420_max_rad_s: float,
        omega_1013_max_rad_s: float,
        rise_time_s: float,
        modulation_frequency_ratio: float,
        phase_amplitude_1_rad: float,
        phase_offset_1_rad: float,
        phase_amplitude_2_rad: float,
        phase_offset_2_rad: float,
        frequency_offset_ratio: float,
        duration_ratio: float,
    ) -> None:
        self._delta = _require_finite("intermediate_detuning_rad_s", intermediate_detuning_rad_s)
        if self._delta == 0.0:
            raise ValueError("ARProtocol requires a nonzero intermediate_detuning_rad_s.")
        self._o420 = _require_positive("omega_420_max_rad_s", omega_420_max_rad_s)
        self._o1013 = _require_positive("omega_1013_max_rad_s", omega_1013_max_rad_s)
        self._rise = _require_positive("rise_time_s", rise_time_s)
        self._mod_ratio = _require_finite("modulation_frequency_ratio", modulation_frequency_ratio)
        self._amp1 = _require_finite("phase_amplitude_1_rad", phase_amplitude_1_rad)
        self._off1 = _require_finite("phase_offset_1_rad", phase_offset_1_rad)
        self._amp2 = _require_finite("phase_amplitude_2_rad", phase_amplitude_2_rad)
        self._off2 = _require_finite("phase_offset_2_rad", phase_offset_2_rad)
        self._freq_off_ratio = _require_finite("frequency_offset_ratio", frequency_offset_ratio)
        self._dur_ratio = _require_positive("duration_ratio", duration_ratio)

    def _resolve(self, system) -> _ResolvedProtocol:
        omega_eff = _effective_rabi(self._o420, self._o1013, self._delta)
        t_gate = self._dur_ratio * 2 * np.pi / omega_eff
        if 2 * self._rise > t_gate:
            raise ValueError(f"need 2*rise_time_s <= t_gate ({2*self._rise} > {t_gate}).")
        omega_mod = self._mod_ratio * omega_eff
        delta_phase = self._freq_off_ratio * omega_eff
        env = lambda t, tg=t_gate: blackman_pulse(t, self._rise, tg)
        phase = lambda t: (
            self._amp1 * np.sin(omega_mod * t + self._off1)
            + self._amp2 * np.sin(2 * omega_mod * t + self._off2)
            + delta_phase * t
        )
        return self._laser_resolve(
            system, t_gate=t_gate, intermediate_detuning_rad_s=self._delta,
            omega_420_max=self._o420, omega_1013_max=self._o1013,
            envelope_420=env, phase_420=phase,
            envelope_1013=lambda t: 1.0, phase_1013=lambda t: 0.0,
        )
