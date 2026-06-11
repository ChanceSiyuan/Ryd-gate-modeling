"""Pulse layer: product Waveform/Pulse objects + kernel Blackman helpers.

Two layers share this module (and the Blackman window mathematics), neither
wraps the other:

- **Product API** (public): :class:`Waveform` and :class:`Pulse` — discrete
  (integer ns), values in rad/us, serializable, validated against
  :class:`~ryd_gate.protocols.channels.ChannelSpec` limits. This is the only
  supported way to express pulse shapes in user code.
- **Kernel helpers** (internal): ``blackman_window`` / ``blackman_pulse`` /
  ``blackman_pulse_sqrt`` — continuous-time functions in seconds that build
  the gate protocols' flat-top envelopes. Excluded from ``__all__``; not part
  of the public contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np

from ryd_gate.core.serialization import check_schema, json_ready, schema_tag
from ryd_gate.core.validation import ValidationIssue

if TYPE_CHECKING:
    from ryd_gate.protocols.channels import ChannelSpec

__all__ = ["Pulse", "Waveform"]

_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")

WaveformKind = Literal["constant", "ramp", "blackman", "interpolated", "custom"]

_WAVEFORM_KINDS = ("constant", "ramp", "blackman", "interpolated", "custom")


# ── Kernel helpers (internal; continuous time in seconds) ───────────────────


def blackman_window(t, t_rise):
    """Evaluate the Blackman window function.

    Internal kernel helper (gate protocols) — not public
    API; no stability guarantee.

    Parameters
    ----------
    t : array_like
        Time values.
    t_rise : float
        Rise time of the window.

    Returns
    -------
    numpy.ndarray
        Window amplitude in [0, 1].
    """
    return (0.42 - 0.5 * np.cos(2 * np.pi * t / (2 * t_rise)) +
            0.08 * np.cos(4 * np.pi * t / (2 * t_rise)))


def blackman_pulse(t, t_rise, t_gate):
    """Blackman-windowed flat-top pulse.

    Internal kernel helper (gate protocols) — not public
    API; no stability guarantee.

    Parameters
    ----------
    t : array_like
        Time values.
    t_rise : float
        Rise/fall time.
    t_gate : float
        Total gate duration (must be >= 2 * t_rise).

    Returns
    -------
    numpy.ndarray
        Pulse envelope.
    """
    if t_gate < 2 * t_rise:
        raise ValueError("t_gate is too small compared to t_rise")
    ret = (blackman_window(t, t_rise) * np.heaviside(t_rise - t, 1) +
           np.heaviside(t - t_rise, 0) * np.heaviside(t_gate - t - t_rise, 0) +
           blackman_window(t_gate - t, t_rise) *
           np.heaviside(t_rise - (t_gate - t), 1))
    return ret


def blackman_pulse_sqrt(t, t_rise, t_gate):
    """Square-root of the Blackman-windowed flat-top pulse.

    Internal kernel helper (gate protocols) — not public
    API; no stability guarantee.

    Parameters
    ----------
    t : array_like
        Time values.
    t_rise : float
        Rise/fall time.
    t_gate : float
        Total gate duration (must be >= 2 * t_rise).

    Returns
    -------
    numpy.ndarray
        Square-root pulse envelope.
    """
    return np.sqrt(np.maximum(blackman_pulse(t, t_rise, t_gate), 0))


# ── Product layer (public; discrete ns, values in rad/us) ───────────────────


def _check_duration(duration_ns) -> int:
    if not isinstance(duration_ns, (int, np.integer)) or isinstance(duration_ns, bool) or duration_ns <= 0:
        raise ValueError(f"duration_ns must be a positive integer, got {duration_ns!r}.")
    return int(duration_ns)


def _check_finite(value, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a float, got {value!r}.") from None
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    return value


@dataclass(frozen=True)
class Waveform:
    """Scalar time-dependent control value: integer-ns duration, rad/us values.

    A kind-tagged immutable dataclass (one class, five constructors), which
    keeps serialization, equality, and validation uniform. Constructed via
    ``Waveform.constant`` / ``ramp`` / ``blackman`` / ``interpolated`` /
    ``custom`` — the direct constructor is not the supported path.
    """

    duration_ns: int
    kind: WaveformKind
    params: Mapping[str, Any] = field(default_factory=dict)
    samples: tuple[float, ...] | None = None
    unit: Literal["rad_per_us"] = "rad_per_us"

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration_ns", _check_duration(self.duration_ns))
        if self.kind not in _WAVEFORM_KINDS:
            raise ValueError(f"kind must be one of {_WAVEFORM_KINDS}, got {self.kind!r}.")
        if self.unit != "rad_per_us":
            raise ValueError(f"unit must be 'rad_per_us', got {self.unit!r}.")

    # ── Constructors ────────────────────────────────────────────────────

    @classmethod
    def constant(cls, duration_ns: int, value: float) -> "Waveform":
        """Fixed value for a fixed duration."""
        return cls(
            duration_ns=_check_duration(duration_ns),
            kind="constant",
            params={"value": _check_finite(value, "value")},
        )

    @classmethod
    def ramp(cls, duration_ns: int, start: float, stop: float) -> "Waveform":
        """Linear sweep from *start* to *stop*."""
        return cls(
            duration_ns=_check_duration(duration_ns),
            kind="ramp",
            params={
                "start": _check_finite(start, "start"),
                "stop": _check_finite(stop, "stop"),
            },
        )

    @classmethod
    def blackman(cls, duration_ns: int, peak: float | None = None, area: float | None = None) -> "Waveform":
        """Pure Blackman window: zero endpoints, maximum *peak* at the center.

        Exactly one of ``peak`` (rad/us) or ``area`` (radians) must be given.
        The area form computes ``peak = area / (0.42 * duration_us)`` (the
        closed-form window integral). Note this is the single pure window —
        the gate protocols' flat-top envelopes are a different (kernel) shape.
        """
        duration_ns = _check_duration(duration_ns)
        if (peak is None) == (area is None):
            raise ValueError("give exactly one of peak= or area=.")
        if peak is None:
            area = _check_finite(area, "area")
            if area == 0.0:
                raise ValueError("area must be nonzero.")
            duration_us = duration_ns * 1e-3
            peak = area / (0.42 * duration_us)
        else:
            peak = _check_finite(peak, "peak")
        return cls(duration_ns=duration_ns, kind="blackman", params={"peak": peak})

    @classmethod
    def interpolated(cls, duration_ns: int, times_ns, values) -> "Waveform":
        """Piecewise-linear waveform through (times_ns, values) control points."""
        duration_ns = _check_duration(duration_ns)
        times = tuple(float(t) for t in times_ns)
        vals = tuple(_check_finite(v, "values") for v in values)
        if len(times) != len(vals):
            raise ValueError(f"times_ns and values must have equal length, got {len(times)} and {len(vals)}.")
        if len(times) < 2:
            raise ValueError("interpolated waveform needs at least two control points.")
        if times[0] != 0.0 or times[-1] != float(duration_ns):
            raise ValueError(f"times_ns must start at 0 and end at duration_ns={duration_ns}.")
        if any(t1 >= t2 for t1, t2 in zip(times, times[1:])):
            raise ValueError("times_ns must be strictly increasing.")
        return cls(
            duration_ns=duration_ns,
            kind="interpolated",
            params={"times_ns": times, "values": vals},
        )

    @classmethod
    def custom(cls, samples, dt_ns: int = 1) -> "Waveform":
        """Explicit sample array on a uniform grid; linear between samples."""
        if not isinstance(dt_ns, (int, np.integer)) or isinstance(dt_ns, bool) or dt_ns <= 0:
            raise ValueError(f"dt_ns must be a positive integer, got {dt_ns!r}.")
        values = tuple(_check_finite(v, "samples") for v in samples)
        if len(values) < 2:
            raise ValueError("custom waveform needs at least two samples.")
        return cls(
            duration_ns=(len(values) - 1) * int(dt_ns),
            kind="custom",
            params={"dt_ns": int(dt_ns)},
            samples=values,
        )

    # ── Evaluation ──────────────────────────────────────────────────────

    def value_at_ns(self, t_ns: float) -> float:
        """Value in rad/us at time *t_ns*, clamped into [0, duration_ns]."""
        t = min(max(float(t_ns), 0.0), float(self.duration_ns))
        if self.kind == "constant":
            return self.params["value"]
        if self.kind == "ramp":
            frac = t / self.duration_ns
            return self.params["start"] + (self.params["stop"] - self.params["start"]) * frac
        if self.kind == "blackman":
            # pure window == kernel blackman_window with t_rise = duration/2
            return self.params["peak"] * float(blackman_window(t, self.duration_ns / 2))
        if self.kind == "interpolated":
            return float(np.interp(t, self.params["times_ns"], self.params["values"]))
        # custom
        assert self.samples is not None  # guaranteed by Waveform.custom
        dt = self.params["dt_ns"]
        grid = np.arange(len(self.samples)) * dt
        return float(np.interp(t, grid, self.samples))

    def value_at_s(self, t_s: float) -> float:
        """Value in rad/s at SI time *t_s* (kernel units)."""
        return self.value_at_ns(t_s * 1e9) * 1e6

    def sample(self, dt_ns: int = 1) -> np.ndarray:
        """Values at 0, dt_ns, ... including duration_ns; dt must divide duration."""
        if not isinstance(dt_ns, (int, np.integer)) or isinstance(dt_ns, bool) or dt_ns <= 0:
            raise ValueError(f"dt_ns must be a positive integer, got {dt_ns!r}.")
        if self.duration_ns % int(dt_ns) != 0:
            raise ValueError(
                f"dt_ns={dt_ns} must divide duration_ns={self.duration_ns}."
            )
        times = np.arange(0, self.duration_ns + int(dt_ns), int(dt_ns), dtype=float)
        return np.array([self.value_at_ns(t) for t in times], dtype=float)

    def integral_rad(self, dt_ns: int = 1) -> float:
        """Pulse area in radians: trapezoid of rad/us samples over microseconds."""
        values = self.sample(dt_ns)
        return float(_trapezoid(values, dx=int(dt_ns) * 1e-3))

    def first_value(self) -> float:
        return self.value_at_ns(0.0)

    def last_value(self) -> float:
        return self.value_at_ns(self.duration_ns)

    # ── Serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "schema": schema_tag("waveform"),
            "kind": self.kind,
            "duration_ns": self.duration_ns,
            "params": json_ready(dict(self.params), "waveform.params"),
            "samples": list(self.samples) if self.samples is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Waveform":
        check_schema(data, "waveform")
        kind = data.get("kind")
        duration = int(data["duration_ns"])
        params = data.get("params", {})
        if kind == "constant":
            return cls.constant(duration, params["value"])
        if kind == "ramp":
            return cls.ramp(duration, params["start"], params["stop"])
        if kind == "blackman":
            return cls.blackman(duration, peak=params["peak"])
        if kind == "interpolated":
            return cls.interpolated(duration, params["times_ns"], params["values"])
        if kind == "custom":
            return cls.custom(data["samples"], dt_ns=params["dt_ns"])
        raise ValueError(f"unknown waveform kind {kind!r}.")


@dataclass(frozen=True)
class Pulse:
    """One laser drive segment: amplitude + detuning waveforms, phase bookkeeping.

    Targets are not stored here — they belong to ``Sequence.add`` (Stage 2).
    """

    amplitude: Waveform
    detuning: Waveform
    phase_rad: float = 0.0
    post_phase_shift_rad: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.amplitude, Waveform) or not isinstance(self.detuning, Waveform):
            raise ValueError("amplitude and detuning must be Waveform objects.")
        if self.amplitude.duration_ns != self.detuning.duration_ns:
            raise ValueError(
                f"amplitude duration {self.amplitude.duration_ns} ns != "
                f"detuning duration {self.detuning.duration_ns} ns."
            )
        object.__setattr__(self, "phase_rad", _check_finite(self.phase_rad, "phase_rad"))
        object.__setattr__(
            self,
            "post_phase_shift_rad",
            _check_finite(self.post_phase_shift_rad, "post_phase_shift_rad"),
        )

    # ── Constructors ────────────────────────────────────────────────────

    @classmethod
    def constant(
        cls,
        duration_ns: int,
        amplitude: float,
        detuning: float,
        phase_rad: float = 0.0,
        post_phase_shift_rad: float = 0.0,
    ) -> "Pulse":
        """Square pulse: fixed amplitude and detuning for *duration_ns*."""
        return cls(
            amplitude=Waveform.constant(duration_ns, amplitude),
            detuning=Waveform.constant(duration_ns, detuning),
            phase_rad=phase_rad,
            post_phase_shift_rad=post_phase_shift_rad,
        )

    @classmethod
    def constant_amplitude(
        cls,
        amplitude: float,
        detuning: Waveform,
        phase_rad: float = 0.0,
        post_phase_shift_rad: float = 0.0,
    ) -> "Pulse":
        """Fixed amplitude under a detuning waveform (e.g. an adiabatic sweep)."""
        return cls(
            amplitude=Waveform.constant(detuning.duration_ns, amplitude),
            detuning=detuning,
            phase_rad=phase_rad,
            post_phase_shift_rad=post_phase_shift_rad,
        )

    @classmethod
    def constant_detuning(
        cls,
        amplitude: Waveform,
        detuning: float,
        phase_rad: float = 0.0,
        post_phase_shift_rad: float = 0.0,
    ) -> "Pulse":
        """Shaped amplitude at fixed detuning (e.g. a Blackman rise/fall)."""
        return cls(
            amplitude=amplitude,
            detuning=Waveform.constant(amplitude.duration_ns, detuning),
            phase_rad=phase_rad,
            post_phase_shift_rad=post_phase_shift_rad,
        )

    # ── Properties / validation ─────────────────────────────────────────

    @property
    def duration_ns(self) -> int:
        return self.amplitude.duration_ns

    def validate(self, channel: "ChannelSpec") -> list[ValidationIssue]:
        """Check this pulse against one channel's limits (no system, no backend)."""
        return _channel_limit_issues(self, channel)

    # ── Serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "schema": schema_tag("pulse"),
            "amplitude": self.amplitude.to_dict(),
            "detuning": self.detuning.to_dict(),
            "phase_rad": self.phase_rad,
            "post_phase_shift_rad": self.post_phase_shift_rad,
            "metadata": json_ready(dict(self.metadata), "pulse.metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Pulse":
        check_schema(data, "pulse")
        return cls(
            amplitude=Waveform.from_dict(data["amplitude"]),
            detuning=Waveform.from_dict(data["detuning"]),
            phase_rad=data.get("phase_rad", 0.0),
            post_phase_shift_rad=data.get("post_phase_shift_rad", 0.0),
            metadata=dict(data.get("metadata", {})),
        )


def _channel_limit_issues(pulse: Pulse, channel: "ChannelSpec") -> list[ValidationIssue]:
    """Shared pulse-vs-channel limit rules.

    The single implementation behind both ``Pulse.validate`` and
    ``DeviceSpec.validate_pulse`` — one rule set, two entry points.
    """
    issues: list[ValidationIssue] = []
    duration = pulse.duration_ns
    if duration < channel.min_duration_ns:
        issues.append(ValidationIssue(
            "error", "pulse.min_duration",
            f"pulse duration {duration} ns is below channel minimum {channel.min_duration_ns} ns.",
            ("pulse", "duration_ns"),
        ))
    if channel.max_duration_ns is not None and duration > channel.max_duration_ns:
        issues.append(ValidationIssue(
            "error", "pulse.max_duration",
            f"pulse duration {duration} ns exceeds channel maximum {channel.max_duration_ns} ns.",
            ("pulse", "duration_ns"),
        ))
    if duration % channel.clock_period_ns != 0:
        issues.append(ValidationIssue(
            "error", "pulse.clock_period",
            f"pulse duration {duration} ns is not a multiple of the channel clock period "
            f"{channel.clock_period_ns} ns.",
            ("pulse", "duration_ns"),
        ))
    if channel.max_abs_amplitude_rad_per_us is not None:
        max_amp = float(np.max(np.abs(pulse.amplitude.sample(1))))
        if max_amp > channel.max_abs_amplitude_rad_per_us:
            issues.append(ValidationIssue(
                "error", "pulse.amplitude_limit",
                f"max |amplitude| {max_amp:.6g} rad/us exceeds channel limit "
                f"{channel.max_abs_amplitude_rad_per_us:.6g} rad/us.",
                ("pulse", "amplitude"),
            ))
    if channel.max_abs_detuning_rad_per_us is not None:
        max_det = float(np.max(np.abs(pulse.detuning.sample(1))))
        if max_det > channel.max_abs_detuning_rad_per_us:
            issues.append(ValidationIssue(
                "error", "pulse.detuning_limit",
                f"max |detuning| {max_det:.6g} rad/us exceeds channel limit "
                f"{channel.max_abs_detuning_rad_per_us:.6g} rad/us.",
                ("pulse", "detuning"),
            ))
    return issues
