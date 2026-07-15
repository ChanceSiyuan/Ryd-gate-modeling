"""CZ gate pulse protocols: a clean pulse container + Time-Optimal /
Amplitude-Robust phase families.

A CZ pulse is the complex 420 nm and 1013 nm laser drives.  The rb87 seven-level
system (``rb87_7_mp`` / ``rb87_7_pm``) builds *unit-Rabi, phase-free* transition
blocks (relative CG/dipole ratios only), so the protocol owns the amplitudes and
phases:

    Omega_420(t)  = omega_420_max  * A_420(s)  * exp(-i * phi_420(s))
    Omega_1013(t) = omega_1013_max * A_1013(s) * exp(-i * phi_1013(s))

with normalized time ``s = t/t_gate in [0, 1]``, ``A_* in [0,1]`` and ``phi_*`` in
radians.  These coefficients multiply the unit blocks at compile time.

Every CZ protocol is **fully specified at construction** and normalized: the
gate duration is ``duration_ratio`` in units of the system's effective-Rabi
time scale ``T_scale = 2*pi / Omega_eff`` (with ``Omega_eff =
omega_420_max * omega_1013_max / 2|Delta|`` on the unit-Rabi rb87 models, or
the system's own ``time_scale`` metadata elsewhere), resolved when the
protocol is bound to a system.  ``system.t_gate`` exposes the resolved value.

- :class:`CZProtocol` — the generic container: four normalized functions +
  the two max Rabis + ``duration_ratio``.  The callable pulse and all its
  parameters are fixed at construction; it is runtime-only (not serializable).
  An adiabatic pulse is just a ``CZProtocol`` whose 420 phase is the chirp
  integral :func:`phase_from_chirp` — no dedicated class.
- :class:`TOProtocol` / :class:`ARProtocol` — the Time-Optimal cosine /
  Amplitude-Robust dual-sine phase families, with explicit normalized fields.
  Neither carries the single-qubit Z ``theta`` — that is a scoring parameter
  computed in analysis scripts, not part of the pulse.
- :func:`phase_from_chirp` — integrate a chirp ``f(t)`` into an optical phase
  (pure pulse construction; no system, no effective theory).
"""

from __future__ import annotations

import numpy as np

from ryd_gate.protocols.base import Protocol

# Protocol-internal default 420/1013 single-photon Rabi maxes (rad/s), used when a
# CZ/TO/AR protocol is bound to a unit-Rabi rb87 seven-level system without an
# explicit ``omega_*_max``.  These are the canonical σ⁻/σ⁺ (``rb87_7_mp``) values
# and are a fixed protocol constant — never read from the system (its Rabi-bearing
# atom-level metadata).  For the σ⁺/σ⁻ (``rb87_7_pm``) manifold, pass omega
# explicitly.
_CZ_DEFAULT_OMEGA_420 = 2 * np.pi * 491e6
_CZ_DEFAULT_OMEGA_1013 = 2 * np.pi * 185e6

# Tags whose primitive E[...] channels are unit-Rabi (the protocol supplies the
# scale via its laser coefficient).  Other systems (analog_3, duck-typed) carry
# the full Rabi in the channel ratio, so the omega scaling is a no-op (1.0) there.
_UNIT_RABI_TAGS = frozenset({"rb87_7_mp", "rb87_7_pm"})


def _laser_ratios(system) -> dict:
    """The system's ``{"420": {...}, "1013": {...}}`` per-channel ratio map."""
    ls = getattr(system, "level_structure", None)
    ratios = getattr(ls, "laser_channel_ratios", None) if ls is not None else None
    return dict(ratios) if ratios else {}


# ── Rabi-scale resolvers ─────────────────────────────────────────────────────


def cz_rabi_maxes(system, omega_420_max=None, omega_1013_max=None) -> tuple[float, float]:
    """Resolve ``(omega_420_max, omega_1013_max)`` in rad/s.

    The Rabi *value* is never inferred from the system: an explicit
    ``omega_*_max`` always wins; otherwise the default is the fixed σ⁻/σ⁺
    protocol constant for unit-Rabi rb87 seven-level systems, or ``1.0`` (a
    no-op) for systems whose blocks already carry the full Rabi (analog_3,
    duck-typed test systems).  The system is consulted only for its
    level-structure *tag*, to choose scale-vs-no-op — not for the amplitude.
    """
    ls = getattr(system, "level_structure", None)
    tag = getattr(ls, "name", None) if ls is not None else None
    if tag in _UNIT_RABI_TAGS:
        base420, base1013 = _CZ_DEFAULT_OMEGA_420, _CZ_DEFAULT_OMEGA_1013
    else:
        base420, base1013 = 1.0, 1.0
    o420 = base420 if omega_420_max is None else float(omega_420_max)
    o1013 = base1013 if omega_1013_max is None else float(omega_1013_max)
    return o420, o1013


def cz_effective_rabi(system, omega_420_max: float, omega_1013_max: float) -> tuple[float, float]:
    """``(rabi_eff, time_scale)`` for a CZ pulse bound to *system*.

    On the unit-Rabi rb87 seven-level models this is ``rabi_eff =
    omega_420_max * omega_1013_max / (2|Delta|)`` with ``time_scale =
    2*pi/rabi_eff``.  Full-Rabi systems (analog_3, duck-typed contexts) carry
    their own ``rabi_eff`` / ``time_scale`` metadata, which is used directly —
    that is also how ``duration_ratio`` resolves on those systems.
    """
    ls = getattr(system, "level_structure", None)
    tag = getattr(ls, "name", None) if ls is not None else None
    if tag in _UNIT_RABI_TAGS:
        rabi_eff = omega_420_max * omega_1013_max / (2 * abs(float(ls.Delta)))
        return rabi_eff, 2 * np.pi / rabi_eff
    rabi_eff = getattr(ls, "rabi_eff", None) if ls is not None else None
    time_scale = getattr(ls, "time_scale", None) if ls is not None else None
    if time_scale is None and rabi_eff is not None:
        time_scale = 2 * np.pi / float(rabi_eff)
    if time_scale is None:
        raise ValueError(
            "cannot resolve the CZ time scale on this system: it is not a "
            "unit-Rabi rb87 seven-level model and carries no "
            "rabi_eff/time_scale metadata."
        )
    return (float(rabi_eff) if rabi_eff is not None else 2 * np.pi / float(time_scale),
            float(time_scale))


# ── pulse-shape closure factories (default-arg capture, no late binding) ─────


def _flat_envelope():
    return lambda s: 1.0


def _blackman_envelope(t_rise: float, t_gate: float):
    from ryd_gate.physics import blackman_pulse

    return lambda s: float(blackman_pulse(s * t_gate, t_rise, t_gate))


def _to_phase(phase_amp: float, omega: float, phase_init: float, delta: float, t_gate: float):
    return lambda s: phase_amp * np.cos(omega * (s * t_gate) + phase_init) + delta * (s * t_gate)


def _ar_phase(
    phase_amp1: float,
    phase_init1: float,
    phase_amp2: float,
    phase_init2: float,
    omega: float,
    delta: float,
    t_gate: float,
):
    return lambda s: (
        phase_amp1 * np.sin(omega * (s * t_gate) + phase_init1)
        + phase_amp2 * np.sin(2 * omega * (s * t_gate) + phase_init2)
        + delta * (s * t_gate)
    )


class _LaserCZProtocol(Protocol):
    """Shared 420/1013 laser machinery for the CZ protocol family (private).

    Subclasses store the normalized pulse definition and implement
    :meth:`_pulse_functions` returning the four ``s``-domain functions for a
    resolved binding.  The resolved context carries everything
    :meth:`get_drive_coefficients` needs, so evaluation never re-touches the
    system.
    """

    _omega_420_max: float | None
    _omega_1013_max: float | None
    _duration_ratio: float

    def __init__(self) -> None:
        # The IR calls get_drive_coefficients once per channel per step (same t);
        # cache the last evaluation so the 12-channel rb87 dict is built once/step.
        self._cache_key: tuple | None = None
        self._cache_coeffs: dict[str, complex] | None = None

    @property
    def duration_ratio(self) -> float:
        return self._duration_ratio

    @property
    def required_channels(self) -> frozenset[str]:
        # Canonical rb87 seven-level forward primitive channels.  The set actually
        # driven on a given system comes from :meth:`drive_channels`.
        return frozenset(
            {
                "E[e1,1]", "E[e2,1]", "E[e3,1]", "E[e1,0]", "E[e2,0]", "E[e3,0]",
                "E[r,e1]", "E[r,e2]", "E[r,e3]", "E[r_garb,e1]", "E[r_garb,e2]", "E[r_garb,e3]",
            }
        )

    def drive_channels(self, system) -> frozenset[str]:
        """Forward primitive ``E[...]`` channels driven on *system*, from its
        ``laser_channel_ratios`` (the 420 group; plus the 1013 group when the
        system has one — rb87 seven-level, but not the static-1013 analog_3)."""
        ratios = _laser_ratios(system)
        return frozenset(set(ratios.get("420", {})) | set(ratios.get("1013", {})))

    def _pulse_functions(self, rabi_eff: float, t_gate: float, system):
        """``(A_420, phi_420, A_1013, phi_1013)`` as functions of ``s``."""
        raise NotImplementedError

    def _resolve(self, system) -> dict:
        o420, o1013 = cz_rabi_maxes(system, self._omega_420_max, self._omega_1013_max)
        rabi_eff, time_scale = cz_effective_rabi(system, o420, o1013)
        t_gate = self._duration_ratio * time_scale
        a420, phi420, a1013, phi1013 = self._pulse_functions(rabi_eff, t_gate, system)
        return {
            "t_gate": t_gate,
            "omega_420_max": o420,
            "omega_1013_max": o1013,
            "laser_channel_ratios": _laser_ratios(system),
            "A_420": a420,
            "phi_420": phi420,
            "A_1013": a1013,
            "phi_1013": phi1013,
        }

    def get_drive_coefficients(self, t: float, ctx: dict) -> dict[str, complex]:
        key = (t, id(ctx))
        if key == self._cache_key:
            return self._cache_coeffs
        s = t / ctx["t_gate"]
        ratios = ctx.get("laser_channel_ratios", {})
        # Expand the physical laser coefficients onto the primitive channels via
        # the manifold CG/dipole ratios; the compiler auto-adds each leg's h.c.
        c420 = ctx["omega_420_max"] * ctx["A_420"](s) * np.exp(-1j * ctx["phi_420"](s))
        coeffs = {chan: c420 * ratio for chan, ratio in ratios.get("420", {}).items()}
        if "1013" in ratios:
            c1013 = ctx["omega_1013_max"] * ctx["A_1013"](s) * np.exp(-1j * ctx["phi_1013"](s))
            for chan, ratio in ratios["1013"].items():
                coeffs[chan] = c1013 * ratio
        self._cache_key, self._cache_coeffs = key, coeffs
        return coeffs

    @staticmethod
    def _dot_phi(phi_fn, s: float, t_gate: float, eps_s: float = 1e-5) -> float:
        """Instantaneous chirp ``dot_phi(t) = d/dt phi(s) = (d phi/ds) / t_gate`` (rad/s).

        ``phi`` is the optical phase in ``Omega(t) = Omega_max * A(s) * exp(-i phi(s))``
        with ``s = t / t_gate``; this is the time derivative of that phase, i.e. the
        instantaneous laser frequency offset / chirp.  Central finite difference away
        from the boundaries, one-sided within ``eps_s`` of ``s=0`` / ``s=1``; ``s`` is
        clamped to ``[0, 1]``.
        """
        s = float(np.clip(s, 0.0, 1.0))
        eps = float(eps_s)
        if s <= eps:
            d_ds = (float(phi_fn(s + eps)) - float(phi_fn(s))) / eps
        elif s >= 1.0 - eps:
            d_ds = (float(phi_fn(s)) - float(phi_fn(s - eps))) / eps
        else:
            d_ds = (float(phi_fn(s + eps)) - float(phi_fn(s - eps))) / (2.0 * eps)
        return d_ds / float(t_gate)

    def _pulse_traces_ctx(self, t: float, ctx: dict) -> dict[str, float]:
        """The 420/1013 laser amplitudes **and** chirps (all in rad/s).

        The ``dot_phi_*`` traces are the time derivative of each optical phase
        ``phi_*`` in ``Omega(t) = Omega_max * A(t) * exp(-i phi(t))`` — the
        instantaneous laser frequency offset (e.g. the 420 detuning sweep of a
        chirped pulse), computed by finite difference (:meth:`_dot_phi`).
        """
        t_gate = ctx["t_gate"]
        s = t / t_gate
        return {
            r"$\Omega_{420}$": ctx["omega_420_max"] * float(ctx["A_420"](s)),
            r"$\Omega_{1013}$": ctx["omega_1013_max"] * float(ctx["A_1013"](s)),
            r"$\dot\phi_{420}$": self._dot_phi(ctx["phi_420"], s, t_gate),
            r"$\dot\phi_{1013}$": self._dot_phi(ctx["phi_1013"], s, t_gate),
        }


class CZProtocol(_LaserCZProtocol):
    """A concrete CZ pulse: four normalized functions of ``s = t/t_gate in [0,1]``
    (420 amplitude/phase, 1013 amplitude/phase), the two max Rabi amplitudes, and
    the normalized ``duration_ratio``.

    The 420/1013 drives are ``omega_*_max * A_*(s) * exp(-i*phi_*(s))`` (+ h.c.);
    they multiply the unit-Rabi system blocks at compile time.  Omit ``A_1013`` /
    ``phi_1013`` for a constant 1013 coupling (``A_1013 = 1``, ``phi_1013 = 0``).
    ``omega_*_max`` default to the fixed σ⁻/σ⁺ (``rb87_7_mp``) canonical Rabis on
    a unit-Rabi rb87 system when left ``None`` (pass them explicitly for the
    ``rb87_7_pm`` manifold); on full-Rabi systems the default is a no-op ``1.0``.

    ``duration_ratio`` is the gate duration in units of the bound system's
    ``T_scale = 2*pi/Omega_eff`` (see :func:`cz_effective_rabi`); the resolved
    duration is ``system.t_gate``.  The callable pulse and all its parameters
    are fixed here at construction — the protocol is runtime-only and not
    serializable.
    """

    def __init__(
        self,
        *,
        duration_ratio: float,
        A_420,
        phi_420,
        A_1013=None,
        phi_1013=None,
        omega_420_max: float | None = None,
        omega_1013_max: float | None = None,
    ) -> None:
        super().__init__()
        if duration_ratio <= 0:
            raise ValueError("duration_ratio must be positive.")
        self._duration_ratio = float(duration_ratio)
        self._A_420 = A_420
        self._phi_420 = phi_420
        self._A_1013 = A_1013 if A_1013 is not None else (lambda s: 1.0)
        self._phi_1013 = phi_1013 if phi_1013 is not None else (lambda s: 0.0)
        self._omega_420_max = None if omega_420_max is None else float(omega_420_max)
        self._omega_1013_max = None if omega_1013_max is None else float(omega_1013_max)

    def _pulse_functions(self, rabi_eff: float, t_gate: float, system):
        del rabi_eff, t_gate, system
        return self._A_420, self._phi_420, self._A_1013, self._phi_1013


class TOProtocol(_LaserCZProtocol):
    """Time-Optimal CZ pulse: cosine 420 phase modulation, fully specified.

    ``phi_420(s) = phase_amplitude * cos(omega * t + phase_offset) + delta * t``
    with ``t = s * t_gate``, ``omega = frequency_ratio * Omega_eff`` and
    ``delta = detuning_ratio * Omega_eff``; the envelope is a Blackman flat-top
    (or flat when ``blackman=False``) and the gate duration is
    ``duration_ratio * T_scale``.  All normalized values resolve against the
    bound system's physical scales.  There is no single-qubit Z ``theta`` here —
    it is a scoring parameter that lives in analysis scripts.
    """

    def __init__(
        self,
        *,
        phase_amplitude: float,
        frequency_ratio: float,
        phase_offset: float,
        detuning_ratio: float,
        duration_ratio: float,
        omega_420_max: float | None = None,
        omega_1013_max: float | None = None,
        blackman: bool = True,
    ) -> None:
        super().__init__()
        if duration_ratio <= 0:
            raise ValueError("duration_ratio must be positive.")
        self.phase_amplitude = float(phase_amplitude)
        self.frequency_ratio = float(frequency_ratio)
        self.phase_offset = float(phase_offset)
        self.detuning_ratio = float(detuning_ratio)
        self._duration_ratio = float(duration_ratio)
        self._omega_420_max = None if omega_420_max is None else float(omega_420_max)
        self._omega_1013_max = None if omega_1013_max is None else float(omega_1013_max)
        self._blackman = bool(blackman)

    def _pulse_functions(self, rabi_eff: float, t_gate: float, system):
        ls = getattr(system, "level_structure", None)
        t_rise = (getattr(ls, "t_rise", None) if ls is not None else None) or 0.0
        omega = self.frequency_ratio * rabi_eff
        delta = self.detuning_ratio * rabi_eff
        envelope = _blackman_envelope(t_rise, t_gate) if self._blackman else _flat_envelope()
        phase = _to_phase(self.phase_amplitude, omega, self.phase_offset, delta, t_gate)
        return envelope, phase, (lambda s: 1.0), (lambda s: 0.0)


class ARProtocol(_LaserCZProtocol):
    """Amplitude-Robust CZ pulse: dual-sine 420 phase modulation, fully specified.

    ``phi_420(s) = phase_amplitude_1 * sin(omega * t + phase_offset_1)
    + phase_amplitude_2 * sin(2 * omega * t + phase_offset_2) + delta * t``
    with ``t = s * t_gate``, ``omega = frequency_ratio * Omega_eff`` and
    ``delta = detuning_ratio * Omega_eff``.  No single-qubit Z ``theta`` — that
    is a scoring parameter in analysis scripts.
    """

    def __init__(
        self,
        *,
        frequency_ratio: float,
        phase_amplitude_1: float,
        phase_offset_1: float,
        phase_amplitude_2: float,
        phase_offset_2: float,
        detuning_ratio: float,
        duration_ratio: float,
        omega_420_max: float | None = None,
        omega_1013_max: float | None = None,
        blackman: bool = True,
    ) -> None:
        super().__init__()
        if duration_ratio <= 0:
            raise ValueError("duration_ratio must be positive.")
        self.frequency_ratio = float(frequency_ratio)
        self.phase_amplitude_1 = float(phase_amplitude_1)
        self.phase_offset_1 = float(phase_offset_1)
        self.phase_amplitude_2 = float(phase_amplitude_2)
        self.phase_offset_2 = float(phase_offset_2)
        self.detuning_ratio = float(detuning_ratio)
        self._duration_ratio = float(duration_ratio)
        self._omega_420_max = None if omega_420_max is None else float(omega_420_max)
        self._omega_1013_max = None if omega_1013_max is None else float(omega_1013_max)
        self._blackman = bool(blackman)

    def _pulse_functions(self, rabi_eff: float, t_gate: float, system):
        ls = getattr(system, "level_structure", None)
        t_rise = (getattr(ls, "t_rise", None) if ls is not None else None) or 0.0
        omega = self.frequency_ratio * rabi_eff
        delta = self.detuning_ratio * rabi_eff
        envelope = _blackman_envelope(t_rise, t_gate) if self._blackman else _flat_envelope()
        phase = _ar_phase(
            self.phase_amplitude_1, self.phase_offset_1,
            self.phase_amplitude_2, self.phase_offset_2,
            omega, delta, t_gate,
        )
        return envelope, phase, (lambda s: 1.0), (lambda s: 0.0)


def phase_from_chirp(chirp_fn, t_gate: float, n_samples: int = 1001):
    """Integrate an instantaneous chirp into an optical phase.

    Returns ``phi(t) = ∫₀^t chirp(t') dt'`` as an O(1) interpolating callable over
    ``t in [0, t_gate]`` (trapezoidal cumulative integral on ``n_samples`` points).
    Use it for a :class:`CZProtocol` phase via ``phi_420 = lambda s: phi(s * t_gate)``.

    This is pure *pulse construction* — ``chirp_fn`` is a plain ``f(t)`` in rad/s
    (e.g. a bare detuning sweep, or a detuning-plus-compensation sum).  It holds no
    physics and needs no system; effective-theory / Stark logic lives elsewhere
    (see :func:`ryd_gate.core.effective_theory.lower_cz_to_effective_01r`).
    """
    t_gate = float(t_gate)
    n = int(n_samples)
    if t_gate <= 0:
        raise ValueError("t_gate must be positive.")
    if n < 2:
        raise ValueError("n_samples must be at least 2.")
    grid = np.linspace(0.0, t_gate, n)
    chirp = np.array([float(chirp_fn(float(t))) for t in grid], dtype=float)
    phase = np.zeros_like(grid)
    phase[1:] = np.cumsum(0.5 * (chirp[:-1] + chirp[1:]) * np.diff(grid))

    def phi(t, _g=grid, _p=phase, _tg=t_gate):
        return float(np.interp(float(np.clip(t, 0.0, _tg)), _g, _p))

    return phi
