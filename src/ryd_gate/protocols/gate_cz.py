"""CZ gate pulse protocols: a clean pulse container + Time-Optimal / Amplitude-
Robust builders + Double-ARP.

A CZ pulse is the complex 420 nm and 1013 nm laser drives.  The rb87 seven-level
system (``rb87_7_mp`` / ``rb87_7_pm``) builds *unit-Rabi, phase-free* transition
blocks (relative CG/dipole ratios only), so the protocol owns the amplitudes and
phases:

    Omega_420(t)  = omega_420_max  * A_420(s)  * exp(-i * phi_420(s))
    Omega_1013(t) = omega_1013_max * A_1013(s) * exp(-i * phi_1013(s))

with normalized time ``s = t/t_gate in [0, 1]``, ``A_* in [0,1]`` and ``phi_*`` in
radians.  These coefficients multiply the unit blocks at compile time.

- :class:`CZProtocol` — the container, and the **only** rb87 seven-level laser-
  domain protocol: four normalized functions + the two max Rabis + ``t_gate``.
  It also runs on ``analog_3`` (where the 1013 leg is a static coupling, not a
  driven block).  An adiabatic pulse is just a ``CZProtocol`` whose 420 phase is the
  chirp integral :func:`phase_from_chirp` — no dedicated class.
- :class:`TOProtocol` / :class:`ARProtocol` — *builders* ``x -> CZProtocol``
  (Time-Optimal cosine phase / Amplitude-Robust dual-sine phase) for the optimizer.
- :func:`phase_from_chirp` — integrate a chirp ``f(t)`` into an optical phase (pure
  pulse construction; no system, no effective theory).
- :class:`EffectiveCZProtocol` — drives the *full* 3x3 ``{0,1,r}`` effective
  Hamiltonian (incl. the ``|0>-|r>`` ``K0r`` leg) directly.  Built by
  :func:`ryd_gate.core.effective_theory.lower_cz_to_effective_01r`.
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
    meta = getattr(system, "meta", None)
    return meta("laser_channel_ratios", {}) if callable(meta) else {}


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
    meta = getattr(system, "meta", None)
    tag = meta("level_structure") if callable(meta) else None
    if tag in _UNIT_RABI_TAGS:
        base420, base1013 = _CZ_DEFAULT_OMEGA_420, _CZ_DEFAULT_OMEGA_1013
    else:
        base420, base1013 = 1.0, 1.0
    o420 = base420 if omega_420_max is None else float(omega_420_max)
    o1013 = base1013 if omega_1013_max is None else float(omega_1013_max)
    return o420, o1013


def cz_effective_rabi(system, omega_420_max: float, omega_1013_max: float) -> tuple[float, float]:
    """``(rabi_eff, time_scale)`` from the two max Rabis and the system's Delta.

    Reproduces the old build-time ``rabi_eff = rabi_420*rabi_1013/(2|Delta|)`` /
    ``time_scale = 2*pi/rabi_eff`` (same operands and order), now that the Rabi
    scale lives in the protocol rather than the Hamiltonian blocks.
    """
    rabi_eff = omega_420_max * omega_1013_max / (2 * abs(float(system.meta("Delta"))))
    return rabi_eff, 2 * np.pi / rabi_eff


# ── builder closure factories (default-arg capture, no late binding) ──────────


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


class CZProtocol(Protocol):
    """A concrete CZ pulse: four normalized functions of ``s = t/t_gate in [0,1]``
    (420 amplitude/phase, 1013 amplitude/phase), the two max Rabi amplitudes, and
    ``t_gate``.

    The 420/1013 drives are ``omega_*_max * A_*(s) * exp(-i*phi_*(s))`` (+ h.c.);
    they multiply the unit-Rabi system blocks at compile time.  Omit ``A_1013`` /
    ``phi_1013`` for a constant 1013 coupling (``A_1013 = 1``, ``phi_1013 = 0``).
    ``omega_*_max`` default to the fixed σ⁻/σ⁺ (``rb87_7_mp``) canonical Rabis on
    a unit-Rabi rb87 system when left ``None`` (pass them explicitly for the
    ``rb87_7_pm`` manifold); on full-Rabi systems the default is a no-op ``1.0``.
    """

    def __init__(
        self,
        *,
        t_gate: float,
        A_420,
        phi_420,
        A_1013=None,
        phi_1013=None,
        omega_420_max: float | None = None,
        omega_1013_max: float | None = None,
        n_steps: int = 200,
    ) -> None:
        if t_gate <= 0:
            raise ValueError("t_gate must be positive.")
        self._t_gate = float(t_gate)
        self._A_420 = A_420
        self._phi_420 = phi_420
        self._A_1013 = A_1013 if A_1013 is not None else (lambda s: 1.0)
        self._phi_1013 = phi_1013 if phi_1013 is not None else (lambda s: 0.0)
        self._omega_420_max = None if omega_420_max is None else float(omega_420_max)
        self._omega_1013_max = None if omega_1013_max is None else float(omega_1013_max)
        self.n_steps = int(n_steps)
        # The IR calls get_drive_coefficients once per channel per step (same t);
        # cache the last evaluation so the 12-channel rb87 dict is built once/step.
        self._cache_key: tuple | None = None
        self._cache_coeffs: dict[str, complex] | None = None

    @property
    def t_gate(self) -> float:
        return self._t_gate

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

    @property
    def n_params(self) -> int:
        return 0

    def validate_params(self, x) -> None:
        if len(x):
            raise ValueError(f"{type(self).__name__} takes no x parameters; got {len(x)}.")

    def unpack_params(self, x, system) -> dict:
        self.validate_params(x)
        o420, o1013 = cz_rabi_maxes(system, self._omega_420_max, self._omega_1013_max)
        return {
            "t_gate": self._t_gate,
            "theta": 0.0,
            "omega_420_max": o420,
            "omega_1013_max": o1013,
            "laser_channel_ratios": _laser_ratios(system),
        }

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        key = (t, id(params))
        if key == self._cache_key:
            return self._cache_coeffs
        s = t / params["t_gate"]
        ratios = params.get("laser_channel_ratios", {})
        # Expand the physical laser coefficients onto the primitive channels via
        # the manifold CG/dipole ratios; the compiler auto-adds each leg's h.c.
        c420 = params["omega_420_max"] * self._A_420(s) * np.exp(-1j * self._phi_420(s))
        coeffs = {chan: c420 * ratio for chan, ratio in ratios.get("420", {}).items()}
        if "1013" in ratios:
            c1013 = params["omega_1013_max"] * self._A_1013(s) * np.exp(-1j * self._phi_1013(s))
            for chan, ratio in ratios["1013"].items():
                coeffs[chan] = c1013 * ratio
        self._cache_key, self._cache_coeffs = key, coeffs
        return coeffs

    def phase_420(self, t: float, params: dict) -> complex:
        return np.exp(-1j * self._phi_420(t / params["t_gate"]))

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

    def pulse_traces(self, t: float, params: dict) -> dict[str, float]:
        """The 420/1013 laser amplitudes **and** chirps (all in rad/s).

        The ``dot_phi_*`` traces are the time derivative of each optical phase
        ``phi_*`` in ``Omega(t) = Omega_max * A(t) * exp(-i phi(t))`` — the
        instantaneous laser frequency offset (e.g. the 420 detuning sweep of a
        chirped pulse), computed by finite difference (:meth:`_dot_phi`).
        """
        t_gate = params["t_gate"]
        s = t / t_gate
        return {
            r"$\Omega_{420}$": params["omega_420_max"] * float(self._A_420(s)),
            r"$\Omega_{1013}$": params["omega_1013_max"] * float(self._A_1013(s)),
            r"$\dot\phi_{420}$": self._dot_phi(self._phi_420, s, t_gate),
            r"$\dot\phi_{1013}$": self._dot_phi(self._phi_1013, s, t_gate),
        }

    def plot(self, system=None, **kwargs):
        """Stacked plot: the two amplitudes and the two phase chirps each get their
        own subplot (shared time axis), since amplitudes (MHz) and chirps live on
        different scales.  Pass ``stacked=False`` for the single-axis view.  See
        :meth:`ryd_gate.protocols.base.Protocol.plot`.
        """
        kwargs.setdefault("stacked", True)
        return super().plot(system, **kwargs)


class TOProtocol:
    """Time-Optimal CZ *builder*: ``x -> CZProtocol`` with cosine phase modulation.

    ``x = [A, omega/Omega_eff, phi0, delta/Omega_eff, theta, T/T_scale]``.  Holds
    only optimization metadata; :meth:`build` constructs the concrete pulse.
    """

    n_params = 6
    theta_index = 4
    t_gate_index = 5

    def __init__(
        self,
        *,
        omega_420_max: float | None = None,
        omega_1013_max: float | None = None,
        blackman: bool = True,
        n_steps: int = 200,
    ) -> None:
        self._omega_420_max = omega_420_max
        self._omega_1013_max = omega_1013_max
        self._blackman = bool(blackman)
        self.n_steps = n_steps

    def validate_params(self, x) -> None:
        if len(x) != 6:
            raise ValueError(f"TO parameters must be a list of 6 elements. Got {len(x)} elements.")

    def get_optimization_bounds(self) -> tuple:
        return (
            (-np.pi, np.pi),
            (-10, 10),
            (-np.pi, np.pi),
            (-2, 2),
            (-np.inf, np.inf),
            (-np.pi, np.pi),
        )

    def unpack_params(self, x, system) -> dict:
        """Back-compat surface: report/MC read ``theta``/``t_gate`` off the builder."""
        self.validate_params(x)
        o420, o1013 = cz_rabi_maxes(system, self._omega_420_max, self._omega_1013_max)
        _, time_scale = cz_effective_rabi(system, o420, o1013)
        return {"t_gate": x[5] * time_scale, "theta": x[4]}

    def build(self, x, system) -> CZProtocol:
        self.validate_params(x)
        o420, o1013 = cz_rabi_maxes(system, self._omega_420_max, self._omega_1013_max)
        rabi_eff, time_scale = cz_effective_rabi(system, o420, o1013)
        t_gate = x[5] * time_scale
        t_rise = system.meta("t_rise", 0.0)
        omega = x[1] * rabi_eff
        delta = x[3] * rabi_eff
        return CZProtocol(
            t_gate=t_gate,
            A_420=_blackman_envelope(t_rise, t_gate) if self._blackman else _flat_envelope(),
            phi_420=_to_phase(x[0], omega, x[2], delta, t_gate),
            omega_420_max=o420,
            omega_1013_max=o1013,
            n_steps=self.n_steps,
        )


class ARProtocol:
    """Amplitude-Robust CZ *builder*: ``x -> CZProtocol`` with dual-sine phase.

    ``x = [omega/Omega_eff, A1, phi1, A2, phi2, delta/Omega_eff, T/T_scale, theta]``.
    """

    n_params = 8
    theta_index = 7
    t_gate_index = 6

    def __init__(
        self,
        *,
        omega_420_max: float | None = None,
        omega_1013_max: float | None = None,
        blackman: bool = True,
        n_steps: int = 200,
    ) -> None:
        self._omega_420_max = omega_420_max
        self._omega_1013_max = omega_1013_max
        self._blackman = bool(blackman)
        self.n_steps = n_steps

    def validate_params(self, x) -> None:
        if len(x) != 8:
            raise ValueError(f"AR parameters must be a list of 8 elements. Got {len(x)} elements.")

    def get_optimization_bounds(self) -> tuple:
        return (
            (-10, 10),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-2, 2),
            (-np.inf, np.inf),
            (-np.pi, np.pi),
        )

    def unpack_params(self, x, system) -> dict:
        self.validate_params(x)
        o420, o1013 = cz_rabi_maxes(system, self._omega_420_max, self._omega_1013_max)
        _, time_scale = cz_effective_rabi(system, o420, o1013)
        return {"t_gate": x[6] * time_scale, "theta": x[7]}

    def build(self, x, system) -> CZProtocol:
        self.validate_params(x)
        o420, o1013 = cz_rabi_maxes(system, self._omega_420_max, self._omega_1013_max)
        rabi_eff, time_scale = cz_effective_rabi(system, o420, o1013)
        t_gate = x[6] * time_scale
        t_rise = system.meta("t_rise", 0.0)
        omega = x[0] * rabi_eff
        delta = x[5] * rabi_eff
        return CZProtocol(
            t_gate=t_gate,
            A_420=_blackman_envelope(t_rise, t_gate) if self._blackman else _flat_envelope(),
            phi_420=_ar_phase(x[1], x[2], x[3], x[4], omega, delta, t_gate),
            omega_420_max=o420,
            omega_1013_max=o1013,
            n_steps=self.n_steps,
        )


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


class EffectiveCZProtocol(Protocol):
    """Drive the *full* 3x3 ``{0,1,r}`` effective Hamiltonian directly.

    The effective pulse is a single matrix-valued callable ``h_eff(t)`` returning
    the 3x3 single-atom Hamiltonian in ``{|0>, |1>, |r>}`` order (the endpoint of
    the 7-level -> ``{0,1,r}`` reduction; build it with
    :func:`ryd_gate.core.effective_theory.lower_cz_to_effective_01r`).  It is
    realized faithfully on the ``01r`` level structure via its primitive
    ``E[r,1]`` / ``E[1,0]`` / ``E[r,0]`` transition channels and ``E[r,r]`` /
    ``E[1,1]`` diagonal-detuning channels — the off-diagonals are the
    ``[upper, lower]`` matrix elements (the compiler adds the h.c.) and the diagonal
    is referenced to ``|0>`` (the global / clock phase is an unobservable
    single-qubit Z):

        E[r,1]  = M[2, 1]   (|r><1|, K1r)        E[1,1] = M[1, 1] - M[0, 0]
        E[1,0]  = M[1, 0]   (|1><0|, K01)        E[r,r] = M[2, 2] - M[0, 0]
        E[r,0]  = M[2, 0]   (|r><0|, K0r)

    ``has_K01`` / ``has_K0r`` declare whether the ``|0>-|1>`` / ``|0>-|r>`` legs are
    driven (so the resonant-flop case stays a pure ``|1>-|r>`` drive).  The full
    effective model (nonzero K0r) is **exact-backend only**.
    """

    def __init__(
        self,
        *,
        t_gate: float,
        h_eff,
        n_steps: int = 200,
        has_K01: bool = True,
        has_K0r: bool = True,
    ) -> None:
        if t_gate <= 0:
            raise ValueError("t_gate must be positive.")
        self.t_gate = float(t_gate)
        self.n_steps = int(n_steps)
        self._h_eff = h_eff
        self._has_K01 = bool(has_K01)
        self._has_K0r = bool(has_K0r)
        self._cache_t: float | None = None
        self._cache_M = None

    @classmethod
    def from_components(
        cls,
        *,
        t_gate: float,
        omega_eff_fn,
        phi_fn,
        delta_R_fn=None,
        D0_fn=None,
        D1_fn=None,
        Dr_fn=None,
        K01_fn=None,
        K0r_fn=None,
        n_steps: int = 200,
    ) -> "EffectiveCZProtocol":
        """Build from the individual effective components (hand-construction).

        Off-diagonals: ``M[2,1] = (Omega_eff/2) e^{-i phi}`` (K1r), ``M[1,0] = K01``,
        ``M[2,0] = K0r``.  Diagonal: ``M[0,0]=D0``, ``M[1,1]=D1``,
        ``M[2,2]=Dr - delta_R``.  ``has_K01`` / ``has_K0r`` follow from whether
        ``K01_fn`` / ``K0r_fn`` are given.
        """

        def _call(fn, t):
            return 0.0 if fn is None else complex(fn(t))

        def h_eff(t):
            M = np.zeros((3, 3), dtype=complex)
            M[0, 0] = _call(D0_fn, t)
            M[1, 1] = _call(D1_fn, t)
            M[2, 2] = _call(Dr_fn, t) - _call(delta_R_fn, t)
            k1r = 0.5 * complex(omega_eff_fn(t)) * np.exp(-1j * float(phi_fn(t)))
            M[2, 1], M[1, 2] = k1r, np.conjugate(k1r)
            if K01_fn is not None:
                k01 = complex(K01_fn(t))
                M[1, 0], M[0, 1] = k01, np.conjugate(k01)
            if K0r_fn is not None:
                k0r = complex(K0r_fn(t))
                M[2, 0], M[0, 2] = k0r, np.conjugate(k0r)
            return M

        return cls(
            t_gate=t_gate,
            h_eff=h_eff,
            n_steps=n_steps,
            has_K01=K01_fn is not None,
            has_K0r=K0r_fn is not None,
        )

    @property
    def n_params(self) -> int:
        return 0

    def validate_params(self, x) -> None:
        if len(x):
            raise ValueError(f"EffectiveCZProtocol takes no x parameters; got {len(x)}.")

    def unpack_params(self, x, system) -> dict:
        self.validate_params(x)
        return {"t_gate": self.t_gate, "theta": 0.0, "n_sites": getattr(system, "N", None)}

    @property
    def required_channels(self) -> frozenset[str]:
        channels = {"E[r,1]", "E[r,r]", "E[1,1]"}
        if self._has_K01:
            channels.add("E[1,0]")
        if self._has_K0r:
            channels.add("E[r,0]")
        return frozenset(channels)

    def _matrix(self, t: float) -> np.ndarray:
        # The IR calls get_drive_coefficients once per channel per step (same t);
        # cache the last evaluation so h_eff (two SW reductions) runs once per t.
        if self._cache_t is None or t != self._cache_t:
            self._cache_M = np.asarray(self._h_eff(t), dtype=complex)
            self._cache_t = t
        return self._cache_M

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        M = self._matrix(t)
        coeffs = {
            "E[r,1]": complex(M[2, 1]),            # |r><1| (K1r)
            "E[1,1]": complex(M[1, 1] - M[0, 0]),  # |1> detuning (delta_hf)
            "E[r,r]": complex(M[2, 2] - M[0, 0]),  # |r> detuning (delta_R)
        }
        if self._has_K01:
            coeffs["E[1,0]"] = complex(M[1, 0])    # |1><0| (K01)
        if self._has_K0r:
            coeffs["E[r,0]"] = complex(M[2, 0])    # |r><0| (K0r)
        return coeffs

    def pulse_traces(self, t: float, params: dict) -> dict[str, float]:
        M = self._matrix(t)
        return {
            r"$|K_{1r}|$": float(abs(M[2, 1])),
            r"$\Delta_R$": float(np.real(M[2, 2] - M[0, 0])),
        }
