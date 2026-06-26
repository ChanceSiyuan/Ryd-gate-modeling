"""CZ gate pulse protocols: Time-Optimal, Amplitude-Robust, and Double-ARP.

- :class:`TOProtocol` — Time-Optimal: φ(t) = A·cos(ωt + φ₀) + δ·t,
  parameters x = [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale]
- :class:`ARProtocol` — Amplitude-Robust:
  φ(t) = A₁·sin(ωt + φ₁) + A₂·sin(2ωt + φ₂) + δ·t,
  parameters x = [ω/Ω_eff, A₁, φ₁, A₂, φ₂, δ/Ω_eff, T/T_scale, θ]
- :class:`DoubleARPProtocol` — two consecutive rapid-adiabatic-passage pulses
  for the seven-level Rb87 model; the effective two-photon Rabi envelope is
  mapped onto the dimensionless 420nm amplitude scale via the system metadata
  ``rabi_eff``, and the detuning sweep is encoded as the time derivative of
  the 420nm optical phase, matching the TO/AR convention.
"""

from __future__ import annotations

import numpy as np

from ryd_gate.protocols.base import Protocol


def _blackman_drive_coefficients(phase: complex, t: float, params: dict) -> dict[str, complex]:
    """Shared TO/AR drive: Blackman-enveloped 420 nm drive + its lightshift.

    ``phase`` is the protocol's ``exp(-i phi(t))`` -- the only piece that differs
    between the time-optimal (cosine) and amplitude-robust (dual-sine) schedules.
    """
    from ryd_gate.physics import blackman_pulse

    amplitude = (
        blackman_pulse(t, params["t_rise"], params["t_gate"])
        if params.get("blackmanflag", True)
        else 1.0
    )
    return {
        "drive_420": amplitude * phase,
        "drive_420_dag": amplitude * np.conjugate(phase),
        "lightshift_zero": amplitude * amplitude,
    }


class _LaserCarrier:
    """Mixin: optional rb87/analog laser overrides carried on a protocol.

    ``Delta_Hz`` / ``rabi_420_Hz`` / ``rabi_1013_Hz`` are the laser intensity /
    frequency knobs.  ``set_protocol`` reads :meth:`laser_kwargs` to forward the
    non-``None`` ones into system materialization; ``None`` means "use the
    ``param_set`` / preset default".
    """

    def __init__(self, *, Delta_Hz=None, rabi_420_Hz=None, rabi_1013_Hz=None, **kwargs) -> None:
        self.Delta_Hz = Delta_Hz
        self.rabi_420_Hz = rabi_420_Hz
        self.rabi_1013_Hz = rabi_1013_Hz
        super().__init__(**kwargs)

    def laser_kwargs(self) -> dict:
        return {
            k: v
            for k, v in (
                ("Delta_Hz", self.Delta_Hz),
                ("rabi_420_Hz", self.rabi_420_Hz),
                ("rabi_1013_Hz", self.rabi_1013_Hz),
            )
            if v is not None
        }


class TOProtocol(_LaserCarrier, Protocol):
    """Time-Optimal pulse protocol with cosine phase modulation."""

    @property
    def n_params(self) -> int:
        return 6

    @property
    def theta_index(self) -> int:
        return 4

    @property
    def t_gate_index(self) -> int:
        return 5

    def validate_params(self, x: list[float]) -> None:
        if len(x) != 6:
            raise ValueError(
                f"TO parameters must be a list of 6 elements. Got {len(x)} elements."
            )

    def unpack_params(self, x: list[float], system) -> dict:
        rabi_eff = system.meta("rabi_eff")
        time_scale = system.meta("time_scale")
        return {
            "phase_amp": x[0],
            "omega": x[1] * rabi_eff,
            "phase_init": x[2],
            "delta": x[3] * rabi_eff,
            "theta": x[4],
            "t_gate": x[5] * time_scale,
            "t_rise": system.meta("t_rise", 0.0),
            "blackmanflag": system.meta("blackmanflag", True),
        }

    def phase_420(self, t: float, params: dict) -> complex:
        return np.exp(
            -1j * (params["phase_amp"] * np.cos(params["omega"] * t + params["phase_init"])
                   + params["delta"] * t)
        )

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        """Blackman-enveloped 420 nm drive for the time-optimal / AR schedule."""
        return _blackman_drive_coefficients(self.phase_420(t, params), t, params)

    def get_optimization_bounds(self) -> tuple:
        return (
            (-np.pi, np.pi),
            (-10, 10),
            (-np.pi, np.pi),
            (-2, 2),
            (-np.inf, np.inf),
            (-np.pi, np.pi),
        )


class ARProtocol(_LaserCarrier, Protocol):
    """Amplitude-Robust pulse protocol with dual-sine phase modulation."""

    @property
    def n_params(self) -> int:
        return 8

    @property
    def theta_index(self) -> int:
        return 7

    @property
    def t_gate_index(self) -> int:
        return 6

    def validate_params(self, x: list[float]) -> None:
        if len(x) != 8:
            raise ValueError(
                f"AR parameters must be a list of 8 elements. Got {len(x)} elements."
            )

    def unpack_params(self, x: list[float], system) -> dict:
        rabi_eff = system.meta("rabi_eff")
        time_scale = system.meta("time_scale")
        return {
            "omega": x[0] * rabi_eff,
            "phase_amp1": x[1],
            "phase_init1": x[2],
            "phase_amp2": x[3],
            "phase_init2": x[4],
            "delta": x[5] * rabi_eff,
            "t_gate": x[6] * time_scale,
            "theta": x[7],
            "t_rise": system.meta("t_rise", 0.0),
            "blackmanflag": system.meta("blackmanflag", True),
        }

    def phase_420(self, t: float, params: dict) -> complex:
        return np.exp(
            -1j * (
                params["phase_amp1"] * np.sin(params["omega"] * t + params["phase_init1"])
                + params["phase_amp2"] * np.sin(2 * params["omega"] * t + params["phase_init2"])
                + params["delta"] * t
            )
        )

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        """Blackman-enveloped 420 nm drive for the time-optimal / AR schedule."""
        return _blackman_drive_coefficients(self.phase_420(t, params), t, params)

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


class DoubleARPProtocol(_LaserCarrier, Protocol):
    """Two consecutive rapid-adiabatic-passage pulses for a Rydberg CZ gate.

    Parameters are angular frequencies in rad/s and time in seconds.
    ``omega_max`` is the target effective two-photon Rabi frequency.  If omitted,
    the system's nominal ``rabi_eff`` is used.  The laser overrides ``Delta_Hz`` /
    ``rabi_420_Hz`` / ``rabi_1013_Hz`` set the rb87/analog operating point when
    this protocol is attached via ``set_protocol``.
    """

    def __init__(
        self,
        *,
        omega_max: float | None = None,
        delta_max: float,
        t_gate: float,
        sigma: float | None = None,
        omega_scale: float = 1.0,
        delta_offset: float = 0.0,
        theta: float = np.pi,
        n_steps: int = 320,
        compensate_stark: bool = False,
        stark_compensation_sign: float = -1.0,
        phase_samples: int | None = None,
        Delta_Hz: float | None = None,
        rabi_420_Hz: float | None = None,
        rabi_1013_Hz: float | None = None,
    ) -> None:
        if t_gate <= 0:
            raise ValueError("t_gate must be positive.")
        if delta_max <= 0:
            raise ValueError("delta_max must be positive.")
        if n_steps < 1:
            raise ValueError("n_steps must be positive.")

        self.omega_max = None if omega_max is None else float(omega_max)
        self.delta_max = float(delta_max)
        self.t_gate = float(t_gate)
        self.sigma = float(0.175 * t_gate if sigma is None else sigma)
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.omega_scale = float(omega_scale)
        self.delta_offset = float(delta_offset)
        self.theta = float(theta)
        self.n_steps = int(n_steps)
        self.compensate_stark = bool(compensate_stark)
        self.stark_compensation_sign = float(stark_compensation_sign)
        self.phase_samples = None if phase_samples is None else int(phase_samples)
        self.t_pulse = 0.5 * self.t_gate
        self.offset_a = np.exp(-((self.t_pulse / 2.0) ** 4) / self.sigma**4)
        super().__init__(
            Delta_Hz=Delta_Hz, rabi_420_Hz=rabi_420_Hz, rabi_1013_Hz=rabi_1013_Hz
        )

    @property
    def n_params(self) -> int:
        return 0

    def validate_params(self, x) -> None:
        if len(x) != 0:
            raise ValueError(
                f"DoubleARPProtocol takes no x parameters; got {len(x)}."
            )

    def unpack_params(self, x, system) -> dict:
        self.validate_params(x)
        rabi_eff = system.meta("rabi_eff", None) if hasattr(system, "meta") else None
        if rabi_eff is None or float(rabi_eff) == 0.0:
            raise ValueError(
                "DoubleARPProtocol requires system metadata 'rabi_eff' to map "
                "effective Rabi frequency onto the 420nm amplitude scale."
            )
        omega_max = float(rabi_eff) if self.omega_max is None else self.omega_max
        params = {
            "t_gate": self.t_gate,
            "theta": self.theta,
            "omega_max": omega_max,
            "delta_max": self.delta_max,
            "delta_offset": self.delta_offset,
            "rabi_eff": float(rabi_eff),
            "n_steps": self.n_steps,
            "compensate_stark": self.compensate_stark,
            "stark_compensation_sign": self.stark_compensation_sign,
            "n_sites": getattr(system, "N", None),
            "pin_deltas": {},
            "scatter_rates": {},
            "static_overlays": [],
        }
        params.update(self._stark_params(system) if self.compensate_stark else self._empty_stark_params())
        phase_t, phase_values = self._build_phase_table(params)
        params["phase_t"] = phase_t
        params["phase_values"] = phase_values
        return params

    @property
    def required_channels(self) -> frozenset[str]:
        return frozenset({"drive_420", "drive_420_dag", "lightshift_zero"})

    def local_time(self, t: float) -> float:
        t_clamped = float(np.clip(t, 0.0, self.t_gate))
        return t_clamped if t_clamped < self.t_pulse else t_clamped - self.t_pulse

    def envelope(self, t: float) -> float:
        """Dimensionless super-Gaussian envelope with zero endpoints."""
        u = self.local_time(t)
        return float(
            (np.exp(-((u - self.t_pulse / 2.0) ** 4) / self.sigma**4) - self.offset_a)
            / (1.0 - self.offset_a)
        )

    def effective_omega(self, t: float, params: dict | None = None) -> float:
        omega_max = self.omega_max if params is None else params["omega_max"]
        if omega_max is None:
            raise ValueError("effective_omega() needs params when omega_max=None.")
        return self.omega_scale * float(omega_max) * self.envelope(t)

    def detuning(self, t: float, params: dict | None = None) -> float:
        delta_max = self.delta_max if params is None else params["delta_max"]
        delta_offset = self.delta_offset if params is None else params["delta_offset"]
        u = self.local_time(t)
        return float(delta_max) * np.sin(np.pi * (u / self.t_pulse - 0.5)) + float(delta_offset)

    def pulse_traces(self, t: float, params: dict) -> dict[str, float]:
        """Physical effective Rabi and ARP detuning sweep at time *t*."""
        return {
            r"$\Omega_{\rm eff}$": self.effective_omega(t, params),
            r"$\Delta$": self.detuning(t, params),
        }

    def stark_shift(self, t: float, params: dict) -> float:
        """Second-order differential Stark shift ``E_r - E_1``."""
        amplitude = self.effective_omega(t, params) / float(params["rabi_eff"])
        return float(params["stark_r"]) - float(params["stark_1_per_amp2"]) * amplitude * amplitude

    def chirp_detuning(self, t: float, params: dict) -> float:
        """420 phase derivative after optional dynamic Stark compensation."""
        desired = self.detuning(t, params)
        if not params.get("compensate_stark", False):
            return desired
        return desired + float(params.get("stark_compensation_sign", -1.0)) * self.stark_shift(t, params)

    def phase(self, t: float, params: dict | None = None) -> float:
        """Return phase whose derivative is the compensated ARP chirp."""
        if params is not None and "phase_t" in params and "phase_values" in params:
            t_clamped = float(np.clip(t, 0.0, self.t_gate))
            return float(np.interp(t_clamped, params["phase_t"], params["phase_values"]))

        # Analytic fallback for plotting before a system-bound params dict exists.
        delta_max = self.delta_max if params is None else params["delta_max"]
        delta_offset = self.delta_offset if params is None else params["delta_offset"]
        t_clamped = float(np.clip(t, 0.0, self.t_gate))
        u = self.local_time(t_clamped)
        arp_part = -float(delta_max) * self.t_pulse / np.pi * np.sin(np.pi * u / self.t_pulse)
        return arp_part + float(delta_offset) * t_clamped

    def phase_420(self, t: float, params: dict) -> complex:
        return np.exp(-1j * self.phase(t, params))

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        phase = self.phase_420(t, params)
        amplitude = self.effective_omega(t, params) / float(params["rabi_eff"])
        return {
            "drive_420": amplitude * phase,
            "drive_420_dag": amplitude * np.conjugate(phase),
            "lightshift_zero": amplitude * amplitude,
        }

    def _build_phase_table(self, params: dict) -> tuple[np.ndarray, np.ndarray]:
        n = self.phase_samples or max(4 * self.n_steps + 1, 1001)
        if n < 2:
            raise ValueError("phase_samples must be at least 2.")
        t_grid = np.linspace(0.0, self.t_gate, int(n))
        chirp = np.array([self.chirp_detuning(t, params) for t in t_grid], dtype=float)
        dt = np.diff(t_grid)
        phase = np.zeros_like(t_grid)
        phase[1:] = np.cumsum(0.5 * (chirp[:-1] + chirp[1:]) * dt)
        return t_grid, phase

    def _empty_stark_params(self) -> dict[str, float]:
        return {
            "stark_1_per_amp2": 0.0,
            "stark_r": 0.0,
            "stark_garb": 0.0,
        }

    def _stark_params(self, system) -> dict[str, float]:
        try:
            h_const = system.blocks.get("H_const").matrix
            h420 = system.blocks.get("drive_420").matrix
            h1013 = system.blocks.get("H_1013").matrix
        except Exception as exc:  # pragma: no cover - defensive message
            raise ValueError(
                "Stark compensation requires rb87_7 local matrix blocks "
                "'H_const', 'drive_420', and 'H_1013'."
            ) from exc

        h_const = np.asarray(h_const)
        h420 = np.asarray(h420)
        h1013 = np.asarray(h1013)
        if h_const.shape[0] < 7 or h420.shape[0] < 7 or h1013.shape[0] < 7:
            raise ValueError("Stark compensation currently targets the rb87_7 level structure.")

        mid_energies = np.real(np.diag(h_const)[2:5])
        if np.any(np.abs(mid_energies) < 1e-15):
            raise ValueError("Intermediate-state energy denominator is zero; cannot compensate Stark shift.")

        stark_1_per_amp2 = -sum(abs(h420[e, 1]) ** 2 / mid_energies[e - 2] for e in (2, 3, 4))
        stark_r = -sum(abs(h1013[5, e]) ** 2 / mid_energies[e - 2] for e in (2, 3, 4))
        stark_garb = -sum(abs(h1013[6, e]) ** 2 / mid_energies[e - 2] for e in (2, 3, 4))
        return {
            "stark_1_per_amp2": float(np.real(stark_1_per_amp2)),
            "stark_r": float(np.real(stark_r)),
            "stark_garb": float(np.real(stark_garb)),
        }
