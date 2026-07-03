"""297 nm single-photon laser protocols for the ``rb87_297_clock_4`` model.

The ``rb87_297_clock_4`` system builds *unit-Rabi* transition blocks whose
``E[r,1]`` / ``E[r_garb,1]`` channel ratios carry only the relative σ⁻ branch
dipole factors, so the protocols own the physical amplitude

    Omega_r(t) = Omega_r_peak * A_297(s) * exp(-i * phi_297(s)),  s = t/t_gate,

with ``Omega_r_peak = single_photon_rabi(target branch)/sqrt(2)`` (the
clock-state factor) computed by :func:`ryd_gate.physics.direct_297_rabis` from
the beam power **at the atoms**, the top-hat beam area, and the Rydberg level
declared on the system.  The garbage-branch Rabi follows automatically from the
system's ``E[r_garb,1]`` channel ratio (same beam, weaker dipole).

- :class:`Direct297PiProtocol` — Blackman π-pulse ``|1⟩ → |r⟩`` (no phase),
  auto-calibrated to a π target pulse area.
- :class:`Direct297CZProtocol` — the CZ pulse container (single-beam analog of
  :class:`~ryd_gate.protocols.gate_cz.CZProtocol`): normalized amplitude and
  phase functions of ``s`` plus ``t_gate``.
- :class:`Direct297TOProtocol` — Time-Optimal *builder* ``x ->
  Direct297CZProtocol`` with cosine phase modulation (single-beam analog of
  :class:`~ryd_gate.protocols.gate_cz.TOProtocol`).
"""

from __future__ import annotations

import numpy as np

from ryd_gate.protocols.base import Protocol


def _system_ryd_level(system) -> int:
    """The bound ``rb87_297_clock_4`` system's Rydberg level (validated)."""
    meta = getattr(system, "meta", None)
    physical_model = meta("physical_model", None) if callable(meta) else None
    if physical_model != "rb87_297_clock_4":
        raise ValueError(
            "297 nm protocols require the `rb87_297_clock_4` atom model."
        )
    ryd_level = meta("ryd_level", None) if callable(meta) else None
    if ryd_level is None:
        raise ValueError(
            "297 nm protocols require a system with `ryd_level` metadata "
            "(use RydbergSystem.set_atom_level('rb87_297_clock_4', ...))."
        )
    return int(ryd_level)


def _laser_ratios(system) -> dict:
    """The system's ``{"297": {...}}`` per-channel ratio map."""
    meta = getattr(system, "meta", None)
    return meta("laser_channel_ratios", {}) if callable(meta) else {}


def _omega_297_peak(system, power_at_atoms_w: float, beam_area_um2: float, cache: dict) -> float:
    """Peak target-branch Rabi (rad/s) for *system*'s Rydberg level, memoized in *cache*."""
    ryd_level = _system_ryd_level(system)
    if ryd_level not in cache:
        from ryd_gate.physics import direct_297_rabis

        cache[ryd_level] = float(
            direct_297_rabis(power_at_atoms_w, beam_area_um2, ryd_level=ryd_level)[0]
        )
    return cache[ryd_level]


class Direct297PiProtocol(Protocol):
    """Blackman-windowed 297 nm σ⁻ π-pulse ``|1⟩ → |r⟩``.

    Parameters
    ----------
    power_at_atoms_w : float
        297 nm beam power (W) reaching the atoms.  Any optics loss between the
        source and the atoms is applied by the caller (notebooks/scripts);
        this protocol carries no optics-loss factor.
    beam_area_um2 : float
        Top-hat beam area (μm²) the power is spread over.
    t_rise_fraction : float, optional
        Blackman rise/fall time as a fraction of ``t_gate`` (default 0.15;
        must be <= 0.5).  ``0`` gives a square pulse.
    t_gate : float or None, optional
        Total pulse duration (s).  ``None`` (default) calibrates ``t_gate``
        against the bound system so the target pulse area is exactly π,
        ``∫ Omega_r(t) dt = pi`` (for a square pulse this is
        ``t_pi = pi / Omega_r_peak``).
    n_steps : int, optional
        Piecewise-constant evolution steps for the exact backends.
    """

    def __init__(
        self,
        power_at_atoms_w: float,
        beam_area_um2: float,
        *,
        t_rise_fraction: float = 0.15,
        t_gate: float | None = None,
        n_steps: int = 200,
    ) -> None:
        if not 0.0 <= t_rise_fraction <= 0.5:
            raise ValueError("t_rise_fraction must be in [0, 0.5].")
        if t_gate is not None and t_gate <= 0:
            raise ValueError("t_gate must be positive.")
        self._power_at_atoms_w = float(power_at_atoms_w)
        self._beam_area_um2 = float(beam_area_um2)
        self._t_rise_fraction = float(t_rise_fraction)
        self._t_gate = None if t_gate is None else float(t_gate)
        self.n_steps = int(n_steps)
        self._omega_r_peak_by_level: dict[int, float] = {}  # lazy (first use pulls in ARC)
        self._auto_t_gate_by_level: dict[int, float] = {}

    def _omega_r_peak(self, system) -> float:
        """Physical peak target-branch Rabi (rad/s), incl. the 1/sqrt(2) clock factor."""
        return _omega_297_peak(
            system, self._power_at_atoms_w, self._beam_area_um2, self._omega_r_peak_by_level
        )

    def _resolved_t_gate(self, system, omega_r_peak: float) -> float:
        """Pulse duration; calibrated to a π target pulse area when not given."""
        if self._t_gate is not None:
            return self._t_gate
        ryd_level = _system_ryd_level(system)
        if ryd_level not in self._auto_t_gate_by_level:
            # The envelope shape in s = t/t_gate is duration-independent (the
            # rise is a fixed fraction), so ∫Ω dt = Ω_peak · F · t_gate with
            # F = ∫₀¹ A(s) ds; a π area gives t_gate = π / (Ω_peak · F).
            s = np.linspace(0.0, 1.0, 2001)
            area_fraction = float(np.trapezoid(self._amplitude(s), s))
            self._auto_t_gate_by_level[ryd_level] = float(
                np.pi / (omega_r_peak * area_fraction)
            )
        return self._auto_t_gate_by_level[ryd_level]

    def _amplitude(self, s):
        """Normalized envelope A(s) in [0, 1] over s = t/t_gate in [0, 1]."""
        if self._t_rise_fraction <= 0.0:
            return np.ones_like(np.asarray(s, dtype=float))
        from ryd_gate.physics import blackman_pulse

        return blackman_pulse(np.asarray(s, dtype=float), self._t_rise_fraction, 1.0)

    @property
    def n_params(self) -> int:
        return 0

    def validate_params(self, x) -> None:
        if len(x):
            raise ValueError(f"Direct297PiProtocol takes no x parameters; got {len(x)}.")

    @property
    def required_channels(self) -> frozenset[str]:
        return frozenset({"E[r,1]", "E[r_garb,1]"})

    def drive_channels(self, system) -> frozenset[str]:
        """The ``E[...]`` channels driven on *system*, from its ``297`` ratio group."""
        return frozenset(_laser_ratios(system).get("297", {}))

    def unpack_params(self, x, system) -> dict:
        self.validate_params(x)
        omega_r_peak = self._omega_r_peak(system)
        t_gate = self._resolved_t_gate(system, omega_r_peak)
        return {
            "t_gate": t_gate,
            "theta": 0.0,
            "omega_297_max": omega_r_peak,
            "laser_channel_ratios": _laser_ratios(system),
            "ryd_level": _system_ryd_level(system),
        }

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        s = t / params["t_gate"]
        c297 = params["omega_297_max"] * float(self._amplitude(s))
        return {
            chan: c297 * ratio
            for chan, ratio in params["laser_channel_ratios"].get("297", {}).items()
        }

    def pulse_traces(self, t: float, params: dict) -> dict[str, float]:
        s = t / params["t_gate"]
        return {r"$\Omega_{297}$": params["omega_297_max"] * float(self._amplitude(s))}


class Direct297CZProtocol(Protocol):
    """A concrete 297 nm CZ pulse: normalized amplitude/phase of the single σ⁻
    beam as functions of ``s = t/t_gate in [0, 1]``, plus ``t_gate``.

    Single-beam analog of :class:`~ryd_gate.protocols.gate_cz.CZProtocol`: the
    drive is ``omega_297_max * A_297(s) * exp(-i*phi_297(s))`` (+ h.c.)
    multiplied onto the system's unit-Rabi ``E[r,1]`` / ``E[r_garb,1]`` channel
    ratios at compile time.  ``omega_297_max`` is not an input — it is computed
    from the beam power **at the atoms** and the top-hat beam area via
    :func:`ryd_gate.physics.direct_297_rabis`, with the Rydberg level read from
    the bound ``rb87_297_clock_4`` system (the protocol carries no independent
    physical-state identity).  Omit ``phi_297`` for a phase-flat pulse.
    """

    def __init__(
        self,
        *,
        t_gate: float,
        A_297,
        phi_297=None,
        power_at_atoms_w: float,
        beam_area_um2: float,
        n_steps: int = 200,
    ) -> None:
        if t_gate <= 0:
            raise ValueError("t_gate must be positive.")
        self._t_gate = float(t_gate)
        self._A_297 = A_297
        self._phi_297 = phi_297 if phi_297 is not None else (lambda s: 0.0)
        self._power_at_atoms_w = float(power_at_atoms_w)
        self._beam_area_um2 = float(beam_area_um2)
        self.n_steps = int(n_steps)
        self._omega_by_level: dict[int, float] = {}  # lazy (first use pulls in ARC)
        # The IR calls get_drive_coefficients once per channel per step (same t);
        # cache the last evaluation so the coefficient dict is built once/step.
        self._cache_key: tuple | None = None
        self._cache_coeffs: dict[str, complex] | None = None

    @property
    def t_gate(self) -> float:
        return self._t_gate

    @property
    def n_params(self) -> int:
        return 0

    def validate_params(self, x) -> None:
        if len(x):
            raise ValueError(f"Direct297CZProtocol takes no x parameters; got {len(x)}.")

    @property
    def required_channels(self) -> frozenset[str]:
        return frozenset({"E[r,1]", "E[r_garb,1]"})

    def drive_channels(self, system) -> frozenset[str]:
        """The ``E[...]`` channels driven on *system*, from its ``297`` ratio group."""
        return frozenset(_laser_ratios(system).get("297", {}))

    def unpack_params(self, x, system) -> dict:
        self.validate_params(x)
        return {
            "t_gate": self._t_gate,
            "theta": 0.0,
            "omega_297_max": _omega_297_peak(
                system, self._power_at_atoms_w, self._beam_area_um2, self._omega_by_level
            ),
            "laser_channel_ratios": _laser_ratios(system),
            "ryd_level": _system_ryd_level(system),
        }

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        key = (t, id(params))
        if key == self._cache_key:
            return self._cache_coeffs
        s = t / params["t_gate"]
        c297 = params["omega_297_max"] * self._A_297(s) * np.exp(-1j * self._phi_297(s))
        coeffs = {
            chan: c297 * ratio
            for chan, ratio in params["laser_channel_ratios"].get("297", {}).items()
        }
        self._cache_key, self._cache_coeffs = key, coeffs
        return coeffs

    def pulse_traces(self, t: float, params: dict) -> dict[str, float]:
        """The 297 laser amplitude **and** chirp (both in rad/s).

        ``dot_phi_297`` is the time derivative of the optical phase ``phi_297``
        in ``Omega(t) = Omega_max * A(t) * exp(-i phi(t))`` — the instantaneous
        laser frequency offset (e.g. a detuning sweep), computed by the shared
        finite-difference helper :meth:`CZProtocol._dot_phi`.
        """
        from ryd_gate.protocols.gate_cz import CZProtocol

        t_gate = params["t_gate"]
        s = t / t_gate
        return {
            r"$\Omega_{297}$": params["omega_297_max"] * float(self._A_297(s)),
            r"$\dot\phi_{297}$": CZProtocol._dot_phi(self._phi_297, s, t_gate),
        }

    def plot(self, system=None, **kwargs):
        """Stacked plot (amplitude and chirp live on different scales); see
        :meth:`ryd_gate.protocols.base.Protocol.plot`."""
        kwargs.setdefault("stacked", True)
        return super().plot(system, **kwargs)


class Direct297TOProtocol:
    """Time-Optimal 297 CZ *builder*: ``x -> Direct297CZProtocol`` with cosine
    phase modulation (single-beam analog of
    :class:`~ryd_gate.protocols.gate_cz.TOProtocol`).

    ``x = [A, omega/Omega_297, phi0, delta/Omega_297, theta, T/T_297]`` with
    ``Omega_297 = omega_297_max`` (from the beam power/area and the bound
    system's Rydberg level) and ``T_297 = 2*pi/Omega_297``.  The phase family is
    ``phi_297(s) = A*cos(omega*s*t_gate + phi0) + delta*s*t_gate``.  Holds only
    optimization metadata; :meth:`build` constructs the concrete pulse.
    """

    n_params = 6
    theta_index = 4
    t_gate_index = 5

    def __init__(
        self,
        power_at_atoms_w: float,
        beam_area_um2: float,
        *,
        blackman: bool = True,
        t_rise_fraction: float = 0.15,
        n_steps: int = 200,
    ) -> None:
        if not 0.0 <= t_rise_fraction <= 0.5:
            raise ValueError("t_rise_fraction must be in [0, 0.5].")
        self._power_at_atoms_w = float(power_at_atoms_w)
        self._beam_area_um2 = float(beam_area_um2)
        self._blackman = bool(blackman)
        self._t_rise_fraction = float(t_rise_fraction)
        self.n_steps = int(n_steps)
        self._omega_by_level: dict[int, float] = {}

    def validate_params(self, x) -> None:
        if len(x) != 6:
            raise ValueError(f"Direct297TO parameters must be a list of 6 elements. Got {len(x)} elements.")

    def get_optimization_bounds(self) -> tuple:
        return (
            (-np.pi, np.pi),
            (-10, 10),
            (-np.pi, np.pi),
            (-2, 2),
            (-np.inf, np.inf),
            (-np.pi, np.pi),
        )

    def _omega_297(self, system) -> float:
        return _omega_297_peak(
            system, self._power_at_atoms_w, self._beam_area_um2, self._omega_by_level
        )

    def unpack_params(self, x, system) -> dict:
        """Back-compat surface: report/metrics read ``theta``/``t_gate`` off the builder."""
        self.validate_params(x)
        time_scale = 2 * np.pi / self._omega_297(system)
        return {"t_gate": x[5] * time_scale, "theta": x[4]}

    def build(self, x, system) -> Direct297CZProtocol:
        self.validate_params(x)
        from ryd_gate.protocols.gate_cz import _blackman_envelope, _flat_envelope, _to_phase

        omega_297 = self._omega_297(system)
        t_gate = x[5] * (2 * np.pi / omega_297)
        omega = x[1] * omega_297
        delta = x[3] * omega_297
        return Direct297CZProtocol(
            t_gate=t_gate,
            A_297=(
                _blackman_envelope(self._t_rise_fraction * t_gate, t_gate)
                if self._blackman
                else _flat_envelope()
            ),
            phi_297=_to_phase(x[0], omega, x[2], delta, t_gate),
            power_at_atoms_w=self._power_at_atoms_w,
            beam_area_um2=self._beam_area_um2,
            n_steps=self.n_steps,
        )
