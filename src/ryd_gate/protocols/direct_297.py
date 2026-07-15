"""297 nm single-photon laser protocols for the ``rb87_297_clock_4`` model.

The ``rb87_297_clock_4`` system builds *unit-Rabi* transition blocks whose
``E[r,1]`` / ``E[r_garb,1]`` channel ratios carry only the relative σ⁻ branch
dipole factors, so the protocols own the physical amplitude

    Omega_r(t) = Omega_r_peak * A_297(s) * exp(-i * phi_297(s)),  s = t/t_gate,

with ``Omega_r_peak = single_photon_rabi(target branch)/sqrt(2)`` (the
clock-state factor) computed by :func:`ryd_gate.physics.direct_297_rabis` from
the beam power **at the atoms**, the top-hat beam area, and the Rydberg level
declared on the system.  The garbage-branch Rabi follows automatically from the
system's ``E[r_garb,1]`` channel ratio (same beam, weaker dipole).  The ARC
computation is lazy: nothing physical is evaluated until the protocol is bound
to a system.

- :class:`Direct297PiProtocol` — Blackman π-pulse ``|1⟩ → |r⟩`` (no phase),
  auto-calibrated to a π target pulse area.
- :class:`Direct297CZProtocol` — the CZ pulse container (single-beam analog of
  :class:`~ryd_gate.protocols.gate_cz.CZProtocol`): normalized amplitude and
  phase functions of ``s`` plus ``t_gate``.
- :class:`Direct297TOProtocol` — Time-Optimal cosine phase family with the
  explicit normalized fields (single-beam analog of
  :class:`~ryd_gate.protocols.gate_cz.TOProtocol`; no ``theta`` — the
  single-qubit Z is a scoring parameter in analysis scripts).
"""

from __future__ import annotations

import numpy as np

from ryd_gate.protocols.base import Protocol


def _system_ryd_level(system) -> int:
    """The bound ``rb87_297_clock_4`` system's Rydberg level (validated)."""
    ls = getattr(system, "level_structure", None)
    physical_model = getattr(ls, "physical_model", None) if ls is not None else None
    if physical_model != "rb87_297_clock_4":
        raise ValueError(
            "297 nm protocols require the `rb87_297_clock_4` atom model."
        )
    ryd_level = getattr(ls, "ryd_level", None)
    if ryd_level is None:
        raise ValueError(
            "297 nm protocols require a system whose level structure carries "
            "`ryd_level` (use level_structure('rb87_297_clock_4', ...))."
        )
    return int(ryd_level)


def _laser_ratios(system) -> dict:
    """The system's ``{"297": {...}}`` per-channel ratio map."""
    ls = getattr(system, "level_structure", None)
    ratios = getattr(ls, "laser_channel_ratios", None) if ls is not None else None
    return dict(ratios) if ratios else {}


def _omega_297_peak(system, power_at_atoms_w: float, beam_area_um2: float, cache: dict) -> float:
    """Peak target-branch Rabi (rad/s) for *system*'s Rydberg level, memoized in *cache*."""
    ryd_level = _system_ryd_level(system)
    if ryd_level not in cache:
        from ryd_gate.physics import direct_297_rabis

        cache[ryd_level] = float(
            direct_297_rabis(power_at_atoms_w, beam_area_um2, ryd_level=ryd_level)[0]
        )
    return cache[ryd_level]


class _Laser297Protocol(Protocol):
    """Shared single-beam 297 machinery (private): resolved context + drive."""

    _power_at_atoms_w: float
    _beam_area_um2: float

    def __init__(self) -> None:
        self._omega_by_level: dict[int, float] = {}  # lazy (first use pulls in ARC)
        # The IR calls get_drive_coefficients once per channel per step (same t);
        # cache the last evaluation so the coefficient dict is built once/step.
        self._cache_key: tuple | None = None
        self._cache_coeffs: dict[str, complex] | None = None

    @property
    def required_channels(self) -> frozenset[str]:
        return frozenset({"E[r,1]", "E[r_garb,1]"})

    def drive_channels(self, system) -> frozenset[str]:
        """The ``E[...]`` channels driven on *system*, from its ``297`` ratio group."""
        return frozenset(_laser_ratios(system).get("297", {}))

    def _omega_297(self, system) -> float:
        return _omega_297_peak(
            system, self._power_at_atoms_w, self._beam_area_um2, self._omega_by_level
        )

    def _pulse_functions(self, omega_297: float, t_gate: float):
        """``(A_297, phi_297)`` as functions of ``s``."""
        raise NotImplementedError

    def _resolved_t_gate(self, system, omega_297: float) -> float:
        raise NotImplementedError

    def _resolve(self, system) -> dict:
        omega_297 = self._omega_297(system)
        t_gate = self._resolved_t_gate(system, omega_297)
        a297, phi297 = self._pulse_functions(omega_297, t_gate)
        return {
            "t_gate": t_gate,
            "omega_297_max": omega_297,
            "laser_channel_ratios": _laser_ratios(system),
            "ryd_level": _system_ryd_level(system),
            "A_297": a297,
            "phi_297": phi297,
        }

    def get_drive_coefficients(self, t: float, ctx: dict) -> dict[str, complex]:
        key = (t, id(ctx))
        if key == self._cache_key:
            return self._cache_coeffs
        s = t / ctx["t_gate"]
        c297 = ctx["omega_297_max"] * ctx["A_297"](s) * np.exp(-1j * ctx["phi_297"](s))
        coeffs = {
            chan: c297 * ratio
            for chan, ratio in ctx["laser_channel_ratios"].get("297", {}).items()
        }
        self._cache_key, self._cache_coeffs = key, coeffs
        return coeffs

    def _pulse_traces_ctx(self, t: float, ctx: dict) -> dict[str, float]:
        """The 297 laser amplitude **and** chirp (both in rad/s).

        ``dot_phi_297`` is the time derivative of the optical phase ``phi_297``
        in ``Omega(t) = Omega_max * A(t) * exp(-i phi(t))`` — the instantaneous
        laser frequency offset (e.g. a detuning sweep), computed by the shared
        finite-difference helper :meth:`_LaserCZProtocol._dot_phi`.
        """
        from ryd_gate.protocols.gate_cz import _LaserCZProtocol

        t_gate = ctx["t_gate"]
        s = t / t_gate
        return {
            r"$\Omega_{297}$": ctx["omega_297_max"] * float(ctx["A_297"](s)),
            r"$\dot\phi_{297}$": _LaserCZProtocol._dot_phi(ctx["phi_297"], s, t_gate),
        }


class Direct297PiProtocol(_Laser297Protocol):
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
    """

    def __init__(
        self,
        power_at_atoms_w: float,
        beam_area_um2: float,
        *,
        t_rise_fraction: float = 0.15,
        t_gate: float | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= t_rise_fraction <= 0.5:
            raise ValueError("t_rise_fraction must be in [0, 0.5].")
        if t_gate is not None and t_gate <= 0:
            raise ValueError("t_gate must be positive.")
        self._power_at_atoms_w = float(power_at_atoms_w)
        self._beam_area_um2 = float(beam_area_um2)
        self._t_rise_fraction = float(t_rise_fraction)
        self._t_gate = None if t_gate is None else float(t_gate)
        self._auto_t_gate_by_level: dict[int, float] = {}

    def _amplitude(self, s):
        """Normalized envelope A(s) in [0, 1] over s = t/t_gate in [0, 1]."""
        if self._t_rise_fraction <= 0.0:
            return np.ones_like(np.asarray(s, dtype=float))
        from ryd_gate.physics import blackman_pulse

        return blackman_pulse(np.asarray(s, dtype=float), self._t_rise_fraction, 1.0)

    def _resolved_t_gate(self, system, omega_297: float) -> float:
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
                np.pi / (omega_297 * area_fraction)
            )
        return self._auto_t_gate_by_level[ryd_level]

    def _pulse_functions(self, omega_297: float, t_gate: float):
        del omega_297, t_gate
        return (lambda s: float(self._amplitude(s))), (lambda s: 0.0)

    def _pulse_traces_ctx(self, t: float, ctx: dict) -> dict[str, float]:
        s = t / ctx["t_gate"]
        return {r"$\Omega_{297}$": ctx["omega_297_max"] * float(self._amplitude(s))}


class Direct297CZProtocol(_Laser297Protocol):
    """A concrete 297 nm CZ pulse: normalized amplitude/phase of the single σ⁻
    beam as functions of ``s = t/t_gate in [0, 1]``, plus ``t_gate``.

    Single-beam analog of :class:`~ryd_gate.protocols.gate_cz.CZProtocol`: the
    drive is ``omega_297_max * A_297(s) * exp(-i*phi_297(s))`` (+ h.c.)
    multiplied onto the system's unit-Rabi ``E[r,1]`` / ``E[r_garb,1]`` channel
    ratios at compile time.  ``omega_297_max`` is not an input — it is computed
    lazily (ARC) from the beam power **at the atoms** and the top-hat beam area
    via :func:`ryd_gate.physics.direct_297_rabis`, with the Rydberg level read
    from the bound ``rb87_297_clock_4`` system (the protocol carries no
    independent physical-state identity).  Omit ``phi_297`` for a phase-flat
    pulse.  Everything is fixed at construction; the protocol is runtime-only.
    """

    def __init__(
        self,
        *,
        t_gate: float,
        A_297,
        phi_297=None,
        power_at_atoms_w: float,
        beam_area_um2: float,
    ) -> None:
        super().__init__()
        if t_gate <= 0:
            raise ValueError("t_gate must be positive.")
        self._t_gate = float(t_gate)
        self._A_297 = A_297
        self._phi_297 = phi_297 if phi_297 is not None else (lambda s: 0.0)
        self._power_at_atoms_w = float(power_at_atoms_w)
        self._beam_area_um2 = float(beam_area_um2)

    @property
    def t_gate(self) -> float:
        return self._t_gate

    def _resolved_t_gate(self, system, omega_297: float) -> float:
        del system, omega_297
        return self._t_gate

    def _pulse_functions(self, omega_297: float, t_gate: float):
        del omega_297, t_gate
        return self._A_297, self._phi_297


class Direct297TOProtocol(_Laser297Protocol):
    """Time-Optimal 297 CZ pulse: cosine phase family, fully specified.

    ``phi_297(s) = phase_amplitude * cos(omega * t + phase_offset) + delta * t``
    with ``t = s * t_gate``, ``omega = frequency_ratio * Omega_297``,
    ``delta = detuning_ratio * Omega_297`` and gate duration
    ``duration_ratio * T_297`` where ``T_297 = 2*pi/Omega_297`` and
    ``Omega_297 = omega_297_max`` (from the beam power/area and the bound
    system's Rydberg level, computed lazily via ARC).  No ``theta`` — the
    single-qubit Z is a scoring parameter in analysis scripts.
    """

    def __init__(
        self,
        power_at_atoms_w: float,
        beam_area_um2: float,
        *,
        phase_amplitude: float,
        frequency_ratio: float,
        phase_offset: float,
        detuning_ratio: float,
        duration_ratio: float,
        blackman: bool = True,
        t_rise_fraction: float = 0.15,
    ) -> None:
        super().__init__()
        if not 0.0 <= t_rise_fraction <= 0.5:
            raise ValueError("t_rise_fraction must be in [0, 0.5].")
        if duration_ratio <= 0:
            raise ValueError("duration_ratio must be positive.")
        self._power_at_atoms_w = float(power_at_atoms_w)
        self._beam_area_um2 = float(beam_area_um2)
        self.phase_amplitude = float(phase_amplitude)
        self.frequency_ratio = float(frequency_ratio)
        self.phase_offset = float(phase_offset)
        self.detuning_ratio = float(detuning_ratio)
        self._duration_ratio = float(duration_ratio)
        self._blackman = bool(blackman)
        self._t_rise_fraction = float(t_rise_fraction)

    @property
    def duration_ratio(self) -> float:
        return self._duration_ratio

    def _resolved_t_gate(self, system, omega_297: float) -> float:
        del system
        return self._duration_ratio * (2 * np.pi / omega_297)

    def _pulse_functions(self, omega_297: float, t_gate: float):
        from ryd_gate.protocols.gate_cz import _blackman_envelope, _flat_envelope, _to_phase

        omega = self.frequency_ratio * omega_297
        delta = self.detuning_ratio * omega_297
        envelope = (
            _blackman_envelope(self._t_rise_fraction * t_gate, t_gate)
            if self._blackman
            else _flat_envelope()
        )
        return envelope, _to_phase(self.phase_amplitude, omega, self.phase_offset, delta, t_gate)
