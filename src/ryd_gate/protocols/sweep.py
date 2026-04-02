"""Unified sweep protocol for detuning sweeps with local addressing.

Works with both AtomicSystem (2-atom 3-level, phase modulation via solve_gate)
and LatticeSystem (N-atom 2-level, piecewise-constant via solve_lattice).

Parameters x = [delta_start, delta_end, t_sweep]

For AtomicSystem: parameters are scaled by rabi_eff / time_scale.
For LatticeSystem: parameters are in absolute units.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.protocols.base import Protocol

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SweepProtocol(Protocol):
    """Detuning sweep protocol with optional local addressing.

    The instantaneous two-photon detuning sweeps from ``delta_start``
    to ``delta_end`` over the gate duration, encoded as a quadratic
    phase on the 420nm laser::

        φ(t) = ω₀·t + (α/2)·t²

    Parameters
    ----------
    addressing : dict mapping atom index → detuning (rad/s), or None.
        For 2-atom systems: ``{0: detuning}`` pins Atom A.
        For lattice systems: ``{i: delta_i, ...}`` pins selected atoms.
    scatter_rate : float
        Scattering rate on addressed atoms' ground state (Hz).
        Only used with AtomicSystem (2-atom).
    omega_ramp_frac : float
        Fraction of t_sweep over which Ω ramps from 0 to 1.
        Only used with LatticeSystem (N-atom).
    n_steps : int
        Number of piecewise-constant time steps for lattice evolution.
    """

    def __init__(
        self,
        addressing: dict[int, float] | None = None,
        scatter_rate: float = 0.0,
        omega_ramp_frac: float = 0.1,
        n_steps: int = 200,
        ac_stark_shift: float = 0.0,
    ) -> None:
        self.addressing = addressing or {}
        self.scatter_rate = scatter_rate
        self.omega_ramp_frac = omega_ramp_frac
        self.n_steps = n_steps
        self.ac_stark_shift = ac_stark_shift
        self._H_pin_2atom: NDArray[np.complexfloating] | None = None
        self._stark_phase_table: tuple | None = None

    # -- Protocol interface ------------------------------------------------

    @property
    def n_params(self) -> int:
        return 3

    def validate_params(self, x: list[float]) -> None:
        if len(x) != 3:
            raise ValueError(
                f"SweepProtocol requires 3 parameters "
                f"[delta_start, delta_end, t_sweep], got {len(x)}."
            )

    def unpack_params(self, x: list[float], system) -> dict:
        """Unpack parameters. Scales by rabi_eff/time_scale for AtomicSystem."""
        if hasattr(system, "rabi_eff"):
            # AtomicSystem: parameters normalized by Ω_eff and T_scale
            return {
                "delta_start": x[0] * system.rabi_eff,
                "delta_end": x[1] * system.rabi_eff,
                "t_gate": x[2] * system.time_scale,
                "t_rise": system.t_rise,
                "blackmanflag": system.blackmanflag,
                "_system_type": "atomic",
            }
        else:
            # LatticeSystem: absolute units
            return {
                "delta_start": x[0],
                "delta_end": x[1],
                "t_gate": x[2],
                "Omega": getattr(system, "Omega", 1.0),
                "omega_ramp_frac": self.omega_ramp_frac,
                "_system_type": "lattice",
            }

    def _ensure_stark_table(self, params: dict) -> None:
        """Precompute cumulative AC Stark phase correction table."""
        if self._stark_phase_table is not None:
            return
        from ryd_gate.blackman import blackman_pulse

        t_gate = params["t_gate"]
        t_rise = params.get("t_rise", 0)
        use_blackman = params.get("blackmanflag", True)
        n_pts = 2000
        ts = np.linspace(0, t_gate, n_pts)
        if use_blackman and t_rise > 0:
            A2 = np.array([blackman_pulse(ti, t_rise, t_gate) ** 2 for ti in ts])
        else:
            A2 = np.ones(n_pts)
        # Cumulative integral of Δ_AC(t') = ac_stark_shift * A²(t')
        from scipy.integrate import cumulative_trapezoid
        cum_phase = np.zeros(n_pts)
        cum_phase[1:] = cumulative_trapezoid(self.ac_stark_shift * A2, ts)
        self._stark_phase_table = (ts, cum_phase)

    def phase_420(self, t: float, params: dict) -> complex:
        """Quadratic chirp phase for 420nm laser.

        When ``ac_stark_shift`` is nonzero, applies feed-forward
        compensation by subtracting the accumulated AC Stark phase
        from the 420nm amplitude envelope::

            φ(t) = ω₀·t + (α/2)·t² − ∫₀ᵗ Δ_AC(t') dt'

        where Δ_AC(t) = ac_stark_shift · A²(t).
        """
        omega_0 = params["delta_start"]
        chirp = (params["delta_end"] - params["delta_start"]) / params["t_gate"]
        base_phase = omega_0 * t + 0.5 * chirp * t * t

        if self.ac_stark_shift != 0:
            self._ensure_stark_table(params)
            ts_table, cum_table = self._stark_phase_table
            correction = float(np.interp(t, ts_table, cum_table))
            base_phase -= correction

        return np.exp(-1j * base_phase)

    @property
    def required_channels(self) -> frozenset[str]:
        """Channels depend on system type; return union of both modes."""
        return frozenset({
            "drive_420", "drive_420_dag", "lightshift_zero",
            "global_X", "global_n",
        })

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        """Return drive coefficients for either atomic or lattice mode."""
        if params.get("_system_type") == "lattice":
            t_gate = params["t_gate"]
            frac = np.clip(t / t_gate, 0, 1) if t_gate > 0 else 0
            Delta_t = params["delta_start"] + (params["delta_end"] - params["delta_start"]) * frac
            Omega = params.get("Omega", 1.0)
            ramp_frac = params.get("omega_ramp_frac", self.omega_ramp_frac)
            ramp_time = ramp_frac * t_gate
            Omega_t = Omega if ramp_time == 0 else Omega * min(1.0, t / ramp_time)
            return {
                "global_X": Omega_t / 2,
                "global_n": -Delta_t,
            }
        else:
            # Atomic mode: quadratic chirp phase + Blackman amplitude
            from ryd_gate.blackman import blackman_pulse

            phase = self.phase_420(t, params)
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

    # -- Extra static Hamiltonian terms ------------------------------------

    def get_ham_const_additions(self) -> "NDArray[np.complexfloating] | None":
        """Return 784nm pinning + scattering operators for 2-atom systems.

        Built lazily on first call so that LatticeSystem users don't pay
        the cost of constructing 2-atom operators they won't use.
        """
        if self._H_pin_2atom is None and 0 in self.addressing:
            from ryd_gate.core.atomic_system import build_atom_a_projector
            N = 3
            H_pin = -self.addressing[0] * build_atom_a_projector(2, n_levels=N)
            H_scat = -1j * self.scatter_rate / 2 * build_atom_a_projector(0, n_levels=N)
            self._H_pin_2atom = H_pin + H_scat
        return self._H_pin_2atom

    def get_pin_deltas(self, N: int) -> np.ndarray:
        """Return per-site detuning array for lattice systems."""
        deltas = np.zeros(N)
        for idx, val in self.addressing.items():
            if idx < N:
                deltas[idx] = val
        return deltas
