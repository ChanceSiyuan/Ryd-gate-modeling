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
    ) -> None:
        self.addressing = addressing or {}
        self.scatter_rate = scatter_rate
        self.omega_ramp_frac = omega_ramp_frac
        self.n_steps = n_steps
        self._H_pin_2atom: NDArray[np.complexfloating] | None = None

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
            }
        else:
            # LatticeSystem: absolute units
            return {
                "delta_start": x[0],
                "delta_end": x[1],
                "t_gate": x[2],
            }

    def phase_420(self, t: float, params: dict) -> complex:
        """Quadratic chirp phase for 420nm laser."""
        omega_0 = params["delta_start"]
        chirp = (params["delta_end"] - params["delta_start"]) / params["t_gate"]
        return np.exp(-1j * (omega_0 * t + 0.5 * chirp * t * t))

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
