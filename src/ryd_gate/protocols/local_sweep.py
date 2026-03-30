"""Local addressing sweep protocol.

Sweeps the two-photon detuning linearly by chirping the 420nm laser
frequency, producing a quadratic phase φ(t) = δ_start·t + α/2·t².

Atom A is pinned to the ground state via a local 784nm light shift.

Parameters x = [δ_start / Ω_eff, δ_end / Ω_eff, T / T_scale]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.core.atomic_system import build_atom_a_projector
from ryd_gate.protocols.base import Protocol

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ryd_gate.core.atomic_system import AtomicSystem


class SweepAddressingProtocol(Protocol):
    """Sweep protocol with local 784nm pinning on Atom A.

    The 420nm laser frequency is linearly chirped so that the
    instantaneous two-photon detuning sweeps from ``delta_start``
    to ``delta_end`` over the gate duration.  This is encoded as a
    quadratic phase on the 420nm laser::

        φ(t) = ω₀·t + (α/2)·t²
        ω₀   = delta_start
        α    = (delta_end − delta_start) / t_gate

    Parameters
    ----------
    local_detuning_A : float
        AC Stark shift on Atom A's Rydberg states from 784nm laser (rad/s).
        Usually negative (e.g., -2π × 12 MHz).
    local_scattering_rate : float
        Scattering rate from 784nm laser on Atom A's ground state (Hz).
    """

    def __init__(
        self,
        local_detuning_A: float,
        local_scattering_rate: float = 35.0,
    ) -> None:
        self.local_detuning_A = local_detuning_A
        self.local_scattering_rate = local_scattering_rate

        # 3-level analog system: |g⟩=0, |e⟩=1, |r⟩=2
        N = 3
        self._H_local_pinning: NDArray[np.complexfloating] = (
            -self.local_detuning_A * build_atom_a_projector(2, n_levels=N)
        )
        self._H_local_scatter: NDArray[np.complexfloating] = (
            -1j * self.local_scattering_rate / 2 * build_atom_a_projector(0, n_levels=N)
        )

    # -- Protocol interface ------------------------------------------------

    @property
    def n_params(self) -> int:
        return 3

    def validate_params(self, x: list[float]) -> None:
        if len(x) != 3:
            raise ValueError(
                f"Sweep parameters must be a list of 3 elements. Got {len(x)} elements."
            )

    def unpack_params(self, x: list[float], system: AtomicSystem) -> dict:
        return {
            "delta_start": x[0] * system.rabi_eff,
            "delta_end": x[1] * system.rabi_eff,
            "t_gate": x[2] * system.time_scale,
        }

    def phase_420(self, t: float, params: dict) -> complex:
        omega_0 = params["delta_start"]
        chirp = (params["delta_end"] - params["delta_start"]) / params["t_gate"]
        return np.exp(-1j * (omega_0 * t + 0.5 * chirp * t * t))

    # -- Extra static Hamiltonian terms ------------------------------------

    def get_ham_const_additions(self) -> "NDArray[np.complexfloating]":
        """Return time-independent 784nm pinning + scattering operators.

        These must be added to ``ham_const_override`` when calling
        :func:`solve_gate`.
        """
        return self._H_local_pinning + self._H_local_scatter
