"""Local addressing sweep protocol.

Sweeps global laser detuning linearly while pinning Atom A
to the ground state via a local 784nm light shift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.core.atomic_system import build_atom_a_projector
from ryd_gate.protocols.base import SweepProtocol

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ryd_gate.core.atomic_system import AtomicSystem


class SweepAddressingProtocol(SweepProtocol):
    """Sweep protocol with local 784nm pinning on Atom A.

    Parameters
    ----------
    omega : float
        Global two-photon Rabi frequency (rad/s).
    delta_start : float
        Initial global detuning (rad/s).
    delta_end : float
        Final global detuning (rad/s).
    t_gate : float
        Total sweep duration (seconds).
    local_detuning_A : float
        AC Stark shift on Atom A's Rydberg states from 784nm laser (rad/s).
        Usually negative (e.g., -2π × 12 MHz).
    local_scattering_rate : float
        Scattering rate from 784nm laser on Atom A's ground state (Hz).
    """

    def __init__(
        self,
        omega: float,
        delta_start: float,
        delta_end: float,
        t_gate: float,
        local_detuning_A: float,
        local_scattering_rate: float = 35.0,
    ) -> None:
        self.omega = omega
        self.delta_start = delta_start
        self.delta_end = delta_end
        self._t_gate = t_gate
        self.local_detuning_A = local_detuning_A
        self.local_scattering_rate = local_scattering_rate

        # Precompute local addressing operator on Atom A (time-independent)
        # Shift Atom A's Rydberg states off resonance
        self._H_local_pinning = (
            -self.local_detuning_A * build_atom_a_projector(5)
            + (-self.local_detuning_A) * build_atom_a_projector(6)
        )
        # Non-Hermitian scattering loss on Atom A's qubit state |1⟩
        self._H_local_scatter = (
            -1j * self.local_scattering_rate / 2 * build_atom_a_projector(1)
        )

    @property
    def t_gate(self) -> float:
        return self._t_gate

    def get_hamiltonian(
        self, t: float, system: AtomicSystem,
    ) -> "NDArray[np.complexfloating]":
        """Build the full time-dependent Hamiltonian at time t.

        H(t) = H_static + H_global_drive(Ω) + H_global_detuning(Δ(t))
               + H_local_pinning + H_local_scatter
        """
        # Linear detuning sweep
        delta_t = self.delta_start + (self.delta_end - self.delta_start) * (t / self._t_gate)

        # Start from the system's static Hamiltonian (decay, vdW, etc.)
        H = system.tq_ham_const.copy()

        # Add 1013nm coupling (always on)
        H = H + system.tq_ham_1013 + system.tq_ham_1013_conj

        # Global drive: 420nm coupling with constant amplitude (no phase modulation)
        H = H + system.tq_ham_420 + system.tq_ham_420_conj

        # Global Rydberg detuning from sweep: Δ(t) on |r⟩ and |r'⟩ of both atoms
        detuning_sq = np.zeros((7, 7), dtype=np.complex128)
        detuning_sq[5, 5] = delta_t
        detuning_sq[6, 6] = delta_t
        H = H + np.kron(np.eye(7), detuning_sq) + np.kron(detuning_sq, np.eye(7))

        # Light shift + AC Stark shift
        H = H + system.tq_ham_lightshift_zero

        # Local pinning on Atom A
        H = H + self._H_local_pinning + self._H_local_scatter

        return H
