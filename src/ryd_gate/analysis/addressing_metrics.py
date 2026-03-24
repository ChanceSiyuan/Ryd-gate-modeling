"""Metrics for local addressing experiments.

Evaluates pinning error, crosstalk, and leakage loss from
Monte Carlo final state vectors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.core.atomic_system import build_atom_a_projector, build_atom_b_projector

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AddressingEvaluator:
    """Evaluate local addressing quality from MC final states.

    Parameters
    ----------
    final_states : list of ndarray
        List of 49-D final state vectors (one per MC shot).
    """

    def __init__(self, final_states: list["NDArray[np.complexfloating]"]) -> None:
        self.final_states = final_states
        self._n_shots = len(final_states)

        # Precompute projection operators
        self._P_A1 = build_atom_a_projector(1)  # |1⟩⟨1| ⊗ I (Atom A in |1⟩)
        self._P_Br = build_atom_b_projector(5)  # I ⊗ |r⟩⟨r| (Atom B in |r⟩)

    def pinning_error(self) -> float:
        """Probability Atom A failed to stay in the ground state |1⟩.

        Returns 1 - ⟨P_{A,1}⟩ averaged over MC shots.
        """
        total = 0.0
        for psi in self.final_states:
            total += np.real(np.vdot(psi, self._P_A1 @ psi))
        return 1.0 - total / self._n_shots

    def crosstalk_error(self) -> float:
        """Probability Atom B failed to reach the Rydberg state |r⟩.

        Returns 1 - ⟨P_{B,r}⟩ averaged over MC shots.
        """
        total = 0.0
        for psi in self.final_states:
            total += np.real(np.vdot(psi, self._P_Br @ psi))
        return 1.0 - total / self._n_shots

    def leakage_loss(self) -> float:
        """Probability lost to non-Hermitian decay (scattering, BBR).

        Returns 1 - ⟨||ψ||²⟩ averaged over MC shots.
        """
        total_norm_sq = 0.0
        for psi in self.final_states:
            total_norm_sq += np.real(np.vdot(psi, psi))
        return 1.0 - total_norm_sq / self._n_shots
