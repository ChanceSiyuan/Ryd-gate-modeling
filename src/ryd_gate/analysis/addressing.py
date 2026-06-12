"""Local-addressing metrics and Monte-Carlo evaluation helpers.

:class:`AddressingEvaluator` computes pinning error, crosstalk, and leakage
loss from Monte Carlo final state vectors; the module also holds the default
sweep parameters and the :func:`evaluate_addressing` wrapper shared by
scripts and notebooks that evaluate local-addressing error budgets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import pi

from ryd_gate.core.operators import build_atom_a_projector, build_atom_b_projector

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AddressingEvaluator:
    """Evaluate local addressing quality from MC final states.

    Parameters
    ----------
    final_states : list of ndarray
        List of 9-D final state vectors (one per MC shot).
    """

    def __init__(self, final_states: list["NDArray[np.complexfloating]"]) -> None:
        self.final_states = final_states
        self._n_shots = len(final_states)

        # 3-level analog system: |g⟩=0, |e⟩=1, |r⟩=2
        N = 3
        self._P_A1 = build_atom_a_projector(0, n_levels=N)  # Atom A in |g⟩
        self._P_Br = build_atom_b_projector(2, n_levels=N)   # Atom B in |r⟩

    def pinning_error(self) -> float:
        """Probability Atom A failed to stay in the ground state |g⟩.

        Returns 1 - ⟨P_{A,g}⟩ averaged over MC shots.
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


# ── Baseline noise for combined sweep ────────────────────────────────
BASELINE_DETUNING_HZ = 130e3
BASELINE_RIN = 0.01
BASELINE_AMP = 0.01
COMBINED_SCALE_MAX = 3.0

# ── Default pinning parameters (784 nm, Manovitz et al.) ────────────
DEFAULT_LOCAL_DETUNING = -2 * pi * 12e6    # rad/s
DEFAULT_LOCAL_SCATTER = 35.0               # Hz

# ── Default sweep range ─────────────────────────────────────────────
SWEEP_DELTA_HZ = 15e6    # Hz, half-range
SWEEP_T_GATE = 1.5e-6    # s


def default_sweep_x(system):
    """Default normalized parameter vector for the standard addressing sweep.

    Sweep from -15 MHz to +15 MHz over 1.5 us.
    """
    return [
        -2 * pi * SWEEP_DELTA_HZ / system.rabi_eff,
         2 * pi * SWEEP_DELTA_HZ / system.rabi_eff,
         SWEEP_T_GATE / system.time_scale,
    ]


def evaluate_addressing(system, initial_state, protocol, x, engine_kwargs,
                         n_mc, seed=42):
    """Run MC addressing sim and return (pinning_err, crosstalk_err, leakage).

    Parameters
    ----------
    system : RydbergSystem
        The 3-level analog system with no protocol bound.
    initial_state : ndarray
        Two-atom initial state vector (e.g. |g,g⟩).
    protocol : SweepProtocol
        Protocol instance with pinning parameters.
    x : list of float
        Parameter vector for the protocol.
    engine_kwargs : dict
        Noise configuration. Supported keys:
        ``sigma_detuning`` (Hz), ``sigma_local_rin`` (fractional),
        ``sigma_amplitude`` (fractional).
    n_mc : int
        Number of Monte-Carlo shots.
    seed : int
        Random seed for reproducibility.
    """
    from ryd_gate.backends.exact import MonteCarloRunner

    engine = MonteCarloRunner(system.with_protocol(protocol), x)
    if engine_kwargs.get("sigma_detuning"):
        engine.setup_detuning_noise(engine_kwargs["sigma_detuning"])
    if engine_kwargs.get("sigma_local_rin"):
        engine.setup_local_rin_noise(engine_kwargs["sigma_local_rin"])
    if engine_kwargs.get("sigma_amplitude"):
        engine.setup_amplitude_noise(engine_kwargs["sigma_amplitude"])
    shots = engine.run_states([initial_state], n_shots=n_mc, seed=seed)
    final_states = [shot[0].psi_final for shot in shots]
    ev = AddressingEvaluator(final_states)
    return ev.pinning_error(), ev.crosstalk_error(), ev.leakage_loss()
