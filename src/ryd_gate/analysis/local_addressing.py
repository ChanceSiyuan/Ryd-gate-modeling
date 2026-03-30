"""Monte-Carlo evaluation helpers for local addressing analysis.

Shared between the CLI script (scripts/scan_local_addressing.py) and
the Streamlit app page (app/pages/4_local_addressing.py).
"""

from __future__ import annotations

import numpy as np
from scipy.constants import pi

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
    system : AtomicSystem
        The 3-level analog system.
    initial_state : ndarray
        Two-atom initial state vector (e.g. |g,g⟩).
    protocol : SweepAddressingProtocol
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
    from ryd_gate.solvers.monte_carlo import MonteCarloEngine
    from ryd_gate.analysis.addressing_metrics import AddressingEvaluator

    engine = MonteCarloEngine(system=system, protocol=protocol, x=x)
    if engine_kwargs.get("sigma_detuning"):
        engine.setup_detuning_noise(engine_kwargs["sigma_detuning"])
    if engine_kwargs.get("sigma_local_rin"):
        engine.setup_rin_noise(engine_kwargs["sigma_local_rin"])
    if engine_kwargs.get("sigma_amplitude"):
        engine.setup_amplitude_noise(engine_kwargs["sigma_amplitude"])
    final_states = engine.run_states(initial_state, n_shots=n_mc, seed=seed)
    ev = AddressingEvaluator(final_states)
    return ev.pinning_error(), ev.crosstalk_error(), ev.leakage_loss()
