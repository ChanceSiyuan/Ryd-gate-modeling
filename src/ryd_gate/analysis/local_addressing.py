"""Local addressing physics and MC evaluation helpers.

Shared between the CLI script (scripts/scan_local_addressing.py) and
the Streamlit app page (app/pages/4_local_addressing.py).
"""

from __future__ import annotations

import numpy as np
from scipy.constants import c, pi

from arc import Rubidium87

# ── ARC-derived atomic constants ────────────────────────────────────────
_atom = Rubidium87()
FREQ_D2 = _atom.getTransitionFrequency(5, 0, 0.5, 5, 1, 1.5)
FREQ_D1 = _atom.getTransitionFrequency(5, 0, 0.5, 5, 1, 0.5)
LAMBDA_D2 = c / FREQ_D2 * 1e9   # nm
LAMBDA_D1 = c / FREQ_D1 * 1e9   # nm
GAMMA_D2 = 2 * pi * 6.065e6     # rad/s
GAMMA_D1 = 2 * pi * 5.746e6     # rad/s

# ── Protocol defaults (Manovitz et al.) ─────────────────────────────────
LAMBDA_PAPER = 784.0                       # nm
CALIBRATION_SHIFT_HZ = -12.2e6             # Hz at 160 uW
CALIBRATION_SCATTER_HZ = 35.0              # Hz at 160 uW
POWER_REF_UW = 160.0                       # uW reference power

DELTA_START = -2 * pi * 15e6               # rad/s
DELTA_END = 2 * pi * 15e6                  # rad/s
T_GATE = 1.5e-6                            # s
LOCAL_DETUNING_784 = -2 * pi * 12e6        # rad/s
LOCAL_SCATTER_784 = 35.0                   # Hz

# Baseline noise for combined sweep
BASELINE_DETUNING_HZ = 130e3
BASELINE_RIN = 0.01
BASELINE_AMP = 0.01
COMBINED_SCALE_MAX = 3.0


# ── AC Stark shift physics ─────────────────────────────────────────────

def _raw_shift_and_scatter(wavelengths_nm):
    """Grimm formula for D1+D2 contributions (unnormalized, vectorized).

    Parameters
    ----------
    wavelengths_nm : float or ndarray
        Laser wavelength(s) in nm.

    Returns
    -------
    shift, scatter : same shape as input, in consistent arbitrary units.
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    omega = 2 * pi * c / (wavelengths_nm * 1e-9)

    shift = np.zeros_like(omega)
    scatter = np.zeros_like(omega)

    for omega_line, Gamma_line, line_strength in [
        (2 * pi * FREQ_D2, GAMMA_D2, 2.0 / 3.0),
        (2 * pi * FREQ_D1, GAMMA_D1, 1.0 / 3.0),
    ]:
        Dr = omega_line - omega
        Dc = omega_line + omega
        shift += line_strength * Gamma_line / omega_line**3 * (1.0 / Dr + 1.0 / Dc)
        scatter += (line_strength * Gamma_line**2 / omega_line**3
                    * (omega / omega_line)**3 * (1.0 / Dr + 1.0 / Dc)**2)

    return shift, scatter


# Calibration: normalize to paper's measured values at 784 nm
_s784, _sc784 = _raw_shift_and_scatter(LAMBDA_PAPER)
_SHIFT_CAL = float(CALIBRATION_SHIFT_HZ / _s784)
_SCATTER_CAL = float(CALIBRATION_SCATTER_HZ / _sc784)


def compute_shift_scatter(wavelengths_nm):
    """Calibrated AC Stark shift (Hz) and scattering rate (Hz).

    Calibrated to the paper's measured values at 784 nm (160 uW, 1 um waist).
    Accepts scalar or array input.

    Returns
    -------
    shift_Hz, scatter_Hz : same shape as input.
    """
    s, sc = _raw_shift_and_scatter(wavelengths_nm)
    return s * _SHIFT_CAL, sc * _SCATTER_CAL


# ── MC evaluation helpers ──────────────────────────────────────────────

def setup_system():
    """Create the atomic system and initial state |01> for addressing."""
    from ryd_gate.core.atomic_system import create_atomic_system, build_sss_state_map
    system = create_atomic_system(param_set="our", detuning_sign=1)
    initial_state = build_sss_state_map()["01"]
    return system, initial_state


def make_protocol(local_detuning=LOCAL_DETUNING_784,
                  scatter_rate=LOCAL_SCATTER_784):
    """Create a SweepAddressingProtocol with standard 784nm settings."""
    from ryd_gate.protocols.local_sweep import SweepAddressingProtocol
    return SweepAddressingProtocol(
        local_detuning_A=local_detuning,
        local_scattering_rate=scatter_rate,
    )


def default_sweep_x(system):
    """Default parameter vector for the standard sweep: [-15, +15] MHz over 1.5 us.

    Returns
    -------
    list of float
        ``[delta_start / rabi_eff, delta_end / rabi_eff, t_gate / time_scale]``
    """
    return [
        DELTA_START / system.rabi_eff,
        DELTA_END / system.rabi_eff,
        T_GATE / system.time_scale,
    ]


def evaluate_addressing(system, initial_state, protocol, x, engine_kwargs,
                         n_mc, seed=42):
    """Run MC addressing sim and return (pinning_err, crosstalk_err, leakage).

    Parameters
    ----------
    x : list of float
        Parameter vector for the protocol.
    engine_kwargs : dict
        Keyword arguments for AddressingMCEngine (sigma_detuning, etc.).
    """
    from ryd_gate.solvers.monte_carlo import AddressingMCEngine
    from ryd_gate.analysis.addressing_metrics import AddressingEvaluator

    engine = AddressingMCEngine(system=system, protocol=protocol, x=x, **engine_kwargs)
    final_states = engine.run(initial_state, n_shots=n_mc, seed=seed)
    ev = AddressingEvaluator(final_states)
    return ev.pinning_error(), ev.crosstalk_error(), ev.leakage_loss()
