"""AC Stark shift physics: ARC-derived constants and calibrated shift/scatter.

Provides the Grimm-formula calculation for D1+D2 contributions and
calibration against Manovitz et al. measured values at 784 nm.
"""

from __future__ import annotations

import numpy as np
from arc import Rubidium87
from scipy.constants import c

# ======================================================================
# ARC-DERIVED ATOMIC CONSTANTS
# ======================================================================

_atom = Rubidium87()
FREQ_D2: float = _atom.getTransitionFrequency(5, 0, 0.5, 5, 1, 1.5)
FREQ_D1: float = _atom.getTransitionFrequency(5, 0, 0.5, 5, 1, 0.5)
LAMBDA_D2: float = c / FREQ_D2 * 1e9   # nm
LAMBDA_D1: float = c / FREQ_D1 * 1e9   # nm
GAMMA_D2: float = 2 * np.pi * 6.065e6  # rad/s
GAMMA_D1: float = 2 * np.pi * 5.746e6  # rad/s

# Calibration reference (Manovitz et al.)
LAMBDA_PAPER: float = 784.0                    # nm
CALIBRATION_SHIFT_HZ: float = -12.2e6          # Hz at 160 uW
CALIBRATION_SCATTER_HZ: float = 35.0           # Hz at 160 uW
POWER_REF_UW: float = 160.0                    # uW reference power


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
    omega = 2 * np.pi * c / (wavelengths_nm * 1e-9)

    shift = np.zeros_like(omega)
    scatter = np.zeros_like(omega)

    for omega_line, Gamma_line, line_strength in [
        (2 * np.pi * FREQ_D2, GAMMA_D2, 2.0 / 3.0),
        (2 * np.pi * FREQ_D1, GAMMA_D1, 1.0 / 3.0),
    ]:
        Dr = omega_line - omega
        Dc = omega_line + omega
        shift += line_strength * Gamma_line / omega_line**3 * (1.0 / Dr + 1.0 / Dc)
        scatter += (line_strength * Gamma_line**2 / omega_line**3
                    * (omega / omega_line)**3 * (1.0 / Dr + 1.0 / Dc)**2)

    return shift, scatter


_s784, _sc784 = _raw_shift_and_scatter(LAMBDA_PAPER)
_SHIFT_CAL: float = float(CALIBRATION_SHIFT_HZ / _s784)
_SCATTER_CAL: float = float(CALIBRATION_SCATTER_HZ / _sc784)


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
