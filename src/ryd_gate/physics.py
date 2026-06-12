"""Physics calculations for Rb87: AC Stark shifts and ARC decay branching.

AC Stark shift physics: ARC-derived constants and calibrated shift/scatter.
Provides the Grimm-formula calculation for D1+D2 contributions, including
the scalar + vector decomposition for the alkali ground state, and
calibration against Manovitz et al. measured values at 784 nm.

Vector light-shift model
------------------------
Following Grimm, Weidemüller & Ovchinnikov (2000), the dipole shift on an
alkali ground-state hyperfine sub-level |F, m_F⟩ in a far-detuned field can
be written

    U ∝ (2 + P g_F m_F)/3 · 1/Δ_{D2}  +  (1 − P g_F m_F)/3 · 1/Δ_{D1}

where P ∈ [−1, 1] is the laser helicity (P = 0 linear, P = ±1 σ±) and
g_F m_F is the Landé factor times the magnetic quantum number of the
ground state. We bundle these into a single dimensionless

    pol  ≡  P · g_F · m_F        (default 0 = scalar / linear-pol limit)

so that the existing scalar weights (2/3, 1/3) are recovered for ``pol=0``
and the calibration to the Manovitz et al. linear-polarization measurement
remains valid. Setting ``pol ≠ 0`` adds the vector contribution; positive
``pol`` enhances the D2 weight at the expense of D1, negative does the
opposite.

For the experimentally relevant ⁸⁷Rb |5S₁/₂, F=2, m_F=−2⟩ state used in
Manovitz et al., g_F = +1/2, so ``pol = -P/1`` (a pure σ⁺ beam gives
``pol=-1``).

Branching ratios: radiative decay branching for Rydberg and intermediate
states, computed from ARC dipole matrix elements and Clebsch-Gordan
coefficients.
"""

from __future__ import annotations

import numpy as np
from arc import Rubidium87
from arc.wigner import CG
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

# Landé g-factor for ⁸⁷Rb 5S_{1/2} F=2 (used to convert pol → P·g_F·m_F).
G_F_5S12_F2: float = 0.5

# Calibration reference (Manovitz et al., linear polarization)
LAMBDA_PAPER: float = 784.0                    # nm
CALIBRATION_SHIFT_HZ: float = -12.2e6          # Hz at 160 uW
CALIBRATION_SCATTER_HZ: float = 35.0           # Hz at 160 uW
POWER_REF_UW: float = 160.0                    # uW reference power


def _raw_shift_and_scatter(wavelengths_nm, pol: float = 0.0):
    """Grimm formula for D1+D2 contributions, with scalar+vector weights.

    Parameters
    ----------
    wavelengths_nm : float or ndarray
        Laser wavelength(s) in nm.
    pol : float, optional
        Polarization parameter ``P · g_F · m_F`` (default 0, linear).
        ``pol=0`` recovers the pure-scalar (2/3, 1/3) D2/D1 weights.

    Returns
    -------
    shift, scatter : same shape as input, in consistent arbitrary units.
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    omega = 2 * np.pi * c / (wavelengths_nm * 1e-9)

    shift = np.zeros_like(omega)
    scatter = np.zeros_like(omega)

    weight_D2 = (2.0 + pol) / 3.0
    weight_D1 = (1.0 - pol) / 3.0

    for omega_line, Gamma_line, w in [
        (2 * np.pi * FREQ_D2, GAMMA_D2, weight_D2),
        (2 * np.pi * FREQ_D1, GAMMA_D1, weight_D1),
    ]:
        Dr = omega_line - omega
        Dc = omega_line + omega
        shift += w * Gamma_line / omega_line**3 * (1.0 / Dr + 1.0 / Dc)
        scatter += (w * Gamma_line**2 / omega_line**3
                    * (omega / omega_line)**3 * (1.0 / Dr + 1.0 / Dc)**2)

    return shift, scatter


# Calibration is performed in the linear-polarization (pol=0) limit, matching
# the experimental conditions of Manovitz et al.
_s784, _sc784 = _raw_shift_and_scatter(LAMBDA_PAPER, pol=0.0)
_SHIFT_CAL: float = float(CALIBRATION_SHIFT_HZ / _s784)
_SCATTER_CAL: float = float(CALIBRATION_SCATTER_HZ / _sc784)


def compute_shift_scatter(wavelengths_nm, pol: float = 0.0):
    """Calibrated AC Stark shift (Hz) and scattering rate (Hz).

    Calibrated to the paper's measured values at 784 nm (160 μW, 1 μm waist,
    linear polarization). Accepts scalar or array input.

    Parameters
    ----------
    wavelengths_nm : float or ndarray
        Laser wavelength(s) in nm.
    pol : float, optional
        Polarization parameter ``P · g_F · m_F`` (default 0 = linear).
        See module docstring for sign conventions. ``pol=0`` matches the
        original scalar model and the experimental calibration condition.

    Returns
    -------
    shift_Hz, scatter_Hz : same shape as input.
    """
    s, sc = _raw_shift_and_scatter(wavelengths_nm, pol=pol)
    return s * _SHIFT_CAL, sc * _SCATTER_CAL


# ======================================================================
# BRANCHING RATIOS
# ======================================================================


def _rydberg_branching_ratios(atom, ryd_level, param_set):
    """Compute branching ratios for Rydberg radiative decay."""
    I = 3 / 2
    mI = 1 / 2
    nr = ryd_level
    lr, jr = 0, 1 / 2
    if param_set == "our":
        mjr = -1 / 2
    else:
        mjr = 1 / 2
    fr_list = [2, 1]
    mfr = mI + mjr

    ne, le = 5, 1
    je_list = [3 / 2, 1 / 2]
    ng, lg, jg = 5, 0, 1 / 2

    a = []
    b = []

    for _je in je_list:
        fe_range = np.arange(abs(I - _je), I + _je + 1, 1)
        for _fe in fe_range:
            mfe_range = np.arange(-_fe, _fe + 1, 1)
            for _mfe in mfe_range:
                t = 0.0
                for _fr in fr_list:
                    if abs(mfr) <= _fr and abs(mfr - _mfe) < 2:
                        t += CG(jr, mjr, I, mI, _fr, mfr) * \
                            atom.getDipoleMatrixElementHFS(
                                ne, le, _je, _fe, _mfe,
                                nr, lr, jr, _fr, mfr,
                                q=mfr - _mfe,
                            )
                a.append(t**2)

                bb = []
                for fg in [2, 1]:
                    mfg_range = np.arange(-fg, fg + 1, 1)
                    for _mfg in mfg_range:
                        if abs(_mfg - _mfe) < 2:
                            bb.append(
                                atom.getDipoleMatrixElementHFS(
                                    ne, le, _je, _fe, _mfe,
                                    ng, lg, jg, fg, _mfg,
                                    q=_mfg - _mfe,
                                ) ** 2
                            )
                        else:
                            bb.append(0.0)
                bb_sum = np.sum(bb)
                bb = [x / bb_sum for x in bb]
                b.append(bb)

    a_sum = np.sum(a)
    a = [x / a_sum for x in a]

    branch_ratio = np.array(
        [a[i] * np.array(b[i]) for i in range(len(a))]
    ).sum(axis=0)

    return {
        "to_0": float(branch_ratio[6]),
        "to_1": float(branch_ratio[2]),
        "to_L0": float(branch_ratio[5] + branch_ratio[7]),
        "to_L1": float(
            branch_ratio[0] + branch_ratio[1]
            + branch_ratio[3] + branch_ratio[4]
        ),
    }


def _mid_branching_ratios(atom, F, mF):
    """Compute branching ratios for 6P3/2 intermediate state decay."""
    ne, le, je, fe, mfe = 6, 1, 3 / 2, F, mF
    ng, lg, jg = 5, 0, 1 / 2

    a = []
    for fg in [2, 1]:
        mfg_range = np.arange(-fg, fg + 1, 1)
        for _mfg in mfg_range:
            if abs(_mfg - mfe) < 2:
                a.append(
                    atom.getDipoleMatrixElementHFS(
                        ne, le, je, fe, mfe,
                        ng, lg, jg, fg, _mfg,
                        q=_mfg - mfe,
                    ) ** 2
                )
            else:
                a.append(0.0)
    a_sum = np.sum(a)
    branch_ratio = [x / a_sum for x in a]

    return {
        "to_0": float(branch_ratio[6]),
        "to_1": float(branch_ratio[2]),
        "to_L0": float(branch_ratio[5] + branch_ratio[7]),
        "to_L1": float(
            branch_ratio[0] + branch_ratio[1]
            + branch_ratio[3] + branch_ratio[4]
        ),
    }
