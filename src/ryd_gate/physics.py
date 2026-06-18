"""Physics helpers for Rb87: pulse envelopes, AC Stark shifts, ARC decay branching.

Pulse envelopes: continuous-time Blackman flat-top windows (``blackman_window`` /
``blackman_pulse`` / ``blackman_pulse_sqrt``) used by the gate protocols'
amplitude shaping (:mod:`ryd_gate.protocols.gate_cz`). These are pure-numpy and
carry no atomic-physics dependency.

AC Stark shift physics: ARC-derived constants and calibrated shift/scatter.
Provides the Grimm-formula calculation for D1+D2 contributions, including
the scalar + vector decomposition for the alkali ground state, and
calibration against Manovitz et al. measured values at 784 nm.

Lazy ARC import
---------------
``arc`` (the alkali-Rydberg calculator) is **imported lazily**: it is pulled in
only when an ARC-derived value is first needed (atomic constants, calibrated
shift/scatter, single-photon Rabi, branching ratios). Importing this module —
and in particular the pulse-envelope helpers used on the CZ gate path — does
*not* initialize ARC. ARC-derived module attributes (``FREQ_D1`` / ``FREQ_D2`` /
``LAMBDA_D1`` / ``LAMBDA_D2`` / ``_atom``) are resolved on first access through
the module-level ``__getattr__`` and cached.

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

import functools

import numpy as np
from scipy.constants import c, epsilon_0

# ======================================================================
# PULSE ENVELOPE KERNELS (pure numpy; no ARC dependency)
# ======================================================================


def blackman_window(t, t_rise):
    """Evaluate the Blackman window function.

    Parameters
    ----------
    t : array_like
        Time values.
    t_rise : float
        Rise time of the window.

    Returns
    -------
    numpy.ndarray
        Window amplitude in [0, 1].
    """
    return (0.42 - 0.5 * np.cos(2 * np.pi * t / (2 * t_rise)) +
            0.08 * np.cos(4 * np.pi * t / (2 * t_rise)))


def blackman_pulse(t, t_rise, t_gate):
    """Blackman-windowed flat-top pulse.

    Parameters
    ----------
    t : array_like
        Time values.
    t_rise : float
        Rise/fall time.
    t_gate : float
        Total gate duration (must be >= 2 * t_rise).

    Returns
    -------
    numpy.ndarray
        Pulse envelope.
    """
    if t_gate < 2 * t_rise:
        raise ValueError("t_gate is too small compared to t_rise")
    ret = (blackman_window(t, t_rise) * np.heaviside(t_rise - t, 1) +
           np.heaviside(t - t_rise, 0) * np.heaviside(t_gate - t - t_rise, 0) +
           blackman_window(t_gate - t, t_rise) *
           np.heaviside(t_rise - (t_gate - t), 1))
    return ret


def blackman_pulse_sqrt(t, t_rise, t_gate):
    """Square-root of the Blackman-windowed flat-top pulse.

    Parameters
    ----------
    t : array_like
        Time values.
    t_rise : float
        Rise/fall time.
    t_gate : float
        Total gate duration (must be >= 2 * t_rise).

    Returns
    -------
    numpy.ndarray
        Square-root pulse envelope.
    """
    return np.sqrt(np.maximum(blackman_pulse(t, t_rise, t_gate), 0))


# ======================================================================
# ARC-DERIVED ATOMIC CONSTANTS (lazy — see module docstring)
# ======================================================================

GAMMA_D2: float = 2 * np.pi * 6.065e6  # rad/s
GAMMA_D1: float = 2 * np.pi * 5.746e6  # rad/s

# Landé g-factor for ⁸⁷Rb 5S_{1/2} F=2 (used to convert pol → P·g_F·m_F).
G_F_5S12_F2: float = 0.5

# Calibration reference (Manovitz et al., linear polarization)
LAMBDA_PAPER: float = 784.0                    # nm
CALIBRATION_SHIFT_HZ: float = -12.2e6          # Hz at 160 uW
CALIBRATION_SCATTER_HZ: float = 35.0           # Hz at 160 uW
POWER_REF_UW: float = 160.0                    # uW reference power


@functools.lru_cache(maxsize=1)
def _get_atom():
    """Lazily construct (and cache) the ARC ``Rubidium87`` atom."""
    from arc import Rubidium87

    return Rubidium87()


@functools.lru_cache(maxsize=1)
def _d_line_frequencies() -> tuple[float, float]:
    """ARC D1/D2 transition frequencies (Hz), computed once and cached."""
    atom = _get_atom()
    freq_d2 = atom.getTransitionFrequency(5, 0, 0.5, 5, 1, 1.5)
    freq_d1 = atom.getTransitionFrequency(5, 0, 0.5, 5, 1, 0.5)
    return freq_d1, freq_d2


@functools.lru_cache(maxsize=1)
def _calibration() -> tuple[float, float]:
    """Shift/scatter calibration factors against the 784 nm reference, cached.

    Calibration is performed in the linear-polarization (pol=0) limit, matching
    the experimental conditions of Manovitz et al.
    """
    s784, sc784 = _raw_shift_and_scatter(LAMBDA_PAPER, pol=0.0)
    return float(CALIBRATION_SHIFT_HZ / s784), float(CALIBRATION_SCATTER_HZ / sc784)


def __getattr__(name: str):
    """Resolve ARC-derived module attributes on first access (then cached)."""
    if name == "_atom":
        return _get_atom()
    if name in {"FREQ_D1", "FREQ_D2", "LAMBDA_D1", "LAMBDA_D2"}:
        freq_d1, freq_d2 = _d_line_frequencies()
        return {
            "FREQ_D1": freq_d1,
            "FREQ_D2": freq_d2,
            "LAMBDA_D1": c / freq_d1 * 1e9,  # nm
            "LAMBDA_D2": c / freq_d2 * 1e9,  # nm
        }[name]
    raise AttributeError(f"module 'ryd_gate.physics' has no attribute {name!r}")


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
    freq_d1, freq_d2 = _d_line_frequencies()
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    omega = 2 * np.pi * c / (wavelengths_nm * 1e-9)

    shift = np.zeros_like(omega)
    scatter = np.zeros_like(omega)

    weight_D2 = (2.0 + pol) / 3.0
    weight_D1 = (1.0 - pol) / 3.0

    for omega_line, Gamma_line, w in [
        (2 * np.pi * freq_d2, GAMMA_D2, weight_D2),
        (2 * np.pi * freq_d1, GAMMA_D1, weight_D1),
    ]:
        Dr = omega_line - omega
        Dc = omega_line + omega
        shift += w * Gamma_line / omega_line**3 * (1.0 / Dr + 1.0 / Dc)
        scatter += (w * Gamma_line**2 / omega_line**3
                    * (omega / omega_line)**3 * (1.0 / Dr + 1.0 / Dc)**2)

    return shift, scatter


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
    shift_cal, scatter_cal = _calibration()
    return s * shift_cal, sc * scatter_cal


# ======================================================================
# LASER POWER -> SINGLE-PHOTON RABI
# ======================================================================

# Default Rydberg level for the 'our' 70S two-photon configuration.
RYD_LEVEL_OUR: int = 70


def electric_field_uniform_beam(power_w: float, beam_area: float) -> float:
    """Peak electric field (V/m) of a laser whose power fills a top-hat aperture.

    The beam is modeled as a top-hat: total power ``power_w`` (W) spread
    uniformly over an area ``beam_area`` (μm²), so the intensity is
    ``I = P / A``. The plane-wave relation ``I = (c ε0 / 2) E0²`` then gives the
    field amplitude, matching the convention ARC uses internally in
    ``getRabiFrequency`` / ``getRabiFrequency2``.
    """
    if power_w < 0.0:
        raise ValueError("power_w must be non-negative.")
    if beam_area <= 0.0:
        raise ValueError("beam_area must be positive.")
    area_m2 = beam_area * (1e-6)**2
    intensity = power_w / area_m2
    return float(np.sqrt(2.0 * intensity / (c * epsilon_0)))


def single_photon_rabi(
    power_w: float,
    beam_area: float,
    *,
    n1: int,
    l1: int,
    j1: float,
    mj1: float,
    n2: int,
    l2: int,
    j2: float,
    q: int,
) -> float:
    """Resonant single-photon Rabi frequency (rad/s) for a uniform top-hat beam.

    Combines :func:`electric_field_uniform_beam` with the ARC dipole matrix
    element of the ``|n1 l1 j1 mj1⟩ → |n2 l2 j2 (mj1+q)⟩`` transition
    (``Ω = |d| E0 / ħ``). ``q`` is the laser polarization (-1, 0, +1 for
    σ⁻, π, σ⁺).
    """
    e0 = electric_field_uniform_beam(power_w, beam_area)
    return float(_get_atom().getRabiFrequency2(n1, l1, j1, mj1, n2, l2, j2, q, e0))


def our_laser_rabis(
    p420_w: float,
    p1013_w: float,
    beam_area: float,
    *,
    ryd_level: int = RYD_LEVEL_OUR,
) -> tuple[float, float]:
    """420/1013 nm single-photon Rabi frequencies (rad/s) for the 'our' path.

    Both beams are top-hats of the given power filling the same
    ``beam_area`` (μm²). Transitions match ``physical_models.py``
    param_set ``our``:

      * 420 nm:  5S₁/₂ (mⱼ=-1/2) --σ⁻--> 6P₃/₂ (mⱼ=-3/2)
      * 1013 nm: 6P₃/₂ (mⱼ=-1/2) --σ⁺--> nS₁/₂ (mⱼ=+1/2)

    Returns
    -------
    (omega_420, omega_1013) : tuple of float
        Single-photon Rabi frequencies in rad/s.
    """
    omega_420 = single_photon_rabi(
        p420_w, beam_area,
        n1=5, l1=0, j1=0.5, mj1=-0.5, n2=6, l2=1, j2=1.5, q=-1,
    )
    omega_1013 = single_photon_rabi(
        p1013_w, beam_area,
        n1=6, l1=1, j1=1.5, mj1=-0.5, n2=ryd_level, l2=0, j2=0.5, q=1,
    )
    return omega_420, omega_1013


# ======================================================================
# BRANCHING RATIOS
# ======================================================================


def _rydberg_branching_ratios(atom, ryd_level, param_set):
    """Compute branching ratios for Rydberg radiative decay."""
    from arc.wigner import CG

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
