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
from scipy.constants import c, epsilon_0, hbar, physical_constants

_MU_B = physical_constants["Bohr magneton"][0]  # Bohr magneton, J/T

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


# ======================================================================
# MAGNETIC FIELD -> RYDBERG ZEEMAN SHIFT
# ======================================================================


def lande_gj(l: int, j: float, s: float = 0.5) -> float:
    """Landé g-factor g_J for a fine-structure level ``|l j⟩`` (spin ``s``).

    Parameters
    ----------
    l : int
        Orbital angular momentum quantum number.
    j : float
        Total (fine-structure) angular momentum quantum number.
    s : float, optional
        Spin quantum number (default 1/2 for an alkali valence electron).

    Returns
    -------
    float
        The Landé g-factor. For an nS_{1/2} level (l=0, j=1/2) this is 2.
    """
    return 1.0 + (j * (j + 1) + s * (s + 1) - l * (l + 1)) / (2 * j * (j + 1))


def zeeman_shift_rad_s(magnetic_field_G: float, *, l: int, j: float, delta_mj: float) -> float:
    """Linear Zeeman shift (rad/s) between two ``m_j`` states of a ``|l j⟩`` level.

    ``Δω = (μ_B / ħ) · g_J(l, j) · Δm_j · B`` with ``B = magnetic_field_G · 1e-4``
    (T) and :func:`lande_gj` for the fine-structure g-factor.

    Parameters
    ----------
    magnetic_field_G : float
        Bias magnetic field in Gauss.
    l : int
        Orbital angular momentum quantum number of the level.
    j : float
        Total (fine-structure) angular momentum quantum number of the level.
    delta_mj : float
        ``m_j`` difference between the two states.

    Returns
    -------
    float
        The Zeeman shift in rad/s.
    """
    B_T = magnetic_field_G * 1e-4
    return (_MU_B / hbar) * lande_gj(l, j) * delta_mj * B_T


def rydberg_zeeman_shift_rad_s(magnetic_field_G: float, *, manifold: str) -> float:
    """Linear Zeeman splitting (rad/s) of the garbage Rydberg state ``r_garb``.

    The seven-level Rb87 gate model reaches an opposite-``m_j`` Rydberg Zeeman
    state ``r_garb`` through polarization leakage. For an ``nS_{1/2}`` Rydberg
    level the two states differ by ``Δm_j = 1`` (m_j = ±1/2), so the energy of
    ``r_garb`` relative to ``r`` is the linear Zeeman shift
    :func:`zeeman_shift_rad_s` with ``g_J = 2`` (``nS_{1/2}``). The result is
    positive for positive ``B``, matching the Hamiltonian convention
    ``h[6, 6] = +ryd_zeeman_shift`` (``r_garb`` above ``r``).

    Parameters
    ----------
    magnetic_field_G : float
        Bias magnetic field in Gauss.
    manifold : str
        ``"mp"`` (σ⁻/σ⁺) or ``"pm"`` (σ⁺/σ⁻); both are ``nS_{1/2}`` states with
        opposite ``m_j = ±1/2`` (``Δm_j = 1``).

    Returns
    -------
    float
        The Zeeman shift in rad/s.
    """
    if manifold not in ("mp", "pm"):
        raise ValueError(f"Unknown rb87 manifold '{manifold}' (expected 'mp' or 'pm').")
    return zeeman_shift_rad_s(magnetic_field_G, l=0, j=0.5, delta_mj=1.0)


def our_laser_rabis(
    p420_w: float,
    p1013_w: float,
    beam_area: float,
    *,
    ryd_level: int = RYD_LEVEL_OUR,
) -> tuple[float, float]:
    """420/1013 nm single-photon Rabi frequencies (rad/s) for the σ⁻/σ⁺ path.

    Both beams are top-hats of the given power filling the same
    ``beam_area`` (μm²). Transitions match the ``rb87_7_mp`` manifold
    (σ⁻/σ⁺; was param_set ``our``):

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
    )/np.sqrt(2)  # sqrt(2) factor for mF=0 splitting into mJ=-1/2 and mJ=+1/2 components
    omega_1013 = single_photon_rabi(
        p1013_w, beam_area,
        n1=6, l1=1, j1=1.5, mj1=-1.5, n2=ryd_level, l2=0, j2=0.5, q=1,
    )
    return omega_420, omega_1013


# Default Rydberg level for the 297 nm single-photon configuration.
RYD_LEVEL_297: int = 53


def direct_297_rabis(
    power_w: float,
    beam_area: float,
    *,
    ryd_level: int = RYD_LEVEL_297,
) -> tuple[float, float]:
    """297 nm σ⁻ single-photon Rabi frequencies (rad/s) from the clock state.

    One top-hat beam of the given power filling ``beam_area`` (μm²) drives both
    Zeeman branches out of the clock-like ground state
    ``|1⟩ = (|m_J=-1/2, m_I=+1/2⟩ + |m_J=+1/2, m_I=-1/2⟩)/√2``:

      * target:  5S₁/₂ (mⱼ=-1/2) --σ⁻--> nP₃/₂ (mⱼ=-3/2)   (m_I=+1/2 spectator)
      * garbage: 5S₁/₂ (mⱼ=+1/2) --σ⁻--> nP₃/₂ (mⱼ=-1/2)   (m_I=-1/2 spectator)

    Both legs carry the clock-state 1/sqrt(2) amplitude factor.

    Returns
    -------
    (omega_r, omega_garb) : tuple of float
        Single-photon Rabi frequencies in rad/s for the target and garbage
        branches.
    """
    omega_r = single_photon_rabi(
        power_w, beam_area,
        n1=5, l1=0, j1=0.5, mj1=-0.5, n2=ryd_level, l2=1, j2=1.5, q=-1,
    ) / np.sqrt(2)  # sqrt(2) factor for mF=0 splitting into mJ=-1/2 and mJ=+1/2 components
    omega_garb = single_photon_rabi(
        power_w, beam_area,
        n1=5, l1=0, j1=0.5, mj1=0.5, n2=ryd_level, l2=1, j2=1.5, q=-1,
    ) / np.sqrt(2)
    return omega_r, omega_garb


# ======================================================================
# ARC PAIR-STATE C6 (VdW) COEFFICIENTS
# ======================================================================


@functools.lru_cache(maxsize=None)
def _arc_pair_c6_cached(
    n1, l1, j1, mj1, n2, l2, j2, mj2, theta, phi, n_range, energy_delta_hz, degenerate,
) -> tuple[float, float]:
    """``(repo C6 in rad/s·μm⁶, bare-channel overlap)`` — see the public wrapper."""
    import warnings

    from arc import PairStateInteractions
    from arc.calculations_atom_pairstate import compositeState, singleAtomState

    calc = PairStateInteractions(_get_atom(), n1, l1, j1, n2, l2, j2, mj1, mj2)
    if not degenerate:
        arc_c6_ghz = float(np.real(calc.getC6perturbatively(theta, phi, n_range, energy_delta_hz)))
        overlap = 1.0
    else:
        values, vectors = calc.getC6perturbatively(
            theta, phi, n_range, energy_delta_hz, degeneratePerturbation=True
        )
        # Bare |mj1, mj2⟩ channel in ARC's {mj1=-j1..+j1} ⊗ {mj2=-j2..+j2}
        # eigenvector basis (vectors are rows); pick the max-overlap eigenchannel.
        bare = compositeState(singleAtomState(j1, mj1), singleAtomState(j2, mj2)).flatten()
        overlaps = np.abs(np.asarray(vectors) @ bare) ** 2
        best = int(np.argmax(overlaps))
        arc_c6_ghz = float(np.real(values[best]))
        overlap = float(overlaps[best])
        if overlap < 0.5:
            warnings.warn(
                f"arc_pair_c6_rad_s_um6: bare pair channel |{mj1},{mj2}⟩ of "
                f"({n1} l={l1} j={j1}, {n2} l={l2} j={j2}) at theta={theta:.3f}, "
                f"phi={phi:.3f} is not a dominant eigenchannel (max overlap "
                f"{overlap:.2f}); returning the max-overlap C6 eigenvalue.",
                stacklevel=2,
            )
    # ARC's perturbative convention is V(R) = -C6/R^6 (getC6perturbatively
    # docstring); this repo uses V(R) = +C6/R^6 (vdw_couplings), hence the flip.
    return -arc_c6_ghz * 2 * np.pi * 1e9, overlap


def arc_pair_c6_rad_s_um6(
    *,
    n1: int,
    l1: int,
    j1: float,
    mj1: float,
    n2: int | None = None,
    l2: int | None = None,
    j2: float | None = None,
    mj2: float | None = None,
    theta: float,
    phi: float,
    n_range: int = 5,
    energy_delta_hz: float = 30e9,
    degenerate: bool = True,
) -> float:
    """Perturbative pair-state C6 (rad/s·μm⁶) in this repo's ``V = +C6/R⁶`` sign.

    Wraps ARC ``PairStateInteractions.getC6perturbatively`` for the
    ``|n1 l1 j1 mj1; n2 l2 j2 mj2⟩`` pair state at inter-atomic axis orientation
    ``(theta, phi)`` relative to the quantization axis, converting from ARC's
    ``V(R) = -C6/R⁶`` GHz·μm⁶ convention. Atom-2 quantum numbers default to
    atom-1's (identical pair state).

    With ``degenerate=True`` (required off-axis / for non-stretched states) the
    C6 matrix over the degenerate ``m_j`` manifold is diagonalized and the
    eigenvalue whose eigenvector has the largest overlap with the bare
    ``|mj1, mj2⟩`` channel is returned; a warning reports the overlap when it is
    not dominant (< 0.5). Results are cached (``theta``/``phi`` rounded to
    1e-9 rad), so repeated per-pair lookups at the same orientation are free.
    """
    if n2 is None:
        n2, l2, j2, mj2 = n1, l1, j1, mj1
    c6, _overlap = _arc_pair_c6_cached(
        int(n1), int(l1), float(j1), float(mj1),
        int(n2), int(l2), float(j2), float(mj2),
        round(float(theta), 9), round(float(phi), 9),
        int(n_range), float(energy_delta_hz), bool(degenerate),
    )
    return c6


# ======================================================================
# BRANCHING RATIOS
# ======================================================================


def _rydberg_branching_ratios(atom, ryd_level, manifold):
    """Compute branching ratios for Rydberg radiative decay.

    ``manifold`` is ``"mp"`` (σ⁻/σ⁺, Rydberg mⱼ=-1/2; was param_set "our") or
    ``"pm"`` (σ⁺/σ⁻, mⱼ=+1/2; was "lukin").
    """
    from arc.wigner import CG

    I = 3 / 2
    mI = 1 / 2
    nr = ryd_level
    lr, jr = 0, 1 / 2
    if manifold == "mp":
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
