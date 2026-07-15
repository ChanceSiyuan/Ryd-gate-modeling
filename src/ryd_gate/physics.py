"""Forward physics helpers for Rb87 (expert module; not a top-level export).

``ryd_gate.physics`` computes Hamiltonian parameters and system physical data
*forward* from experimental/atomic inputs — single-photon Rabi frequencies,
linear Zeeman shifts and ARC pair-state C6 coefficients. Nothing here consumes
an ``EvolutionResult``, expectation, amplitude, sample or ensemble statistic;
that post-processing lives in scripts/notebooks.

The public surface is exactly the five functions in :data:`__all__`. Everything
else (the ARC atom cache, branching-ratio builders, the top-hat field helper,
the Landé/Rydberg-Zeeman helpers) is private implementation.

Lazy ARC import
---------------
``arc`` is imported lazily: it is pulled in only when an ARC-derived value is
first needed (Rabi frequency, pair C6, branching ratios). Importing this module
does not initialize ARC.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.constants import c, epsilon_0, hbar, physical_constants

_MU_B = physical_constants["Bohr magneton"][0]  # Bohr magneton, J/T

__all__ = [
    "single_photon_rabi",
    "rb87_7_mp_rabi_frequencies",
    "rb87_297_clock_rabi_frequencies",
    "zeeman_shift_rad_s",
    "arc_pair_c6_rad_s_um6",
]


# ======================================================================
# ARC atom (lazy)
# ======================================================================


@functools.lru_cache(maxsize=1)
def _get_atom():
    """Lazily construct (and cache) the ARC ``Rubidium87`` atom."""
    from arc import Rubidium87

    return Rubidium87()


# ======================================================================
# LASER POWER -> SINGLE-PHOTON RABI
# ======================================================================


def _electric_field_uniform_beam(power_w: float, beam_area_um2: float) -> float:
    """Peak electric field (V/m) of a top-hat beam of ``power_w`` over ``beam_area_um2``.

    Intensity ``I = P / A`` and the plane-wave relation ``I = (c ε0 / 2) E0²``,
    matching the convention ARC uses internally in ``getRabiFrequency2``.
    """
    area_m2 = beam_area_um2 * (1e-6) ** 2
    intensity = power_w / area_m2
    return float(np.sqrt(2.0 * intensity / (c * epsilon_0)))


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _check_transition_level(tag: str, n: int, l: int, j: float, mj: float) -> None:
    if not (_is_int(n) and n >= 1):
        raise ValueError(f"{tag}: n must be a positive integer; got {n!r}.")
    if not (_is_int(l) and 0 <= l < n):
        raise ValueError(f"{tag}: l must be an integer in [0, n); got {l!r}.")
    if j not in (abs(l - 0.5), l + 0.5):
        raise ValueError(f"{tag}: j must be l±1/2 for a single valence electron; got {j!r}.")
    if abs(mj) > j or (j - abs(mj)) % 1 != 0:
        raise ValueError(f"{tag}: |mj| must be <= j with mj in the j ladder; got mj={mj!r}, j={j!r}.")


def single_photon_rabi(
    power_w: float,
    beam_area_um2: float,
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

    Combines the top-hat field of ``power_w`` (W) over ``beam_area_um2`` (μm²)
    with the ARC dipole matrix element of the
    ``|n1 l1 j1 mj1⟩ → |n2 l2 j2 (mj1+q)⟩`` transition (``Ω = |d| E0 / ħ``).
    ``q`` is the laser polarization (-1, 0, +1 for σ⁻, π, σ⁺).
    """
    if not np.isfinite(power_w) or power_w < 0.0:
        raise ValueError(f"power_w must be finite and non-negative; got {power_w!r}.")
    if not np.isfinite(beam_area_um2) or beam_area_um2 <= 0.0:
        raise ValueError(f"beam_area_um2 must be finite and positive; got {beam_area_um2!r}.")
    if isinstance(q, bool) or q not in (-1, 0, 1):
        raise ValueError(f"q (polarization) must be -1, 0 or +1; got {q!r}.")
    _check_transition_level("lower state", n1, l1, j1, mj1)
    _check_transition_level("upper state", n2, l2, j2, mj1 + q)
    e0 = _electric_field_uniform_beam(power_w, beam_area_um2)
    return float(_get_atom().getRabiFrequency2(n1, l1, j1, mj1, n2, l2, j2, q, e0))


def rb87_7_mp_rabi_frequencies(
    power_420_w: float,
    power_1013_w: float,
    beam_area_um2: float,
    *,
    ryd_level: int = 70,
) -> tuple[float, float]:
    """420/1013 nm single-photon Rabi frequencies (rad/s) for the σ⁻/σ⁺ path.

    Both beams are top-hats of the given power filling the same
    ``beam_area_um2`` (μm²). Transitions match the ``rb87_7_mp`` manifold
    (σ⁻/σ⁺; was param_set ``our``):

      * 420 nm:  5S₁/₂ (mⱼ=-1/2) --σ⁻--> 6P₃/₂ (mⱼ=-3/2)
      * 1013 nm: 6P₃/₂ (mⱼ=-1/2) --σ⁺--> nS₁/₂ (mⱼ=+1/2)

    Returns ``(omega_420, omega_1013)`` in rad/s. The 420 leg carries the
    clock-state ``1/√2`` amplitude factor (mF=0 splitting into mJ=±1/2).
    """
    omega_420 = single_photon_rabi(
        power_420_w, beam_area_um2,
        n1=5, l1=0, j1=0.5, mj1=-0.5, n2=6, l2=1, j2=1.5, q=-1,
    ) / np.sqrt(2)
    omega_1013 = single_photon_rabi(
        power_1013_w, beam_area_um2,
        n1=6, l1=1, j1=1.5, mj1=-1.5, n2=ryd_level, l2=0, j2=0.5, q=1,
    )
    return omega_420, omega_1013


def rb87_297_clock_rabi_frequencies(
    power_297_w: float,
    beam_area_um2: float,
    *,
    ryd_level: int = 53,
) -> tuple[float, float]:
    """297 nm σ⁻ single-photon Rabi frequencies (rad/s) from the clock state.

    One top-hat beam of the given power filling ``beam_area_um2`` (μm²) drives
    both Zeeman branches out of the clock-like ground state
    ``|1⟩ = (|m_J=-1/2, m_I=+1/2⟩ + |m_J=+1/2, m_I=-1/2⟩)/√2``:

      * target:  5S₁/₂ (mⱼ=-1/2) --σ⁻--> nP₃/₂ (mⱼ=-3/2)   (m_I=+1/2 spectator)
      * garbage: 5S₁/₂ (mⱼ=+1/2) --σ⁻--> nP₃/₂ (mⱼ=-1/2)   (m_I=-1/2 spectator)

    Both legs carry the clock-state ``1/√2`` amplitude factor. Returns
    ``(omega_r, omega_r_garb)`` in rad/s.
    """
    omega_r = single_photon_rabi(
        power_297_w, beam_area_um2,
        n1=5, l1=0, j1=0.5, mj1=-0.5, n2=ryd_level, l2=1, j2=1.5, q=-1,
    ) / np.sqrt(2)
    omega_r_garb = single_photon_rabi(
        power_297_w, beam_area_um2,
        n1=5, l1=0, j1=0.5, mj1=0.5, n2=ryd_level, l2=1, j2=1.5, q=-1,
    ) / np.sqrt(2)
    return omega_r, omega_r_garb


# ======================================================================
# MAGNETIC FIELD -> ZEEMAN SHIFT
# ======================================================================


def _lande_gj(l: int, j: float, s: float = 0.5) -> float:
    """Landé g-factor g_J for a fine-structure level ``|l j⟩`` (spin ``s``).

    For an nS_{1/2} level (l=0, j=1/2) this is 2.
    """
    return 1.0 + (j * (j + 1) + s * (s + 1) - l * (l + 1)) / (2 * j * (j + 1))


def zeeman_shift_rad_s(magnetic_field_G: float, *, l: int, j: float, delta_mj: float) -> float:
    """Linear Zeeman shift (rad/s) between two ``m_j`` states of a ``|l j⟩`` level.

    ``Δω = (μ_B / ħ) · g_J(l, j) · Δm_j · B`` with ``B = magnetic_field_G · 1e-4`` (T).
    """
    if not np.isfinite(magnetic_field_G):
        raise ValueError(f"magnetic_field_G must be finite; got {magnetic_field_G!r}.")
    B_T = magnetic_field_G * 1e-4
    return (_MU_B / hbar) * _lande_gj(l, j) * delta_mj * B_T


def _rydberg_zeeman_shift_rad_s(magnetic_field_G: float, *, manifold: str) -> float:
    """Linear Zeeman splitting (rad/s) of ``r_garb`` relative to ``r`` (nS_{1/2}).

    Both ``r`` and ``r_garb`` are ``nS_{1/2}`` states with opposite
    ``m_j = ±1/2`` (``Δm_j = 1``, ``g_J = 2``). Positive for positive ``B``,
    matching ``h[6, 6] = +ryd_zeeman_shift``.
    """
    if manifold not in ("mp", "pm"):
        raise ValueError(f"Unknown rb87 manifold '{manifold}' (expected 'mp' or 'pm').")
    return zeeman_shift_rad_s(magnetic_field_G, l=0, j=0.5, delta_mj=1.0)


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
    1e-9 rad).
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
# BRANCHING RATIOS (private; consumed by core.physical_models)
# ======================================================================


def _rydberg_branching_ratios(atom, ryd_level, manifold):
    """Branching ratios for Rydberg radiative decay.

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
    """Branching ratios for 6P3/2 intermediate-state decay."""
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
