"""Tests for the laser-power -> single-photon Rabi helpers in ``ryd_gate.physics``.

Covers the ARC-backed single-photon Rabi and the ``rb87_7_mp`` 70S 420/1013
convenience wrapper (PH03/PH04). The top-hat E-field conversion is a private
helper; it is exercised through ``single_photon_rabi``.
"""

import numpy as np
import pytest
from scipy.constants import hbar, physical_constants

from ryd_gate.physics import (
    _electric_field_uniform_beam,
    _get_atom,
    rb87_7_mp_rabi_frequencies,
    single_photon_rabi,
)

_BOHR = physical_constants["Bohr radius"][0]
_ECHARGE = physical_constants["elementary charge"][0]

# Rectangular array footprint used by error_budget_sweep / error_buget notebook.
_BEAM_SHORT_AXIS_UM = 6.0


def test_single_photon_rabi_matches_manual_dipole_formula():
    """Ω = |d| E0 / ħ with d from the ARC dipole matrix element (a0·e units)."""
    power_w = 6.41
    beam_area_um2 = 100.0**2
    e0 = _electric_field_uniform_beam(power_w, beam_area_um2)
    # 420 nm: 5S_1/2(mj=-1/2) --σ⁻--> 6P_3/2(mj=-3/2)
    dipole_au = _get_atom().getDipoleMatrixElement(5, 0, 0.5, -0.5, 6, 1, 1.5, -1.5, -1)
    expected = abs(dipole_au) * _ECHARGE * _BOHR * e0 / hbar

    got = single_photon_rabi(
        power_w, beam_area_um2,
        n1=5, l1=0, j1=0.5, mj1=-0.5, n2=6, l2=1, j2=1.5, q=-1,
    )
    assert got == pytest.approx(expected, rel=1e-9)


def test_single_photon_rabi_sqrt_power_and_area_scaling():
    area_um2 = 100.0**2
    kw = dict(n1=5, l1=0, j1=0.5, mj1=-0.5, n2=6, l2=1, j2=1.5, q=-1)
    base = single_photon_rabi(1.0, area_um2, **kw)
    assert single_photon_rabi(4.0, area_um2, **kw) == pytest.approx(2.0 * base, rel=1e-9)
    assert single_photon_rabi(1.0, 4.0 * area_um2, **kw) == pytest.approx(base / 2.0, rel=1e-9)


def test_single_photon_rabi_validates_inputs():
    kw = dict(n1=5, l1=0, j1=0.5, mj1=-0.5, n2=6, l2=1, j2=1.5, q=-1)
    with pytest.raises(ValueError):
        single_photon_rabi(-1.0, 100.0, **kw)          # negative power
    with pytest.raises(ValueError):
        single_photon_rabi(1.0, 0.0, **kw)             # non-positive area
    with pytest.raises(ValueError):
        single_photon_rabi(1.0, 100.0, **{**kw, "q": 2})   # bad polarization
    with pytest.raises(ValueError):
        single_photon_rabi(1.0, 100.0, **{**kw, "mj1": 2.5})  # |mj| > j


def test_rb87_7_mp_rabi_known_values():
    """6.41 W / 100 W over a 100 µm × 100 µm top-hat -> ~1701 / ~317 MHz (ARC, 70S)."""
    beam_area_um2 = 100.0**2
    omega_420, omega_1013 = rb87_7_mp_rabi_frequencies(6.41, 100.0, beam_area_um2, ryd_level=70)
    assert omega_420 / (2 * np.pi) / 1e6 == pytest.approx(1700.9, rel=2e-2)
    assert omega_1013 / (2 * np.pi) / 1e6 == pytest.approx(317.3, rel=2e-2)


def test_rb87_7_mp_rabi_delegates_to_single_photon_rabi():
    beam_area_um2 = 100.0**2
    omega_420, omega_1013 = rb87_7_mp_rabi_frequencies(6.41, 100.0, beam_area_um2, ryd_level=70)
    assert omega_420 == pytest.approx(
        single_photon_rabi(
            6.41, beam_area_um2,
            n1=5, l1=0, j1=0.5, mj1=-0.5, n2=6, l2=1, j2=1.5, q=-1,
        )
        / np.sqrt(2)  # mF=0 splitting into mJ=±1/2
    )
    assert omega_1013 == pytest.approx(
        single_photon_rabi(
            100.0, beam_area_um2,
            n1=6, l1=1, j1=1.5, mj1=-1.5, n2=70, l2=0, j2=0.5, q=1,
        )
    )


def test_rb87_7_mp_rabi_rectangular_array_footprint():
    """Matches error_budget_sweep: area = sqrt(N) * a * s with s = 6 µm."""
    n_beam_atoms, a_um = 200, 7.0
    beam_length_um = float(np.sqrt(n_beam_atoms) * a_um)
    beam_area_um2 = beam_length_um * _BEAM_SHORT_AXIS_UM

    p420_eff = 6.4 * (1.0 - 0.90)
    p1013_eff = 100.0 * (1.0 - 0.90)
    omega_420, omega_1013 = rb87_7_mp_rabi_frequencies(
        p420_eff, p1013_eff, beam_area_um2, ryd_level=70,
    )
    assert omega_420 / (2 * np.pi) / 1e6 == pytest.approx(2205.2, rel=2e-2)
    assert omega_1013 / (2 * np.pi) / 1e6 == pytest.approx(411.7, rel=2e-2)


def test_beam_area_param_is_um2():
    """The area parameter is named beam_area_um2 (keyword-only spelling check)."""
    kw = dict(n1=5, l1=0, j1=0.5, mj1=-0.5, n2=6, l2=1, j2=1.5, q=-1)
    a = single_photon_rabi(1.0, beam_area_um2=100.0, **kw)
    b = single_photon_rabi(1.0, 100.0, **kw)
    assert a == pytest.approx(b)
