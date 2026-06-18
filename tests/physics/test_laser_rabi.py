"""Tests for the laser-power -> single-photon Rabi helpers in ``ryd_gate.physics``.

Covers the top-hat E-field conversion, the ARC-backed single-photon Rabi, and
the ``our`` 70S 420/1013 convenience wrapper used by
``scripts/error_budget_sweep.py``.
"""

import numpy as np
import pytest
from scipy.constants import c, epsilon_0, hbar, physical_constants

from ryd_gate.physics import (
    _atom,
    electric_field_uniform_beam,
    our_laser_rabis,
    single_photon_rabi,
)

_BOHR = physical_constants["Bohr radius"][0]
_ECHARGE = physical_constants["elementary charge"][0]

# Rectangular array footprint used by error_budget_sweep / error_buget notebook.
_BEAM_SHORT_AXIS_UM = 6.0


def test_electric_field_matches_plane_wave_formula():
    power_w = 6.41
    beam_area_um2 = 100.0**2
    area_m2 = beam_area_um2 * (1e-6) ** 2
    intensity = power_w / area_m2
    expected = np.sqrt(2.0 * intensity / (c * epsilon_0))
    assert electric_field_uniform_beam(power_w, beam_area_um2) == pytest.approx(expected)


def test_electric_field_scalings():
    area_ref_um2 = 100.0**2
    e_ref = electric_field_uniform_beam(1.0, area_ref_um2)
    # E0 ∝ sqrt(P)
    assert electric_field_uniform_beam(4.0, area_ref_um2) == pytest.approx(2.0 * e_ref)
    # E0 ∝ 1 / sqrt(area)
    assert electric_field_uniform_beam(1.0, 4.0 * area_ref_um2) == pytest.approx(e_ref / 2.0)


def test_electric_field_rejects_bad_inputs():
    with pytest.raises(ValueError):
        electric_field_uniform_beam(-1.0, 100.0)
    with pytest.raises(ValueError):
        electric_field_uniform_beam(1.0, 0.0)


def test_single_photon_rabi_matches_manual_dipole_formula():
    """Ω = |d| E0 / ħ with d from the ARC dipole matrix element (a0·e units)."""
    power_w = 6.41
    beam_area_um2 = 100.0**2
    e0 = electric_field_uniform_beam(power_w, beam_area_um2)
    # 420 nm: 5S_1/2(mj=-1/2) --σ⁻--> 6P_3/2(mj=-3/2)
    dipole_au = _atom.getDipoleMatrixElement(5, 0, 0.5, -0.5, 6, 1, 1.5, -1.5, -1)
    expected = abs(dipole_au) * _ECHARGE * _BOHR * e0 / hbar

    got = single_photon_rabi(
        power_w, beam_area_um2,
        n1=5, l1=0, j1=0.5, mj1=-0.5, n2=6, l2=1, j2=1.5, q=-1,
    )
    assert got == pytest.approx(expected, rel=1e-9)


def test_single_photon_rabi_sqrt_power_scaling():
    area_um2 = 100.0**2
    kw = dict(n1=5, l1=0, j1=0.5, mj1=-0.5, n2=6, l2=1, j2=1.5, q=-1)
    base = single_photon_rabi(1.0, area_um2, **kw)
    assert single_photon_rabi(4.0, area_um2, **kw) == pytest.approx(2.0 * base, rel=1e-9)
    # Ω ∝ E0 ∝ 1/sqrt(area)
    assert single_photon_rabi(1.0, 4.0 * area_um2, **kw) == pytest.approx(base / 2.0, rel=1e-9)


def test_our_laser_rabis_known_values():
    """6.41 W / 100 W over a 100 µm × 100 µm top-hat -> ~2405 / ~183 MHz (ARC, 70S)."""
    beam_area_um2 = 100.0**2
    omega_420, omega_1013 = our_laser_rabis(6.41, 100.0, beam_area_um2, ryd_level=70)
    f420_mhz = omega_420 / (2 * np.pi) / 1e6
    f1013_mhz = omega_1013 / (2 * np.pi) / 1e6
    assert f420_mhz == pytest.approx(2405.0, rel=2e-2)
    assert f1013_mhz == pytest.approx(183.0, rel=2e-2)


def test_our_laser_rabis_delegates_to_single_photon_rabi():
    beam_area_um2 = 100.0**2
    omega_420, omega_1013 = our_laser_rabis(6.41, 100.0, beam_area_um2, ryd_level=70)
    assert omega_420 == pytest.approx(
        single_photon_rabi(
            6.41, beam_area_um2,
            n1=5, l1=0, j1=0.5, mj1=-0.5, n2=6, l2=1, j2=1.5, q=-1,
        )
    )
    assert omega_1013 == pytest.approx(
        single_photon_rabi(
            100.0, beam_area_um2,
            n1=6, l1=1, j1=1.5, mj1=-0.5, n2=70, l2=0, j2=0.5, q=1,
        )
    )


def test_our_laser_rabis_rectangular_array_footprint():
    """Matches error_budget_sweep: area = sqrt(N) * a * s with s = 6 µm."""
    n_beam_atoms, a_um = 200, 7.0
    beam_length_um = float(np.sqrt(n_beam_atoms) * a_um)
    beam_area_um2 = beam_length_um * _BEAM_SHORT_AXIS_UM

    p420_eff = 6.4 * (1.0 - 0.90)
    p1013_eff = 100.0 * (1.0 - 0.90)
    omega_420, omega_1013 = our_laser_rabis(
        p420_eff, p1013_eff, beam_area_um2, ryd_level=70,
    )

    f420_mhz = omega_420 / (2 * np.pi) / 1e6
    f1013_mhz = omega_1013 / (2 * np.pi) / 1e6
    assert f420_mhz == pytest.approx(3118.6, rel=2e-2)
    assert f1013_mhz == pytest.approx(237.7, rel=2e-2)

    # Same intensity as spreading the same power over the area directly.
    expected_e0 = electric_field_uniform_beam(p420_eff, beam_area_um2)
    assert omega_420 == pytest.approx(
        _atom.getRabiFrequency2(5, 0, 0.5, -0.5, 6, 1, 1.5, -1, expected_e0),
        rel=1e-9,
    )
