"""PSD model + phase-trace generation (PRA 107, 042611).

One-sided densities throughout: sigma_nu**2 = int_0^inf S_dnu df. The paper uses
two-sided densities, so its closed forms are halved when compared here.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ryd_gate.phase_noise import PhaseNoisePSD

NOISE_DIR = Path(__file__).resolve().parents[1] / "results" / "297_laser_noise"


def test_white_psd_is_flat_and_s_phi_falls_as_f_squared():
    psd = PhaseNoisePSD.white(100.0)
    f = np.array([1.0, 1e3, 1e6])
    assert np.allclose(psd.s_dnu(f), 100.0)
    assert np.allclose(psd.s_phi(f), 100.0 / f**2)


def test_white_psd_sigma_nu_is_sqrt_h0_times_bandwidth():
    psd = PhaseNoisePSD.white(100.0)
    assert psd.sigma_nu(1e-6, 1e4) == pytest.approx(np.sqrt(100.0 * 1e4), rel=1e-6)


def test_sigma_nu_rejects_a_nonpositive_lower_edge():
    # f_lo is the modelling parameter f_min, not a formality: below the measurement
    # edge S_dnu is extrapolated, so an unbounded band fabricates power silently.
    psd = PhaseNoisePSD.white(100.0)
    with pytest.raises(ValueError, match="f_lo must be positive"):
        psd.sigma_nu(0.0, 1e4)


def test_harmonic_scales_s_dnu_by_its_square():
    base = PhaseNoisePSD.white(100.0)
    quad = PhaseNoisePSD.white(100.0, harmonic=4)
    assert quad.s_dnu(np.array([1e6])) == pytest.approx(16 * base.s_dnu(np.array([1e6])))


def test_servo_bump_adds_a_gaussian_peak_of_the_requested_total_power():
    # s_g is the dimensionless total phase-noise power of the bump (paper Eq. 41);
    # the bump's S_dnu integral over positive f is therefore s_g * f_g**2.
    s_g, f_g, sigma_g = 1e-4, 1.0e6, 2.0e4
    psd = PhaseNoisePSD.white(0.0, servo_bump=(s_g, f_g, sigma_g))
    f = np.linspace(f_g - 6 * sigma_g, f_g + 6 * sigma_g, 20001)
    assert np.trapezoid(psd.s_dnu(f), f) == pytest.approx(s_g * f_g**2, rel=1e-3)


def test_measured_psd_interpolates_in_log_log_and_holds_flat_above_the_edge():
    f = np.array([1e3, 1e4, 1e5, 1e6])
    asd = np.array([1e3, 1e2, 1e1, 1e0])            # exactly ASD ~ f^-1
    psd = PhaseNoisePSD(f, asd**2, extrapolation="flat")
    assert psd.s_dnu(np.array([10**3.5]))[0] == pytest.approx((10**2.5) ** 2, rel=1e-9)
    assert psd.s_dnu(np.array([1e8]))[0] == pytest.approx(1.0, rel=1e-9)


def test_below_the_lowest_sample_the_lowest_measured_slope_continues():
    f = np.array([1e3, 1e4, 1e5, 1e6])
    asd = np.array([1e3, 1e2, 1e1, 1e0])            # ASD ~ f^-1, so S_dnu ~ f^-2
    psd = PhaseNoisePSD(f, asd**2)
    # slope continued, not clamped: a flat hold would give the 1 kHz value 1e6.
    assert psd.s_dnu(np.array([1e2]))[0] == pytest.approx(1e8, rel=1e-9)


def test_power_law_extrapolation_continues_the_fitted_slope():
    f = np.array([1e3, 1e4, 1e5, 1e6])
    asd = np.array([1e3, 1e2, 1e1, 1e0])            # p = 1 exactly
    psd = PhaseNoisePSD(f, asd**2, extrapolation="power",
                        power_law_fit_hz=(1e5, 1e6))
    assert psd.power_law_exponent == pytest.approx(1.0, abs=1e-9)
    assert psd.s_dnu(np.array([1e7]))[0] == pytest.approx(1e-2, rel=1e-9)


def test_from_csv_reads_the_digitized_measurement(tmp_path):
    p = tmp_path / "psd.csv"
    p.write_text("f_Hz,asd_mean_Hz_per_rtHz,asd_lo,asd_hi\n"
                 "# comment line\n"
                 "1e3,1e3,1,1\n1e6,1e0,1,1\n")
    psd = PhaseNoisePSD.from_csv(p, harmonic=4)
    assert psd.s_dnu(np.array([1e6]))[0] == pytest.approx(16.0, rel=1e-9)


def test_from_csv_reproduces_the_committed_digitizer_model():
    # scripts/laser_noise_psd.py writes psd_model.json through this class, so the
    # committed artifact and the library cannot drift apart unnoticed.
    model = json.loads((NOISE_DIR / "psd_model.json").read_text())
    assert set(model) == {"ECDL", "seed"}
    for entry in model.values():
        psd = PhaseNoisePSD.from_csv(NOISE_DIR / entry["csv"], harmonic=entry["harmonic"])
        assert psd.f_hz[-1] == pytest.approx(entry["f_edge_hz"], rel=1e-12)
        assert psd.power_law_exponent == pytest.approx(entry["power_law_exponent"], rel=1e-9)
        assert psd.s_dnu(np.array([entry["f_edge_hz"]]))[0] == pytest.approx(
            entry["s_dnu_edge_297"], rel=1e-9)
