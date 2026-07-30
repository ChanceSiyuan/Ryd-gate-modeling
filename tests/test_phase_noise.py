"""PSD model + phase-trace generation (PRA 107, 042611).

One-sided densities throughout: sigma_nu**2 = int_0^inf S_dnu df. The paper uses
two-sided densities, so its closed forms are halved when compared here.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ryd_gate.phase_noise import PhaseNoisePSD, log_frequency_bins, phase_trace

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


def test_trace_is_reproducible_and_seed_dependent():
    psd = PhaseNoisePSD.white(1e4)
    a = phase_trace(psd, 1e-6, seed=7, f_max=1e8)
    b = phase_trace(psd, 1e-6, seed=7, f_max=1e8)
    c = phase_trace(psd, 1e-6, seed=8, f_max=1e8)
    assert np.array_equal(a.values, b.values)
    assert not np.allclose(a.values, c.values)


def test_trace_callable_matches_its_samples():
    psd = PhaseNoisePSD.white(1e4)
    tr = phase_trace(psd, 1e-6, seed=3, f_max=1e8)
    mid = 0.5 * (tr.times[:-1] + tr.times[1:])
    # the spline reproduces its own nodes exactly and stays close between them
    assert np.max(np.abs(tr(tr.times) - tr.values)) < 1e-12
    assert np.max(np.abs(tr(mid) - np.interp(mid, tr.times, tr.values))) < 1e-3


def test_trace_rejects_times_outside_the_gate():
    # clipping an arbitrary overshoot would freeze the 2 pi dnu_0 t ramp at its final
    # value, i.e. hand a solver a plausible trace and a silently optimistic error.
    tr = phase_trace(PhaseNoisePSD.white(1e4), 1e-6, seed=5, f_max=1e8)
    with pytest.raises(ValueError, match="must lie in"):
        tr(1.1e-6)
    with pytest.raises(ValueError, match="must lie in"):
        tr(np.array([0.0, -1e-9]))
    # but a rounding-level overshoot, which solver times do produce, is still the endpoint
    over = np.nextafter(np.nextafter(tr.times[-1], np.inf), np.inf)
    assert tr(over) == pytest.approx(tr.values[-1], abs=1e-12)


def test_f_min_is_forwarded_to_the_quasi_static_band():
    # ASD ~ 1/f, so S_dnu ~ f^-2 and the frozen band is dominated by its lower edge:
    # moving f_min from 1 Hz to 100 Hz must shrink dnu_0 about tenfold.
    psd = PhaseNoisePSD(np.array([1e0, 1e6]), np.array([1e6, 1e0]) ** 2)
    t_gate, f_split = 1e-6, 1e6
    wide = phase_trace(psd, t_gate, seed=11, f_max=1e8)
    narrow = phase_trace(psd, t_gate, seed=11, f_max=1e8, f_min=100.0)
    # one seed draws one standard normal, so the ratio is exactly the band sigma ratio
    assert narrow.dnu_0 / wide.dnu_0 == pytest.approx(
        psd.sigma_nu(100.0, f_split) / psd.sigma_nu(1.0, f_split), rel=1e-12)
    assert narrow.dnu_0 / wide.dnu_0 == pytest.approx(0.1, rel=0.02)


def test_log_frequency_bins_tile_the_span_with_interior_centres():
    f_lo, f_hi = 1e3, 1e6
    f, df = log_frequency_bins(f_lo, f_hi, 40)
    assert f.size == 120                                    # 3 decades x 40 points
    assert float(np.sum(df)) == pytest.approx(f_hi - f_lo, rel=1e-12)
    edges = np.concatenate(([f_lo], f_lo + np.cumsum(df)))
    assert np.all(edges[:-1] < f) and np.all(f < edges[1:])


def test_log_frequency_bins_reject_a_non_increasing_span():
    # f_max below 1/t_gate would otherwise give descending edges, negative widths and
    # sqrt of a negative amplitude: an all-NaN trace with no exception anywhere.
    with pytest.raises(ValueError, match="0 < f_lo < f_hi"):
        log_frequency_bins(1e6, 1e3, 40)


def test_resolved_band_variance_matches_the_psd_integral():
    # var(phi) over the explicitly summed band equals int 2 S_phi df, since each
    # term sqrt(2 S_phi df) cos(...) contributes half its squared amplitude.
    psd = PhaseNoisePSD.white(1e6)
    t_gate, f_max = 1e-6, 1e8
    traces = [phase_trace(psd, t_gate, seed=s, f_max=f_max) for s in range(400)]
    resolved = np.asarray([tr.values - 2 * np.pi * tr.dnu_0 * tr.times for tr in traces])
    expected = float(np.sum(psd.s_phi(traces[0].f_grid) * traces[0].df_grid))
    assert np.var(resolved) == pytest.approx(expected, rel=0.15)


def test_quasi_static_offset_matches_the_unresolved_band():
    psd = PhaseNoisePSD.white(1e6)
    t_gate = 1e-6
    offsets = np.asarray([phase_trace(psd, t_gate, seed=s, f_max=1e8).dnu_0
                          for s in range(2000)])
    expected = psd.sigma_nu(1.0, 1.0 / t_gate)
    assert np.std(offsets) == pytest.approx(expected, rel=0.1)


def test_generated_traces_reproduce_the_input_spectrum():
    from scipy.signal import welch
    psd = PhaseNoisePSD.white(1e4)
    t_gate, n = 4e-6, 16384
    traces = [phase_trace(psd, t_gate, seed=s, f_max=2e8, n_samples=n)
              for s in range(60)]
    dt = traces[0].times[1] - traces[0].times[0]
    dnu = np.gradient(np.asarray([tr.values for tr in traces]), dt, axis=1) / (2 * np.pi)
    f, pxx = welch(dnu, fs=1.0 / dt, nperseg=n // 8, axis=1)
    band = (f > 5e6) & (f < 5e7)
    assert np.median(pxx.mean(axis=0)[band]) == pytest.approx(1e4, rel=0.35)
