"""PSD model + phase-trace generation (PRA 107, 042611).

One-sided densities throughout: sigma_nu**2 = int_0^inf S_dnu df. The paper uses
two-sided densities, so its closed forms are halved when compared here.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ryd_gate.phase_noise import (
    PhaseNoisePSD,
    error_from_kernel,
    filter_kernel,
    log_frequency_bins,
    phase_trace,
)

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


def test_trace_derivative_is_the_instantaneous_frequency_and_is_guarded():
    # A time-domain solver needs d(phi)/dt = 2 pi dnu(t), not phi. CubicSpline's own
    # derivative inherits extrapolate=True, so reaching for it privately would
    # silently continue a quadratic past t_gate instead of raising -- exactly what
    # __call__'s guard exists to prevent.
    tr = phase_trace(PhaseNoisePSD.white(1e4), 1e-6, seed=5, f_max=1e8)
    mid = 0.5 * (tr.times[:-1] + tr.times[1:])
    fd = np.diff(tr.values) / np.diff(tr.times)
    assert np.max(np.abs(tr.derivative(mid) - fd)) < 1e-3 * np.max(np.abs(fd))
    with pytest.raises(ValueError, match="must lie in"):
        tr.derivative(1.1e-6)
    with pytest.raises(ValueError, match="must lie in"):
        tr.derivative(np.array([0.0, -1e-9]))
    # a rounding-level overshoot, which solver times do produce, is still accepted
    over = np.nextafter(np.nextafter(tr.times[-1], np.inf), np.inf)
    assert np.isfinite(tr.derivative(over))


def test_f_min_is_forwarded_to_the_quasi_static_band():
    # ASD ~ 1/f, so S_dnu ~ f^-2 and the frozen band is dominated by its lower edge:
    # moving f_min from 1 Hz to 100 Hz must shrink dnu_0 about tenfold.
    psd = PhaseNoisePSD(np.array([1e0, 1e6]), np.array([1e6, 1e0]) ** 2)
    t_gate, f_split = 1e-6, 1e4                     # the collapse sits at 0.01/t_gate
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
    expected = psd.sigma_nu(1.0, 0.01 / t_gate)
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


def _rabi_pi_pulse_components(omega0, n_rot, n_t):
    """<perp|A(t)> for a resonant two-level Rabi drive of N rotations.

    A(t) = U(T,t) N_e psi(t) with N_e = |e><e|; the metric projects onto the
    single direction orthogonal to the ideal final state, so there is one
    component. Everything is available in closed form for a constant drive:
    U(t) = cos(omega0 t/2) I - i sin(omega0 t/2) sigma_x.
    """
    t_gate = 2.0 * np.pi * n_rot / omega0
    t = np.linspace(0.0, t_gate, n_t)

    def u(tau):
        c, s = np.cos(0.5 * omega0 * tau), np.sin(0.5 * omega0 * tau)
        return np.array([[c, -1j * s], [-1j * s, c]])

    psi0 = np.array([1.0, 0.0], dtype=complex)
    psi_T = u(t_gate) @ psi0
    perp = np.array([-np.conj(psi_T[1]), np.conj(psi_T[0])])
    comp = np.empty((n_t, 1), dtype=complex)
    for k, tk in enumerate(t):
        psi_k = u(tk) @ psi0
        a_k = u(t_gate - tk) @ (np.array([0.0, 1.0]) * psi_k)   # N_e psi
        comp[k, 0] = np.vdot(perp, a_k)
    return t, comp, t_gate


class _ZERO_TRACE:
    """Noiseless stand-in with the PhaseTrace call signature."""

    def __init__(self, t_gate):
        self.t_gate = t_gate

    def __call__(self, t):
        return 0.0 * np.asarray(t, dtype=float)


def test_filter_kernel_reproduces_the_paper_white_noise_gate_error():
    # Paper Eq. 79 (initial state |0>): eps = pi**3 h0 N / Omega_0, with h0
    # TWO-SIDED. Against our one-sided h0 the target is half that.
    omega0 = 2 * np.pi * 1e6
    h0_onesided = 200.0
    for n_rot in (0.5, 1.0):
        t, comp, t_gate = _rabi_pi_pulse_components(omega0, n_rot, 16385)
        f_bins, df_bins = log_frequency_bins(1e2, 1e9, 60)
        kernel = filter_kernel(t, comp, f_bins, df_bins)
        eps = error_from_kernel(PhaseNoisePSD.white(h0_onesided), f_bins, kernel)
        expected = np.pi**3 * (h0_onesided / 2) * n_rot / omega0
        assert eps == pytest.approx(expected, rel=0.05)


def test_filter_kernel_agrees_with_direct_monte_carlo():
    """Same two-level pi pulse, integrated with real phase traces.

    ``n_rot`` is 4, not the 0.5 of the closed-form test above, because the two legs
    are limited by different things. ``phase_trace`` collapses everything below
    ``1/t_gate`` into one static offset, which reproduces the gate's response there
    only while that response is still flat -- and a T-long gate's response varies on
    exactly the ``1/T`` scale. At half a rotation 97% of ``int ||G||**2 df`` sits
    below ``1/t_gate`` and the collapse mismodels it by 65%; by four rotations the
    response has moved out to the Rabi peak ``Omega_0/2pi = n_rot/T`` and only 1.3%
    is left in the collapsed band. ``f_max`` is 10x that peak, above which the
    response is nil, so the traces carry a decade of out-of-band noise for the
    kernel to reject.
    """
    from scipy.integrate import solve_ivp

    omega0, h0 = 2 * np.pi * 1e6, 200.0
    n_rot = 4.0
    t_gate = 2.0 * np.pi * n_rot / omega0
    psd = PhaseNoisePSD.white(h0)

    def run(trace):
        def rhs(tt, y):
            c = 0.5 * omega0 * np.exp(-1j * trace(tt))
            return -1j * np.array([c * y[1], np.conj(c) * y[0]])
        sol = solve_ivp(rhs, (0.0, t_gate), np.array([1.0, 0.0], dtype=complex),
                        method="DOP853", rtol=1e-10, atol=1e-13)
        return sol.y[:, -1]

    ideal = run(_ZERO_TRACE(t_gate))
    errs = []
    for s in range(300):
        psi = run(phase_trace(psd, t_gate, seed=s, f_max=1e7, n_samples=8192))
        errs.append(1.0 - abs(np.vdot(ideal, psi)) ** 2)
    mc = float(np.mean(errs))

    t, comp, _ = _rabi_pi_pulse_components(omega0, n_rot, 16385)
    f_bins, df_bins = log_frequency_bins(1e2, 1e9, 60)
    predicted = error_from_kernel(psd, f_bins,
                                  filter_kernel(t, comp, f_bins, df_bins))
    stderr = float(np.std(errs) / np.sqrt(len(errs)))
    assert abs(mc - predicted) < 4 * stderr + 0.05 * predicted


def test_subtracting_the_whole_response_leaves_exactly_zero():
    """``Q = 1`` must give zero: norm conservation, no simulation needed.

    This is the refutation of the first version of the metric. With a single
    component and ``subtract`` equal to it, ``Q = 1 - |psi_0><psi_0|`` annihilates the
    only direction there is, so every bin cancels to rounding — whereas the
    unsubtracted sum is a positive number, which is what a fixed projector that does
    not annihilate ``psi_0(T)`` silently reports.
    """
    t = np.linspace(0.0, 5e-7, 4097)
    comp = (np.cos(2 * np.pi * 3e6 * t) + 1j * t / 5e-7)[:, None]
    f_bins, df_bins = log_frequency_bins(1e2, 1e9, 30)
    unsubtracted = filter_kernel(t, comp, f_bins, df_bins)
    assert np.max(unsubtracted) > 0.0
    zeroed = filter_kernel(t, comp, f_bins, df_bins, subtract=comp[:, 0])
    assert np.max(np.abs(zeroed)) < 1e-10 * np.max(unsubtracted)


def test_subtracted_kernel_is_the_orthogonal_complement_and_stays_non_negative():
    """``K`` with ``subtract`` is ``||G||**2 - |<psi_0|G>|**2``, bin by bin."""
    t = np.linspace(0.0, 5e-7, 4097)
    rng = np.random.default_rng(4)
    comp = (rng.standard_normal((t.size, 3)) + 1j * rng.standard_normal((t.size, 3)))
    psi0 = np.array([0.6, 0.8j, 0.0])                       # a unit direction
    sub = comp @ psi0.conj()
    f_bins, df_bins = log_frequency_bins(1e2, 1e9, 30)
    full = filter_kernel(t, comp, f_bins, df_bins)
    perp = filter_kernel(t, comp, f_bins, df_bins, subtract=sub)
    only = filter_kernel(t, sub[:, None], f_bins, df_bins)
    assert np.allclose(perp, full - only, rtol=1e-10, atol=0.0)
    assert np.all(perp >= 0.0)


def test_filter_kernel_counts_both_signs_of_the_frequency_axis():
    """``K_b`` integrates ``||G(f)||**2 + ||G(-f)||**2``, not twice ``||G(f)||**2``.

    The Rabi components above are ``i`` times a real function, so ``|G(-f)| = |G(f)|``
    identically and the two readings coincide to the last bit -- neither test above
    would notice a kernel that dropped one sign and doubled the other. A pure
    ``exp(+2j pi f0 t)`` breaks the symmetry: all its weight sits at ``+f0``, so
    Parseval puts ``sum_b K_b`` at ``T`` while doubling the positive branch alone
    gives ``2T``.
    """
    t_gate, f0 = 5e-7, 4e6
    t = np.linspace(0.0, t_gate, 16385)
    comp = np.exp(2j * np.pi * f0 * t)[:, None]
    f_bins, df_bins = log_frequency_bins(1e2, 1e9, 60)
    total = float(np.sum(filter_kernel(t, comp, f_bins, df_bins)))
    assert total == pytest.approx(t_gate, rel=0.01)


def test_the_evaluation_grid_must_resolve_the_1_over_T_fringes():
    """``fine_per_decade`` is a convergence parameter, and 200 is not enough at 4.5 us.

    ``sum_b K_b`` for a pure tone is ``T`` by Parseval whatever the grid *can see*, so
    this measures the evaluation grid alone with no solver in the loop.  ``K(f)`` has
    sinc fringes of width ``1/T`` while a logarithmic grid of ``p`` points per decade
    has spacing ``f ln10 / p``, so the grid must satisfy ``p >= ln10 f T`` across the
    band.  At 4.5 us the default 200 overstates a 50 MHz tone by 41% and loses 91% of
    a 150 MHz one — the two failure directions of straddling a fringe — and it is
    what put the n=73 / 4.5 us kernel 13% high before
    ``max_leakage_297_sweep.kernel_fine_per_decade`` sized the grid to ``T``.
    """
    f_bins, df_bins = log_frequency_bins(1.0, 2.0e8, 30)
    t_gate = 4.5e-6
    t = np.linspace(0.0, t_gate, 4096)
    for f0, needed in ((5.0e7, 800), (1.5e8, 1600)):
        comp = np.exp(2j * np.pi * f0 * t)[:, None]

        def total(fpd):
            return float(np.sum(filter_kernel(t, comp, f_bins, df_bins,
                                              fine_per_decade=fpd))) / t_gate

        assert abs(total(200) - 1.0) > 0.4                # the default is badly wrong
        assert total(needed) == pytest.approx(1.0, rel=0.01)
        assert total(2 * needed) == pytest.approx(1.0, rel=0.01)
