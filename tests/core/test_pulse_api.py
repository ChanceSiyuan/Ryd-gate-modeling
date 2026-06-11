"""Tests for the product Waveform/Pulse API (pulse.py)."""

import numpy as np
import pytest

from ryd_gate.protocols.channels import ChannelSpec
from ryd_gate.pulse import Pulse, Waveform


def _channel(**overrides):
    base = dict(
        channel_id="test_channel",
        kind="rydberg",
        transition="1_r",
        addressing="global",
    )
    base.update(overrides)
    return ChannelSpec(**base)


class TestWaveformConstructors:
    def test_constant_samples_all_equal(self):
        wf = Waveform.constant(1000, 2.5)
        assert wf.kind == "constant"
        assert np.all(wf.sample(100) == 2.5)
        assert wf.value_at_ns(123.4) == 2.5

    def test_ramp_endpoints_and_linearity(self):
        wf = Waveform.ramp(1000, start=0.0, stop=5.0)
        assert wf.first_value() == 0.0
        assert wf.last_value() == 5.0
        assert wf.value_at_ns(500) == pytest.approx(2.5)

    def test_blackman_peak_form(self):
        wf = Waveform.blackman(1000, peak=3.0)
        assert wf.value_at_ns(0) == pytest.approx(0.0, abs=1e-12)
        assert wf.value_at_ns(1000) == pytest.approx(0.0, abs=1e-12)
        assert wf.value_at_ns(500) == pytest.approx(3.0)
        # window formula at the quarter point
        t, T, peak = 250.0, 1000.0, 3.0
        expected = peak * (0.42 - 0.5 * np.cos(2 * np.pi * t / T) + 0.08 * np.cos(4 * np.pi * t / T))
        assert wf.value_at_ns(t) == pytest.approx(expected)

    def test_blackman_area_form(self):
        area = np.pi
        wf = Waveform.blackman(1000, area=area)
        assert abs(wf.integral_rad(1) - area) <= 1e-3 * abs(area)
        neg = Waveform.blackman(1000, area=-np.pi)
        assert neg.params["peak"] < 0

    def test_blackman_exactly_one_of_peak_area(self):
        with pytest.raises(ValueError, match="exactly one"):
            Waveform.blackman(1000)
        with pytest.raises(ValueError, match="exactly one"):
            Waveform.blackman(1000, peak=1.0, area=np.pi)

    def test_interpolated_piecewise_linear(self):
        wf = Waveform.interpolated(1000, times_ns=[0, 500, 1000], values=[0.0, 2.0, 0.0])
        assert wf.value_at_ns(250) == pytest.approx(1.0)
        assert wf.value_at_ns(750) == pytest.approx(1.0)

    def test_interpolated_invalid_grids_raise(self):
        with pytest.raises(ValueError):
            Waveform.interpolated(1000, [0, 1000, 500], [0.0, 1.0, 0.0])
        with pytest.raises(ValueError):
            Waveform.interpolated(1000, [0, 500], [0.0, 1.0, 0.0])
        with pytest.raises(ValueError):
            Waveform.interpolated(1000, [100, 1000], [0.0, 1.0])
        with pytest.raises(ValueError):
            Waveform.interpolated(1000, [0, 500, 1000], [0.0, np.inf, 0.0])

    def test_custom_duration_and_interpolation(self):
        wf = Waveform.custom([0.0, 1.0, 0.0], dt_ns=10)
        assert wf.duration_ns == 20
        assert wf.samples == (0.0, 1.0, 0.0)
        assert wf.value_at_ns(5) == pytest.approx(0.5)

    def test_custom_invalid_raises(self):
        with pytest.raises(ValueError):
            Waveform.custom([1.0], dt_ns=10)
        with pytest.raises(ValueError):
            Waveform.custom([0.0, np.nan], dt_ns=10)
        with pytest.raises(ValueError):
            Waveform.custom([0.0, 1.0], dt_ns=0)

    def test_invalid_duration_raises(self):
        with pytest.raises(ValueError):
            Waveform.constant(0, 1.0)
        with pytest.raises(ValueError):
            Waveform.constant(-5, 1.0)
        with pytest.raises(ValueError):
            Waveform.constant(10.5, 1.0)


class TestWaveformEvaluation:
    def test_value_at_ns_clamps(self):
        wf = Waveform.ramp(1000, 0.0, 5.0)
        assert wf.value_at_ns(-50) == 0.0
        assert wf.value_at_ns(2000) == 5.0

    def test_value_at_s_conversion(self):
        wf = Waveform.constant(1000, 2.5)
        assert wf.value_at_s(250e-9) == pytest.approx(2.5e6)
        ramp = Waveform.ramp(1000, 0.0, 4.0)
        assert ramp.value_at_s(500e-9) == pytest.approx(ramp.value_at_ns(500) * 1e6)

    def test_sample_includes_endpoint(self):
        wf = Waveform.ramp(1000, 0.0, 5.0)
        values = wf.sample(250)
        assert len(values) == 5
        assert values[-1] == 5.0

    def test_sample_divisibility_enforced(self):
        wf = Waveform.constant(1000, 1.0)
        with pytest.raises(ValueError, match="divide"):
            wf.sample(300)
        with pytest.raises(ValueError, match="divide"):
            wf.integral_rad(300)

    def test_integral_constant(self):
        wf = Waveform.constant(1000, 2.0)  # 2 rad/us over 1 us -> 2 rad
        assert wf.integral_rad(10) == pytest.approx(2.0)


class TestPulse:
    def test_constant_pulse_matching_waveforms(self):
        pulse = Pulse.constant(1000, amplitude=1.0, detuning=0.5, phase_rad=0.25)
        assert pulse.amplitude.kind == "constant"
        assert pulse.detuning.kind == "constant"
        assert pulse.duration_ns == 1000
        assert pulse.phase_rad == 0.25

    def test_constant_amplitude_and_detuning_helpers(self):
        sweep = Pulse.constant_amplitude(1.0, Waveform.ramp(800, -5.0, 5.0))
        assert sweep.amplitude.kind == "constant"
        assert sweep.amplitude.duration_ns == 800
        rise = Pulse.constant_detuning(Waveform.blackman(600, area=np.pi), 0.0)
        assert rise.detuning.kind == "constant"
        assert rise.detuning.duration_ns == 600

    def test_mismatched_durations_raise(self):
        with pytest.raises(ValueError, match="duration"):
            Pulse(amplitude=Waveform.constant(1000, 1.0), detuning=Waveform.constant(500, 0.0))

    def test_non_finite_phase_raises(self):
        with pytest.raises(ValueError, match="phase"):
            Pulse.constant(1000, 1.0, 0.0, phase_rad=np.inf)

    def test_validate_unconstrained_channel_passes(self):
        pulse = Pulse.constant(1000, 1.0, 0.0)
        assert pulse.validate(_channel()) == []

    def test_validate_limits(self):
        channel = _channel(
            min_duration_ns=100,
            max_duration_ns=2000,
            clock_period_ns=4,
            max_abs_amplitude_rad_per_us=2.0,
            max_abs_detuning_rad_per_us=1.0,
        )
        too_short = Pulse.constant(96, 1.0, 0.0)
        assert {i.code for i in too_short.validate(channel)} == {"pulse.min_duration"}
        too_long = Pulse.constant(2400, 1.0, 0.0)
        assert {i.code for i in too_long.validate(channel)} == {"pulse.max_duration"}
        off_clock = Pulse.constant(1002, 1.0, 0.0)
        assert {i.code for i in off_clock.validate(channel)} == {"pulse.clock_period"}
        too_strong = Pulse.constant(1000, 3.0, 0.0)
        assert {i.code for i in too_strong.validate(channel)} == {"pulse.amplitude_limit"}
        too_detuned = Pulse.constant(1000, 1.0, 5.0)
        assert {i.code for i in too_detuned.validate(channel)} == {"pulse.detuning_limit"}


class TestKernelHelperClosure:
    def test_kernel_helpers_importable_by_name(self):
        from ryd_gate.pulse import blackman_pulse, blackman_pulse_sqrt, blackman_window

        assert callable(blackman_window)
        assert callable(blackman_pulse)
        assert callable(blackman_pulse_sqrt)

    def test_top_level_blackman_import_fails(self):
        with pytest.raises(ImportError):
            from ryd_gate import blackman_pulse  # noqa: F401

    def test_pulse_module_all_is_product_only(self):
        import ryd_gate.pulse as pulse_module

        assert set(pulse_module.__all__) == {"Pulse", "Waveform"}

    def test_star_import_excludes_kernel_helpers(self):
        namespace: dict = {}
        exec("from ryd_gate.pulse import *", namespace)
        assert "Waveform" in namespace
        assert "Pulse" in namespace
        assert not any(name.startswith("blackman") for name in namespace)
