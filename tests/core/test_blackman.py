"""Tests for the public Blackman pulse envelope (ryd_gate.protocols.blackman_pulse)."""

import numpy as np
import pytest

from ryd_gate.protocols import blackman_pulse


class TestBlackmanPulse:
    def test_flat_top(self):
        """Pulse is ~1.0 in the flat-top region."""
        rise, gate = 0.1, 1.0
        t = np.linspace(rise + 0.01, gate - rise - 0.01, 100)
        np.testing.assert_allclose(blackman_pulse(t, rise, gate), 1.0, atol=1e-10)

    def test_rise_and_fall(self):
        rise, gate = 0.2, 1.0
        t = np.linspace(0, gate, 500)
        vals = blackman_pulse(t, rise, gate)
        assert vals[0] == pytest.approx(0.0, abs=0.01)
        assert vals[-1] == pytest.approx(0.0, abs=0.01)
        assert np.max(vals) == pytest.approx(1.0, abs=0.01)

    def test_symmetric_about_midpoint(self):
        rise, gate = 0.2, 1.0
        t = np.linspace(0, gate, 401)
        vals = blackman_pulse(t, rise, gate)
        np.testing.assert_allclose(vals, vals[::-1], atol=1e-12)

    def test_scalar_input_returns_float(self):
        v = blackman_pulse(0.5, 0.1, 1.0)
        assert isinstance(v, float)
        assert v == pytest.approx(1.0, abs=1e-10)

    def test_array_shape_preserved(self):
        t = np.linspace(0, 1.0, 37)
        assert blackman_pulse(t, 0.1, 1.0).shape == t.shape

    def test_raises_on_short_gate(self):
        with pytest.raises(ValueError):
            blackman_pulse(0.5, 1.0, 1.0)  # 2*rise > gate

    def test_raises_on_nonpositive(self):
        with pytest.raises(ValueError):
            blackman_pulse(0.5, 0.0, 1.0)
        with pytest.raises(ValueError):
            blackman_pulse(0.5, 0.1, 0.0)

    def test_raises_on_nonfinite(self):
        with pytest.raises(ValueError):
            blackman_pulse(0.5, 0.1, np.inf)
