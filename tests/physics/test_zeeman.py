"""Tests for the magnetic-field / Zeeman helpers in ``ryd_gate.physics``.

Pure numpy + scipy.constants (no ARC), so these are fast. ``zeeman_shift_rad_s``
is the public helper (PH03); ``_lande_gj`` / ``_rydberg_zeeman_shift_rad_s`` are
private implementation exercised here for the manifold logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate.physics import _lande_gj, _rydberg_zeeman_shift_rad_s, zeeman_shift_rad_s


def test_lande_gj_ns12_is_two():
    assert _lande_gj(0, 0.5, 0.5) == pytest.approx(2.0)


def test_lande_gj_p32_is_four_thirds():
    assert _lande_gj(1, 1.5, 0.5) == pytest.approx(4.0 / 3.0)


def test_general_zeeman_ns12_56mhz_at_20G():
    # nS_1/2 (g_J=2, Δmj=1): 20 G -> 2pi*~56 MHz.
    shift = zeeman_shift_rad_s(20.0, l=0, j=0.5, delta_mj=1.0)
    assert shift / (2 * np.pi * 1e6) == pytest.approx(56.0, rel=2e-3)


def test_general_zeeman_scales_with_delta_mj():
    base = zeeman_shift_rad_s(10.0, l=1, j=1.5, delta_mj=1.0)
    assert zeeman_shift_rad_s(10.0, l=1, j=1.5, delta_mj=-2.0) == pytest.approx(-2.0 * base)


def test_general_zeeman_linear_in_field():
    base = zeeman_shift_rad_s(10.0, l=0, j=0.5, delta_mj=1.0)
    assert zeeman_shift_rad_s(30.0, l=0, j=0.5, delta_mj=1.0) == pytest.approx(3.0 * base)
    assert zeeman_shift_rad_s(0.0, l=0, j=0.5, delta_mj=1.0) == pytest.approx(0.0)


def test_general_zeeman_rejects_nonfinite_field():
    with pytest.raises(ValueError):
        zeeman_shift_rad_s(np.inf, l=0, j=0.5, delta_mj=1.0)


def test_rydberg_zeeman_manifolds_agree_and_positive():
    a = _rydberg_zeeman_shift_rad_s(13.0, manifold="mp")
    b = _rydberg_zeeman_shift_rad_s(13.0, manifold="pm")
    assert a == pytest.approx(b)
    assert _rydberg_zeeman_shift_rad_s(20.0, manifold="mp") > 0.0


def test_rydberg_zeeman_matches_general_helper():
    assert _rydberg_zeeman_shift_rad_s(20.0, manifold="mp") == pytest.approx(
        zeeman_shift_rad_s(20.0, l=0, j=0.5, delta_mj=1.0)
    )


def test_rydberg_zeeman_rejects_unknown_manifold():
    with pytest.raises(ValueError, match="Unknown rb87 manifold"):
        _rydberg_zeeman_shift_rad_s(20.0, manifold="xy")
