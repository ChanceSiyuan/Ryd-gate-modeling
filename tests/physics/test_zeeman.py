"""Tests for the magnetic-field / Zeeman helpers in ``ryd_gate.physics``.

Pure numpy + scipy.constants (no ARC), so these are fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate.physics import lande_gj, rydberg_zeeman_shift_rad_s


def test_lande_gj_ns12_is_two():
    # nS_{1/2}: l=0, j=1/2, s=1/2 -> g_J = 2 exactly.
    assert lande_gj(0, 0.5, 0.5) == pytest.approx(2.0)


def test_lande_gj_p32_is_four_thirds():
    # sanity: 6P_{3/2} (l=1, j=3/2) -> g_J = 4/3.
    assert lande_gj(1, 1.5, 0.5) == pytest.approx(4.0 / 3.0)


@pytest.mark.parametrize("manifold", ["mp", "pm"])
def test_zeeman_shift_56mhz_at_20G(manifold):
    # 20 G -> 2pi * ~56 MHz for both manifolds (g_J=2, Delta_mj=1 for nS_1/2).
    shift = rydberg_zeeman_shift_rad_s(20.0, manifold=manifold)
    assert shift / (2 * np.pi * 1e6) == pytest.approx(56.0, rel=2e-3)


def test_zeeman_shift_manifolds_agree():
    # Both mp and pm are nS_1/2 opposite-m_j states -> identical splitting.
    assert rydberg_zeeman_shift_rad_s(13.0, manifold="mp") == pytest.approx(
        rydberg_zeeman_shift_rad_s(13.0, manifold="pm")
    )


def test_zeeman_shift_is_linear_in_field():
    base = rydberg_zeeman_shift_rad_s(10.0, manifold="mp")
    assert rydberg_zeeman_shift_rad_s(30.0, manifold="mp") == pytest.approx(3.0 * base)
    assert rydberg_zeeman_shift_rad_s(0.0, manifold="mp") == pytest.approx(0.0)


def test_zeeman_shift_positive_for_positive_field():
    # Positive B -> positive shift (r_garb above r), matching the h[6,6] convention.
    assert rydberg_zeeman_shift_rad_s(20.0, manifold="mp") > 0.0


def test_zeeman_shift_rejects_unknown_manifold():
    with pytest.raises(ValueError, match="Unknown rb87 manifold"):
        rydberg_zeeman_shift_rad_s(20.0, manifold="xy")
