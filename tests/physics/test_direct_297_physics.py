"""Tests for the 297 nm single-photon physics helpers (ARC-backed).

Covers ``direct_297_rabis`` (branch Rabis with the clock-state 1/sqrt(2)) and
``arc_pair_c6_rad_s_um6`` (perturbative pair C6 in the repo ``V = +C6/R^6``
sign convention, with degenerate-manifold eigenchannel selection).
"""

import numpy as np
import pytest

from ryd_gate.core.level_structures import DEFAULT_C6
from ryd_gate.physics import (
    arc_pair_c6_rad_s_um6,
    direct_297_rabis,
    single_photon_rabi,
)

_POWER_W = 1e-3
_AREA_UM2 = 100.0


def test_direct_297_rabis_delegates_to_single_photon_rabi():
    omega_r, omega_garb = direct_297_rabis(_POWER_W, _AREA_UM2, ryd_level=53)
    assert omega_r == pytest.approx(
        single_photon_rabi(
            _POWER_W, _AREA_UM2,
            n1=5, l1=0, j1=0.5, mj1=-0.5, n2=53, l2=1, j2=1.5, q=-1,
        )
        / np.sqrt(2)  # mF=0 clock-state splitting into mJ=-1/2 and mJ=+1/2
    )
    assert omega_garb == pytest.approx(
        single_photon_rabi(
            _POWER_W, _AREA_UM2,
            n1=5, l1=0, j1=0.5, mj1=0.5, n2=53, l2=1, j2=1.5, q=-1,
        )
        / np.sqrt(2)
    )


def test_direct_297_branch_ratio_is_cg_ratio():
    # Same radial integral for both sigma- branches, so the ratio is the pure
    # CG ratio (1/sqrt(3)) / 1 of the mJ=+1/2->-1/2 vs mJ=-1/2->-3/2 legs.
    omega_r, omega_garb = direct_297_rabis(_POWER_W, _AREA_UM2)
    assert 0.0 < omega_garb < omega_r
    assert omega_garb / omega_r == pytest.approx(1.0 / np.sqrt(3), rel=1e-6)


def test_arc_c6_70s_sign_and_magnitude():
    # 70S1/2 pair on axis: the repo convention V = +C6/R^6 must give a positive
    # (repulsive) C6 close to the repo default 2pi*874e9 rad/s um^6 (ARC returns
    # a negative GHz um^6 value in its V = -C6/R^6 convention).
    c6 = arc_pair_c6_rad_s_um6(
        n1=70, l1=0, j1=0.5, mj1=-0.5, theta=0.0, phi=0.0, degenerate=False
    )
    assert np.isfinite(c6) and c6 > 0.0
    assert c6 == pytest.approx(DEFAULT_C6, rel=0.05)


def test_arc_c6_53p_degenerate_finite_and_warns_on_weak_overlap():
    # The 53P3/2 (mj=-3/2, mj=-3/2) channel at theta=pi/2 is not a dominant
    # eigenchannel (overlap ~0.46), so the helper warns but still returns the
    # max-overlap eigenvalue. phi=0.123 keeps the cache key unique so the
    # (once-per-key) warning fires in this test.
    with pytest.warns(UserWarning, match="not a dominant eigenchannel"):
        c6 = arc_pair_c6_rad_s_um6(
            n1=53, l1=1, j1=1.5, mj1=-1.5, theta=np.pi / 2, phi=0.123
        )
    assert np.isfinite(c6) and c6 != 0.0


@pytest.mark.filterwarnings("ignore:arc_pair_c6_rad_s_um6:UserWarning")
def test_arc_c6_cache_keys_on_rounded_angles():
    kw = dict(n1=53, l1=1, j1=1.5, mj1=-1.5, theta=np.pi / 2, phi=0.321)
    a = arc_pair_c6_rad_s_um6(**kw)
    # A sub-rounding perturbation of theta maps to the same cache key/value.
    b = arc_pair_c6_rad_s_um6(**{**kw, "theta": np.pi / 2 + 1e-12})
    assert a == b
