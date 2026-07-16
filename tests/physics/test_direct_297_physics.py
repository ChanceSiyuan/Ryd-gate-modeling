"""Tests for the 297 nm single-photon physics helpers (ARC-backed).

Covers ``rb87_297_clock_rabi_frequencies`` (branch Rabis with the clock-state
1/sqrt(2)) and ``arc_pair_c6_rad_s_um6`` (perturbative pair C6 in the repo
``V = +C6/R^6`` sign convention, with degenerate-manifold eigenchannel
selection).
"""

import numpy as np
import pytest

from ryd_gate.physics import (
    arc_pair_c6_rad_s_um6,
    rb87_297_clock_rabi_frequencies,
    single_photon_rabi,
)

_POWER_W = 1e-3
_AREA_UM2 = 100.0

# ARC 70S1/2 on-axis C6 (rad/s·μm⁶); ~2π·862.7 GHz (close to the old 2π·874 GHz).
_ARC_70S_C6 = 2 * np.pi * 862.7e9


def test_297_rabis_delegate_to_single_photon_rabi():
    omega_r, omega_garb = rb87_297_clock_rabi_frequencies(_POWER_W, _AREA_UM2, ryd_level=53)
    assert omega_r == pytest.approx(
        single_photon_rabi(
            _POWER_W, _AREA_UM2,
            n1=5, l1=0, j1=0.5, mj1=-0.5, n2=53, l2=1, j2=1.5, q=-1,
        )
        / np.sqrt(2)
    )
    assert omega_garb == pytest.approx(
        single_photon_rabi(
            _POWER_W, _AREA_UM2,
            n1=5, l1=0, j1=0.5, mj1=0.5, n2=53, l2=1, j2=1.5, q=-1,
        )
        / np.sqrt(2)
    )


def test_297_branch_ratio_is_cg_ratio():
    # Same radial integral for both σ⁻ branches, so the ratio is the pure CG
    # ratio (1/sqrt(3)) of the mJ=+1/2->-1/2 vs mJ=-1/2->-3/2 legs.
    omega_r, omega_garb = rb87_297_clock_rabi_frequencies(_POWER_W, _AREA_UM2)
    assert 0.0 < omega_garb < omega_r
    assert omega_garb / omega_r == pytest.approx(1.0 / np.sqrt(3), rel=1e-6)


def test_arc_c6_70s_sign_and_magnitude():
    # 70S1/2 pair on axis: repo convention V = +C6/R^6 gives a positive
    # (repulsive) C6 (ARC returns a negative GHz·μm⁶ value in its V = -C6/R^6).
    c6 = arc_pair_c6_rad_s_um6(
        n1=70, l1=0, j1=0.5, mj1=-0.5, theta=0.0, phi=0.0, degenerate=False
    )
    assert np.isfinite(c6) and c6 > 0.0
    assert c6 == pytest.approx(_ARC_70S_C6, rel=0.02)


def test_arc_c6_53p_degenerate_finite_and_warns_on_weak_overlap():
    with pytest.warns(UserWarning, match="not a dominant eigenchannel"):
        c6 = arc_pair_c6_rad_s_um6(
            n1=53, l1=1, j1=1.5, mj1=-1.5, theta=np.pi / 2, phi=0.123
        )
    assert np.isfinite(c6) and c6 != 0.0


def test_arc_c6_cache_keys_on_rounded_angles(monkeypatch):
    # The 1e-9 angle rounding (physics.py:288-289) is pure Python; stub the cached
    # ARC solve so we test only that a 1e-12 theta jitter collapses to one cache key
    # (no third full degenerate 53P C6 solve).
    import ryd_gate.physics as physics_mod

    calls = []

    def fake_cached(*args):
        calls.append(args)
        return (1.23, 1.0)

    monkeypatch.setattr(physics_mod, "_arc_pair_c6_cached", fake_cached)

    kw = dict(n1=53, l1=1, j1=1.5, mj1=-1.5, theta=np.pi / 2, phi=0.321)
    a = arc_pair_c6_rad_s_um6(**kw)
    b = arc_pair_c6_rad_s_um6(**{**kw, "theta": np.pi / 2 + 1e-12})
    assert a == b == 1.23
    assert calls[0] == calls[1]  # identical rounded args reach the cached solver


@pytest.mark.filterwarnings("ignore:arc_pair_c6_rad_s_um6:UserWarning")
def test_arc_c6_explicit_atom2_quantum_numbers():
    # Explicit atom-2 quantum numbers (mj2 != mj1) exercise the non-default
    # (heteronuclear-style) path of arc_pair_c6_rad_s_um6 (physics.py:283-284).
    # (The mixed mj pair sits at the 0.5 overlap boundary, so the eigenchannel
    # warning is expected and suppressed.)
    c6 = arc_pair_c6_rad_s_um6(
        n1=70, l1=0, j1=0.5, mj1=-0.5,
        n2=70, l2=0, j2=0.5, mj2=0.5,
        theta=0.0, phi=0.0,
    )
    assert np.isfinite(c6)
