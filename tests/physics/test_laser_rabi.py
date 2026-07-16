"""Tests for the laser-power -> single-photon Rabi helpers in ``ryd_gate.physics``.

Covers the ARC-backed single-photon Rabi and the ``rb87_7_mp`` 70S 420/1013
convenience wrapper (PH03/PH04). The top-hat E-field conversion is a private
helper; it is exercised through ``single_photon_rabi``.
"""

import numpy as np
import pytest
from scipy.constants import hbar, physical_constants

from ryd_gate.physics import (
    _check_transition_level,
    _electric_field_uniform_beam,
    _get_atom,
    _mid_branching_ratios,
    _rydberg_branching_ratios,
    rb87_7_mp_rabi_frequencies,
    single_photon_rabi,
)

_BOHR = physical_constants["Bohr radius"][0]
_ECHARGE = physical_constants["elementary charge"][0]


def test_single_photon_rabi_matches_manual_dipole_formula():
    """Ω = |d| E0 / ħ with d from the ARC dipole matrix element (a0·e units)."""
    power_w = 6.41
    beam_area_um2 = 100.0**2
    e0 = _electric_field_uniform_beam(power_w, beam_area_um2)
    # 420 nm: 5S_1/2(mj=-1/2) --σ⁻--> 6P_3/2(mj=-3/2)
    dipole_au = _get_atom().getDipoleMatrixElement(5, 0, 0.5, -0.5, 6, 1, 1.5, -1.5, -1)
    expected = abs(dipole_au) * _ECHARGE * _BOHR * e0 / hbar

    got = single_photon_rabi(
        power_w, beam_area_um2,
        n1=5, l1=0, j1=0.5, mj1=-0.5, n2=6, l2=1, j2=1.5, q=-1,
    )
    assert got == pytest.approx(expected, rel=1e-9)


def test_single_photon_rabi_sqrt_power_and_area_scaling():
    area_um2 = 100.0**2
    kw = dict(n1=5, l1=0, j1=0.5, mj1=-0.5, n2=6, l2=1, j2=1.5, q=-1)
    base = single_photon_rabi(1.0, area_um2, **kw)
    assert single_photon_rabi(4.0, area_um2, **kw) == pytest.approx(2.0 * base, rel=1e-9)
    assert single_photon_rabi(1.0, 4.0 * area_um2, **kw) == pytest.approx(base / 2.0, rel=1e-9)


def test_single_photon_rabi_validates_inputs():
    kw = dict(n1=5, l1=0, j1=0.5, mj1=-0.5, n2=6, l2=1, j2=1.5, q=-1)
    with pytest.raises(ValueError):
        single_photon_rabi(-1.0, 100.0, **kw)          # negative power
    with pytest.raises(ValueError):
        single_photon_rabi(1.0, 0.0, **kw)             # non-positive area
    with pytest.raises(ValueError):
        single_photon_rabi(1.0, 100.0, **{**kw, "q": 2})   # bad polarization
    with pytest.raises(ValueError):
        single_photon_rabi(1.0, 100.0, **{**kw, "mj1": 2.5})  # |mj| > j


def test_rb87_7_mp_rabi_known_values():
    """6.41 W / 100 W over a 100 µm × 100 µm top-hat -> ~1701 / ~317 MHz (ARC, 70S)."""
    beam_area_um2 = 100.0**2
    omega_420, omega_1013 = rb87_7_mp_rabi_frequencies(6.41, 100.0, beam_area_um2, ryd_level=70)
    assert omega_420 / (2 * np.pi) / 1e6 == pytest.approx(1700.9, rel=2e-2)
    assert omega_1013 / (2 * np.pi) / 1e6 == pytest.approx(317.3, rel=2e-2)


def test_rb87_7_mp_rabi_delegates_to_single_photon_rabi():
    beam_area_um2 = 100.0**2
    omega_420, omega_1013 = rb87_7_mp_rabi_frequencies(6.41, 100.0, beam_area_um2, ryd_level=70)
    assert omega_420 == pytest.approx(
        single_photon_rabi(
            6.41, beam_area_um2,
            n1=5, l1=0, j1=0.5, mj1=-0.5, n2=6, l2=1, j2=1.5, q=-1,
        )
        / np.sqrt(2)  # mF=0 splitting into mJ=±1/2
    )
    assert omega_1013 == pytest.approx(
        single_photon_rabi(
            100.0, beam_area_um2,
            n1=6, l1=1, j1=1.5, mj1=-1.5, n2=70, l2=0, j2=0.5, q=1,
        )
    )


def test_check_transition_level_rejects_bad_quantum_numbers():
    # The remaining validation branches of _check_transition_level (physics.py:72-79),
    # reached before any ARC call.
    with pytest.raises(ValueError, match="n must be"):
        _check_transition_level("s", 0, 0, 0.5, 0.5)       # n < 1
    with pytest.raises(ValueError, match="l must be"):
        _check_transition_level("s", 5, 5, 0.5, 0.5)       # l not in [0, n)
    with pytest.raises(ValueError, match="j must be"):
        _check_transition_level("s", 5, 0, 1.5, 0.5)       # j != l ± 1/2
    with pytest.raises(ValueError, match="mj"):
        _check_transition_level("s", 5, 0, 0.5, 0.0)       # mj off the j ladder


def test_rydberg_branching_ratios_partition_unity():
    # _rydberg_branching_ratios (physics.py:299-374) normalizes the Rydberg->6P->5S
    # cascade; the four reported channels partition the branch ratio, so they lie
    # in [0, 1] and sum to 1 for both σ⁻/σ⁺ manifolds.
    atom = _get_atom()
    for manifold in ("mp", "pm"):
        br = _rydberg_branching_ratios(atom, 70, manifold)
        vals = [br["to_0"], br["to_1"], br["to_L0"], br["to_L1"]]
        assert all(0.0 <= v <= 1.0 for v in vals)
        assert sum(vals) == pytest.approx(1.0)


def test_mid_branching_ratios_partition_unity():
    # _mid_branching_ratios (physics.py:377-407) normalizes the 6P_3/2 -> 5S decay;
    # its four channels also partition unity.
    atom = _get_atom()
    br = _mid_branching_ratios(atom, F=3, mF=0)
    vals = [br["to_0"], br["to_1"], br["to_L0"], br["to_L1"]]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert sum(vals) == pytest.approx(1.0)
