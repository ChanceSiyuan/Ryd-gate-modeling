"""Tests for ``Direct297CZProtocol`` and the ``Direct297TOProtocol`` builder.

ARC-backed: the protocols compute their physical target Rabi from the beam
power/area and the Rydberg level of the bound ``rb87_297_clock_4`` system.
"""

import numpy as np
import pytest

from ryd_gate import RydbergSystem, simulate
from ryd_gate.physics import direct_297_rabis
from ryd_gate.protocols import (
    Direct297CZProtocol,
    Direct297TOProtocol,
    phase_from_chirp,
)

_POWER_W = 1e-3     # 1 mW at the atoms -> Omega_r ~ 2pi * 1.4 MHz
_AREA_UM2 = 100.0
_T_GATE = 0.4e-6


def _flat_cz(**kwargs):
    kwargs.setdefault("t_gate", _T_GATE)
    kwargs.setdefault("A_297", lambda s: 1.0)
    kwargs.setdefault("power_at_atoms_w", _POWER_W)
    kwargs.setdefault("beam_area_um2", _AREA_UM2)
    return Direct297CZProtocol(**kwargs)


def _clock4_system(**level_kwargs):
    return RydbergSystem.set_atom_level("rb87_297_clock_4", **level_kwargs)


def test_ryd_level_comes_from_bound_system(monkeypatch):
    seen = []

    def fake_direct_297_rabis(power_w, beam_area_um2, *, ryd_level):
        seen.append((power_w, beam_area_um2, ryd_level))
        return 123.0, 45.0

    monkeypatch.setattr("ryd_gate.physics.direct_297_rabis", fake_direct_297_rabis)
    params = _clock4_system(ryd_level=61).set_protocol(_flat_cz()).unpack_params(())

    assert seen == [(_POWER_W, _AREA_UM2, 61)]
    assert params["ryd_level"] == 61
    assert params["omega_297_max"] == 123.0
    assert params["t_gate"] == _T_GATE
    assert params["theta"] == 0.0


def test_rejects_non_297_system():
    system = RydbergSystem.set_atom_level("1r").set_protocol(_flat_cz())
    with pytest.raises(ValueError, match="rb87_297_clock_4"):
        system.unpack_params(())


def test_drive_coefficients_scale_the_297_ratios():
    system = _clock4_system()
    p = _flat_cz(phi_297=lambda s: 0.5)
    params = p.unpack_params((), system)
    coeffs = p.get_drive_coefficients(0.5 * _T_GATE, params)
    ratios = system.meta("laser_channel_ratios")["297"]
    expected = params["omega_297_max"] * np.exp(-0.5j)
    assert coeffs["E[r,1]"] == pytest.approx(expected * ratios["E[r,1]"])
    assert coeffs["E[r_garb,1]"] == pytest.approx(expected * ratios["E[r_garb,1]"])


def test_zero_state_is_dark_spectator():
    system = _clock4_system(
        enable_rydberg_decay=False, magnetic_field_G=100.0
    ).set_protocol(_flat_cz())
    result = simulate(
        system,
        psi0=["0"],
        backend="exact_dense",
        observables=["sum_n_0", "sum_n_r", "sum_n_r_garb"],
    )
    assert result.expectation("sum_n_0") == pytest.approx(1.0, abs=1e-12)
    assert result.expectation("sum_n_r") == pytest.approx(0.0, abs=1e-12)
    assert result.expectation("sum_n_r_garb") == pytest.approx(0.0, abs=1e-12)


def test_dot_phi_recovers_chirp_in_interior():
    # A phase built by phase_from_chirp must differentiate back to the input
    # chirp away from the pulse edges (finite-difference on the interpolant).
    chirp = lambda t: 2 * np.pi * 5e6 * np.sin(2 * np.pi * t / _T_GATE)
    phi = phase_from_chirp(chirp, _T_GATE)
    p = _flat_cz(phi_297=lambda s: phi(s * _T_GATE))
    params = p.unpack_params((), _clock4_system())
    for s in (0.25, 0.4, 0.6, 0.75):
        traces = p.pulse_traces(s * _T_GATE, params)
        assert traces[r"$\dot\phi_{297}$"] == pytest.approx(
            chirp(s * _T_GATE), rel=2e-2, abs=2e4
        )


def test_cz_rejects_bad_t_gate():
    with pytest.raises(ValueError, match="t_gate"):
        _flat_cz(t_gate=-1.0)


def test_to_builder_shape():
    b = Direct297TOProtocol(_POWER_W, _AREA_UM2)
    assert b.n_params == 6
    assert b.theta_index == 4
    assert b.t_gate_index == 5
    assert len(b.get_optimization_bounds()) == 6
    with pytest.raises(ValueError, match="6 elements"):
        b.validate_params([0.0])


def test_to_builder_t_gate_scales_with_omega_297():
    system = _clock4_system()
    x = [0.5, 1.2, 0.3, -0.1, 0.7, 7.6]
    b = Direct297TOProtocol(_POWER_W, _AREA_UM2)
    pulse = b.build(x, system)
    omega_297 = direct_297_rabis(_POWER_W, _AREA_UM2, ryd_level=53)[0]

    assert isinstance(pulse, Direct297CZProtocol)
    assert pulse.t_gate == pytest.approx(x[5] * 2 * np.pi / omega_297)
    params = b.unpack_params(x, system)
    assert params["t_gate"] == pytest.approx(pulse.t_gate)
    assert params["theta"] == 0.7
