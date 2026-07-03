"""End-to-end tests for ``Direct297PiProtocol`` on ``rb87_297_clock_4``.

ARC-backed: the protocol computes its physical target Rabi from the beam
power/area via ``direct_297_rabis``.
"""

import numpy as np
import pytest

from ryd_gate import RydbergSystem, simulate
from ryd_gate.protocols import Direct297PiProtocol

_POWER_W = 1e-3     # 1 mW at the atoms -> Omega_r ~ 2pi * 1.4 MHz
_AREA_UM2 = 100.0


def _system(protocol, **level_kwargs):
    return RydbergSystem.set_atom_level("rb87_297_clock_4", **level_kwargs).set_protocol(protocol)


def _params(protocol, **level_kwargs):
    return _system(protocol, **level_kwargs).unpack_params(())


def test_square_pulse_t_gate_is_pi_over_omega():
    p = Direct297PiProtocol(_POWER_W, _AREA_UM2, t_rise_fraction=0.0)
    params = _params(p)
    assert params["t_gate"] == pytest.approx(np.pi / params["omega_297_max"], rel=1e-9)


def test_blackman_t_gate_compensates_envelope_area():
    # Blackman flat-top area fraction is 1 - 2f + 2f*0.42 = 1 - 1.16 f.
    f = 0.15
    p = Direct297PiProtocol(_POWER_W, _AREA_UM2, t_rise_fraction=f)
    params = _params(p)
    assert params["t_gate"] == pytest.approx(
        np.pi / (params["omega_297_max"] * (1.0 - 1.16 * f)), rel=1e-4
    )


def test_explicit_t_gate_wins():
    p = Direct297PiProtocol(_POWER_W, _AREA_UM2, t_gate=1e-6)
    assert _params(p)["t_gate"] == 1e-6


def test_ryd_level_comes_from_bound_system(monkeypatch):
    seen = []

    def fake_direct_297_rabis(power_w, beam_area_um2, *, ryd_level):
        seen.append((power_w, beam_area_um2, ryd_level))
        return 123.0, 45.0

    monkeypatch.setattr("ryd_gate.physics.direct_297_rabis", fake_direct_297_rabis)
    p = Direct297PiProtocol(_POWER_W, _AREA_UM2, t_rise_fraction=0.0)
    params = _params(p, ryd_level=61)

    assert seen == [(_POWER_W, _AREA_UM2, 61)]
    assert params["ryd_level"] == 61
    assert params["omega_297_max"] == 123.0


def test_rejects_non_297_system():
    system = RydbergSystem.set_atom_level("1r").set_protocol(
        Direct297PiProtocol(_POWER_W, _AREA_UM2)
    )
    with pytest.raises(ValueError, match="rb87_297_clock_4"):
        system.unpack_params(())


def test_rejects_bad_construction():
    with pytest.raises(ValueError, match="t_rise_fraction"):
        Direct297PiProtocol(_POWER_W, _AREA_UM2, t_rise_fraction=0.6)
    with pytest.raises(ValueError, match="t_gate"):
        Direct297PiProtocol(_POWER_W, _AREA_UM2, t_gate=-1.0)


def test_near_pi_transfer_with_decay_off_and_large_field():
    # Decay off, 100 G bias: |Delta_garb| ~ 2pi*93 MHz >> Omega_garb ~ 2pi*0.8 MHz,
    # so the calibrated pi pulse moves |1> -> |r> with negligible garbage leakage.
    system = RydbergSystem.set_atom_level(
        "rb87_297_clock_4", enable_rydberg_decay=False, magnetic_field_G=100.0
    ).set_protocol(Direct297PiProtocol(_POWER_W, _AREA_UM2))
    result = simulate(
        system,
        backend="exact_dense",
        observables=["sum_n_r", "sum_n_r_garb", "sum_n_1"],
    )
    assert result.expectation("sum_n_r") > 0.999
    assert result.expectation("sum_n_r_garb") < 1e-6
    assert result.expectation("sum_n_1") < 1e-3


def test_zero_state_is_dark_spectator():
    # No 297 leg touches the logical |0>: starting there, the pi pulse moves
    # nothing (population stays in |0>, no Rydberg population appears).
    system = RydbergSystem.set_atom_level(
        "rb87_297_clock_4", enable_rydberg_decay=False, magnetic_field_G=100.0
    ).set_protocol(Direct297PiProtocol(_POWER_W, _AREA_UM2))
    result = simulate(
        system,
        psi0=["0"],
        backend="exact_dense",
        observables=["sum_n_0", "sum_n_r", "sum_n_r_garb"],
    )
    assert result.expectation("sum_n_0") == pytest.approx(1.0, abs=1e-12)
    assert result.expectation("sum_n_r") == pytest.approx(0.0, abs=1e-12)
    assert result.expectation("sum_n_r_garb") == pytest.approx(0.0, abs=1e-12)


def test_norm_loss_matches_gamma_integral():
    # No-renormalization ODE evolution: d||psi||^2/dt = -Gamma*(n_r + n_r_garb),
    # so the final norm loss must equal Gamma * integral of the Rydberg
    # population (the p_decay integral), up to time-grid quadrature error.
    from ryd_gate.backends.exact import simulate as simulate_exact
    from ryd_gate.backends.exact.dense_ode import DenseODEBackend

    system = RydbergSystem.set_atom_level(
        "rb87_297_clock_4", enable_rydberg_decay=True, magnetic_field_G=50.0
    ).set_protocol(Direct297PiProtocol(_POWER_W, _AREA_UM2))
    t_gate = system.unpack_params(())["t_gate"]
    result = simulate_exact(
        system,
        (),
        system.product_state(["1"]),
        t_eval=np.linspace(0.0, t_gate, 801),
        backend=DenseODEBackend(),
    )

    n_ryd = np.array(
        [
            system.expectation("sum_n_r", psi) + system.expectation("sum_n_r_garb", psi)
            for psi in result.states
        ]
    )
    p_decay_integral = system.meta("ryd_state_decay_rate") * np.trapezoid(n_ryd, result.times)
    norm_loss = 1.0 - float(np.linalg.norm(result.psi_final)) ** 2
    assert norm_loss > 1e-4  # decay actually happened over the pulse
    assert p_decay_integral == pytest.approx(norm_loss, rel=1e-3)
