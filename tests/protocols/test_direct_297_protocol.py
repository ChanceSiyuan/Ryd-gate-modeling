"""End-to-end tests for ``Direct297PiProtocol`` on ``rb87_297_clock_4``.

ARC-backed: the protocol computes its physical target Rabi from the beam
power/area via ``direct_297_rabis``.
"""

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.protocols import Direct297PiProtocol

_POWER_W = 1e-3     # 1 mW at the atoms -> Omega_r ~ 2pi * 1.4 MHz
_AREA_UM2 = 100.0


def _system(protocol, **level_kwargs):
    return RydbergSystem(
        level_structure=level_structure("rb87_297_clock_4", **level_kwargs),
        register=Register.chain(1),
        protocol=protocol,
    )


def _ctx(protocol, **level_kwargs):
    return protocol._resolve(_system(protocol, **level_kwargs))


def test_square_pulse_t_gate_is_pi_over_omega():
    p = Direct297PiProtocol(_POWER_W, _AREA_UM2, t_rise_fraction=0.0)
    ctx = _ctx(p)
    assert ctx["t_gate"] == pytest.approx(np.pi / ctx["omega_297_max"], rel=1e-9)


def test_blackman_t_gate_compensates_envelope_area():
    # Blackman flat-top area fraction is 1 - 2f + 2f*0.42 = 1 - 1.16 f.
    f = 0.15
    p = Direct297PiProtocol(_POWER_W, _AREA_UM2, t_rise_fraction=f)
    ctx = _ctx(p)
    assert ctx["t_gate"] == pytest.approx(
        np.pi / (ctx["omega_297_max"] * (1.0 - 1.16 * f)), rel=1e-4
    )


def test_explicit_t_gate_wins():
    p = Direct297PiProtocol(_POWER_W, _AREA_UM2, t_gate=1e-6)
    assert _ctx(p)["t_gate"] == 1e-6


def test_ryd_level_comes_from_bound_system(monkeypatch):
    seen = []

    def fake_direct_297_rabis(power_w, beam_area_um2, *, ryd_level):
        seen.append((power_w, beam_area_um2, ryd_level))
        return 123.0, 45.0

    monkeypatch.setattr("ryd_gate.physics.direct_297_rabis", fake_direct_297_rabis)
    p = Direct297PiProtocol(_POWER_W, _AREA_UM2, t_rise_fraction=0.0)
    ctx = _ctx(p, ryd_level=61)

    assert seen == [(_POWER_W, _AREA_UM2, 61)]
    assert ctx["ryd_level"] == 61
    assert ctx["omega_297_max"] == 123.0


def test_rejects_non_297_system():
    system = RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(1),
        protocol=Direct297PiProtocol(_POWER_W, _AREA_UM2),
    )
    with pytest.raises(ValueError, match="rb87_297_clock_4"):
        system.t_gate  # noqa: B018 — resolving against the wrong model must raise


def test_rejects_bad_construction():
    with pytest.raises(ValueError, match="t_rise_fraction"):
        Direct297PiProtocol(_POWER_W, _AREA_UM2, t_rise_fraction=0.6)
    with pytest.raises(ValueError, match="t_gate"):
        Direct297PiProtocol(_POWER_W, _AREA_UM2, t_gate=-1.0)


def test_near_pi_transfer_with_decay_off_and_large_field():
    # Decay off, 100 G bias: |Delta_garb| ~ 2pi*93 MHz >> Omega_garb ~ 2pi*0.8 MHz,
    # so the calibrated pi pulse moves |1> -> |r> with negligible garbage leakage.
    system = _system(
        Direct297PiProtocol(_POWER_W, _AREA_UM2),
        enable_rydberg_decay=False, magnetic_field_G=100.0,
    )
    obs = system.observables
    result = simulate(
        system,
        observables={
            "n_r": obs.level_sum("r"),
            "n_r_garb": obs.level_sum("r_garb"),
            "n_1": obs.level_sum("1"),
        },
    )
    assert result.expectation("n_r")[0].real > 0.999
    assert result.expectation("n_r_garb")[0].real < 1e-6
    assert result.expectation("n_1")[0].real < 1e-3


def test_zero_state_is_dark_spectator():
    # No 297 leg touches the logical |0>: starting there, the pi pulse moves
    # nothing — no Rydberg population ever appears (exactly zero: the RHS never
    # couples out of |0>).  The |0> population itself carries only the adaptive
    # integrator's amplitude drift from tracking the ~6.8 GHz clock phase
    # (~1e-5 at the default tolerances), not physical leakage.
    system = _system(
        Direct297PiProtocol(_POWER_W, _AREA_UM2),
        enable_rydberg_decay=False, magnetic_field_G=100.0,
    )
    obs = system.observables
    result = simulate(
        system,
        ["0"],
        observables={
            "n_0": obs.level_sum("0"),
            "n_r": obs.level_sum("r"),
            "n_r_garb": obs.level_sum("r_garb"),
        },
    )
    assert result.expectation("n_0")[0].real == pytest.approx(1.0, abs=1e-4)
    assert result.expectation("n_r")[0].real == pytest.approx(0.0, abs=1e-12)
    assert result.expectation("n_r_garb")[0].real == pytest.approx(0.0, abs=1e-12)


def test_norm_loss_matches_gamma_integral():
    # No-renormalization ODE evolution: d||psi||^2/dt = -Gamma*(n_r + n_r_garb),
    # so the final norm loss must equal Gamma * integral of the Rydberg
    # population (the p_decay integral), up to time-grid quadrature error.
    # Raw (non-Hermitian-safe) expectations: identity = survival norm, and the
    # trapezoid convention on the t_eval grid is the pinned decay bookkeeping.
    from ryd_gate.backends.exact import simulate as simulate_exact

    system = _system(
        Direct297PiProtocol(_POWER_W, _AREA_UM2),
        enable_rydberg_decay=True, magnetic_field_G=50.0,
    )
    obs = system.observables
    t_gate = system.t_gate
    result = simulate_exact(
        system,
        system.product_state(["1"]),
        t_eval=np.linspace(0.0, t_gate, 801),
        observables={
            "n_ryd": obs.level_sum("r") + obs.level_sum("r_garb"),
            "one": obs.identity(),
        },
    )

    n_ryd = np.real(result.expectation("n_ryd"))
    p_decay_integral = system.level_structure.ryd_state_decay_rate * np.trapezoid(
        n_ryd, result.times
    )
    norm_loss = 1.0 - float(np.linalg.norm(result.final_state)) ** 2
    # the raw identity expectation at the last requested time is the survival norm
    survival = result.expectation("one")[-1].real
    assert survival == pytest.approx(float(np.linalg.norm(result.final_state)) ** 2, rel=1e-9)
    assert norm_loss > 1e-4  # decay actually happened over the pulse
    assert p_decay_integral == pytest.approx(norm_loss, rel=1e-3)
