"""Tests for ``Direct297CZProtocol`` and the ``Direct297TOProtocol`` phase family.

ARC-backed: the protocols compute their physical target Rabi from the beam
power/area and the Rydberg level of the bound ``rb87_297_clock_4`` system.
"""

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.physics import direct_297_rabis
from ryd_gate.protocols import (
    Direct297CZProtocol,
    Direct297TOProtocol,
)
from ryd_gate.protocols.gate_cz import phase_from_chirp

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
    return RydbergSystem(
        level_structure=level_structure("rb87_297_clock_4", **level_kwargs),
        register=Register.chain(1),
    )


def test_ryd_level_comes_from_bound_system(monkeypatch):
    seen = []

    def fake_direct_297_rabis(power_w, beam_area_um2, *, ryd_level):
        seen.append((power_w, beam_area_um2, ryd_level))
        return 123.0, 45.0

    monkeypatch.setattr("ryd_gate.physics.direct_297_rabis", fake_direct_297_rabis)
    p = _flat_cz()
    ctx = p._resolve(_clock4_system(ryd_level=61))

    assert seen == [(_POWER_W, _AREA_UM2, 61)]
    assert ctx["ryd_level"] == 61
    assert ctx["omega_297_max"] == 123.0
    assert ctx["t_gate"] == _T_GATE


def test_rejects_non_297_system():
    system = RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(1),
        protocol=_flat_cz(),
    )
    with pytest.raises(ValueError, match="rb87_297_clock_4"):
        system.t_gate  # noqa: B018 — resolving against the wrong model must raise


def test_drive_coefficients_scale_the_297_ratios():
    system = _clock4_system()
    p = _flat_cz(phi_297=lambda s: 0.5)
    ctx = p._resolve(system)
    coeffs = p.get_drive_coefficients(0.5 * _T_GATE, ctx)
    ratios = system.level_structure.laser_channel_ratios["297"]
    expected = ctx["omega_297_max"] * np.exp(-0.5j)
    assert coeffs["E[r,1]"] == pytest.approx(expected * ratios["E[r,1]"])
    assert coeffs["E[r_garb,1]"] == pytest.approx(expected * ratios["E[r_garb,1]"])


def test_zero_state_is_dark_spectator():
    # Rydberg populations stay exactly zero (the RHS never couples out of |0>);
    # the |0> population carries only the adaptive integrator's amplitude drift
    # from tracking the ~6.8 GHz clock phase (~1e-5 at default tolerances).
    system = _clock4_system(
        enable_rydberg_decay=False, magnetic_field_G=100.0
    ).with_protocol(_flat_cz())
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


def test_dot_phi_recovers_chirp_in_interior():
    # A phase built by phase_from_chirp must differentiate back to the input
    # chirp away from the pulse edges (finite-difference on the interpolant).
    chirp = lambda t: 2 * np.pi * 5e6 * np.sin(2 * np.pi * t / _T_GATE)
    phi = phase_from_chirp(chirp, _T_GATE)
    p = _flat_cz(phi_297=lambda s: phi(s * _T_GATE))
    system = _clock4_system()
    for s in (0.25, 0.4, 0.6, 0.75):
        traces = p.pulse_traces(s * _T_GATE, system)
        assert traces[r"$\dot\phi_{297}$"] == pytest.approx(
            chirp(s * _T_GATE), rel=2e-2, abs=2e4
        )


def test_cz_rejects_bad_t_gate():
    with pytest.raises(ValueError, match="t_gate"):
        _flat_cz(t_gate=-1.0)


def test_to_protocol_t_gate_scales_with_omega_297():
    # Old x = [A, w, phi0, d, theta, T]; theta (x[4]) was a scoring parameter,
    # not part of the pulse, so it has no protocol field.
    system = _clock4_system()
    x = [0.5, 1.2, 0.3, -0.1, 0.7, 7.6]
    p = Direct297TOProtocol(
        _POWER_W, _AREA_UM2,
        phase_amplitude=x[0], frequency_ratio=x[1], phase_offset=x[2],
        detuning_ratio=x[3], duration_ratio=x[5],
    )
    omega_297 = direct_297_rabis(_POWER_W, _AREA_UM2, ryd_level=53)[0]

    assert p.resolve_t_gate(system) == pytest.approx(x[5] * 2 * np.pi / omega_297)
