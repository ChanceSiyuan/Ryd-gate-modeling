"""End-to-end tests for ``Direct297PiProtocol`` on ``rb87_297_clock_4``.

The protocol takes an explicit target |1>-|r> Rabi frequency
``omega_297_max_rad_s`` (the power->Rabi conversion is done in the calling
script via ``ryd_gate.physics.rb87_297_clock_rabi_frequencies``) and auto-times a
pi-area pulse ``t_gate = pi / (omega * F)`` where ``F`` is the envelope area.

ARC-backed: constructing the ``rb87_297_clock_4`` preset needs ARC.
"""

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.core.lowering import lower_drives
from ryd_gate.core.operators import parse_E
from ryd_gate.physics import rb87_297_clock_rabi_frequencies
from ryd_gate.protocols import Direct297PiProtocol, blackman_pulse
from ryd_gate.protocols._resolved import _LaserDrive

OMEGA = 2 * np.pi * 1.4e6  # representative target Rabi (rad/s)


def _system(protocol, **level_kwargs):
    return RydbergSystem(
        level_structure=level_structure("rb87_297_clock_4", **level_kwargs),
        register=Register.chain(1),
        protocol=protocol,
    )


@pytest.fixture(scope="module")
def sys297():
    return _system(Direct297PiProtocol(omega_297_max_rad_s=OMEGA), magnetic_field_G=100.0)


def test_square_pulse_t_gate_is_pi_over_omega(sys297):
    p = Direct297PiProtocol(omega_297_max_rad_s=OMEGA, rise_fraction=0.0)
    assert p._resolve(sys297).t_gate == pytest.approx(np.pi / OMEGA, rel=1e-9)


def test_blackman_t_gate_compensates_envelope_area(sys297):
    # Blackman flat-top area fraction is F = 1 - 1.16 f.
    f = 0.15
    p = Direct297PiProtocol(omega_297_max_rad_s=OMEGA, rise_fraction=f)
    assert p._resolve(sys297).t_gate == pytest.approx(
        np.pi / (OMEGA * (1.0 - 1.16 * f)), rel=1e-3
    )


def test_construction_validation():
    with pytest.raises(ValueError, match="rise_fraction"):
        Direct297PiProtocol(omega_297_max_rad_s=OMEGA, rise_fraction=0.6)
    with pytest.raises(ValueError, match="rise_fraction"):
        Direct297PiProtocol(omega_297_max_rad_s=OMEGA, rise_fraction=-0.1)
    with pytest.raises(ValueError, match="omega_297_max_rad_s"):
        Direct297PiProtocol(omega_297_max_rad_s=-1.0)


def test_resolved_single_297_laser_drive(sys297):
    resolved = Direct297PiProtocol(omega_297_max_rad_s=OMEGA)._resolve(sys297)
    assert len(resolved.drives) == 1
    (drive,) = resolved.drives
    assert isinstance(drive, _LaserDrive)
    assert drive.group == "297"
    # at the flat-top centre the envelope is 1 and the phase is 0
    assert drive.coefficient(0.5 * resolved.t_gate) == pytest.approx(OMEGA + 0.0j)


def test_incompatible_preset_rejected():
    with pytest.raises(ValueError, match="not compatible"):
        RydbergSystem(
            level_structure=level_structure("1r"),
            register=Register.chain(1),
            protocol=Direct297PiProtocol(omega_297_max_rad_s=OMEGA),
        )


def test_lowered_expands_to_target_and_garbage_legs(sys297):
    legs = {leg.channel: leg.factor for leg in sys297.level_structure._laser_legs}
    assert set(legs) == {"E[r,1]", "E[r_garb,1]"}
    assert legs["E[r,1]"] == pytest.approx(0.5)

    _t_gate, channels = lower_drives(sys297)
    by = {c.channel: c for c in channels}
    tc = 0.5 * sys297.t_gate  # flat-top centre: envelope == 1
    env = float(blackman_pulse(0.5, 0.15, 1.0))
    assert env == pytest.approx(1.0)

    assert by["E[r,1]"].coefficient(tc) == pytest.approx(0.5 * OMEGA)
    assert by["E[r_garb,1]"].coefficient(tc) == pytest.approx(legs["E[r_garb,1]"] * OMEGA)
    # the garbage/target ratio is exactly the preset leg ratio
    ratio = by["E[r_garb,1]"].coefficient(tc) / by["E[r,1]"].coefficient(tc)
    assert ratio == pytest.approx(legs["E[r_garb,1]"] / legs["E[r,1]"])


def test_near_pi_transfer_with_large_field():
    # 100 G bias: |Delta_garb| ~ 2pi*93 MHz >> Omega_garb, so the calibrated pi
    # pulse moves |1> -> |r> with negligible garbage leakage.
    omega_r, _garb = rb87_297_clock_rabi_frequencies(1e-3, 100.0)
    system = _system(
        Direct297PiProtocol(omega_297_max_rad_s=omega_r), magnetic_field_G=100.0
    )
    obs = system.observables
    result = simulate(
        system,
        observables={
            "n_r": obs.n("r", 0),
            "n_r_garb": obs.n("r_garb", 0),
            "n_1": obs.n("1", 0),
        },
    )
    assert result.expectation("n_r")[0] > 0.99
    assert result.expectation("n_r_garb")[0] < 1e-3
    assert result.expectation("n_1")[0] < 1e-2


def test_zero_state_is_dark_spectator():
    # |0> is a dark spectator by construction: structurally, no lowered 297 leg
    # names "0" as ket or bra, so the single-atom drive Hamiltonian's |0> row and
    # column are exactly zero at every t (starting in |0>, the RHS never couples
    # out of it and no Rydberg population can ever appear).  This is the "exactly
    # zero coupling" property the old evolve-in-|0> test pinned, without paying a
    # DOP853 solve of ~2700 optical cycles of the clock diagonal to evolve nothing.
    system = _system(
        Direct297PiProtocol(omega_297_max_rad_s=OMEGA), magnetic_field_G=100.0
    )
    levels = system._basis.local_levels
    idx = {lvl: i for i, lvl in enumerate(levels)}
    _t_gate, channels = lower_drives(system)

    # no lowered channel involves |0>
    for ch in channels:
        ket, bra = parse_E(ch.channel)
        assert "0" not in (ket, bra), ch.channel

    # ... hence the |0> row/column of the single-atom drive Hamiltonian is zero
    i0 = idx["0"]
    d = len(levels)
    for t in (0.0, 0.25 * system.t_gate, 0.5 * system.t_gate, system.t_gate):
        H = np.zeros((d, d), dtype=complex)
        for ch in channels:
            ket, bra = parse_E(ch.channel)
            c = complex(ch.coefficient(float(t)))
            H[idx[ket], idx[bra]] += c
            H[idx[bra], idx[ket]] += np.conj(c)
        assert np.all(H[i0, :] == 0.0)
        assert np.all(H[:, i0] == 0.0)
