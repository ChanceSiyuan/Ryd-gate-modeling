"""Tests for ``Direct297CZProtocol`` and the ``Direct297TOProtocol`` phase family.

Both take an explicit target |1>-|r> Rabi frequency ``omega_297_max_rad_s`` and
produce a single ``297`` laser drive; the garbage-branch coupling comes from the
preset's private leg ratio. ARC-backed: constructing ``rb87_297_clock_4`` needs
ARC.
"""

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.core.lowering import lower_drives
from ryd_gate.protocols import (
    Direct297CZProtocol,
    Direct297TOProtocol,
    blackman_pulse,
    phase_from_chirp,
)
from ryd_gate.protocols._resolved import _LaserDrive

OMEGA = 2 * np.pi * 1.4e6
T_GATE = 0.4e-6


@pytest.fixture(scope="module")
def sys297():
    return RydbergSystem(
        level_structure=level_structure("rb87_297_clock_4", magnetic_field_G=100.0),
        register=Register.chain(1),
        protocol=Direct297CZProtocol(
            t_gate_s=T_GATE, omega_297_max_rad_s=OMEGA, envelope_297=lambda t: 1.0
        ),
    )


# ── Direct297CZProtocol ──────────────────────────────────────────────────────


def test_cz_construction_validation():
    with pytest.raises(ValueError, match="t_gate_s"):
        Direct297CZProtocol(t_gate_s=-1.0, omega_297_max_rad_s=OMEGA, envelope_297=lambda t: 1.0)
    with pytest.raises(ValueError, match="omega_297_max_rad_s"):
        Direct297CZProtocol(t_gate_s=T_GATE, omega_297_max_rad_s=0.0, envelope_297=lambda t: 1.0)
    with pytest.raises(TypeError, match="envelope_297"):
        Direct297CZProtocol(t_gate_s=T_GATE, omega_297_max_rad_s=OMEGA, envelope_297=1.0)
    with pytest.raises(TypeError, match="phase_297_rad"):
        Direct297CZProtocol(
            t_gate_s=T_GATE, omega_297_max_rad_s=OMEGA,
            envelope_297=lambda t: 1.0, phase_297_rad=2.0,
        )


def test_cz_resolved_t_gate_and_single_laser_drive(sys297):
    env = lambda t: np.sin(np.pi * t / T_GATE) ** 2
    phase = lambda t: 2.0 * t / T_GATE
    resolved = Direct297CZProtocol(
        t_gate_s=T_GATE, omega_297_max_rad_s=OMEGA, envelope_297=env, phase_297_rad=phase
    )._resolve(sys297)

    assert resolved.t_gate == pytest.approx(T_GATE)
    (drive,) = resolved.drives
    assert isinstance(drive, _LaserDrive)
    assert drive.group == "297"
    t = 0.3 * T_GATE
    assert drive.coefficient(t) == pytest.approx(OMEGA * env(t) * np.exp(-1j * phase(t)))


def test_cz_phase_from_chirp_sets_297_phase(sys297):
    # A physical-time phase built from a chirp integral shows up on the compiled
    # E[r,1] channel as exp(-i phi(t)) on top of the leg factor.
    chirp = lambda t: 2 * np.pi * 5e6 * np.sin(2 * np.pi * t / T_GATE)
    phi = phase_from_chirp(chirp, T_GATE)
    proto = Direct297CZProtocol(
        t_gate_s=T_GATE, omega_297_max_rad_s=OMEGA,
        envelope_297=lambda t: 1.0, phase_297_rad=phi,
    )
    _t_gate, channels = lower_drives(sys297.with_protocol(proto))
    by = {c.channel: c for c in channels}
    for s in (0.25, 0.4, 0.6, 0.75):
        t = s * T_GATE
        assert by["E[r,1]"].coefficient(t) == pytest.approx(0.5 * OMEGA * np.exp(-1j * phi(t)))


# ── Direct297TOProtocol ──────────────────────────────────────────────────────


def _to_kwargs(**overrides):
    kwargs = dict(
        omega_297_max_rad_s=OMEGA,
        rise_time_s=0.05e-6,
        phase_amplitude_rad=0.5,
        modulation_frequency_ratio=1.2,
        phase_offset_rad=0.3,
        frequency_offset_ratio=-0.1,
        duration_ratio=7.6,
    )
    kwargs.update(overrides)
    return kwargs


def test_to_t_gate_scales_with_duration_ratio_and_omega(sys297):
    p = Direct297TOProtocol(**_to_kwargs())
    assert p._resolve(sys297).t_gate == pytest.approx(7.6 * 2 * np.pi / OMEGA)


def test_to_construction_validation():
    with pytest.raises(ValueError, match="omega_297_max_rad_s"):
        Direct297TOProtocol(**_to_kwargs(omega_297_max_rad_s=-1.0))
    with pytest.raises(ValueError, match="rise_time_s"):
        Direct297TOProtocol(**_to_kwargs(rise_time_s=0.0))
    with pytest.raises(ValueError, match="duration_ratio"):
        Direct297TOProtocol(**_to_kwargs(duration_ratio=-2.0))
    with pytest.raises(ValueError, match="phase_amplitude_rad"):
        Direct297TOProtocol(**_to_kwargs(phase_amplitude_rad=np.inf))


def test_to_resolve_requires_two_rise_within_gate(sys297):
    # duration_ratio -> t_gate = 0.01*2pi/OMEGA which is far below 2*rise_time.
    p = Direct297TOProtocol(**_to_kwargs(duration_ratio=0.01, rise_time_s=1.0e-6))
    with pytest.raises(ValueError, match="rise_time_s"):
        p._resolve(sys297)


def test_to_resolved_single_laser_drive_matches_blackman_phase_family(sys297):
    p = Direct297TOProtocol(**_to_kwargs())
    resolved = p._resolve(sys297)
    (drive,) = resolved.drives
    assert isinstance(drive, _LaserDrive)
    assert drive.group == "297"

    t_gate = resolved.t_gate
    omega_mod = 1.2 * OMEGA
    delta = -0.1 * OMEGA
    for s in (0.3, 0.5):
        t = s * t_gate
        env = blackman_pulse(t, 0.05e-6, t_gate)
        phase = 0.5 * np.cos(omega_mod * t + 0.3) + delta * t
        assert drive.coefficient(t) == pytest.approx(OMEGA * env * np.exp(-1j * phase))


def test_cz_non_finite_envelope_raises(sys297):
    # _laser_297 guards against a NaN/inf envelope or phase (direct_297.py:42-44).
    proto = Direct297CZProtocol(
        t_gate_s=T_GATE, omega_297_max_rad_s=OMEGA, envelope_297=lambda t: np.inf
    )
    (drive,) = proto._resolve(sys297).drives
    with pytest.raises(ValueError, match="non-finite"):
        drive.coefficient(0.5 * T_GATE)
