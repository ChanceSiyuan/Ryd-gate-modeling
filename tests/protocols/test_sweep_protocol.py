"""Tests for the function-defined global SweepProtocol (new resolve seam).

SweepProtocol produces only effective ``_ChannelDrive`` primitives on the
|1>-|r> channels::

    H[r,1](t) = omega_half_rad_s(t)          # already Omega/2 (no extra 1/2)
    H[r,r](t) = -detuning_rad_s(t)  [ - local_detuning_rad_s(t, i) per site ]
"""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.core.lowering import lower_drives
from ryd_gate.protocols import SweepProtocol
from ryd_gate.protocols._resolved import _ChannelDrive

MHZ = 2 * np.pi * 1e6


def _system(protocol, *, preset="1r", n=1, cutoff=None):
    return RydbergSystem(
        level_structure=level_structure(preset),
        register=Register.chain(n),
        protocol=protocol,
        interaction_cutoff_um=cutoff,
    )


def _channels_by_name(system):
    _t_gate, channels = lower_drives(system)
    return {c.channel: c for c in channels}


def test_binds_to_1r_and_01r_and_exposes_t_gate():
    t_gate = 0.5e-6
    proto = SweepProtocol(
        t_gate_s=t_gate,
        omega_half_rad_s=lambda t: 1.0 * MHZ,
        detuning_rad_s=lambda t: 2.0 * MHZ,
    )
    for preset in ("1r", "01r"):
        system = _system(proto, preset=preset)
        assert system.t_gate == pytest.approx(t_gate)


def test_resolved_channel_drives_use_user_functions():
    t_gate = 2.0e-6
    proto = SweepProtocol(
        t_gate_s=t_gate,
        omega_half_rad_s=lambda t: 10.0 * MHZ * (t / t_gate),
        detuning_rad_s=lambda t: -5.0 * MHZ + 20.0 * MHZ * (t / t_gate),
    )
    resolved = proto._resolve(_system(proto))

    assert resolved.t_gate == pytest.approx(t_gate)
    drives = {d.channel: d for d in resolved.drives}
    assert set(drives) == {"E[r,1]", "E[r,r]"}
    assert all(isinstance(d, _ChannelDrive) for d in resolved.drives)

    t = 0.5 * t_gate
    assert complex(drives["E[r,1]"].coefficient(t)) == pytest.approx(5.0 * MHZ)
    assert float(drives["E[r,r]"].coefficient(t)) == pytest.approx(-5.0 * MHZ)


def test_lowered_channels_merge_global_and_local_detuning():
    t_gate = 1.0e-6
    proto = SweepProtocol(
        t_gate_s=t_gate,
        omega_half_rad_s=lambda t: 1.0 * MHZ,
        detuning_rad_s=lambda t: 3.0 * MHZ,
        local_detuning_rad_s=lambda t, i: [-1.0 * MHZ, 1.0 * MHZ][i],
    )
    by = _channels_by_name(_system(proto, n=2))

    t = 0.5 * t_gate
    # global |1>-|r> drive is site-uniform
    assert complex(by["E[r,1]"].coefficient(t)) == pytest.approx(1.0 * MHZ)
    # E[r,r] = -detuning - local_detuning, resolved to a per-site profile
    err = np.asarray(by["E[r,r]"].coefficient(t))
    assert err.shape == (2,)
    np.testing.assert_allclose(err, [-3.0 * MHZ + 1.0 * MHZ, -3.0 * MHZ - 1.0 * MHZ])


def test_construction_validation():
    ok = dict(omega_half_rad_s=lambda t: 0.0, detuning_rad_s=lambda t: 0.0)
    with pytest.raises(ValueError, match="t_gate_s"):
        SweepProtocol(t_gate_s=0.0, **ok)
    with pytest.raises(ValueError, match="t_gate_s"):
        SweepProtocol(t_gate_s=-1.0, **ok)
    with pytest.raises(TypeError, match="omega_half_rad_s"):
        SweepProtocol(t_gate_s=1.0, omega_half_rad_s=1.0, detuning_rad_s=lambda t: 0.0)
    with pytest.raises(TypeError, match="detuning_rad_s"):
        SweepProtocol(t_gate_s=1.0, omega_half_rad_s=lambda t: 0.0, detuning_rad_s=2.0)
    with pytest.raises(TypeError, match="local_detuning_rad_s"):
        SweepProtocol(t_gate_s=1.0, **ok, local_detuning_rad_s=3.0)


def test_rabi_flop_pi_pulse():
    # H[r,1] = omega_half => 2-level Rabi Omega = 2*omega_half; P_r(t) = sin^2(omega_half t).
    # omega_half = pi/(2 t_gate) makes a full |1> -> |r> transfer.
    t_gate = 0.5e-6
    omega_half = np.pi / (2 * t_gate)
    proto = SweepProtocol(
        t_gate_s=t_gate,
        omega_half_rad_s=lambda t: omega_half,
        detuning_rad_s=lambda t: 0.0,
    )
    system = _system(proto, n=1)
    obs = system.observables
    result = simulate(system, observables={"n_r": obs.n("r", 0)})
    assert result.expectation("n_r")[0] == pytest.approx(1.0, abs=1e-3)


def test_local_detuning_suppresses_one_site():
    # Global pi pulse on both sites; a large local detuning on site 1 detunes it
    # off resonance so only site 0 transfers. Interactions disabled (cutoff 0).
    t_gate = 0.5e-6
    omega_half = np.pi / (2 * t_gate)
    big = 2 * np.pi * 20e6
    proto = SweepProtocol(
        t_gate_s=t_gate,
        omega_half_rad_s=lambda t: omega_half,
        detuning_rad_s=lambda t: 0.0,
        local_detuning_rad_s=lambda t, i: 0.0 if i == 0 else big,
    )
    system = _system(proto, n=2, cutoff=0.0)
    obs = system.observables
    result = simulate(
        system, observables={"n_r_0": obs.n("r", 0), "n_r_1": obs.n("r", 1)}
    )
    assert result.expectation("n_r_0")[0] > 0.99
    assert result.expectation("n_r_1")[0] < 0.05


def test_plot_returns_single_figure():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    proto = SweepProtocol(
        t_gate_s=1.0e-6,
        omega_half_rad_s=lambda t: 1.0 * MHZ,
        detuning_rad_s=lambda t: 2.0 * MHZ,
        local_detuning_rad_s=lambda t, i: float(i) * MHZ,
    )
    fig = proto.plot(_system(proto, n=2))
    assert isinstance(fig, Figure)


def test_plot_rejects_too_few_points():
    # Protocol.plot guards n_points < 2 (base.py:39-40), before any resolve/draw.
    proto = SweepProtocol(
        t_gate_s=1e-6, omega_half_rad_s=lambda t: 0.0, detuning_rad_s=lambda t: 0.0
    )
    with pytest.raises(ValueError, match="n_points >= 2"):
        proto.plot(None, n_points=1)


def test_plot_rejects_nonpositive_t_gate():
    # Protocol.plot guards a non-positive resolved t_gate (base.py:43-44). The
    # concrete protocols validate t_gate at construction, so a minimal subclass
    # that resolves to t_gate = 0 exercises the branch.
    from ryd_gate.protocols._resolved import _ResolvedProtocol
    from ryd_gate.protocols.base import Protocol

    class _ZeroGate(Protocol):
        def _resolve(self, system):
            return _ResolvedProtocol(t_gate=0.0, drives=())

    with pytest.raises(ValueError, match="positive t_gate"):
        _ZeroGate().plot(None)
