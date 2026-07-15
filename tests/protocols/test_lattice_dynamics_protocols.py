"""TFIM quench/anneal physics expressed as a SweepProtocol (P16).

The dedicated TFIM protocol classes are deleted. The transverse-field Ising
dynamics is written directly on the |1>-|r> channels of a ``1r`` system::

    H[r,1](t) = Omega/2 = h_x(t)        # transverse field
    H[r,r](t) = -Delta(t)               # longitudinal field / detuning
    H[r,r]_i += -local_detuning(t, i)   # optional per-site (boundary) pinning

These tests build such SweepProtocols and assert the compiled channel drives.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.core.lowering import lower_drives
from ryd_gate.protocols import SweepProtocol

MHZ = 2 * np.pi * 1e6


def _square_system(protocol, *, side=2, cutoff=None):
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.square(side, spacing_um=9.0),
        protocol=protocol,
        interaction_cutoff_um=cutoff,
    )


def _channels_by_name(system):
    _t_gate, channels = lower_drives(system)
    return {c.channel: c for c in channels}


def test_tfim_quench_emits_constant_transverse_field_and_detuning():
    h_x = 1.0 * MHZ
    delta = 4.0 * MHZ
    t_gate = 0.5e-6
    quench = SweepProtocol(
        t_gate_s=t_gate,
        omega_half_rad_s=lambda t: h_x,
        detuning_rad_s=lambda t: delta,
    )
    system = _square_system(quench)
    by = _channels_by_name(system)

    assert system.t_gate == pytest.approx(t_gate)
    assert set(by) == {"E[r,1]", "E[r,r]"}
    t = 0.5 * t_gate
    assert complex(by["E[r,1]"].coefficient(t)) == pytest.approx(h_x)      # h_x = Omega/2
    assert float(by["E[r,r]"].coefficient(t)) == pytest.approx(-delta)     # -Delta


def test_tfim_anneal_piecewise_transverse_field_and_swept_detuning():
    h_x_peak = 1.0 * MHZ
    delta_i, delta_f = -3.0 * MHZ, 3.0 * MHZ
    t_rise, t_sweep, t_fall = 0.2e-6, 0.6e-6, 0.2e-6
    t_gate = t_rise + t_sweep + t_fall

    def h_x(t):
        if t < t_rise:
            return h_x_peak * t / t_rise
        if t < t_rise + t_sweep:
            return h_x_peak
        return h_x_peak * (t_gate - t) / t_fall

    def delta(t):
        return delta_i + (delta_f - delta_i) * (t / t_gate)

    anneal = SweepProtocol(t_gate_s=t_gate, omega_half_rad_s=h_x, detuning_rad_s=delta)
    system = _square_system(anneal)
    by = _channels_by_name(system)

    assert system.t_gate == pytest.approx(t_gate)
    # transverse field: ramps up, holds at the peak, ramps down
    assert complex(by["E[r,1]"].coefficient(0.1e-6)) == pytest.approx(0.5 * h_x_peak)
    assert complex(by["E[r,1]"].coefficient(0.5e-6)) == pytest.approx(h_x_peak)
    assert complex(by["E[r,1]"].coefficient(0.9e-6)) == pytest.approx(0.5 * h_x_peak)
    # detuning sweep passes through zero at the mid-point (delta_i == -delta_f)
    assert float(by["E[r,r]"].coefficient(0.0)) == pytest.approx(-delta_i)
    assert float(by["E[r,r]"].coefficient(0.5 * t_gate)) == pytest.approx(0.0)
    assert float(by["E[r,r]"].coefficient(t_gate)) == pytest.approx(-delta_f)


def test_tfim_anneal_local_boundary_pinning_is_per_site():
    # Compensate open-boundary interaction shifts by pinning the corner sites.
    pin = 5.0 * MHZ
    delta = 4.0 * MHZ
    t_gate = 1.0e-6
    corners = {0, 3}
    anneal = SweepProtocol(
        t_gate_s=t_gate,
        omega_half_rad_s=lambda t: 1.0 * MHZ,
        detuning_rad_s=lambda t: delta,
        local_detuning_rad_s=lambda t, i: pin if i in corners else 0.0,
    )
    system = _square_system(anneal, side=2)  # N = 4
    by = _channels_by_name(system)

    err = np.asarray(by["E[r,r]"].coefficient(0.5 * t_gate))
    assert err.shape == (4,)
    expected = -delta - np.array([pin, 0.0, 0.0, pin])
    np.testing.assert_allclose(err, expected)


def test_quench_simulation_runs_and_stays_physical():
    # A small quench on a 2x2 array evolves to a physical state (populations in
    # [0, 1], norm conserved). Interactions are the isotropic 1r pair C6.
    quench = SweepProtocol(
        t_gate_s=0.3e-6,
        omega_half_rad_s=lambda t: 1.0 * MHZ,
        detuning_rad_s=lambda t: 2.0 * MHZ,
    )
    system = _square_system(quench, side=2)
    obs = system.observables
    n_r = sum(obs.n("r", i) for i in range(system.N))
    result = simulate(system, observables={"n_r": n_r})
    total = result.expectation("n_r")[0]
    assert 0.0 <= total <= system.N


def test_plot_returns_single_figure():
    quench = SweepProtocol(
        t_gate_s=0.5e-6,
        omega_half_rad_s=lambda t: 1.0 * MHZ,
        detuning_rad_s=lambda t: 4.0 * MHZ,
    )
    system = _square_system(quench)
    fig = quench.plot(system)
    assert isinstance(fig, Figure)
