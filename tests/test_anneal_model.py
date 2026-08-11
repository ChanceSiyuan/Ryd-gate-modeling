"""Tests for the scripts-local shared TFIM anneal construction."""

from types import SimpleNamespace

import numpy as np
from scripts import anneal_model


def test_peps_options_preserve_the_calibrated_tier():
    options = anneal_model.peps_options(
        dt_w0=0.1, w0=2.0, bond_dimension=6,
        measurement_method="belief_propagation")
    assert options == {
        "time_step_s": 0.05,
        "bond_dimension": 6,
        "svd_tolerance": 1e-8,
        "ntu_max_iterations": 20,
        "ntu_iteration_tolerance": 1e-10,
        "measurement_method": "belief_propagation",
        "environment_bond_dimension": 32,
        "environment_tolerance": 1e-8,
        "environment_max_iterations": 50,
        "device": "cpu",
    }


def test_build_anneal_system_maps_schedule_and_boundary_pins(monkeypatch):
    coords = anneal_model.A_UM * np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    geometry = SimpleNamespace(coords=coords, N=4)
    monkeypatch.setattr(
        anneal_model.Register, "rectangle",
        lambda lx, ly, spacing_um: geometry)
    monkeypatch.setattr(anneal_model, "level_structure", lambda *args, **kwargs: "1r")

    captured = {}

    def fake_system(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(anneal_model, "RydbergSystem", fake_system)
    c6 = 24.0 * anneal_model.A_UM**6  # w0 = 1 rad/s
    system, n_sites, t_gate, w0, sublattice = anneal_model.build_anneal_system(
        2, 2, c6, hz_i_w0=24.0, hx_peak_w0=1.0, t_hold_w0=12.0)

    assert system is not None and n_sites == 4
    assert w0 == 1.0 and t_gate == 16.0
    np.testing.assert_array_equal(sublattice, [1.0, -1.0, -1.0, 1.0])
    protocol = captured["protocol"]
    assert protocol._omega_half(0.0) == 0.0
    assert protocol._omega_half(1.0) == 0.5
    assert protocol._omega_half(3.0) == 1.0
    assert protocol._omega_half(15.0) == 0.5
    # Every 2x2 corner has two neighbours, so all boundary pins cancel.
    assert protocol._local(0.0, 0) == 0.0
    assert protocol._detuning(0.0) == -24.0
    assert protocol._detuning(8.0) == 0.0


def test_staggered_magnetization_uses_shared_sublattice_convention():
    occupations = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_array_equal(
        anneal_model.staggered_magnetization(occupations, np.array([1.0, -1.0])),
        [1.0, -1.0])
