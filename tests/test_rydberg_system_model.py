"""Tests for the unified RydbergSystem API."""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate import RydbergSystem, RydbergSystemModel, SweepProtocol, simulate
from ryd_gate.core.rydberg_system import InteractionSpec, level_structure
from ryd_gate.lattice import make_chain, make_square_lattice
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol


def test_1r_lattice_basis_blocks_and_observables():
    model = RydbergSystemModel.from_lattice(make_square_lattice(2, 2), "1r")

    assert model.basis.local_levels == ("1", "r")
    assert model.basis.n_sites == 4
    assert model.blocks.has("H_vdw")
    assert model.blocks.has("global_X")
    assert model.blocks.has("global_n")
    assert model.observables.get("n_r_0").per_site is True


def test_all_pair_vdw_is_default():
    geom = make_chain(4, spacing_um=4.0)
    model = RydbergSystemModel.from_lattice(geom, "1r")

    assert len(model.meta("interaction_pairs")) == 6


def test_nnn_interaction_mode_truncates_pairs():
    geom = make_square_lattice(3, 3, spacing_um=1.0)
    model = RydbergSystemModel.from_lattice(
        geom,
        "1r",
        interaction=InteractionSpec(C6=1.0, mode="nnn"),
    )

    assert len(model.meta("interaction_pairs")) == 20


def test_vdw_energy_on_double_rydberg_state():
    geom = make_chain(2, spacing_um=4.0)
    model = RydbergSystemModel.from_lattice(geom, "1r")
    psi = model.product_state("rr")
    pair = model.meta("interaction_pairs")[0]

    energy = np.real(np.vdot(psi, model.blocks.get("H_vdw") @ psi))
    assert np.isclose(energy, pair[2])


def test_sweep_simulation_with_unified_model():
    model = RydbergSystem.from_lattice(
        make_square_lattice(2, 2),
        "1r",
        interaction=InteractionSpec(C6=0.0),
        protocol=SweepProtocol(n_steps=10),
    )
    psi0 = model.ground_state()
    result = simulate(model, [-1.0, 1.0, 0.1], psi0)

    assert np.isclose(np.linalg.norm(result.psi_final), 1.0)


def test_01r_digital_analog_simulation():
    protocol = DigitalAnalogProtocol.constant(omega_R=1.0, t_gate=0.1, n_steps=10)
    model = RydbergSystem.from_preset(
        "01r", protocol=protocol, N=2, spacing_um=4.0, C6=0.0,
    )
    psi0 = model.product_state("11")
    result = simulate(model, [], psi0)

    assert model.basis.local_levels == ("0", "1", "r")
    assert np.isclose(np.linalg.norm(result.psi_final), 1.0)


def test_rydberg_system_alias_back_compat():
    """RydbergSystemModel remains importable as an alias for RydbergSystem."""
    assert RydbergSystemModel is RydbergSystem


def test_level_structure_presets():
    assert level_structure("1r").levels == ("1", "r")
    assert level_structure("01r").levels == ("0", "1", "r")
    assert level_structure("1er").levels == ("1", "e", "r")
    assert level_structure("rb87_7").levels == ("0", "1", "e1", "e2", "e3", "r", "r_garb")


@pytest.mark.slow
def test_dense_rb87_preset_constructs_model():
    model = RydbergSystemModel.from_preset("our")

    assert model.basis.local_dim == 7
    assert model.basis.local_levels == ("0", "1", "e1", "e2", "e3", "r", "r_garb")
    assert model.blocks.has("H_const")
    assert model.blocks.has("drive_420")
    assert model.meta("rabi_eff") > 0
