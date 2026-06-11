"""Tests for the unified RydbergSystem API."""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate import RydbergSystem, SweepProtocol
from ryd_gate.backends.exact import simulate
from ryd_gate.backends.exact.compiler import ExactSparseCompiler
from ryd_gate.core.level_structures import InteractionSpec, level_structure
from ryd_gate.core.operator_spec import RydbergPairInteractionSpec
from ryd_gate.lattice import Register
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol


def _sweep(t_gate=0.1, omega=1.0, delta=0.0, n_steps=10):
    return SweepProtocol(
        t_gate=t_gate,
        omega_half_fn=lambda t: 0.5 * omega,
        delta_fn=lambda t: delta,
        n_steps=n_steps,
    )


class _GerProtocol:
    n_params = 0

    def validate_params(self, x):
        if x:
            raise ValueError("no params")

    def unpack_params(self, x, system):
        self.validate_params(x)
        return {"t_gate": 0.1, "pin_deltas": {}, "scatter_rates": {}, "static_overlays": []}

    def drive_channels(self, system):
        return frozenset({"drive_420", "H_1013", "delta_e", "delta_R"})

    def get_drive_coefficients(self, t, params):
        return {"drive_420": 1.0, "H_1013": 1.0, "delta_e": 0.0, "delta_R": 0.0}


def test_1r_lattice_basis_blocks_and_observables():
    model = RydbergSystem.from_lattice(Register.rectangle(2, 2), "1r")

    assert model.basis.local_levels == ("1", "r")
    assert model.basis.n_sites == 4
    assert model.blocks.has("H_vdw")
    assert model.blocks.has("global_X")
    assert model.blocks.has("global_n")
    assert model.observables.get("n_r_0").per_site is True
    assert isinstance(model.blocks.get("H_vdw"), RydbergPairInteractionSpec)


def test_all_pair_vdw_is_default():
    geom = Register.chain(4, spacing_um=4.0)
    model = RydbergSystem.from_lattice(geom, "1r")

    assert len(model.meta("interaction_pairs")) == 6


def test_nnn_interaction_mode_truncates_pairs():
    geom = Register.rectangle(3, 3, spacing_um=1.0)
    model = RydbergSystem.from_lattice(
        geom,
        "1r",
        interaction=InteractionSpec(C6=1.0, mode="nnn"),
    )

    assert len(model.meta("interaction_pairs")) == 20


def test_vdw_energy_on_double_rydberg_state():
    geom = Register.chain(2, spacing_um=4.0)
    model = RydbergSystem.from_lattice(geom, "1r")
    psi = model.product_state("rr")
    pair = model.meta("interaction_pairs")[0]

    H_vdw = ExactSparseCompiler().materialize_block(model, "H_vdw")
    energy = np.real(np.vdot(psi, H_vdw @ psi))
    assert np.isclose(energy, pair[2])


def test_large_lattice_construction_does_not_materialize_exact_matrices():
    geom = Register.rectangle(20, 20, spacing_um=1.0)
    model = RydbergSystem.from_lattice(
        geom,
        "1r",
        interaction=InteractionSpec(C6=1.0, mode="nn"),
    )

    assert model.N == 400
    assert isinstance(model.blocks.get("H_vdw"), RydbergPairInteractionSpec)


def test_exact_sparse_compiler_rejects_too_large_hilbert_space():
    model = RydbergSystem.from_lattice(
        Register.chain(8),
        "1r",
        interaction=InteractionSpec(C6=0.0),
        protocol=_sweep(n_steps=2),
    )
    params = model.unpack_params([])

    with pytest.raises(ValueError, match="Exact sparse compilation"):
        ExactSparseCompiler(max_dim=16).compile(model, params)


def test_sweep_simulation_with_unified_model():
    model = RydbergSystem.from_lattice(
        Register.rectangle(2, 2),
        "1r",
        interaction=InteractionSpec(C6=0.0),
        protocol=_sweep(delta=0.0, n_steps=10),
    )
    psi0 = model.ground_state()
    result = simulate(model, [], psi0)

    assert np.isclose(np.linalg.norm(result.psi_final), 1.0)


def test_sparse_expm_t_eval_array_records_requested_steps_only():
    model = RydbergSystem.from_lattice(
        Register.chain(1),
        "1r",
        interaction=InteractionSpec(C6=0.0),
        protocol=_sweep(n_steps=10),
    )
    psi0 = model.ground_state()
    t_eval = np.array([0.0, 0.05, 0.1])

    result = simulate(model, [], psi0, t_eval=t_eval)

    np.testing.assert_allclose(result.times, t_eval)
    assert result.states.shape == (len(t_eval), model.dim)


def test_sparse_expm_t_eval_true_records_internal_steps_for_compatibility():
    model = RydbergSystem.from_lattice(
        Register.chain(1),
        "1r",
        interaction=InteractionSpec(C6=0.0),
        protocol=_sweep(n_steps=4),
    )
    psi0 = model.ground_state()

    result = simulate(model, [], psi0, t_eval=True)

    assert result.times.shape == (4,)
    assert result.states.shape == (4, model.dim)


def test_01r_digital_analog_simulation():
    protocol = DigitalAnalogProtocol(
        t_gate=0.1,
        omega_R_fn=lambda t: 1.0,
        n_steps=10,
    )
    model = RydbergSystem.from_lattice(
        Register.chain(2, spacing_um=4.0),
        "01r",
        interaction=InteractionSpec(C6=0.0),
        protocol=protocol,
    )
    psi0 = model.product_state("11")
    result = simulate(model, [], psi0)

    assert model.basis.local_levels == ("0", "1", "r")
    assert np.isclose(np.linalg.norm(result.psi_final), 1.0)


def test_level_structure_presets():
    assert level_structure("1r").levels == ("1", "r")
    assert level_structure("01r").levels == ("0", "1", "r")
    assert level_structure("ger").levels == ("g", "e", "r")
    assert level_structure("rb87_7").levels == ("0", "1", "e1", "e2", "e3", "r", "r_garb")

    with pytest.raises(ValueError, match="Unknown level-structure"):
        level_structure("1er")


def test_ger_lattice_builds_g_e_r_levels():
    model = RydbergSystem.from_lattice(
        Register.chain(1),
        "ger",
        interaction=InteractionSpec(C6=0.0),
    )

    assert model.basis.local_levels == ("g", "e", "r")
    assert model.blocks.has("drive_420")
    assert model.blocks.has("H_1013")


def test_ger_transition_blocks_are_not_compiled_as_static_dense_terms():
    model = RydbergSystem.from_lattice(
        Register.chain(1),
        "ger",
        interaction=InteractionSpec(C6=0.0),
        protocol=_GerProtocol(),
    )
    params = model.unpack_params([])
    ir = ExactSparseCompiler().compile(model, params)

    assert "H_1013" not in {term.name for term in ir.static_terms}


@pytest.mark.slow
def test_rb87_7_lattice_constructs_our_model():
    model = RydbergSystem.from_lattice(
        Register.chain(2, spacing_um=3.0),
        "rb87_7",
        param_set="our",
    )

    assert model.basis.local_dim == 7
    assert model.basis.local_levels == ("0", "1", "e1", "e2", "e3", "r", "r_garb")
    assert model.blocks.has("H_const")
    assert model.blocks.has("drive_420")
    assert model.meta("rabi_eff") > 0
