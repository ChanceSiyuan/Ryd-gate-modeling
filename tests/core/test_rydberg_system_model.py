"""Tests for the unified RydbergSystem API."""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate import RydbergSystem, SweepProtocol
from ryd_gate.backends.exact import simulate
from ryd_gate.backends.exact.compiler import ExactSparseCompiler
from ryd_gate.core.level_structures import (
    InteractionSpec,
    LevelStructureSpec,
    TransitionSpec,
    level_structure,
)
from ryd_gate.core.operators import RydbergPairInteractionSpec
from ryd_gate.core.physical_models import _rb87_zero_420_couplings
from ryd_gate.lattice import Register
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol
from ryd_gate.protocols.gate_cz import TOProtocol


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
    model = RydbergSystem.set_atom_level("1r").set_atom_geom(Register.rectangle(2, 2)).build()

    assert model.basis.local_levels == ("1", "r")
    assert model.basis.n_sites == 4
    assert model.blocks.has("H_vdw")
    assert model.blocks.has("global_X")
    assert model.blocks.has("global_n")
    assert model.observables.get("n_r_0").per_site is True
    assert isinstance(model.blocks.get("H_vdw"), RydbergPairInteractionSpec)


def test_all_pair_vdw_is_default():
    geom = Register.chain(4, spacing_um=4.0)
    model = RydbergSystem.set_atom_level("1r").set_atom_geom(geom).build()

    assert len(model.meta("interaction_pairs")) == 6


def test_nnn_interaction_mode_truncates_pairs():
    geom = Register.rectangle(3, 3, spacing_um=1.0)
    model = RydbergSystem.set_atom_level("1r").set_atom_geom(
        geom, interaction=InteractionSpec(C6=1.0, mode="nnn")
    ).build()

    assert len(model.meta("interaction_pairs")) == 20


def test_vdw_energy_on_double_rydberg_state():
    geom = Register.chain(2, spacing_um=4.0)
    model = RydbergSystem.set_atom_level("1r").set_atom_geom(geom).build()
    psi = model.product_state("rr")
    pair = model.meta("interaction_pairs")[0]

    H_vdw = ExactSparseCompiler().materialize_block(model, "H_vdw")
    energy = np.real(np.vdot(psi, H_vdw @ psi))
    assert np.isclose(energy, pair[2])


def test_large_lattice_construction_does_not_materialize_exact_matrices():
    geom = Register.rectangle(20, 20, spacing_um=1.0)
    model = RydbergSystem.set_atom_level("1r").set_atom_geom(
        geom, interaction=InteractionSpec(C6=1.0, mode="nn")
    ).build()

    assert model.N == 400
    assert isinstance(model.blocks.get("H_vdw"), RydbergPairInteractionSpec)


def test_exact_sparse_compiler_rejects_too_large_hilbert_space():
    model = RydbergSystem.set_atom_level("1r").set_atom_geom(
        Register.chain(8), interaction=InteractionSpec(C6=0.0)
    ).set_protocol(_sweep(n_steps=2))
    params = model.unpack_params([])

    with pytest.raises(ValueError, match="Exact sparse compilation"):
        ExactSparseCompiler(max_dim=16).compile(model, params)


def test_sweep_simulation_with_unified_model():
    model = RydbergSystem.set_atom_level("1r").set_atom_geom(
        Register.rectangle(2, 2), interaction=InteractionSpec(C6=0.0)
    ).set_protocol(_sweep(delta=0.0, n_steps=10))
    psi0 = model.ground_state()
    result = simulate(model, [], psi0)

    assert np.isclose(np.linalg.norm(result.psi_final), 1.0)


def test_sparse_expm_t_eval_array_records_requested_steps_only():
    model = RydbergSystem.set_atom_level("1r").set_atom_geom(
        Register.chain(1), interaction=InteractionSpec(C6=0.0)
    ).set_protocol(_sweep(n_steps=10))
    psi0 = model.ground_state()
    t_eval = np.array([0.0, 0.05, 0.1])

    result = simulate(model, [], psi0, t_eval=t_eval)

    np.testing.assert_allclose(result.times, t_eval)
    assert result.states.shape == (len(t_eval), model.dim)


def test_sparse_expm_t_eval_true_records_internal_steps_for_compatibility():
    model = RydbergSystem.set_atom_level("1r").set_atom_geom(
        Register.chain(1), interaction=InteractionSpec(C6=0.0)
    ).set_protocol(_sweep(n_steps=4))
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
    model = RydbergSystem.set_atom_level("01r").set_atom_geom(
        Register.chain(2, spacing_um=4.0), interaction=InteractionSpec(C6=0.0)
    ).set_protocol(protocol)
    psi0 = model.product_state("11")
    result = simulate(model, [], psi0)

    assert model.basis.local_levels == ("0", "1", "r")
    assert np.isclose(np.linalg.norm(result.psi_final), 1.0)




def _symbolic_ger_spec() -> LevelStructureSpec:
    """Hand-built symbolic three-level ladder (the custom-model escape hatch)."""
    return LevelStructureSpec(
        name="ger_symbolic",
        levels=("g", "e", "r"),
        rydberg_levels=("r",),
        transitions=(
            TransitionSpec("420", "g", "e", "drive_420"),
            TransitionSpec("1013", "e", "r", "H_1013"),
        ),
        detuning_levels={"delta_e": "e", "delta_R": "r"},
        initial_level="g",
    )

def test_level_structure_presets():
    assert level_structure("1r").levels == ("1", "r")
    assert level_structure("01r").levels == ("0", "1", "r")
    assert level_structure("analog_3").levels == ("g", "e", "r")
    assert level_structure("rb87_7").levels == ("0", "1", "e1", "e2", "e3", "r", "r_garb")

    with pytest.raises(ValueError, match="Unknown level-structure"):
        level_structure("1er")


def test_ger_preset_removed():
    """D13: the symbolic `ger` preset is gone; hand-built specs replace it."""
    with pytest.raises(ValueError, match="Unknown level-structure"):
        level_structure("ger")


def test_custom_symbolic_spec_builds_g_e_r_levels():
    model = RydbergSystem.set_atom_level(_symbolic_ger_spec()).set_atom_geom(
        Register.chain(1), interaction=InteractionSpec(C6=0.0)
    ).build()

    assert model.basis.local_levels == ("g", "e", "r")
    assert model.blocks.has("drive_420")
    assert model.blocks.has("H_1013")


def test_custom_spec_is_symbolic_regardless_of_param_set():
    """D11/D13: names carry semantics — only `analog_3` mounts physical blocks."""
    symbolic = RydbergSystem.set_atom_level(
        _symbolic_ger_spec(), param_set="analog_3"
    ).set_atom_geom(Register.chain(1), interaction=InteractionSpec(C6=0.0)).build()
    assert not symbolic.blocks.has("H_const")
    assert symbolic.meta("physical_model", None) is None

    physical = RydbergSystem.set_atom_level("analog_3").set_atom_geom(
        Register.chain(1), interaction=InteractionSpec(C6=0.0)
    ).build()
    assert physical.blocks.has("H_const")
    assert physical.meta("physical_model") == "analog_3"


def test_symbolic_transition_blocks_are_not_compiled_as_static_dense_terms():
    model = RydbergSystem.set_atom_level(_symbolic_ger_spec()).set_atom_geom(
        Register.chain(1), interaction=InteractionSpec(C6=0.0)
    ).set_protocol(_GerProtocol())
    params = model.unpack_params([])
    ir = ExactSparseCompiler().compile(model, params)

    assert "H_1013" not in {term.name for term in ir.static_terms}


@pytest.mark.slow
def test_rb87_7_lattice_constructs_our_model():
    model = RydbergSystem.set_atom_level("rb87_7", param_set="our").set_atom_geom(
        Register.chain(2, spacing_um=3.0)
    ).build()

    assert model.basis.local_dim == 7
    assert model.basis.local_levels == ("0", "1", "e1", "e2", "e3", "r", "r_garb")
    assert model.blocks.has("H_const")
    assert model.blocks.has("drive_420")
    assert model.meta("rabi_eff") > 0


def test_rb87_7_hz_overrides_match_defaults_when_omitted():
    base = RydbergSystem.set_atom_level("rb87_7", param_set="our").set_atom_geom(
        Register.chain(2, spacing_um=3.0)
    ).build()
    explicit = RydbergSystem.set_atom_level("rb87_7", param_set="our").set_atom_geom(
        Register.chain(2, spacing_um=3.0)
    ).set_protocol(
        TOProtocol(Delta_Hz=9.1e9, rabi_420_Hz=491e6, rabi_1013_Hz=185e6)
    )
    assert base.meta("rabi_eff") == pytest.approx(explicit.meta("rabi_eff"))
    assert base.meta("Delta") == pytest.approx(explicit.meta("Delta"))
    assert base.meta("rabi_420") == pytest.approx(explicit.meta("rabi_420"))
    assert base.meta("rabi_1013") == pytest.approx(explicit.meta("rabi_1013"))


def test_rb87_7_zero_state_is_modeled_explicitly():
    """|0> is always modeled explicitly: clock-detuned energy + |0>->|e> 420
    legs in drive_420 (no separate lightshift_zero block)."""
    system = RydbergSystem.set_atom_level("rb87_7", param_set="our").set_atom_geom(
        Register.chain(1, spacing_um=3.0)
    ).build()

    h_const = system.blocks.get("H_const").matrix
    h420 = system.blocks.get("drive_420").matrix

    assert "zero_state_model" not in system.metadata
    assert h_const[0, 0].real == pytest.approx(-2 * np.pi * 6.835e9)
    assert not np.allclose(h420[2:5, 0], 0.0)  # off-resonant |0>->|e> legs
    assert not np.allclose(h420[2:5, 1], 0.0)  # |1>->|e> drive


def test_rb87_7_zero_state_model_kwarg_is_rejected():
    with pytest.raises(TypeError, match="zero_state_model"):
        RydbergSystem.set_atom_level(
            "rb87_7", param_set="our", zero_state_model="explicit"
        ).set_atom_geom(Register.chain(1, spacing_um=3.0)).build()


def test_rb87_7_zero_state_explicit_couplings_match_tex_phase_convention():
    couplings = 2 * np.asarray(_rb87_zero_420_couplings("our", 1.0, np.sqrt(1 / 3)))
    expected = np.array([
        np.sqrt(3 / 10) + np.sqrt(2 / 15),
        -np.sqrt(1 / 2),
        0.0,
    ])

    np.testing.assert_allclose(couplings, expected, atol=1e-12)


def test_rb87_7_hz_overrides_rescale_rabi_eff():
    import numpy as np

    system = RydbergSystem.set_atom_level("rb87_7", param_set="lukin").set_atom_geom(
        Register.chain(2, spacing_um=3.0)
    ).set_protocol(TOProtocol(rabi_420_Hz=300e6, rabi_1013_Hz=300e6))
    Delta = float(system.meta("Delta"))
    rabi_420 = float(system.meta("rabi_420"))
    rabi_1013 = float(system.meta("rabi_1013"))
    expected = rabi_420 * rabi_1013 / (2 * abs(Delta))
    assert system.meta("rabi_eff") == pytest.approx(expected)
    assert rabi_420 == pytest.approx(2 * np.pi * 300e6)
