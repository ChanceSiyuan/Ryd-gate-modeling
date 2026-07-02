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
from ryd_gate.core.effective_theory import single_atom_hamiltonian_parts
from ryd_gate.core.physical_models import _rb87_zero_420_couplings


def _pair_op(model):
    return next(t.operator for t in model.static_hamiltonian_terms if t.name == "H_pair")
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
        return frozenset({"E[e,g]", "E[r,e]", "E[e,e]", "E[r,r]"})

    def get_drive_coefficients(self, t, params):
        return {"E[e,g]": 1.0, "E[r,e]": 1.0, "E[e,e]": 0.0, "E[r,r]": 0.0}


def test_1r_lattice_basis_blocks_and_observables():
    model = RydbergSystem.set_atom_level("1r").set_atom_geom(Register.rectangle(2, 2))

    assert model.basis.local_levels == ("1", "r")
    assert model.basis.n_sites == 4
    assert any(t.name == "H_pair" for t in model.static_hamiltonian_terms)
    assert "E[r,1]" in model.hamiltonian_channels
    assert "E[r,r]" in model.hamiltonian_channels
    assert model.observables.get("n_r_0").per_site is True
    assert isinstance(_pair_op(model), RydbergPairInteractionSpec)


def test_all_pair_vdw_is_default():
    geom = Register.chain(4, spacing_um=4.0)
    model = RydbergSystem.set_atom_level("1r").set_atom_geom(geom)

    assert len(model.meta("interaction_pairs")) == 6


def test_nnn_interaction_mode_truncates_pairs():
    geom = Register.rectangle(3, 3, spacing_um=1.0)
    model = RydbergSystem.set_atom_level("1r").set_atom_geom(
        geom, interaction=InteractionSpec(C6=1.0, mode="nnn")
    )

    assert len(model.meta("interaction_pairs")) == 20


def test_vdw_energy_on_double_rydberg_state():
    geom = Register.chain(2, spacing_um=4.0)
    model = RydbergSystem.set_atom_level("1r").set_atom_geom(geom)
    psi = model.product_state("rr")
    pair = model.meta("interaction_pairs")[0]

    H_pair = ExactSparseCompiler().materialize_operator(model, _pair_op(model))
    energy = np.real(np.vdot(psi, H_pair @ psi))
    assert np.isclose(energy, pair[2])


def test_large_lattice_construction_does_not_materialize_exact_matrices():
    geom = Register.rectangle(20, 20, spacing_um=1.0)
    model = RydbergSystem.set_atom_level("1r").set_atom_geom(
        geom, interaction=InteractionSpec(C6=1.0, mode="nn")
    )

    assert model.N == 400
    assert isinstance(_pair_op(model), RydbergPairInteractionSpec)


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
            TransitionSpec("420", "g", "e", "E[e,g]"),
            TransitionSpec("1013", "e", "r", "E[r,e]"),
        ),
        detuning_levels={"E[e,e]": "e", "E[r,r]": "r"},
        initial_level="g",
    )

def test_level_structure_presets():
    assert level_structure("1r").levels == ("1", "r")
    assert level_structure("01r").levels == ("0", "1", "r")
    assert level_structure("analog_3").levels == ("g", "e", "r")
    assert level_structure("rb87_7_mp").levels == ("0", "1", "e1", "e2", "e3", "r", "r_garb")

    with pytest.raises(ValueError, match="Unknown level-structure"):
        level_structure("1er")


def test_ger_preset_removed():
    """D13: the symbolic `ger` preset is gone; hand-built specs replace it."""
    with pytest.raises(ValueError, match="Unknown level-structure"):
        level_structure("ger")


def test_custom_symbolic_spec_builds_g_e_r_levels():
    model = RydbergSystem.set_atom_level(_symbolic_ger_spec()).set_atom_geom(
        Register.chain(1), interaction=InteractionSpec(C6=0.0)
    )

    assert model.basis.local_levels == ("g", "e", "r")
    assert "E[e,g]" in model.hamiltonian_channels
    assert "E[r,e]" in model.hamiltonian_channels


def test_custom_spec_is_symbolic():
    """D11/D13: names carry semantics — only physical preset tags mount blocks."""
    symbolic = RydbergSystem.set_atom_level(
        _symbolic_ger_spec()
    ).set_atom_geom(Register.chain(1), interaction=InteractionSpec(C6=0.0))
    assert not any(t.name.startswith("E[") for t in symbolic.static_hamiltonian_terms)
    assert symbolic.meta("physical_model", None) is None

    physical = RydbergSystem.set_atom_level("analog_3").set_atom_geom(
        Register.chain(1), interaction=InteractionSpec(C6=0.0)
    )
    assert any(t.name.startswith("E[") for t in physical.static_hamiltonian_terms)
    assert physical.meta("physical_model") == "analog_3"


def test_symbolic_transition_blocks_are_not_compiled_as_static_dense_terms():
    model = RydbergSystem.set_atom_level(_symbolic_ger_spec()).set_atom_geom(
        Register.chain(1), interaction=InteractionSpec(C6=0.0)
    ).set_protocol(_GerProtocol())
    params = model.unpack_params([])
    ir = ExactSparseCompiler().compile(model, params)

    assert "E[r,e]" not in {term.name for term in ir.static_terms}


@pytest.mark.slow
def test_rb87_7_lattice_constructs_mp_model():
    model = RydbergSystem.set_atom_level("rb87_7_mp").set_atom_geom(
        Register.chain(2, spacing_um=3.0)
    )

    assert model.basis.local_dim == 7
    assert model.basis.local_levels == ("0", "1", "e1", "e2", "e3", "r", "r_garb")
    assert model.static_hamiltonian_terms  # static diagonal energies
    assert "E[e1,1]" in model.hamiltonian_channels  # 420 leg
    assert "E[r,e1]" in model.hamiltonian_channels  # 1013 leg
    assert model.meta("rb87_manifold") == "mp"
    # The Rabi scale lives in the CZ protocol now; only Delta is a system property.
    assert model.meta("Delta") != 0
    assert model.meta("rabi_eff") is None


def test_rb87_7_protocol_rabi_defaults_match_canonical():
    """rb87 builds unit blocks; the CZ protocol's default Rabis are the fixed
    σ⁻/σ⁺ (rb87_7_mp) canonical values, and cz_effective_rabi reproduces the
    operating point."""
    from ryd_gate.protocols.gate_cz import cz_effective_rabi, cz_rabi_maxes

    system = RydbergSystem.set_atom_level("rb87_7_mp").set_atom_geom(
        Register.chain(2, spacing_um=3.0)
    ).set_protocol(TOProtocol())
    o420, o1013 = cz_rabi_maxes(system)
    assert (o420, o1013) == pytest.approx((2 * np.pi * 491e6, 2 * np.pi * 185e6))
    rabi_eff, _ = cz_effective_rabi(system, o420, o1013)
    assert rabi_eff == pytest.approx(o420 * o1013 / (2 * abs(float(system.meta("Delta")))))


def test_rb87_7_zero_state_is_modeled_explicitly():
    """|0> is always modeled explicitly: clock-detuned energy + |0>->|e> 420
    legs (the E[e_k,0] channels; no separate lightshift_zero term)."""
    system = RydbergSystem.set_atom_level("rb87_7_mp").set_atom_geom(
        Register.chain(1, spacing_um=3.0)
    )

    h_const, h420, _ = single_atom_hamiltonian_parts(system)

    assert "zero_state_model" not in system.metadata
    assert h_const[0, 0].real == pytest.approx(-2 * np.pi * 6.835e9)
    assert not np.allclose(h420[2:5, 0], 0.0)  # off-resonant |0>->|e> legs
    assert not np.allclose(h420[2:5, 1], 0.0)  # |1>->|e> drive


def test_rb87_7_zero_state_model_kwarg_is_rejected():
    with pytest.raises(TypeError, match="zero_state_model"):
        RydbergSystem.set_atom_level(
            "rb87_7_mp", zero_state_model="explicit"
        ).set_atom_geom(Register.chain(1, spacing_um=3.0))


def test_rb87_7_zero_state_explicit_couplings_match_tex_phase_convention():
    couplings = 2 * np.asarray(_rb87_zero_420_couplings("mp", 1.0, np.sqrt(1 / 3)))
    expected = np.array([
        np.sqrt(3 / 10) + np.sqrt(2 / 15),
        -np.sqrt(1 / 2),
        0.0,
    ])

    np.testing.assert_allclose(couplings, expected, atol=1e-12)


def test_protocol_rabi_overrides_rescale_effective_rabi():
    """Per-protocol ``omega_*_max`` overrides set the operating point used to
    de-dimensionalize ``x`` (the Rabi knob moved from the system to the protocol)."""
    system = RydbergSystem.set_atom_level("rb87_7_pm").set_atom_geom(
        Register.chain(2, spacing_um=3.0)
    )
    o420 = o1013 = 2 * np.pi * 300e6
    over = TOProtocol(omega_420_max=o420, omega_1013_max=o1013)
    default = TOProtocol()
    x = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # t_gate = 1.0 * time_scale
    Delta = abs(float(system.meta("Delta")))
    expected_ts = 2 * np.pi / (o420 * o1013 / (2 * Delta))
    assert over.unpack_params(x, system)["t_gate"] == pytest.approx(expected_ts)
    # the bare protocol uses the canonical lukin Rabis -> a different time scale
    assert default.unpack_params(x, system)["t_gate"] != pytest.approx(expected_ts)
