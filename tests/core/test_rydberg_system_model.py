"""Tests for the unified RydbergSystem API."""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem, SweepProtocol, level_structure
from ryd_gate.backends.exact import simulate
from ryd_gate.backends.exact.compiler import ExactCompiler
from ryd_gate.core.effective_theory import single_atom_hamiltonian_parts
from ryd_gate.core.level_structures import InteractionSpec
from ryd_gate.core.observables import ObservableExpr
from ryd_gate.core.operators import RydbergPairInteractionSpec
from ryd_gate.core.physical_models import _rb87_zero_420_couplings
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol
from ryd_gate.protocols.gate_cz import TOProtocol


def _build(name, register, *, interaction=None, protocol=None):
    return RydbergSystem(
        level_structure=level_structure(name),
        register=register,
        interaction=interaction,
        protocol=protocol,
    )


def _pair_op(model):
    return next(t.operator for t in model.static_hamiltonian_terms if t.name == "H_pair")


def _sweep(t_gate=0.1, omega=1.0, delta=0.0):
    return SweepProtocol(
        t_gate=t_gate,
        omega_half_fn=lambda t: 0.5 * omega,
        delta_fn=lambda t: delta,
    )


def test_1r_lattice_basis_blocks_and_observables():
    model = _build("1r", Register.rectangle(2, 2))

    assert model.basis.local_levels == ("1", "r")
    assert model.basis.n_sites == 4
    assert any(t.name == "H_pair" for t in model.static_hamiltonian_terms)
    assert "E[r,1]" in model.hamiltonian_channels
    assert "E[r,r]" in model.hamiltonian_channels
    assert isinstance(model.observables.n("r", 0), ObservableExpr)
    assert isinstance(_pair_op(model), RydbergPairInteractionSpec)


def test_all_pair_vdw_is_default():
    model = _build("1r", Register.chain(4, spacing_um=4.0))

    assert len(model.interaction_pairs) == 6


def test_nnn_interaction_mode_truncates_pairs():
    model = _build(
        "1r",
        Register.rectangle(3, 3, spacing_um=1.0),
        interaction=InteractionSpec(C6=1.0, mode="nnn"),
    )

    assert len(model.interaction_pairs) == 20


def test_vdw_energy_on_double_rydberg_state():
    model = _build("1r", Register.chain(2, spacing_um=4.0))
    psi = model.product_state("rr")
    pair = model.interaction_pairs[0]

    H_pair = ExactCompiler().materialize_operator(model, _pair_op(model))
    energy = np.real(np.vdot(psi, H_pair @ psi))
    assert np.isclose(energy, pair[2])


def test_large_lattice_construction_does_not_materialize_exact_matrices():
    model = _build(
        "1r",
        Register.rectangle(20, 20, spacing_um=1.0),
        interaction=InteractionSpec(C6=1.0, mode="nn"),
    )

    assert model.N == 400
    assert isinstance(_pair_op(model), RydbergPairInteractionSpec)


def test_exact_compiler_rejects_too_large_hilbert_space():
    model = _build(
        "1r", Register.chain(8), interaction=InteractionSpec(C6=0.0), protocol=_sweep()
    )

    with pytest.raises(ValueError, match="Exact sparse compilation"):
        ExactCompiler(max_dim=16).compile(model)


def test_sweep_simulation_with_unified_model():
    model = _build(
        "1r",
        Register.rectangle(2, 2),
        interaction=InteractionSpec(C6=0.0),
        protocol=_sweep(delta=0.0),
    )
    psi0 = model.ground_state()
    result = simulate(model, psi0)

    assert np.isclose(np.linalg.norm(result.final_state), 1.0)


def test_t_eval_array_returns_requested_times_exactly():
    model = _build(
        "1r", Register.chain(1), interaction=InteractionSpec(C6=0.0), protocol=_sweep()
    )
    psi0 = model.ground_state()
    t_eval = np.array([0.0, 0.05, 0.1])

    result = simulate(
        model, psi0, t_eval=t_eval,
        observables={"n_r": model.observables.level_sum("r")},
    )

    np.testing.assert_array_equal(result.times, t_eval)
    assert result.expectation("n_r").shape == (len(t_eval),)


def test_01r_digital_analog_simulation():
    protocol = DigitalAnalogProtocol(
        t_gate=0.1,
        coupling_r1=lambda t: 0.5,  # Omega_R/2 matrix element (old omega_R_fn = 1.0)
    )
    model = _build(
        "01r",
        Register.chain(2, spacing_um=4.0),
        interaction=InteractionSpec(C6=0.0),
        protocol=protocol,
    )
    psi0 = model.product_state("11")
    result = simulate(model, psi0)

    assert model.basis.local_levels == ("0", "1", "r")
    assert np.isclose(np.linalg.norm(result.final_state), 1.0)


def test_level_structure_presets():
    assert level_structure("1r").levels == ("1", "r")
    assert level_structure("01r").levels == ("0", "1", "r")
    assert level_structure("analog_3").levels == ("g", "e", "r")
    assert level_structure("rb87_7_mp").levels == ("0", "1", "e1", "e2", "e3", "r", "r_garb")

    with pytest.raises(ValueError, match="Unknown level-structure"):
        level_structure("1er")


def test_ger_preset_removed():
    """D13: the symbolic `ger` preset is gone (no custom-spec escape hatch either)."""
    with pytest.raises(ValueError, match="Unknown level-structure"):
        level_structure("ger")


@pytest.mark.slow
def test_rb87_7_lattice_constructs_mp_model():
    model = _build("rb87_7_mp", Register.chain(2, spacing_um=3.0))

    assert model.basis.local_dim == 7
    assert model.basis.local_levels == ("0", "1", "e1", "e2", "e3", "r", "r_garb")
    assert model.static_hamiltonian_terms  # static diagonal energies
    assert "E[e1,1]" in model.hamiltonian_channels  # 420 leg
    assert "E[r,e1]" in model.hamiltonian_channels  # 1013 leg
    assert model.level_structure.physical_model == "rb87_7_mp"
    # The Rabi scale lives in the CZ protocol now; only Delta is a system property.
    assert model.level_structure.Delta != 0
    assert model.level_structure.rabi_eff is None


def test_rb87_7_protocol_rabi_defaults_match_canonical():
    """rb87 builds unit blocks; the CZ protocol's default Rabis are the fixed
    σ⁻/σ⁺ (rb87_7_mp) canonical values, and cz_effective_rabi reproduces the
    operating point."""
    from ryd_gate.protocols.gate_cz import cz_effective_rabi, cz_rabi_maxes

    system = _build("rb87_7_mp", Register.chain(2, spacing_um=3.0))
    o420, o1013 = cz_rabi_maxes(system)
    assert (o420, o1013) == pytest.approx((2 * np.pi * 491e6, 2 * np.pi * 185e6))
    rabi_eff, _ = cz_effective_rabi(system, o420, o1013)
    assert rabi_eff == pytest.approx(
        o420 * o1013 / (2 * abs(float(system.level_structure.Delta)))
    )


def test_rb87_7_zero_state_is_modeled_explicitly():
    """|0> is always modeled explicitly: clock-detuned energy + |0>->|e> 420
    legs (the E[e_k,0] channels; no separate lightshift_zero term)."""
    system = _build("rb87_7_mp", Register.chain(1, spacing_um=3.0))

    h_const, h420, _ = single_atom_hamiltonian_parts(system)

    assert h_const[0, 0].real == pytest.approx(-2 * np.pi * 6.835e9)
    assert not np.allclose(h420[2:5, 0], 0.0)  # off-resonant |0>->|e> legs
    assert not np.allclose(h420[2:5, 1], 0.0)  # |1>->|e> drive


def test_rb87_7_zero_state_model_kwarg_is_rejected():
    with pytest.raises(TypeError, match="zero_state_model"):
        level_structure("rb87_7_mp", zero_state_model="explicit")


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
    resolve ``duration_ratio`` (the Rabi knob lives on the protocol, not the
    system)."""
    system = _build("rb87_7_pm", Register.chain(2, spacing_um=3.0))
    o420 = o1013 = 2 * np.pi * 300e6
    fields = dict(
        phase_amplitude=0.0, frequency_ratio=0.0, phase_offset=0.0,
        detuning_ratio=0.0, duration_ratio=1.0,  # t_gate = 1.0 * time_scale
    )
    over = TOProtocol(omega_420_max=o420, omega_1013_max=o1013, **fields)
    default = TOProtocol(**fields)
    Delta = abs(float(system.level_structure.Delta))
    expected_ts = 2 * np.pi / (o420 * o1013 / (2 * Delta))
    assert over.resolve_t_gate(system) == pytest.approx(expected_ts)
    # system.t_gate is the bound protocol's duration resolved against the system
    assert system.with_protocol(over).t_gate == pytest.approx(expected_ts)
    # the bare protocol uses the canonical lukin Rabis -> a different time scale
    assert default.resolve_t_gate(system) != pytest.approx(expected_ts)
