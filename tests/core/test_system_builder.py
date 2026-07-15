"""Contract tests for the explicit RydbergSystem constructor.

``RydbergSystem(level_structure=..., register=..., interaction=..., protocol=...)``
replaces the fluent ``set_atom_*`` builder: the level structure is resolved up
front by ``level_structure(name, **physical_kwargs)`` (an immutable preset),
the register is explicit, and protocols bind at construction or via
``with_protocol``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.core.level_structures import InteractionSpec
from ryd_gate.core.physical_models import analog_3_local_blocks
from ryd_gate.protocols.gate_cz import TOProtocol


def _to_protocol(**overrides):
    """A fully specified TOProtocol (binding-only tests; values arbitrary)."""
    fields = dict(
        phase_amplitude=0.0, frequency_ratio=1.0, phase_offset=0.0,
        detuning_ratio=0.0, duration_ratio=1.0,
    )
    fields.update(overrides)
    return TOProtocol(**fields)


# ── happy paths per preset kind ──────────────────────────────────────────────


def test_1r_chain_builds_usable_system():
    system = RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(3, spacing_um=5.0),
    )

    assert system.N == 3
    assert system.dim == 8
    assert any(t.name == "H_pair" for t in system.static_hamiltonian_terms)
    assert system.protocol is None


def test_single_atom_register_is_a_complete_system():
    system = RydbergSystem(
        level_structure=level_structure("1r"), register=Register.chain(1)
    )

    assert system.N == 1
    assert system.dim == 2
    assert system.interaction_pairs == ()
    assert system.protocol is None


def test_01r_channels_and_observables():
    system = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(2, spacing_um=4.0),
    )

    assert system.basis.local_levels == ("0", "1", "r")
    assert set(system.hamiltonian_channels) == {
        "E[r,1]", "E[1,0]", "E[r,0]", "E[r,r]", "E[1,1]"
    }
    # observables is the immutable expression factory bound to this basis
    from ryd_gate.core.observables import ObservableExpr, ObservableFactory

    assert isinstance(system.observables, ObservableFactory)
    assert isinstance(system.observables.n("r", 0), ObservableExpr)
    assert isinstance(system.observables.level_sum("r"), ObservableExpr)


def test_interaction_spec_controls_pair_set():
    all_pairs = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.rectangle(2, 2, spacing_um=6.0),
    )
    nn_pairs = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.rectangle(2, 2, spacing_um=6.0),
        interaction=InteractionSpec(C6=4.0, mode="nn"),
    )

    assert len(all_pairs.interaction_pairs) == 6  # C(4, 2)
    assert len(nn_pairs.interaction_pairs) == 4  # square edges only


def test_analog_3_laser_kwargs_set_the_operating_point():
    system = RydbergSystem(
        level_structure=level_structure(
            "analog_3", detuning_sign=1, Delta_Hz=5.0e9,
            rabi_420_Hz=300e6, rabi_1013_Hz=200e6,
        ),
        register=Register.chain(2, spacing_um=5.0),
        protocol=_to_protocol(),
    )

    expected = (2 * np.pi * 300e6) * (2 * np.pi * 200e6) / (2 * abs(2 * np.pi * 5.0e9))
    assert system.level_structure.rabi_eff == pytest.approx(expected)
    assert system.level_structure.Delta == pytest.approx(2 * np.pi * 5.0e9)


def test_analog_3_operating_point_matches_source_of_truth():
    # The public construction must reproduce the analog_3 operating point that
    # ``analog_3_local_blocks`` (the single source of truth) computes from the
    # same laser knobs.
    laser = dict(Delta_Hz=5.0e9, rabi_420_Hz=300e6, rabi_1013_Hz=200e6)
    built = RydbergSystem(
        level_structure=level_structure("analog_3", detuning_sign=1, **laser),
        register=Register.chain(2, spacing_um=5.0),
        protocol=_to_protocol(),
    )
    blk = analog_3_local_blocks(detuning_sign=1, **laser)

    assert built.basis.local_levels == ("g", "e", "r")
    assert built.level_structure.physical_model == "analog_3"
    assert built.level_structure.rabi_eff == pytest.approx(blk.rabi_eff)
    assert built.level_structure.Delta == pytest.approx(blk.Delta)
    assert built.static_hamiltonian_terms  # diagonal g/e/r energies
    assert "E[e,g]" in built.hamiltonian_channels


@pytest.mark.parametrize("tag", ["rb87_7_mp", "rb87_7_pm"])
def test_rb87_7_manifold_tags_build_seven_level(tag):
    s = RydbergSystem(level_structure=level_structure(tag), register=Register.chain(1))

    assert s.basis.local_dim == 7
    assert s.basis.local_levels == ("0", "1", "e1", "e2", "e3", "r", "r_garb")
    assert s.protocol is None
    assert s.static_hamiltonian_terms  # static diagonal energies
    assert "E[e1,1]" in s.hamiltonian_channels  # 420 leg
    assert "E[r,e1]" in s.hamiltonian_channels  # 1013 leg
    assert s.level_structure.name == tag
    assert s.level_structure.physical_model == tag


@pytest.mark.slow
def test_rb87_7_default_operating_point():
    system = RydbergSystem(
        level_structure=level_structure("rb87_7_mp"),
        register=Register.chain(2, spacing_um=3.0),
    )

    assert system.basis.local_dim == 7
    assert system.static_hamiltonian_terms  # static diagonal energies
    assert "E[e1,1]" in system.hamiltonian_channels  # 420 leg
    assert "E[r,e1]" in system.hamiltonian_channels  # 1013 leg
    assert system.level_structure.Delta != 0
    # The Rabi scale lives in the CZ protocol, not on the level structure.
    assert system.level_structure.rabi_eff is None


def test_rb87_7_static_overrides():
    ls = level_structure(
        "rb87_7_mp", Delta_Hz=12e9, ryd_level=80, C6_rad_s_um6=2 * np.pi * 900e9,
        t_rise=30e-9,
    )
    assert ls.Delta == pytest.approx(2 * np.pi * 12e9)
    assert ls.ryd_level == 80
    assert ls.t_rise == pytest.approx(30e-9)
    # C6 override flows into the default rb87 interaction (v_ryd at the nominal 3 um).
    s = RydbergSystem(level_structure=ls, register=Register.chain(2, spacing_um=3.0))
    v_ryd = max(abs(v) for *_, v in s.interaction_pairs)
    assert v_ryd == pytest.approx(2 * np.pi * 900e9 / 3**6)


@pytest.mark.slow
def test_rb87_7_magnetic_field_sets_zeeman_shift():
    # The garbage-Rydberg splitting is a physical linear Zeeman shift set by
    # magnetic_field_G: 20 G -> 2pi * 56 MHz (g_J=2, Delta_mj=1 for nS_1/2).
    ls = level_structure("rb87_7_mp", detuning_sign=1, magnetic_field_G=20.0)

    assert ls.magnetic_field_G == 20.0
    assert ls.ryd_zeeman_shift / (2 * np.pi * 1e6) == pytest.approx(56.0, rel=2e-3)


# ── argument-type and preset validation ─────────────────────────────────────


def test_wrong_argument_types_raise_typeerror():
    reg = Register.chain(2, spacing_um=5.0)
    with pytest.raises(TypeError, match="level_structure"):
        RydbergSystem(level_structure="1r", register=reg)
    with pytest.raises(TypeError, match="register"):
        RydbergSystem(
            level_structure=level_structure("1r"), register=[(0.0, 0.0), (4.0, 0.0)]
        )
    with pytest.raises(TypeError, match="interaction"):
        RydbergSystem(
            level_structure=level_structure("1r"), register=reg, interaction=874e9
        )


@pytest.mark.parametrize("name", ["1r", "01r"])
def test_symbolic_presets_reject_physical_kwargs(name):
    with pytest.raises(TypeError, match="does not accept physical parameter") as exc:
        level_structure(name, Delta_Hz=5.0e9)
    message = str(exc.value)
    assert "Delta_Hz" in message
    assert "Allowed parameters: none" in message


def test_rb87_7_rejects_unknown_physical_kwargs():
    # param_set (old manifold selector) and enable_polarization_leakage are
    # gone; both hit the generic unknown-parameter check.
    with pytest.raises(TypeError, match="param_set"):
        level_structure("rb87_7_mp", param_set="our")
    with pytest.raises(TypeError, match="param_set"):
        level_structure("rb87_7_pm", param_set="lukin")
    with pytest.raises(TypeError, match="enable_polarization_leakage"):
        level_structure("rb87_7_mp", enable_polarization_leakage=True)


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown level-structure"):
        level_structure("1er")


def test_removed_preset_names_raise():
    # The bare 'rb87_7' tag points at the manifold split; '01' is deleted.
    with pytest.raises(ValueError, match="rb87_7_mp"):
        level_structure("rb87_7")
    with pytest.raises(ValueError, match="removed"):
        level_structure("01")


# ── protocol binding ─────────────────────────────────────────────────────────


def test_protocol_binds_at_construction():
    proto = _to_protocol()
    system = RydbergSystem(
        level_structure=level_structure("analog_3", detuning_sign=1),
        register=Register.chain(2, spacing_um=5.0),
        protocol=proto,
    )

    assert system.protocol is proto


def test_with_protocol_returns_new_bound_system():
    proto = _to_protocol()
    s = RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(2, spacing_um=4.0),
    )
    s2 = s.with_protocol(proto)

    assert isinstance(s2, RydbergSystem)
    assert s2 is not s
    assert s2.protocol is proto
    assert s.protocol is None  # receiver untouched


# ── level_structure immutability ─────────────────────────────────────────────


def test_level_structure_results_are_immutable():
    ls = level_structure("1r")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ls.name = "other"
    analog = level_structure("analog_3")
    with pytest.raises(dataclasses.FrozenInstanceError):
        analog.Delta = 0.0
