"""Tests for the RydbergSystem fluent builder.

``set_atom_level(...)`` → ``set_atom_geom(...)`` → ``set_protocol(...)`` replaces
the old monolithic ``from_lattice`` constructor.  Atom-level flags and the laser
parameters (``Delta_Hz`` and, for analog_3, ``rabi_420_Hz``/``rabi_1013_Hz``) live
on ``set_atom_level``; the geometry + Rydberg interaction on ``set_atom_geom``; and
the pulse protocol enters at ``set_protocol``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate import RydbergSystem
from ryd_gate.core.level_structures import InteractionSpec
from ryd_gate.core.physical_models import analog_3_local_blocks
from ryd_gate.core.system import _ATOM_LEVELS
from ryd_gate.lattice import Register
from ryd_gate.protocols.gate_cz import TOProtocol


def test_builder_chain_builds_usable_lattice():
    system = RydbergSystem.set_atom_level("1r").set_atom_geom(
        Register.chain(3, spacing_um=5.0)
    )

    assert system.N == 3
    assert system.dim == 8
    assert any(t.name == "H_pair" for t in system.static_hamiltonian_terms)
    assert system.protocol is None


def test_build_without_geom_is_single_atom():
    system = RydbergSystem.set_atom_level("1r")

    assert system.N == 1
    assert system.dim == 2


def test_set_protocol_is_terminal_and_binds_protocol():
    proto = TOProtocol()
    system = (
        RydbergSystem.set_atom_level("analog_3", detuning_sign=1)
        .set_atom_geom(Register.chain(2, spacing_um=5.0))
        .set_protocol(proto)
    )

    assert isinstance(system, RydbergSystem)
    assert system.protocol is proto


def test_interaction_is_forwarded_to_geom():
    all_pairs = RydbergSystem.set_atom_level("01r").set_atom_geom(
        Register.rectangle(2, 2, spacing_um=6.0)
    )
    nn_pairs = RydbergSystem.set_atom_level("01r").set_atom_geom(
        Register.rectangle(2, 2, spacing_um=6.0),
        interaction=InteractionSpec(C6=4.0, mode="nn"),
    )

    assert len(all_pairs.meta("interaction_pairs")) == 6  # C(4, 2)
    assert len(nn_pairs.meta("interaction_pairs")) == 4  # square edges only


def test_laser_params_on_set_atom_level_set_the_operating_point():
    geom = Register.chain(2, spacing_um=5.0)
    system = (
        RydbergSystem.set_atom_level(
            "analog_3", detuning_sign=1, Delta_Hz=5.0e9, rabi_420_Hz=300e6, rabi_1013_Hz=200e6
        )
        .set_atom_geom(geom)
        .set_protocol(TOProtocol())
    )

    expected = (2 * np.pi * 300e6) * (2 * np.pi * 200e6) / (2 * abs(2 * np.pi * 5.0e9))
    assert system.meta("rabi_eff") == pytest.approx(expected)
    assert system.meta("Delta") == pytest.approx(2 * np.pi * 5.0e9)


def test_bare_protocol_falls_back_to_preset_defaults():
    geom = Register.chain(2, spacing_um=5.0)
    default = RydbergSystem.set_atom_level("analog_3", detuning_sign=1).set_atom_geom(geom)
    via_bare_protocol = (
        RydbergSystem.set_atom_level("analog_3", detuning_sign=1)
        .set_atom_geom(geom)
        .set_protocol(TOProtocol())
    )

    assert via_bare_protocol.meta("rabi_eff") == pytest.approx(default.meta("rabi_eff"))


def test_analog_3_only_receives_user_supplied_level_flags():
    # The builder forwards only the flags the user actually passed, so an
    # analog_3 build never sees an rb87-only default (e.g. magnetic_field_G)
    # that ``_apply_analog_3_lattice_blocks`` would reject.
    system = RydbergSystem.set_atom_level("analog_3", detuning_sign=1).set_atom_geom(
        Register.chain(1, spacing_um=5.0)
    )

    assert system.meta("physical_model") == "analog_3"


def test_builder_operating_point_matches_source_of_truth():
    # The public fluent build must reproduce the analog_3 operating point that
    # ``analog_3_local_blocks`` (the single source of truth) computes from the
    # same laser knobs.
    geom = Register.chain(2, spacing_um=5.0)
    laser = dict(Delta_Hz=5.0e9, rabi_420_Hz=300e6, rabi_1013_Hz=200e6)

    built = (
        RydbergSystem.set_atom_level("analog_3", detuning_sign=1, **laser)
        .set_atom_geom(geom)
        .set_protocol(TOProtocol())
    )
    blk = analog_3_local_blocks(detuning_sign=1, **laser)

    assert built.basis.local_levels == ("g", "e", "r")
    assert built.meta("physical_model") == "analog_3"
    assert built.meta("rabi_eff") == pytest.approx(blk.rabi_eff)
    assert built.meta("Delta") == pytest.approx(blk.Delta)
    assert built.static_hamiltonian_terms  # diagonal g/e/r energies
    assert "E[e,g]" in built.hamiltonian_channels


def test_duck_typed_protocol_without_laser_kwargs():
    # Protocols need not subclass Protocol; the builder must not assume
    # laser_kwargs() exists on whatever object is passed to set_protocol.
    class _Duck:
        n_params = 0

        def validate_params(self, x):
            pass

        def unpack_params(self, x, system):
            return {"t_gate": 0.1}

        def drive_channels(self, system):
            return frozenset({"E[e,g]"})

        def get_drive_coefficients(self, t, params):
            return {"E[e,g]": 1.0}

    system = (
        RydbergSystem.set_atom_level("analog_3", detuning_sign=1)
        .set_atom_geom(Register.chain(1, spacing_um=5.0))
        .set_protocol(_Duck())
    )

    assert system.meta("physical_model") == "analog_3"
    assert isinstance(system.protocol, _Duck)


@pytest.mark.slow
def test_builder_rb87_7_default_operating_point():
    system = RydbergSystem.set_atom_level("rb87_7_mp").set_atom_geom(
        Register.chain(2, spacing_um=3.0)
    )

    assert system.basis.local_dim == 7
    assert system.static_hamiltonian_terms  # static diagonal energies
    assert "E[e1,1]" in system.hamiltonian_channels  # 420 leg
    assert "E[r,e1]" in system.hamiltonian_channels  # 1013 leg
    assert system.meta("Delta") != 0
    # The Rabi scale now lives in the CZ protocol, not in system metadata.
    assert system.meta("rabi_eff") is None


def test_atom_level_table_lists_builtins():
    assert set(_ATOM_LEVELS) == {"01", "1r", "01r", "analog_3", "rb87_7_mp", "rb87_7_pm"}
    for entry in _ATOM_LEVELS.values():
        assert entry["kind"] in {"symbolic", "analog_3", "rb87_7"}
        assert isinstance(entry["level_kwargs"], frozenset)
        assert isinstance(entry["description"], str) and entry["description"]


@pytest.mark.parametrize("name", ["01", "1r", "01r"])
def test_symbolic_models_reject_atom_level_kwargs(name):
    # Validation is eager: it fires at set_atom_level, before any geometry.
    with pytest.raises(TypeError, match="does not accept atom-level parameter") as exc:
        RydbergSystem.set_atom_level(name, Delta_Hz=5.0e9)
    message = str(exc.value)
    assert "Delta_Hz" in message
    assert "Allowed parameters: none" in message


def test_old_rb87_api_is_rejected():
    # The bare 'rb87_7' tag and the param_set kwarg are both gone (breaking change):
    # the dropped tag is an unknown level structure (ValueError); the dropped kwarg
    # is now an unknown atom-level parameter, rejected by the generic kwarg check.
    with pytest.raises(ValueError, match="rb87_7_mp"):
        RydbergSystem.set_atom_level("rb87_7")
    with pytest.raises(TypeError, match="param_set"):
        RydbergSystem.set_atom_level("rb87_7_mp", param_set="our")
    with pytest.raises(TypeError, match="param_set"):
        RydbergSystem.set_atom_level("rb87_7_pm", param_set="lukin")


@pytest.mark.parametrize("tag,manifold", [("rb87_7_mp", "mp"), ("rb87_7_pm", "pm")])
def test_rb87_7_manifold_tag_builds_seven_level(tag, manifold):
    s = RydbergSystem.set_atom_level(tag)
    assert s.basis.local_dim == 7
    assert s.basis.local_levels == ("0", "1", "e1", "e2", "e3", "r", "r_garb")
    assert s.protocol is None
    assert s.static_hamiltonian_terms  # static diagonal energies
    assert "E[e1,1]" in s.hamiltonian_channels  # 420 leg
    assert "E[r,e1]" in s.hamiltonian_channels  # 1013 leg
    assert s.meta("rb87_manifold") == manifold
    assert s.meta("level_structure") == tag


def test_rb87_7_static_overrides():
    s = RydbergSystem.set_atom_level(
        "rb87_7_mp", Delta_Hz=12e9, ryd_level=80, C6_rad_s_um6=2 * np.pi * 900e9, t_rise=30e-9
    )
    assert s.meta("Delta") == pytest.approx(2 * np.pi * 12e9)
    assert s.meta("ryd_level") == 80
    assert s.meta("t_rise") == pytest.approx(30e-9)
    # C6 override flows into the default rb87 interaction (v_ryd at the nominal 3 um).
    s2 = s.set_atom_geom(Register.chain(2, spacing_um=3.0))
    assert s2.meta("v_ryd") == pytest.approx(2 * np.pi * 900e9 / 3**6)


@pytest.mark.slow
def test_rb87_7_magnetic_field_sets_zeeman_shift():
    # The garbage-Rydberg splitting is now a physical linear Zeeman shift set by
    # magnetic_field_G: 20 G -> 2pi * 56 MHz (g_J=2, Delta_mj=1 for nS_1/2).
    s = RydbergSystem.set_atom_level("rb87_7_mp", detuning_sign=1, magnetic_field_G=20.0)

    assert s.meta("magnetic_field_G") == 20.0
    assert s.meta("ryd_zeeman_shift") / (2 * np.pi * 1e6) == pytest.approx(56.0, rel=2e-3)


def test_rb87_7_rejects_enable_polarization_leakage():
    # The old boolean is gone; passing it hits the generic unknown-parameter check
    # (eager, before any ARC/build), not the old fake far-detuning behavior.
    with pytest.raises(TypeError, match="does not accept atom-level parameter"):
        RydbergSystem.set_atom_level("rb87_7_mp", enable_polarization_leakage=True)


def test_set_atom_level_returns_materialized_system():
    # set_atom_level alone yields a complete, usable single-atom system: default
    # geometry (chain of 1), no interaction pairs, no protocol.
    s = RydbergSystem.set_atom_level("rb87_7_mp")

    assert isinstance(s, RydbergSystem)
    assert s.N == 1
    assert s.protocol is None
    assert s.model_tag == "rb87_7_mp"
    assert s.meta("interaction_pairs") == ()
    assert s.static_hamiltonian_terms  # static diagonal energies
    assert "E[e1,1]" in s.hamiltonian_channels  # 420 leg
    assert "E[r,e1]" in s.hamiltonian_channels  # 1013 leg


def test_set_atom_geom_returns_materialized_system():
    s = RydbergSystem.set_atom_level("rb87_7_mp")
    s2 = s.set_atom_geom(Register.chain(2, spacing_um=3.0))

    assert isinstance(s2, RydbergSystem)
    assert s2.N == 2
    assert len(s2.meta("interaction_pairs")) == 1
    # set_atom_geom preserves the atom-level config and leaves the receiver intact.
    assert s2.model_tag == "rb87_7_mp"
    assert s.N == 1
    assert s.meta("interaction_pairs") == ()


def test_set_protocol_binds_and_returns_usable_system():
    proto = TOProtocol()
    s = RydbergSystem.set_atom_level("rb87_7_mp").set_atom_geom(Register.chain(2, spacing_um=3.0))
    s3 = s.set_protocol(proto)

    assert isinstance(s3, RydbergSystem)
    assert s3.protocol is proto
    assert s.protocol is None  # receiver untouched
