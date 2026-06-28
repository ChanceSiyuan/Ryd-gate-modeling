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
from ryd_gate.core.system import _build_from_lattice
from ryd_gate.lattice import Register
from ryd_gate.protocols.gate_cz import TOProtocol


def test_builder_chain_builds_usable_lattice():
    system = (
        RydbergSystem.set_atom_level("1r")
        .set_atom_geom(Register.chain(3, spacing_um=5.0))
        .build()
    )

    assert system.N == 3
    assert system.dim == 8
    assert system.blocks.has("H_vdw")
    assert system.protocol is None


def test_build_without_geom_is_single_atom():
    system = RydbergSystem.set_atom_level("1r").build()

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
    all_pairs = (
        RydbergSystem.set_atom_level("01r")
        .set_atom_geom(Register.rectangle(2, 2, spacing_um=6.0))
        .build()
    )
    nn_pairs = (
        RydbergSystem.set_atom_level("01r")
        .set_atom_geom(
            Register.rectangle(2, 2, spacing_um=6.0),
            interaction=InteractionSpec(C6=4.0, mode="nn"),
        )
        .build()
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
    default = (
        RydbergSystem.set_atom_level("analog_3", detuning_sign=1).set_atom_geom(geom).build()
    )
    via_bare_protocol = (
        RydbergSystem.set_atom_level("analog_3", detuning_sign=1)
        .set_atom_geom(geom)
        .set_protocol(TOProtocol())
    )

    assert via_bare_protocol.meta("rabi_eff") == pytest.approx(default.meta("rabi_eff"))


def test_analog_3_only_receives_user_supplied_level_flags():
    # The builder forwards only the flags the user actually passed, so an
    # analog_3 build never sees an rb87-only default (e.g. enable_polarization_leakage)
    # that ``_apply_analog_3_lattice_blocks`` would reject.
    system = (
        RydbergSystem.set_atom_level("analog_3", detuning_sign=1)
        .set_atom_geom(Register.chain(1, spacing_um=5.0))
        .build()
    )

    assert system.meta("physical_model") == "analog_3"


def test_builder_matches_direct_engine():
    geom = Register.chain(2, spacing_um=5.0)
    proto = TOProtocol()
    laser = dict(Delta_Hz=5.0e9, rabi_420_Hz=300e6, rabi_1013_Hz=200e6)

    built = (
        RydbergSystem.set_atom_level("analog_3", detuning_sign=1, **laser)
        .set_atom_geom(geom)
        .set_protocol(proto)
    )
    direct = _build_from_lattice(
        RydbergSystem,
        geom,
        "analog_3",
        None,
        protocol=proto,
        param_set=None,
        Omega=1.0,
        detuning_sign=1,
        **laser,
    )

    assert built.basis.local_levels == direct.basis.local_levels
    assert set(built.blocks.list()) == set(direct.blocks.list())
    assert built.meta("rabi_eff") == pytest.approx(direct.meta("rabi_eff"))
    assert built.meta("Delta") == pytest.approx(direct.meta("Delta"))


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
            return frozenset({"drive_420"})

        def get_drive_coefficients(self, t, params):
            return {"drive_420": 1.0}

    system = (
        RydbergSystem.set_atom_level("analog_3", detuning_sign=1)
        .set_atom_geom(Register.chain(1, spacing_um=5.0))
        .set_protocol(_Duck())
    )

    assert system.meta("physical_model") == "analog_3"
    assert isinstance(system.protocol, _Duck)


@pytest.mark.slow
def test_builder_rb87_7_default_operating_point():
    system = (
        RydbergSystem.set_atom_level("rb87_7", param_set="our")
        .set_atom_geom(Register.chain(2, spacing_um=3.0))
        .build()
    )

    assert system.basis.local_dim == 7
    assert system.blocks.has("H_const")
    assert system.blocks.has("drive_420")
    assert system.blocks.has("drive_1013")
    assert system.meta("Delta") != 0
    # The Rabi scale now lives in the CZ protocol, not in system metadata.
    assert system.meta("rabi_eff") is None
