"""Tests for site-dependent DigitalAnalogProtocol profiles."""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate import RydbergSystem
from ryd_gate.backends.exact import simulate
from ryd_gate.backends.exact.compiler import compile_expm_ir
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.protocol_context import TNProtocolContext
from ryd_gate.core.level_structures import (
    InteractionSpec,
    LevelStructureSpec,
    TransitionSpec,
    three_level_profiles_from_coeffs,
    two_level_drive_and_detuning_from_coeffs,
)
from ryd_gate.ir import compile_hamiltonian_ir
from ryd_gate.lattice import Register
from ryd_gate.protocols.digital_analog import (
    DigitalAnalogProtocol,
    as_site_profile,
    is_scalar_profile,
)
from ryd_gate.protocols.sweep import SweepProtocol


def test_is_scalar_profile():
    assert is_scalar_profile(1.0)
    assert is_scalar_profile(np.float64(2.0))
    assert not is_scalar_profile([1.0, 0.0])
    assert not is_scalar_profile(np.array([1.0, 0.0]))


def test_as_site_profile_broadcast():
    np.testing.assert_allclose(as_site_profile(3.0, 2), [3.0, 3.0])
    np.testing.assert_allclose(as_site_profile([1.0, 2.0], 2), [1.0, 2.0])


def test_as_site_profile_wrong_length_raises():
    with pytest.raises(ValueError, match="length-2"):
        as_site_profile([1.0, 2.0, 3.0], 2)


def test_unpack_params_accepts_tn_context():
    proto = DigitalAnalogProtocol(t_gate=0.1, omega_R_fn=lambda t: 1.0)
    spec = create_tn_lattice_spec(2, 2)

    params = proto.unpack_params([], TNProtocolContext(spec))

    assert params == {"t_gate": 0.1, "n_sites": 4}


def test_tn_channel_mapping_for_sweep_protocol_on_1r_spec():
    spec = create_tn_lattice_spec(2, 2)
    proto = SweepProtocol(
        t_gate=0.1,
        omega_half_fn=lambda t: 0.5 * spec.Omega,
        delta_fn=lambda t: 3.0,
    )
    params = proto.unpack_params([], TNProtocolContext(spec))
    coeffs = proto.get_drive_coefficients(0.05, params)

    Omega, Delta, pin = two_level_drive_and_detuning_from_coeffs(coeffs, spec)

    assert np.isclose(Omega, spec.Omega)
    assert np.isclose(Delta, 3.0)
    assert pin is None


def test_tn_channel_mapping_combines_global_and_site_detuning_terms():
    spec = create_tn_lattice_spec(1, 2)
    proto = SweepProtocol(
        t_gate=0.1,
        omega_half_fn=lambda t: 0.5 * spec.Omega,
        delta_fn=lambda t: 3.0,
        address_fn=lambda t, i: [-1.0, 1.0][i],
    )
    params = proto.unpack_params([], TNProtocolContext(spec))
    coeffs = proto.get_drive_coefficients(0.05, params)

    Omega, Delta, pin = two_level_drive_and_detuning_from_coeffs(coeffs, spec)

    assert np.isclose(Omega, spec.Omega)
    assert np.isclose(Delta, 3.0)
    np.testing.assert_allclose(pin, [-1.0, 1.0])


def test_digital_analog_channels_rejected_on_1r_tn_spec():
    spec = create_tn_lattice_spec(1, 2)
    proto = DigitalAnalogProtocol(t_gate=0.1, omega_R_fn=lambda t: 1.0)
    params = proto.unpack_params([], TNProtocolContext(spec))
    coeffs = proto.get_drive_coefficients(0.05, params)

    with pytest.raises(ValueError, match="not declared"):
        two_level_drive_and_detuning_from_coeffs(coeffs, spec)


def test_tn_channel_mapping_rejects_hyperfine_drive():
    spec = create_tn_lattice_spec(1, 2, level_structure="01r")
    proto = DigitalAnalogProtocol(
        t_gate=0.1,
        omega_R_fn=lambda t: 1.0,
        omega_hf_fn=lambda t: 1.0,
    )
    params = proto.unpack_params([], TNProtocolContext(spec))
    coeffs = proto.get_drive_coefficients(0.05, params)

    with pytest.raises(ValueError, match="omega_hf"):
        two_level_drive_and_detuning_from_coeffs(coeffs, spec)


def test_three_level_tn_profiles_for_digital_analog_function_schedule():
    spec = create_tn_lattice_spec(1, 2, level_structure="01r")
    proto = DigitalAnalogProtocol(
        t_gate=0.1,
        omega_R_fn=lambda t: [2.0, 4.0],
        omega_hf_fn=lambda t: [6.0, 8.0],
        delta_R_fn=lambda t: [1.0, 2.0],
        delta_hf_fn=lambda t: [0.25, 0.5],
    )
    params = proto.unpack_params([], TNProtocolContext(spec))
    coeffs = proto.get_drive_coefficients(0.05, params)

    profiles = three_level_profiles_from_coeffs(coeffs, spec)

    np.testing.assert_allclose(profiles["omega_R"], [2.0, 4.0])
    np.testing.assert_allclose(profiles["omega_hf"], [6.0, 8.0])
    np.testing.assert_allclose(profiles["delta_R"], [1.0, 2.0])
    np.testing.assert_allclose(profiles["delta_hf"], [0.25, 0.5])


def test_three_level_tn_profiles_follow_shared_level_spec_channels():
    custom_level_spec = LevelStructureSpec(
        name="01r",
        levels=("0", "1", "r"),
        rydberg_levels=("r",),
        transitions=(
            TransitionSpec("R_custom", "1", "r", "rydberg_drive"),
            TransitionSpec("hf_custom", "0", "1", "hyperfine_drive"),
        ),
        detuning_levels={"rydberg_detuning": "r", "hyperfine_detuning": "1"},
    )
    spec = create_tn_lattice_spec(1, 2, level_structure=custom_level_spec)

    profiles = three_level_profiles_from_coeffs(
        {
            "rydberg_drive": 1.0,
            "hyperfine_drive": 2.0,
            "rydberg_detuning": -3.0,
            "hyperfine_detuning": -4.0,
        },
        spec,
    )

    np.testing.assert_allclose(profiles["omega_R"], [2.0, 2.0])
    np.testing.assert_allclose(profiles["omega_hf"], [4.0, 4.0])
    np.testing.assert_allclose(profiles["delta_R"], [3.0, 3.0])
    np.testing.assert_allclose(profiles["delta_hf"], [4.0, 4.0])


def test_drive_channels_scalar_uses_global():
    proto = DigitalAnalogProtocol(t_gate=0.1, omega_R_fn=lambda t: 1.0)
    system = RydbergSystem.from_lattice(
        Register.chain(2),
        "01r",
        interaction=InteractionSpec(C6=0.0),
        protocol=proto,
    )
    assert proto.drive_channels(system) == frozenset({"drive_R", "drive_hf", "delta_R", "delta_hf"})


def test_drive_channels_site_profile_uses_per_site():
    proto = DigitalAnalogProtocol(
        t_gate=0.1,
        omega_R_fn=lambda t: [1.0, 0.0],
        n_steps=20,
    )
    system = RydbergSystem.from_lattice(
        Register.chain(2),
        "01r",
        interaction=InteractionSpec(C6=0.0),
        protocol=proto,
    )
    channels = proto.drive_channels(system)
    assert "drive_R" not in channels
    assert "drive_R_0" in channels
    assert "drive_R_1" in channels


def test_site_dependent_omega_R_drives_one_site_only():
    omega = 2 * np.pi * 1e6
    t_pi2 = np.pi / (2 * omega)
    proto = DigitalAnalogProtocol(
        t_gate=t_pi2,
        omega_R_fn=lambda t: [omega, 0.0],
        n_steps=50,
    )
    system = RydbergSystem.from_lattice(
        Register.chain(2),
        "01r",
        interaction=InteractionSpec(C6=0.0),
        protocol=proto,
    )
    psi0 = system.product_state(["1", "1"])
    result = simulate(system, [], psi0, t_eval=True)
    psi_final = result.states[-1]

    n_r_0 = system.expectation("n_r_0", psi_final)
    n_r_1 = system.expectation("n_r_1", psi_final)
    assert np.isclose(n_r_0, 0.5, atol=0.05)
    assert n_r_1 < 0.05


def test_compile_expm_ir_includes_per_site_drive_terms():
    proto = DigitalAnalogProtocol(
        t_gate=0.1,
        omega_R_fn=lambda t: [1.0, 0.0],
        n_steps=10,
    )
    system = RydbergSystem.from_lattice(
        Register.chain(2),
        "01r",
        interaction=InteractionSpec(C6=0.0),
        protocol=proto,
    )
    params = system.unpack_params([])
    ham = compile_hamiltonian_ir(system, params)
    ir = compile_expm_ir(ham)
    names = {term.name for term in ir.drive_terms}
    assert "drive_R_0" in names
    assert "drive_R_1" in names
    assert "drive_R" not in names
