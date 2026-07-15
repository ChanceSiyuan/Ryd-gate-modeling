"""Tests for site-dependent DigitalAnalogProtocol profiles (matrix-element API)."""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate import RydbergSystem, level_structure
from ryd_gate.backends.exact import simulate
from ryd_gate.backends.exact.compiler import _compile_exact_ir
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.protocol_context import TNProtocolContext
from ryd_gate.core.level_structures import (
    InteractionSpec,
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


def test_resolve_accepts_tn_context():
    proto = DigitalAnalogProtocol(t_gate=0.1, coupling_r1=lambda t: 1.0)
    spec = create_tn_lattice_spec(2, 2)

    params = proto._resolve(TNProtocolContext(spec))

    assert params == {"t_gate": 0.1, "n_sites": 4}


def test_tn_channel_mapping_for_sweep_protocol_on_1r_spec():
    spec = create_tn_lattice_spec(2, 2)
    proto = SweepProtocol(
        t_gate=0.1,
        omega_half_fn=lambda t: 0.5 * spec.Omega,
        delta_fn=lambda t: 3.0,
    )
    params = proto._resolve(TNProtocolContext(spec))
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
    params = proto._resolve(TNProtocolContext(spec))
    coeffs = proto.get_drive_coefficients(0.05, params)

    Omega, Delta, pin = two_level_drive_and_detuning_from_coeffs(coeffs, spec)

    assert np.isclose(Omega, spec.Omega)
    assert np.isclose(Delta, 3.0)
    np.testing.assert_allclose(pin, [-1.0, 1.0])


def test_digital_analog_undeclared_channels_rejected_on_1r_tn_spec():
    # 1r declares no |0> level, so a hyperfine E[1,0] coefficient is unknown.
    spec = create_tn_lattice_spec(1, 2)
    proto = DigitalAnalogProtocol(t_gate=0.1, coupling_10=lambda t: 1.0)
    params = proto._resolve(TNProtocolContext(spec))
    coeffs = proto.get_drive_coefficients(0.05, params)

    with pytest.raises(ValueError, match="not declared"):
        two_level_drive_and_detuning_from_coeffs(coeffs, spec)


def test_tn_two_level_reduction_rejects_hyperfine_drive():
    spec = create_tn_lattice_spec(1, 2, level_structure="01r")
    proto = DigitalAnalogProtocol(
        t_gate=0.1,
        coupling_r1=lambda t: 0.5,
        coupling_10=lambda t: 0.5,
    )
    params = proto._resolve(TNProtocolContext(spec))
    coeffs = proto.get_drive_coefficients(0.05, params)

    with pytest.raises(ValueError, match="coupling_10"):
        two_level_drive_and_detuning_from_coeffs(coeffs, spec)


def test_tn_two_level_reduction_rejects_k0r_drive():
    spec = create_tn_lattice_spec(1, 2, level_structure="01r")
    proto = DigitalAnalogProtocol(
        t_gate=0.1,
        coupling_r1=lambda t: 0.5,
        coupling_r0=lambda t: 0.5,
    )
    params = proto._resolve(TNProtocolContext(spec))
    coeffs = proto.get_drive_coefficients(0.05, params)

    with pytest.raises(ValueError, match="coupling_r0"):
        two_level_drive_and_detuning_from_coeffs(coeffs, spec)


def test_three_level_profiles_support_k0r_matrix_element():
    # The |0>-|r> (K0r) leg lowers to a per-site matrix-element profile on the
    # TN 01r path (the old K0r rejection guard is gone).
    spec = create_tn_lattice_spec(1, 2, level_structure="01r")
    proto = DigitalAnalogProtocol(
        t_gate=0.1,
        coupling_r0=lambda t: 0.25 + 0.5j,
    )
    params = proto._resolve(TNProtocolContext(spec))
    coeffs = proto.get_drive_coefficients(0.05, params)

    profiles = three_level_profiles_from_coeffs(coeffs, spec)

    np.testing.assert_allclose(profiles["coupling_r0"], [0.25 + 0.5j, 0.25 + 0.5j])
    np.testing.assert_allclose(profiles["coupling_r1"], 0.0)


def test_three_level_tn_profiles_for_digital_analog_function_schedule():
    spec = create_tn_lattice_spec(1, 2, level_structure="01r")
    proto = DigitalAnalogProtocol(
        t_gate=0.1,
        coupling_r1=lambda t: [1.0, 2.0 + 1.0j],
        coupling_10=lambda t: [3.0, 4.0],
        coupling_r0=lambda t: [0.5j, 0.0],
        energy_r=lambda t: [-1.0, -2.0],
        energy_1=lambda t: [-0.25, -0.5],
    )
    params = proto._resolve(TNProtocolContext(spec))
    coeffs = proto.get_drive_coefficients(0.05, params)

    profiles = three_level_profiles_from_coeffs(coeffs, spec)

    np.testing.assert_allclose(profiles["coupling_r1"], [1.0, 2.0 + 1.0j])
    np.testing.assert_allclose(profiles["coupling_10"], [3.0, 4.0])
    np.testing.assert_allclose(profiles["coupling_r0"], [0.5j, 0.0])
    np.testing.assert_allclose(profiles["energy_r"], [-1.0, -2.0])
    np.testing.assert_allclose(profiles["energy_1"], [-0.25, -0.5])


def test_drive_channels_scalar_uses_global():
    proto = DigitalAnalogProtocol(t_gate=0.1, coupling_r1=lambda t: 1.0)
    system = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(2),
        interaction=InteractionSpec(C6=0.0),
        protocol=proto,
    )
    # Absent control functions declare no channels.
    assert proto.drive_channels(system) == frozenset({"E[r,1]"})


def test_drive_channels_site_profile_uses_per_site():
    proto = DigitalAnalogProtocol(
        t_gate=0.1,
        coupling_r1=lambda t: [1.0, 0.0],
    )
    system = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(2),
        interaction=InteractionSpec(C6=0.0),
        protocol=proto,
    )
    channels = proto.drive_channels(system)
    assert "E[r,1]" not in channels
    assert "E[r,1]_0" in channels
    assert "E[r,1]_1" in channels


def test_site_dependent_coupling_drives_one_site_only():
    omega = 2 * np.pi * 1e6
    t_pi2 = np.pi / (2 * omega)
    proto = DigitalAnalogProtocol(
        t_gate=t_pi2,
        coupling_r1=lambda t: [0.5 * omega, 0.0],
    )
    system = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(2),
        interaction=InteractionSpec(C6=0.0),
        protocol=proto,
    )
    psi0 = system.product_state(["1", "1"])
    obs = system.observables
    result = simulate(
        system, psi0,
        observables={"n_r_0": obs.n("r", 0), "n_r_1": obs.n("r", 1)},
    )

    n_r_0 = result.expectation("n_r_0")[0].real
    n_r_1 = result.expectation("n_r_1")[0].real
    assert np.isclose(n_r_0, 0.5, atol=0.05)
    assert n_r_1 < 0.05


def test_exact_ir_includes_per_site_drive_terms():
    proto = DigitalAnalogProtocol(
        t_gate=0.1,
        coupling_r1=lambda t: [1.0, 0.0],
    )
    system = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(2),
        interaction=InteractionSpec(C6=0.0),
        protocol=proto,
    )
    ham = compile_hamiltonian_ir(system)
    ir = _compile_exact_ir(ham)
    names = {term.name for term in ir.drive_terms}
    assert "E[r,1]_0" in names
    assert "E[r,1]_1" in names
    assert "E[r,1]" not in names
