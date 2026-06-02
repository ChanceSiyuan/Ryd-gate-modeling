"""Tests for site-dependent DigitalAnalogProtocol profiles."""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate import RydbergSystem, simulate
from ryd_gate.compilers.exact_sparse import compile_expm_ir
from ryd_gate.model.system import InteractionSpec
from ryd_gate.lattice import make_chain
from ryd_gate.protocols.digital_analog import (
    DigitalAnalogProtocol,
    Segment,
    as_site_profile,
    is_scalar_profile,
)


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


def test_drive_channels_scalar_uses_global():
    proto = DigitalAnalogProtocol.constant(omega_R=1.0, t_gate=0.1)
    system = RydbergSystem.from_lattice(
        make_chain(2), "01r", interaction=InteractionSpec(C6=0.0), protocol=proto,
    )
    assert proto.drive_channels(system) == frozenset(
        {"drive_R", "drive_hf", "delta_R", "delta_hf"}
    )


def test_drive_channels_site_profile_uses_per_site():
    proto = DigitalAnalogProtocol(
        [Segment(duration=0.1, omega_R=[1.0, 0.0])],
        n_steps=20,
    )
    system = RydbergSystem.from_lattice(
        make_chain(2), "01r", interaction=InteractionSpec(C6=0.0), protocol=proto,
    )
    channels = proto.drive_channels(system)
    assert "drive_R" not in channels
    assert "drive_R_0" in channels
    assert "drive_R_1" in channels


def test_site_dependent_omega_R_drives_one_site_only():
    omega = 2 * np.pi * 1e6
    t_pi2 = np.pi / (2 * omega)
    proto = DigitalAnalogProtocol(
        [Segment(duration=t_pi2, omega_R=[omega, 0.0])],
        n_steps=50,
    )
    system = RydbergSystem.from_lattice(
        make_chain(2), "01r", interaction=InteractionSpec(C6=0.0), protocol=proto,
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
        [Segment(duration=0.1, omega_R=[1.0, 0.0])],
        n_steps=10,
    )
    system = RydbergSystem.from_lattice(
        make_chain(2), "01r", interaction=InteractionSpec(C6=0.0), protocol=proto,
    )
    params = system.unpack_params([])
    ir = compile_expm_ir(system, params)
    names = {term.name for term in ir.drive_terms}
    assert "drive_R_0" in names
    assert "drive_R_1" in names
    assert "drive_R" not in names
