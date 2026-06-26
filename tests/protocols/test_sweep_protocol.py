"""Tests for the function-defined SweepProtocol."""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.protocol_context import TNProtocolContext
from ryd_gate.protocols.sweep import SweepProtocol


def test_sweep_protocol_takes_no_x_parameters():
    proto = SweepProtocol(
        t_gate=2.0,
        omega_half_fn=lambda t: 0.5 * t,
        delta_fn=lambda t: 1.0 + t,
    )
    spec = create_tn_lattice_spec(1, 1)

    params = proto.unpack_params([], TNProtocolContext(spec))

    assert params["t_gate"] == 2.0
    assert np.isclose(params["Omega"], 2.0)
    assert np.isclose(params["Delta"], 3.0)
    with pytest.raises(ValueError, match="no x parameters"):
        proto.unpack_params([0.0], TNProtocolContext(spec))


def test_lattice_coefficients_use_user_functions():
    proto = SweepProtocol(
        t_gate=2.0,
        omega_half_fn=lambda t: 10.0 * t / 2.0,
        delta_fn=lambda t: 1.0 + 2.0 * t,
    )

    coeffs = proto.get_drive_coefficients(0.5, {})

    assert np.isclose(coeffs["global_X"], 2.5)
    assert np.isclose(coeffs["global_n"], -2.0)


def test_lattice_coefficients_accept_per_site_detuning():
    proto = SweepProtocol(
        t_gate=2.0,
        omega_half_fn=lambda t: 1.0,
        delta_fn=lambda t: 3.0,
        address_fn=lambda t, i: [-1.0, 1.0][i],
    )
    params = {"n_sites": 2}

    coeffs = proto.get_drive_coefficients(0.5, params)

    assert np.isclose(coeffs["global_n"], -3.0)
    assert np.isclose(coeffs["global_n_0"], 1.0)
    assert np.isclose(coeffs["global_n_1"], -1.0)
    assert np.isclose(-(coeffs["global_n"] + coeffs["global_n_0"]), 2.0)
    assert np.isclose(-(coeffs["global_n"] + coeffs["global_n_1"]), 4.0)


def test_delta_fn_must_be_scalar():
    proto = SweepProtocol(
        t_gate=2.0,
        omega_half_fn=lambda t: 1.0,
        delta_fn=lambda t: [1.0, 2.0, 3.0],
    )

    with pytest.raises(ValueError, match="scalar global"):
        proto.get_drive_coefficients(0.5, {"n_sites": 2})


def test_address_fn_requires_n_sites_for_coefficients():
    proto = SweepProtocol(
        t_gate=2.0,
        omega_half_fn=lambda t: 1.0,
        delta_fn=lambda t: 3.0,
        address_fn=lambda t, i: 0.5,
    )

    with pytest.raises(ValueError, match="n_sites"):
        proto.get_drive_coefficients(0.5, {})


def test_time_arguments_are_clamped_to_gate_window():
    proto = SweepProtocol(
        t_gate=2.0,
        omega_half_fn=lambda t: t,
        delta_fn=lambda t: t,
    )

    assert np.isclose(proto.omega_half_at(-1.0), 0.0)
    assert np.isclose(proto.delta_at(3.0), 2.0)


def test_phase_420_uses_integral_of_detuning_function():
    proto = SweepProtocol(
        t_gate=1.0,
        omega_half_fn=lambda t: 0.0,
        delta_fn=lambda t: 2.0 * t,
    )

    phase = proto.phase_420(0.5, {})

    np.testing.assert_allclose(phase, np.exp(-1j * 0.25), atol=1e-6)


def test_plot_smoke_uses_address_profile():
    import matplotlib

    matplotlib.use("Agg")
    from ryd_gate import InteractionSpec, RydbergSystem
    from ryd_gate.lattice import Register

    proto = SweepProtocol(
        t_gate=1.0,
        omega_half_fn=lambda t: 1.0,
        delta_fn=lambda t: 2.0,
        address_fn=lambda t, i: float(i),
        n_steps=4,
    )
    system = RydbergSystem.set_atom_level("1r").set_atom_geom(
        Register.rectangle(1, 2), interaction=InteractionSpec(C6=0.0)
    ).set_protocol(proto)

    fig_p, ax_p = proto.plot(system=system, show=False)
    fig_s, ax_s = proto.plot_address_map(system=system, show=False)

    assert len(fig_p.axes) == 1            # single overlaid pulse axis
    assert len(fig_s.axes) == 2            # heatmap + colorbar
