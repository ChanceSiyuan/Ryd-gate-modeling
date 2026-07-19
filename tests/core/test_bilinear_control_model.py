"""Bilinear control model export: structure, plain types, and the parity contract (ADR-0024)."""

import numpy as np
import pytest
from scipy.linalg import expm

from ryd_gate import Register, RydbergSystem, bilinear_control_model, level_structure, simulate
from ryd_gate.core.lowering import lower_drives
from ryd_gate.protocols import SweepProtocol

MHZ = 2 * np.pi * 1e6


def _constant_1r_system(omega_half=0.6 * MHZ, detuning=-0.35 * MHZ, t_gate=0.7e-6):
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(1),
        protocol=SweepProtocol(
            t_gate_s=t_gate,
            omega_half_rad_s=lambda t: omega_half,
            detuning_rad_s=lambda t: detuning,
        ),
    )


def test_export_structure_and_plain_types():
    """The export is quadrature-split, Hermitian, and made of plain arrays and mappings only."""
    h0, controls, states = bilinear_control_model(_constant_1r_system(), states=[["1"], ["r"]])

    assert type(h0) is np.ndarray and h0.shape == (2, 2)
    np.testing.assert_allclose(h0, h0.conj().T, atol=1e-12)
    assert type(controls) is dict
    assert set(controls) == {"E[r,1]:x", "E[r,1]:y", "E[r,r]"}
    for name, operator in controls.items():
        assert isinstance(name, str) and type(operator) is np.ndarray and operator.shape == (2, 2)
        np.testing.assert_allclose(operator, operator.conj().T, atol=1e-12, err_msg=name)
    assert type(states) is dict and set(states) == {("1",), ("r",)}
    for vector in states.values():
        assert type(vector) is np.ndarray and vector.shape == (2,)
        np.testing.assert_allclose(np.linalg.norm(vector), 1.0, atol=1e-12)


def test_constant_control_parity_with_simulate():
    """The parity contract: propagating the exported model with the lowered constant
    channel values reproduces ``simulate``'s endpoint amplitudes. This pins basis
    ordering, rad/s units, and the quadrature conventions without exported metadata."""
    t_gate = 0.7e-6
    system = _constant_1r_system(t_gate=t_gate)
    h0, controls, states = bilinear_control_model(system, states=[["1"], ["r"]])

    _t, lowered = lower_drives(system)
    coefficient = {entry.channel: complex(entry.coefficient(0.31 * t_gate)) for entry in lowered}

    h = (
        h0
        + coefficient["E[r,1]"].real * controls["E[r,1]:x"]
        + coefficient["E[r,1]"].imag * controls["E[r,1]:y"]
        + coefficient["E[r,r]"].real * controls["E[r,r]"]
    )
    psi = expm(-1j * h * t_gate) @ states[("1",)]

    result = simulate(system, ["1"])
    for labels in (["1"], ["r"]):
        index = int(np.argmax(np.abs(states[tuple(labels)])))
        assert abs(psi[index] - result.amplitude(labels)) < 1e-6


def test_complex_drive_parity_pins_the_imaginary_quadrature():
    """The real-drive parity case leaves the :y operator multiplied by zero, so a
    wrong sign on 1j*(A - A^H) or on u_y = Im c would pass it. This constant
    COMPLEX drive makes both quadratures load-bearing against ``simulate``."""
    from ryd_gate.protocols import DigitalAnalogProtocol

    t_gate = 0.6e-6
    coupling = (0.5 + 0.35j) * MHZ
    system = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(1),
        protocol=DigitalAnalogProtocol(t_gate_s=t_gate, coupling_r1_rad_s=lambda t: coupling),
    )
    h0, controls, states = bilinear_control_model(system, states=[["1"], ["r"]])
    assert {"E[r,1]:x", "E[r,1]:y"} <= set(controls)

    _t, lowered = lower_drives(system)
    coefficient = {entry.channel: complex(entry.coefficient(0.5 * t_gate)) for entry in lowered}
    assert abs(coefficient["E[r,1]"].imag) > 0.0  # the y quadrature must actually engage

    h = h0.copy()
    for label, value in coefficient.items():
        if f"{label}:x" in controls:
            h = h + value.real * controls[f"{label}:x"] + value.imag * controls[f"{label}:y"]
        else:
            h = h + value.real * controls[label]
    psi = expm(-1j * h * t_gate) @ states[("1",)]

    result = simulate(system, ["1"])
    for labels in (["1"], ["r"]):
        index = int(np.argmax(np.abs(states[tuple(labels)])))
        assert abs(psi[index] - result.amplitude(labels)) < 1e-6


def test_plus_state_export_matches_the_product_states():
    system = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(1),
        protocol=SweepProtocol(t_gate_s=1e-6, omega_half_rad_s=lambda t: 0.0, detuning_rad_s=lambda t: 0.0),
    )
    _h0, _controls, states = bilinear_control_model(system, states=[["0"], ["1"], "plus"])
    np.testing.assert_allclose(states["plus"], (states[("0",)] + states[("1",)]) / np.sqrt(2.0), atol=1e-12)


def test_noise_realizations_are_refused():
    system = _constant_1r_system()
    system._realization = {"laser_amplitude_scales": {}}
    with pytest.raises(ValueError, match="noiseless"):
        bilinear_control_model(system)


def test_per_site_coefficients_are_refused():
    from ryd_gate.protocols import DigitalAnalogProtocol

    w = MHZ
    protocol = DigitalAnalogProtocol(
        t_gate_s=0.2e-6,
        coupling_r1_rad_s=lambda t: [1.0 * w, (0.8 + 0.5j) * w],
        coupling_10_rad_s=lambda t: 0.6 * w,
        coupling_r0_rad_s=lambda t: [0.4j * w, 0.25 * w],
        energy_r_rad_s=lambda t: [-0.7 * w, 0.5 * w],
        energy_1_rad_s=lambda t: 0.3 * w,
    )
    system = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(2, spacing_um=8.0),
        protocol=protocol,
    )
    with pytest.raises(ValueError, match="per-site"):
        bilinear_control_model(system)


def test_state_entries_are_validated():
    system = _constant_1r_system()
    with pytest.raises(ValueError, match="one label string per site"):
        bilinear_control_model(system, states=[["1", "1"]])
    with pytest.raises(ValueError, match="'plus'"):
        bilinear_control_model(system, states=["minus"])
