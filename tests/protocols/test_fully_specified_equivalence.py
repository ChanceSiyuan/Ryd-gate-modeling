"""Old-seam -> fully-specified protocol equivalence characterization.

The pinned tables in ``_patch2_pins.py`` were generated from the pre-Patch-2
implementations (x-vector builders, four-field DigitalAnalog, protocol-owned
n_steps) immediately before the switch.  Each test constructs the equivalent
fully specified protocol and asserts the resolved ``t_gate`` and the drive
coefficients at five sample times reproduce the old numbers exactly.

Field mappings under test:

- ``TOProtocol``: old ``x = [A, omega/Omega_eff, phi0, delta/Omega_eff, theta,
  T/T_scale]`` -> ``phase_amplitude, frequency_ratio, phase_offset,
  detuning_ratio, duration_ratio`` (theta was a scoring parameter, not pulse
  shape — it is gone).
- ``ARProtocol``: old ``x = [omega/Omega_eff, A1, phi1, A2, phi2,
  delta/Omega_eff, T/T_scale, theta]`` -> ``frequency_ratio,
  phase_amplitude_1, phase_offset_1, phase_amplitude_2, phase_offset_2,
  detuning_ratio, duration_ratio``.
- ``CZProtocol``: old absolute ``t_gate`` -> ``duration_ratio = t_gate /
  T_scale(system)``.
- ``DigitalAnalogProtocol``: old full-Rabi/detuning fields -> direct matrix
  elements: ``coupling_r1 = omega_R/2``, ``coupling_10 = omega_hf/2``,
  ``energy_r = -delta_R``, ``energy_1 = -delta_hf`` (the emitted channel
  coefficients are unchanged).
"""

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.protocols import (
    ARProtocol,
    CZProtocol,
    DigitalAnalogProtocol,
    Direct297CZProtocol,
    Direct297PiProtocol,
    Direct297TOProtocol,
    SweepProtocol,
    TFIMAnnealProtocol,
    TFIMQuenchProtocol,
    TOProtocol,
)
from ryd_gate.protocols.gate_cz import cz_effective_rabi, cz_rabi_maxes

from ._patch2_pins import PINS

MHZ = 2 * np.pi * 1e6


def _assert_matches_pin(protocol, system, pin, rtol=1e-12):
    ctx = protocol._resolve(system)
    t_gate = float(ctx["t_gate"])
    assert t_gate == pytest.approx(pin["t_gate"], rel=rtol)
    for s_key, table in pin["coeffs"].items():
        t = float(s_key) * t_gate
        coeffs = protocol.get_drive_coefficients(t, ctx)
        assert sorted(coeffs) == sorted(table), (
            f"channel set changed at s={s_key}: {sorted(coeffs)} != {sorted(table)}"
        )
        for chan, (re, im) in table.items():
            val = complex(coeffs[chan])
            assert val.real == pytest.approx(re, rel=rtol, abs=1e-9), (chan, s_key)
            assert val.imag == pytest.approx(im, rel=rtol, abs=1e-9), (chan, s_key)


@pytest.fixture(scope="module")
def sys7():
    return RydbergSystem(
        level_structure=level_structure("rb87_7_mp", detuning_sign=1),
        register=Register.chain(2, spacing_um=3.0),
    )


def test_to_dark_equivalence(sys7):
    pin = PINS["to_dark_rb87_7_mp"]
    x = pin["x"]
    protocol = TOProtocol(
        phase_amplitude=x[0], frequency_ratio=x[1], phase_offset=x[2],
        detuning_ratio=x[3], duration_ratio=x[5],
    )
    _assert_matches_pin(protocol, sys7, pin)
    # The resolved duration is exposed read-only on the bound system.
    assert sys7.with_protocol(protocol).t_gate == pytest.approx(pin["t_gate"], rel=1e-12)


def test_to_flat_explicit_omegas_equivalence(sys7):
    pin = PINS["to_flat_pm_omegas"]
    x = PINS["to_dark_rb87_7_mp"]["x"]
    protocol = TOProtocol(
        phase_amplitude=x[0], frequency_ratio=x[1], phase_offset=x[2],
        detuning_ratio=x[3], duration_ratio=x[5],
        blackman=False,
        omega_420_max=2 * np.pi * 237e6, omega_1013_max=2 * np.pi * 303e6,
    )
    _assert_matches_pin(protocol, sys7, pin)


def test_ar_equivalence(sys7):
    pin = PINS["ar_our_rb87_7_mp"]
    x = pin["x"]
    protocol = ARProtocol(
        frequency_ratio=x[0],
        phase_amplitude_1=x[1], phase_offset_1=x[2],
        phase_amplitude_2=x[3], phase_offset_2=x[4],
        detuning_ratio=x[5], duration_ratio=x[6],
    )
    _assert_matches_pin(protocol, sys7, pin)


def test_cz_generic_duration_ratio_equivalence(sys7):
    pin = PINS["cz_generic_rb87_7_mp"]
    o420, o1013 = cz_rabi_maxes(sys7)
    _, time_scale = cz_effective_rabi(sys7, o420, o1013)
    assert time_scale == pytest.approx(PINS["scales_rb87_7_mp"]["time_scale"], rel=1e-12)
    protocol = CZProtocol(
        duration_ratio=pin["abs_t_gate"] / time_scale,
        A_420=lambda s: float(np.sin(np.pi * s) ** 2),
        phi_420=lambda s: 5.0 * s * s,
        A_1013=lambda s: 1.0 - 0.1 * s,
        phi_1013=lambda s: 0.3 * s,
    )
    _assert_matches_pin(protocol, sys7, pin)


@pytest.fixture(scope="module")
def sys297():
    return RydbergSystem(
        level_structure=level_structure("rb87_297_clock_4", magnetic_field_G=100.0),
        register=Register.chain(2, spacing_um=3.0),
    )


def test_direct297_pi_equivalence(sys297):
    pin = PINS["direct297_pi"]
    protocol = Direct297PiProtocol(
        t_gate=1.0e-6, power_at_atoms_w=0.56, beam_area_um2=420.0
    )
    _assert_matches_pin(protocol, sys297, pin)


def test_direct297_cz_equivalence(sys297):
    pin = PINS["direct297_cz"]
    protocol = Direct297CZProtocol(
        t_gate=1.0e-6,
        A_297=lambda s: float(np.sin(np.pi * s) ** 2),
        phi_297=lambda s: 2.0 * s,
        power_at_atoms_w=0.56,
        beam_area_um2=420.0,
    )
    _assert_matches_pin(protocol, sys297, pin)


def test_direct297_to_equivalence(sys297):
    pin = PINS["direct297_to"]
    x = pin["x"]
    protocol = Direct297TOProtocol(
        0.56, 420.0,
        phase_amplitude=x[0], frequency_ratio=x[1], phase_offset=x[2],
        detuning_ratio=x[3], duration_ratio=x[5],
    )
    _assert_matches_pin(protocol, sys297, pin)
    assert protocol.resolve_t_gate(sys297) == pytest.approx(
        pin["unpacked_t_gate"], rel=1e-12
    )


@pytest.fixture(scope="module")
def sys01r():
    return RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(2, spacing_um=3.0),
    )


def test_digital_analog_scalar_equivalence(sys01r):
    """Old four-field scalar schedule == new five-field matrix elements."""
    pin = PINS["da_scalar_01r"]
    Tda = 2.0e-7
    protocol = DigitalAnalogProtocol(
        t_gate=Tda,
        coupling_r1=lambda t: 0.5 * 3.0 * MHZ * float(np.sin(np.pi * t / Tda)),
        coupling_10=lambda t: 0.5 * 1.0 * MHZ,
        energy_r=lambda t: -2.0 * MHZ * (t / Tda),
        energy_1=lambda t: +0.5 * MHZ,
    )
    _assert_matches_pin(protocol, sys01r, pin)


def test_digital_analog_profile_equivalence(sys01r):
    """Old four-field site-profile schedule == new five-field matrix elements."""
    pin = PINS["da_profile_01r"]
    Tda = 2.0e-7
    protocol = DigitalAnalogProtocol(
        t_gate=Tda,
        coupling_r1=lambda t: 0.5 * np.array([2.0, 4.0]) * MHZ,
        coupling_10=lambda t: 0.5 * np.array([6.0, 8.0]) * MHZ * (t / Tda),
        energy_r=lambda t: -np.array([1.0, 2.0]) * MHZ,
        energy_1=lambda t: -np.array([0.25, 0.5]) * MHZ,
    )
    _assert_matches_pin(protocol, sys01r, pin)


def test_sweep_equivalence():
    pin = PINS["sweep_1r"]
    Tsw = 0.5e-6
    protocol = SweepProtocol(
        t_gate=Tsw,
        omega_half_fn=lambda t: 0.5 * 2.0 * MHZ * float(np.sin(np.pi * t / Tsw) ** 2),
        delta_fn=lambda t: -5.0 * MHZ + 10.0 * MHZ * (t / Tsw),
        address_fn=lambda t, i: 0.5 * MHZ * i * (t / Tsw),
    )
    system = RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(3, spacing_um=9.0),
    )
    _assert_matches_pin(protocol, system, pin)
    # Private phase-integration accuracy: the accumulated detuning phase table
    # must reproduce the old default-resolution values.
    for s_key, (re, im) in pin["phase_420"].items():
        val = complex(protocol.phase_420(float(s_key) * Tsw))
        assert val.real == pytest.approx(re, rel=1e-9, abs=1e-12)
        assert val.imag == pytest.approx(im, rel=1e-9, abs=1e-12)


def test_tfim_equivalence():
    system = RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.square(2, spacing_um=9.0),
    )
    quench = TFIMQuenchProtocol(hx=2 * np.pi * 1e6, t_gate=0.5e-6)
    _assert_matches_pin(quench, system, PINS["tfim_quench"])
    anneal = TFIMAnnealProtocol(
        hx_peak=2 * np.pi * 1e6, hz_initial=-2 * np.pi * 3e6,
        hz_final=2 * np.pi * 3e6, t_rise=0.2e-6, t_sweep=0.6e-6, t_fall=0.2e-6,
    )
    _assert_matches_pin(anneal, system, PINS["tfim_anneal"])
