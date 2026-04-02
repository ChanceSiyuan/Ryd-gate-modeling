"""Tests for Protocol.get_drive_coefficients() generalized interface."""

import numpy as np
import pytest

from ryd_gate.core.atomic_system import create_our_system, create_analog_system, create_lattice_system
from ryd_gate.protocols.gate_cz_to import TOProtocol
from ryd_gate.protocols.gate_cz_ar import ARProtocol
from ryd_gate.protocols.sweep import SweepProtocol


class TestTOProtocolChannels:
    """Test TOProtocol.get_drive_coefficients()."""

    def test_returns_correct_keys(self):
        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]
        params = proto.unpack_params(x, system)
        t = params["t_gate"] / 2
        coeffs = proto.get_drive_coefficients(t, params)
        assert set(coeffs.keys()) == {"drive_420", "drive_420_dag", "lightshift_zero"}

    def test_coefficients_are_complex(self):
        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]
        params = proto.unpack_params(x, system)
        t = params["t_gate"] / 2
        coeffs = proto.get_drive_coefficients(t, params)
        assert isinstance(coeffs["drive_420"], (complex, np.complexfloating, float, np.floating))

    def test_drive_420_dag_is_conjugate(self):
        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]
        params = proto.unpack_params(x, system)
        t = params["t_gate"] / 2
        coeffs = proto.get_drive_coefficients(t, params)
        np.testing.assert_allclose(
            coeffs["drive_420_dag"],
            np.conjugate(coeffs["drive_420"]),
            atol=1e-15,
        )

    def test_lightshift_is_amplitude_squared(self):
        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]
        params = proto.unpack_params(x, system)
        t = params["t_gate"] / 2  # Mid-pulse: Blackman ~ 1
        coeffs = proto.get_drive_coefficients(t, params)
        amp = abs(coeffs["drive_420"])
        np.testing.assert_allclose(coeffs["lightshift_zero"], amp ** 2, atol=1e-12)

    def test_phase_420_still_works(self):
        """Legacy phase_420() interface unchanged."""
        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]
        params = proto.unpack_params(x, system)
        phase = proto.phase_420(params["t_gate"] / 2, params)
        assert abs(phase) == pytest.approx(1.0, abs=1e-12)

    def test_required_channels(self):
        proto = TOProtocol()
        assert proto.required_channels == frozenset({"drive_420", "drive_420_dag", "lightshift_zero"})

    def test_unpack_params_includes_pulse_shape(self):
        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]
        params = proto.unpack_params(x, system)
        assert "t_rise" in params
        assert "blackmanflag" in params
        assert params["t_rise"] == system.t_rise
        assert params["blackmanflag"] == system.blackmanflag


class TestARProtocolChannels:
    """Test ARProtocol.get_drive_coefficients()."""

    def test_returns_correct_keys(self):
        system = create_our_system()
        proto = ARProtocol()
        x = [2.0, 1.0, 0.5, 0.5, 0.3, 0.1, 2.0, 0.0]
        params = proto.unpack_params(x, system)
        t = params["t_gate"] / 2
        coeffs = proto.get_drive_coefficients(t, params)
        assert set(coeffs.keys()) == {"drive_420", "drive_420_dag", "lightshift_zero"}

    def test_drive_420_dag_is_conjugate(self):
        system = create_our_system()
        proto = ARProtocol()
        x = [2.0, 1.0, 0.5, 0.5, 0.3, 0.1, 2.0, 0.0]
        params = proto.unpack_params(x, system)
        t = params["t_gate"] / 2
        coeffs = proto.get_drive_coefficients(t, params)
        np.testing.assert_allclose(
            coeffs["drive_420_dag"],
            np.conjugate(coeffs["drive_420"]),
            atol=1e-15,
        )

    def test_unpack_params_includes_pulse_shape(self):
        system = create_our_system()
        proto = ARProtocol()
        x = [2.0, 1.0, 0.5, 0.5, 0.3, 0.1, 2.0, 0.0]
        params = proto.unpack_params(x, system)
        assert "t_rise" in params
        assert "blackmanflag" in params


class TestSweepProtocolChannels:
    """Test SweepProtocol.get_drive_coefficients() for both system modes."""

    def test_atomic_mode_returns_420_channels(self):
        system = create_analog_system()
        proto = SweepProtocol()
        x = [-5.0, 5.0, 1.5]
        params = proto.unpack_params(x, system)
        t = params["t_gate"] / 2
        coeffs = proto.get_drive_coefficients(t, params)
        assert set(coeffs.keys()) == {"drive_420", "drive_420_dag", "lightshift_zero"}

    def test_atomic_mode_phase_matches_phase_420(self):
        system = create_analog_system()
        proto = SweepProtocol()
        x = [-5.0, 5.0, 1.5]
        params = proto.unpack_params(x, system)
        t = params["t_gate"] / 2
        coeffs = proto.get_drive_coefficients(t, params)
        phase = proto.phase_420(t, params)
        np.testing.assert_allclose(coeffs["drive_420"], phase, atol=1e-15)

    def test_lattice_mode_returns_lattice_channels(self):
        system = create_lattice_system(Lx=2, Ly=2)
        proto = SweepProtocol()
        x = [-5.0, 5.0, 1.5]
        params = proto.unpack_params(x, system)
        t = params["t_gate"] / 2
        coeffs = proto.get_drive_coefficients(t, params)
        assert set(coeffs.keys()) == {"global_X", "global_n"}

    def test_lattice_mode_detuning_interpolation(self):
        system = create_lattice_system(Lx=2, Ly=2)
        proto = SweepProtocol()
        x = [-5.0, 5.0, 1.5]
        params = proto.unpack_params(x, system)
        # At t=0: Delta = delta_start
        coeffs_start = proto.get_drive_coefficients(0, params)
        assert coeffs_start["global_n"] == pytest.approx(5.0, abs=1e-12)  # -delta_start
        # At t=t_gate: Delta = delta_end
        coeffs_end = proto.get_drive_coefficients(params["t_gate"], params)
        assert coeffs_end["global_n"] == pytest.approx(-5.0, abs=1e-12)  # -delta_end

    def test_lattice_mode_omega_ramp(self):
        system = create_lattice_system(Lx=2, Ly=2)
        proto = SweepProtocol(omega_ramp_frac=0.1)
        x = [-5.0, 5.0, 10.0]
        params = proto.unpack_params(x, system)
        # At t=0: Omega should be 0 (start of ramp)
        coeffs_0 = proto.get_drive_coefficients(0, params)
        assert coeffs_0["global_X"] == pytest.approx(0.0, abs=1e-12)
        # At t=t_gate/2: Omega should be full (past ramp)
        coeffs_mid = proto.get_drive_coefficients(params["t_gate"] / 2, params)
        assert coeffs_mid["global_X"] == pytest.approx(system.Omega / 2, abs=1e-12)

    def test_required_channels_includes_both_modes(self):
        proto = SweepProtocol()
        channels = proto.required_channels
        assert "drive_420" in channels
        assert "global_X" in channels
        assert "global_n" in channels

    def test_lattice_unpack_includes_omega(self):
        system = create_lattice_system(Lx=2, Ly=2)
        proto = SweepProtocol()
        x = [-5.0, 5.0, 1.5]
        params = proto.unpack_params(x, system)
        assert "Omega" in params
        assert params["Omega"] == system.Omega
