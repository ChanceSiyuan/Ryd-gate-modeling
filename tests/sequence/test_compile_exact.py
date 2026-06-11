"""Tests for SequenceProtocol lowering and compile_sequence_to_system."""

import pytest

from ryd_gate import DeviceSpec, Pulse, Register, RydbergSystem, Sequence, Waveform
from ryd_gate.protocols.sequence_protocol import SequenceProtocol, compile_sequence_to_system


def _seq(level="1r"):
    seq = Sequence(Register.chain(2, 4.0), DeviceSpec.virtual_rb87(), level)
    seq.declare_channel("ryd", "rydberg_global")
    return seq


class TestSequenceProtocol:
    def test_n_params_zero_and_param_validation(self):
        proto = SequenceProtocol(_seq())
        assert proto.n_params == 0
        proto.validate_params([])
        proto.validate_params(None)
        with pytest.raises(ValueError):
            proto.validate_params([0.1])

    def test_unpack_params(self):
        seq = _seq()
        seq.add(Pulse.constant(1000, 1.0, 0.0), "ryd")
        params = SequenceProtocol(seq).unpack_params([], None)
        assert params == {"t_gate": 1000 * 1e-9, "n_sites": 2}

    def test_channels_1r(self):
        seq = _seq("1r")
        seq.add(Pulse.constant(1000, 1.0, 0.0), "ryd")
        proto = SequenceProtocol(seq)
        assert proto.required_channels == {"global_X", "global_n"}
        assert proto.drive_channels(None) == {"global_X", "global_n"}

    def test_channels_01r(self):
        seq = _seq("01r")
        seq.add(Pulse.constant(1000, 1.0, 0.0), "ryd")
        proto = SequenceProtocol(seq)
        assert proto.required_channels == {"drive_R", "delta_R"}

    def test_coefficient_conventions(self):
        """amp channel = Omega/2 in rad/s; det channel = -Delta in rad/s."""
        seq = _seq("1r")
        seq.add(Pulse.constant(1000, 1.0, 2.0), "ryd")
        proto = SequenceProtocol(seq)
        coeffs = proto.get_drive_coefficients(0.5e-6, {})
        assert coeffs["global_X"] == pytest.approx(1.0e6 / 2.0)
        assert coeffs["global_n"] == pytest.approx(-2.0e6)

    def test_zero_outside_intervals(self):
        seq = _seq("1r")
        seq.add(Pulse.constant(1000, 1.0, 2.0), "ryd")
        proto = SequenceProtocol(seq)
        coeffs = proto.get_drive_coefficients(1.5e-6, {})
        assert coeffs["global_X"] == 0.0
        assert coeffs["global_n"] == 0.0

    def test_time_varying_waveform_lowering(self):
        seq = _seq("1r")
        seq.add(
            Pulse(
                amplitude=Waveform.ramp(1000, 0.0, 4.0),
                detuning=Waveform.constant(1000, 0.0),
            ),
            "ryd",
        )
        proto = SequenceProtocol(seq)
        coeffs = proto.get_drive_coefficients(0.25e-6, {})
        assert coeffs["global_X"] == pytest.approx(1.0e6 / 2.0)  # ramp value 1.0 at t=250ns

    def test_pulse_traces_physical_units(self):
        seq = _seq("1r")
        seq.add(Pulse.constant(1000, 1.0, -3.0), "ryd")
        traces = SequenceProtocol(seq).pulse_traces(0.5e-6, {})
        assert traces == {"ryd.amp": 1.0, "ryd.det": -3.0}


class TestCompile:
    def test_compile_1r_system(self):
        seq = _seq("1r")
        seq.add(Pulse.constant(1000, 1.0, 0.0), "ryd")
        system = compile_sequence_to_system(seq)
        assert isinstance(system, RydbergSystem)
        assert system.basis.local_levels == ("1", "r")
        assert system.geometry is seq.register  # Register IS the geometry (no copy)
        assert isinstance(system.protocol, SequenceProtocol)
        assert system.blocks.has("global_X") and system.blocks.has("global_n")

    def test_compile_01r_system(self):
        seq = _seq("01r")
        seq.add(Pulse.constant(1000, 1.0, 0.0), "ryd")
        system = compile_sequence_to_system(seq)
        assert system.basis.local_levels == ("0", "1", "r")
        assert system.blocks.has("drive_R") and system.blocks.has("delta_R")

    def test_phase_not_stage2(self):
        seq = _seq("1r")
        seq.add(Pulse.constant(1000, 1.0, 0.0, phase_rad=0.3), "ryd")
        with pytest.raises(NotImplementedError, match="phase_not_stage2"):
            compile_sequence_to_system(seq)
        seq2 = _seq("1r")
        seq2.add(Pulse.constant(1000, 1.0, 0.0, post_phase_shift_rad=0.3), "ryd")
        with pytest.raises(NotImplementedError, match="phase_not_stage2"):
            compile_sequence_to_system(seq2)

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="sequence.empty"):
            compile_sequence_to_system(_seq("1r"))
