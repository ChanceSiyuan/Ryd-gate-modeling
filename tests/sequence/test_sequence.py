"""Tests for the Sequence builder (sequence.py)."""

import json

import matplotlib

matplotlib.use("Agg")

import pytest

from ryd_gate import (
    DeviceSpec,
    MeasureOp,
    Pulse,
    PulseOp,
    Register,
    Sequence,
    level_structure,
)
from ryd_gate.protocols.channels import ChannelSpec


def _device():
    return DeviceSpec.virtual_rb87()


def _seq(level="1r", n=2, spacing=4.0):
    seq = Sequence(Register.chain(n, spacing), _device(), level)
    seq.declare_channel("ryd", "rydberg_global")
    return seq


class TestConstruction:
    def test_default_level_structure_from_device(self):
        seq = Sequence(Register.chain(2, 4.0), _device())
        assert seq.level_structure.name == _device().default_level_structure == "01r"

    def test_string_and_spec_forms(self):
        assert Sequence(Register.chain(2, 4.0), _device(), "1r").level_structure.name == "1r"
        spec = level_structure("1r")
        assert Sequence(Register.chain(2, 4.0), _device(), spec).level_structure is spec

    def test_invalid_register_raises_at_construction(self):
        with pytest.raises(ValueError, match="register.min_distance"):
            Sequence(Register.chain(2, 1.0), _device())

    def test_invalid_level_structure_type_raises(self):
        with pytest.raises(TypeError):
            Sequence(Register.chain(2, 4.0), _device(), 123)


class TestDeclareChannel:
    def test_unknown_channel_id_raises(self):
        seq = Sequence(Register.chain(2, 4.0), _device(), "1r")
        with pytest.raises(ValueError, match="unknown channel id"):
            seq.declare_channel("ryd", "nope")

    def test_duplicate_name_raises(self):
        seq = _seq()
        with pytest.raises(ValueError, match="already declared"):
            seq.declare_channel("ryd", "rydberg_global")

    def test_local_channel_not_stage2(self):
        seq = Sequence(Register.chain(2, 4.0), _device(), "01r")
        with pytest.raises(NotImplementedError, match="local_not_stage2"):
            seq.declare_channel("loc", "rydberg_local")

    def test_hyperfine_channel_not_stage2(self):
        seq = Sequence(Register.chain(2, 4.0), _device(), "01r")
        with pytest.raises(NotImplementedError, match="hyperfine_not_stage2"):
            seq.declare_channel("hf", "hyperfine_global")

    def test_model_mismatch(self):
        seq = Sequence(Register.chain(2, 4.0), _device(), "ger")
        with pytest.raises(ValueError, match="channel_model_mismatch"):
            seq.declare_channel("ryd", "rydberg_global")

    def test_compiler_channel_collision(self):
        seq = _seq()
        with pytest.raises(ValueError, match="compiler_channel_collision"):
            seq.declare_channel("ryd2", "rydberg_global")


class TestScheduling:
    def test_pulses_append_back_to_back(self):
        seq = _seq()
        seq.add(Pulse.constant(1000, 1.0, 0.0), "ryd")
        seq.add(Pulse.constant(500, 0.5, 0.0), "ryd")
        ops = seq.operations
        assert isinstance(ops[0], PulseOp) and ops[0].t_start_ns == 0
        assert isinstance(ops[1], PulseOp) and ops[1].t_start_ns == 1000
        assert seq.duration_ns == 1500

    def test_delay_extends_only_its_channel(self):
        # custom device with two non-colliding global rydberg channels
        device = DeviceSpec(
            name="dual",
            dimensions=2,
            atom_species="Rb87",
            allowed_level_structures=("1r",),
            default_level_structure="1r",
            min_atom_distance_um=2.0,
            channels={
                "ryd_a": ChannelSpec(
                    channel_id="ryd_a", kind="rydberg", transition="1_r", addressing="global",
                    amplitude_channels={"1r": "global_X"}, detuning_channels={"1r": "global_n"},
                ),
                "ryd_b": ChannelSpec(
                    channel_id="ryd_b", kind="rydberg", transition="1_r", addressing="global",
                    amplitude_channels={"1r": "drive_R"}, detuning_channels={"1r": "delta_R"},
                ),
            },
        )
        seq = Sequence(Register.chain(2, 4.0), device, "1r")
        seq.declare_channel("a", "ryd_a")
        seq.declare_channel("b", "ryd_b")
        seq.delay(300, "a")
        seq.add(Pulse.constant(1000, 1.0, 0.0), "a")   # starts at 300 on channel a
        seq.add(Pulse.constant(200, 1.0, 0.0), "b")    # starts at 0 on channel b
        starts = {op.channel: op.t_start_ns for op in seq.operations if isinstance(op, PulseOp)}
        assert starts == {"a": 300, "b": 0}
        assert seq.duration_ns == 1300

    def test_add_on_undeclared_channel_raises(self):
        seq = Sequence(Register.chain(2, 4.0), _device(), "1r")
        with pytest.raises(ValueError, match="not declared"):
            seq.add(Pulse.constant(1000, 1.0, 0.0), "ryd")

    def test_add_enforces_channel_limits(self):
        device = DeviceSpec(
            name="limited",
            dimensions=2,
            atom_species="Rb87",
            allowed_level_structures=("1r",),
            default_level_structure="1r",
            min_atom_distance_um=2.0,
            channels={
                "ryd": ChannelSpec(
                    channel_id="ryd", kind="rydberg", transition="1_r", addressing="global",
                    amplitude_channels={"1r": "global_X"}, detuning_channels={"1r": "global_n"},
                    max_abs_amplitude_rad_per_us=2.0,
                ),
            },
        )
        seq = Sequence(Register.chain(2, 4.0), device, "1r")
        seq.declare_channel("ryd", "ryd")
        with pytest.raises(ValueError, match="pulse.amplitude_limit"):
            seq.add(Pulse.constant(1000, 5.0, 0.0), "ryd")

    def test_invalid_delay_raises(self):
        seq = _seq()
        with pytest.raises(ValueError, match="positive integer"):
            seq.delay(0, "ryd")


class TestMeasure:
    def test_measure_appends_at_duration(self):
        seq = _seq()
        seq.add(Pulse.constant(1000, 1.0, 0.0), "ryd")
        seq.measure("rydberg")
        op = seq.operations[-1]
        assert isinstance(op, MeasureOp)
        assert op.basis == "rydberg"
        assert op.t_start_ns == 1000
        assert seq.is_measured()

    def test_locked_after_measure(self):
        seq = _seq()
        seq.add(Pulse.constant(1000, 1.0, 0.0), "ryd")
        seq.measure()
        with pytest.raises(ValueError, match="sequence.measured"):
            seq.add(Pulse.constant(100, 1.0, 0.0), "ryd")
        with pytest.raises(ValueError, match="sequence.measured"):
            seq.delay(100, "ryd")
        with pytest.raises(ValueError, match="sequence.measured"):
            seq.measure()

    def test_invalid_basis_raises(self):
        seq = _seq()
        with pytest.raises(ValueError, match="basis"):
            seq.measure("parity")


class TestSerialization:
    def test_roundtrip_preserves_schedule(self):
        seq = Sequence(Register.chain(2, 4.0), _device(), "01r")
        seq.declare_channel("ryd", "rydberg_global")
        seq.add(Pulse.constant(1000, 1.0, 0.5), "ryd")
        seq.delay(200, "ryd")
        seq.add(Pulse.constant(400, 2.0, 0.0), "ryd")
        seq.measure("rydberg")

        data = seq.to_dict()
        assert data["schema"] == "ryd-gate/sequence/v1"
        json.dumps(data)

        back = Sequence.from_dict(data)
        assert back.operations == seq.operations
        assert {n: c.channel_id for n, c in back.declared_channels.items()} == {
            "ryd": "rydberg_global"
        }
        assert back.level_structure == seq.level_structure
        assert back.duration_ns == seq.duration_ns == 1600
        assert back.is_measured()

    def test_wrong_schema_raises(self):
        data = _seq().to_dict()
        data["schema"] = "ryd-gate/register/v1"
        with pytest.raises(ValueError, match="schema"):
            Sequence.from_dict(data)


class TestDraw:
    def test_draw_returns_figure(self):
        import matplotlib.pyplot as plt

        seq = _seq()
        seq.add(Pulse.constant(1000, 1.0, -2.0), "ryd")
        fig, ax = seq.draw(show=False)
        assert fig is not None and ax is not None
        plt.close(fig)
