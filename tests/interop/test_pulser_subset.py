"""Pulser abstract-repr subset: import, export, typed refusals."""

import json

import numpy as np
import pytest

from ryd_gate import DeviceSpec, NoiseModel, Pulse, Register, Sequence, Waveform
from ryd_gate.interop import (
    PulserInteropError,
    from_pulser_abstract_repr,
    noise_from_pulser_abstract_repr,
    noise_to_pulser_abstract_repr,
    to_pulser_abstract_repr,
)
from ryd_gate.sequence import DelayOp, PulseOp


def _minimal_payload(**overrides):
    payload = {
        "version": "1",
        "name": "afm-demo",
        "register": [
            {"name": "q0", "x": 0.0, "y": 0.0},
            {"name": "q1", "x": 6.0, "y": 0.0},
        ],
        "channels": {"ryd": "rydberg_global"},
        "operations": [
            {
                "op": "pulse",
                "channel": "ryd",
                "protocol": "min-delay",
                "amplitude": {"kind": "blackman", "duration": 1000, "area": float(np.pi)},
                "detuning": {"kind": "constant", "duration": 1000, "value": 0.0},
                "phase": 0.0,
                "post_phase_shift": 0.0,
            },
            {"op": "delay", "channel": "ryd", "time": 200},
        ],
        "measurement": "ground-rydberg",
    }
    payload.update(overrides)
    return payload


class TestImport:
    def test_minimal_global_rydberg_sequence(self):
        seq = from_pulser_abstract_repr(_minimal_payload())
        assert isinstance(seq, Sequence)
        assert seq.register.ids == ("q0", "q1")
        assert seq.level_structure.name == "1r"
        assert list(seq.declared_channels) == ["ryd"]
        ops = seq.operations
        assert isinstance(ops[0], PulseOp)
        assert ops[0].pulse.amplitude.kind == "blackman"
        assert ops[0].pulse.amplitude.integral_rad() == pytest.approx(np.pi, rel=1e-3)
        assert isinstance(ops[1], DelayOp) and ops[1].duration_ns == 200
        assert seq.is_measured()

    def test_json_string_input(self):
        seq = from_pulser_abstract_repr(json.dumps(_minimal_payload()))
        assert seq.register.n_atoms == 2

    def test_layout_trap_mapping(self):
        payload = _minimal_payload(
            layout={
                "coordinates": [[0.0, 0.0], [6.0, 0.0], [0.0, 6.0], [6.0, 6.0]],
                "slug": "square4",
            },
            register=[
                {"name": "a", "x": 6.0, "y": 0.0},
                {"name": "b", "x": 6.0, "y": 6.0},
            ],
        )
        seq = from_pulser_abstract_repr(payload)
        register = seq.register
        assert register.ids == ("a", "b")
        assert register.layout is not None
        assert register.layout.name == "square4"
        assert register.layout.trap_coords_um == ((0.0, 0.0), (6.0, 0.0), (0.0, 6.0), (6.0, 6.0))
        assert register.metadata["trap_ids"] == [1, 3]
        np.testing.assert_allclose(register.coords, [[6.0, 0.0], [6.0, 6.0]])

    def test_all_waveform_kinds_map(self):
        shapes = {
            "constant": {"kind": "constant", "duration": 1000, "value": 1.0},
            "ramp": {"kind": "ramp", "duration": 1000, "start": 0.0, "stop": 2.0},
            "blackman": {"kind": "blackman", "duration": 1000, "area": 3.14},
            "interpolated": {
                "kind": "interpolated", "duration": 1000,
                "times": [0.0, 0.4, 1.0], "values": [0.0, 2.0, 0.0],
            },
            "custom": {"kind": "custom", "samples": [0.0, 1.0, 0.5, 0.0]},
        }
        for kind, shape in shapes.items():
            payload = _minimal_payload(operations=[{
                "op": "pulse", "channel": "ryd",
                "amplitude": shape,
                "detuning": {"kind": "constant", "duration": shape.get("duration", 3), "value": 0.0},
                "phase": 0.0, "post_phase_shift": 0.0,
            }], measurement=None)
            seq = from_pulser_abstract_repr(payload)
            assert seq.operations[0].pulse.amplitude.kind == kind


class TestImportRefusals:
    @pytest.mark.parametrize(
        ("overrides", "code"),
        [
            ({"channels": {"mw": "mw_global"}}, "pulser.xy_not_supported"),
            ({"channels": {"loc": "rydberg_local"}}, "pulser.local_addressing_not_supported"),
            ({"channels": {"ram": "raman_global"}}, "pulser.channel_not_supported"),
            ({"slm_mask_targets": ["q0"]}, "pulser.slm_not_supported"),
            ({"dmm_channels": {"dmm_0": {}}}, "pulser.dmm_not_supported"),
            ({"measurement": "digital"}, "pulser.measurement_not_supported"),
        ],
    )
    def test_unsupported_top_level_constructs(self, overrides, code):
        payload = _minimal_payload(**overrides)
        if "channels" in overrides:
            payload["operations"] = []
            payload["measurement"] = None
        with pytest.raises(PulserInteropError) as err:
            from_pulser_abstract_repr(payload)
        assert err.value.code == code

    def test_eom_operation_refused_with_path(self):
        payload = _minimal_payload(
            operations=[{"op": "enable_eom_mode", "channel": "ryd"}], measurement=None
        )
        with pytest.raises(PulserInteropError) as err:
            from_pulser_abstract_repr(payload)
        assert err.value.code == "pulser.eom_not_supported"
        assert err.value.path == ("operations", "0")

    def test_target_operation_is_local_addressing(self):
        payload = _minimal_payload(
            operations=[{"op": "target", "channel": "ryd", "target": "q0"}],
            measurement=None,
        )
        with pytest.raises(PulserInteropError) as err:
            from_pulser_abstract_repr(payload)
        assert err.value.code == "pulser.local_addressing_not_supported"

    def test_nonzero_phase_refused_with_path(self):
        payload = _minimal_payload()
        payload["operations"][0]["phase"] = 1.0
        with pytest.raises(PulserInteropError) as err:
            from_pulser_abstract_repr(payload)
        assert err.value.code == "pulser.phase_not_supported"
        assert err.value.path == ("operations", "0", "phase")

    def test_unknown_waveform_refused_with_path(self):
        payload = _minimal_payload()
        payload["operations"][0]["amplitude"] = {"kind": "kaiser", "duration": 100, "beta": 2}
        with pytest.raises(PulserInteropError) as err:
            from_pulser_abstract_repr(payload)
        assert err.value.code == "pulser.waveform_not_supported"
        assert err.value.path == ("operations", "0", "amplitude")


class TestExportRoundTrip:
    def _native_sequence(self):
        seq = Sequence(Register.chain(2, spacing_um=6.0), DeviceSpec.virtual_rb87(), "1r")
        seq.declare_channel("ryd", "rydberg_global")
        seq.add(Pulse.constant_detuning(Waveform.blackman(1000, area=np.pi), -1.5), "ryd")
        seq.delay(300, "ryd")
        seq.add(Pulse.constant(500, 2.0, 0.5), "ryd")
        seq.measure("rydberg")
        return seq

    def test_export_then_import_round_trips(self):
        native = self._native_sequence()
        payload = to_pulser_abstract_repr(native)
        assert payload["channels"] == {"ryd": "rydberg_global"}
        assert payload["measurement"] == "ground-rydberg"

        back = from_pulser_abstract_repr(payload)
        assert back.register.ids == native.register.ids
        np.testing.assert_allclose(back.register.coords, native.register.coords)
        assert len(back.operations) == len(native.operations)
        for ours, theirs in zip(native.operations, back.operations):
            assert type(ours) is type(theirs)
            if isinstance(ours, PulseOp):
                assert theirs.pulse.amplitude.kind == ours.pulse.amplitude.kind
                assert theirs.pulse.amplitude.duration_ns == ours.pulse.amplitude.duration_ns
                assert theirs.pulse.amplitude.value_at_ns(500) == pytest.approx(
                    ours.pulse.amplitude.value_at_ns(500), rel=1e-12
                )
        assert back.is_measured()

    def test_export_refuses_nonzero_phase(self):
        seq = Sequence(Register.chain(1, spacing_um=6.0), DeviceSpec.virtual_rb87(), "1r")
        seq.declare_channel("ryd", "rydberg_global")
        seq.add(Pulse.constant(100, 1.0, 0.0, phase_rad=0.5), "ryd")
        with pytest.raises(PulserInteropError) as err:
            to_pulser_abstract_repr(seq)
        assert err.value.code == "pulser.phase_not_supported"

    def test_export_includes_layout(self):
        from ryd_gate import RegisterLayout

        layout = RegisterLayout(
            name="square4", kind="custom",
            trap_coords_um=((0.0, 0.0), (6.0, 0.0), (0.0, 6.0), (6.0, 6.0)),
        )
        register = layout.define_register([0, 1])
        seq = Sequence(register, DeviceSpec.virtual_rb87(), "1r")
        payload = to_pulser_abstract_repr(seq)
        assert payload["layout"]["slug"] == "square4"
        assert payload["layout"]["coordinates"] == [[0.0, 0.0], [6.0, 0.0], [0.0, 6.0], [6.0, 6.0]]


class TestNoiseBridge:
    def test_import_maps_matching_fields(self):
        noise = noise_from_pulser_abstract_repr({
            "runs": 16,
            "state_prep_error": 0.005,
            "p_false_pos": 0.01,
            "p_false_neg": 0.02,
            "amp_sigma": 0.01,
            "detuning_sigma": 0.02,
            "noise_types": ["SPAM", "amplitude"],
            "samples_per_run": 1,
        })
        assert noise == NoiseModel(
            runs=16, state_prep_error=0.005, p_false_pos=0.01, p_false_neg=0.02,
            amp_sigma=0.01, detuning_sigma_rad_per_us=0.02,
        )

    def test_import_accepts_n_trajectories(self):
        assert noise_from_pulser_abstract_repr({"n_trajectories": 5}).runs == 5

    def test_import_refuses_unsupported_fields(self):
        with pytest.raises(PulserInteropError) as err:
            noise_from_pulser_abstract_repr({"runs": 4, "temperature": 30.0})
        assert err.value.code == "pulser.noise_field_not_supported"
        assert err.value.path == ("temperature",)

    def test_import_ignores_zero_valued_unknown_fields(self):
        noise = noise_from_pulser_abstract_repr({"runs": 2, "temperature": 0.0})
        assert noise.runs == 2

    def test_export_round_trip(self):
        native = NoiseModel(runs=8, amp_sigma=0.01, detuning_sigma_rad_per_us=0.02)
        payload = noise_to_pulser_abstract_repr(native)
        assert noise_from_pulser_abstract_repr(payload) == native

    def test_export_refuses_microscopic_extensions(self):
        with pytest.raises(PulserInteropError) as err:
            noise_to_pulser_abstract_repr(NoiseModel(rydberg_decay=True))
        assert err.value.code == "pulser.noise_field_not_supported"
