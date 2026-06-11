"""NoiseModel data contract: types, validation, serialization."""

import numpy as np
import pytest

from ryd_gate import NoiseModel


def _codes(issues):
    return [issue.code for issue in issues]


class TestDefaults:
    def test_default_model(self):
        noise = NoiseModel()
        assert noise.runs == 1
        assert noise.n_trajectories == 1
        assert noise.noise_types == ()
        assert noise.validate() == []

    def test_default_summary_names_no_active_noise(self):
        assert "no active noise" in NoiseModel().summary()


class TestNoiseTypes:
    def test_active_types_in_fixed_order(self):
        noise = NoiseModel(
            runs=3,
            detuning_sigma_rad_per_us=0.02,
            amp_sigma=0.01,
            local_rin_sigma=0.05,
            position_sigma_um=0.07,
            rydberg_decay=True,
            intermediate_decay=True,
            state_prep_error=0.1,
            p_false_neg=0.2,
            temperature_uK=10.0,
            laser_waist_um=100.0,
        )
        assert noise.noise_types == (
            "state_prep", "readout", "detuning", "amplitude", "local_rin",
            "position", "rydberg_decay", "intermediate_decay", "temperature",
            "laser_waist",
        )

    def test_readout_active_from_either_probability(self):
        assert NoiseModel(p_false_pos=0.01).noise_types == ("readout",)
        assert NoiseModel(p_false_neg=0.01).noise_types == ("readout",)

    def test_position_tuple_activity(self):
        assert NoiseModel(position_sigma_um=(0.0, 0.0, 0.0)).noise_types == ()
        assert NoiseModel(position_sigma_um=(0.0, 0.0, 0.13)).noise_types == ("position",)

    def test_summary_contains_each_active_type_exactly_once(self):
        noise = NoiseModel(
            detuning_sigma_rad_per_us=0.02,
            amp_sigma=0.01,
            rydberg_decay=True,
            intermediate_decay=True,
        )
        text = noise.summary()
        for name in noise.noise_types:
            assert text.count(name) == 1, name
        assert text == noise.summary()


class TestValidation:
    @pytest.mark.parametrize("runs", [0, -1, 2.5, True])
    def test_invalid_runs(self, runs):
        assert "noise.runs" in _codes(NoiseModel(runs=runs).validate())

    @pytest.mark.parametrize("field", ["state_prep_error", "p_false_pos", "p_false_neg"])
    def test_probability_range(self, field):
        assert "noise.probability_range" in _codes(
            NoiseModel(**{field: 1.5}).validate()
        )
        assert "noise.probability_range" in _codes(
            NoiseModel(**{field: -0.1}).validate()
        )

    @pytest.mark.parametrize(
        "field", ["detuning_sigma_rad_per_us", "amp_sigma", "local_rin_sigma"]
    )
    def test_negative_sigma(self, field):
        assert "noise.nonnegative" in _codes(NoiseModel(**{field: -0.1}).validate())

    def test_position_sigma_shape(self):
        assert "noise.position_sigma_shape" in _codes(
            NoiseModel(position_sigma_um=(0.1, 0.2)).validate()
        )

    def test_position_sigma_entries(self):
        assert "noise.nonnegative" in _codes(
            NoiseModel(position_sigma_um=(0.1, np.inf, 0.1)).validate()
        )
        assert "noise.nonnegative" in _codes(
            NoiseModel(position_sigma_um=-0.1).validate()
        )

    def test_temperature_and_waist_nonnegative(self):
        assert "noise.nonnegative" in _codes(NoiseModel(temperature_uK=-1.0).validate())
        assert "noise.nonnegative" in _codes(NoiseModel(laser_waist_um=-1.0).validate())

    def test_non_json_metadata(self):
        assert "noise.metadata_json" in _codes(
            NoiseModel(metadata={"bad": object()}).validate()
        )


class TestSerialization:
    def test_round_trip_with_schema_tag(self):
        noise = NoiseModel(
            runs=5,
            detuning_sigma_rad_per_us=0.02,
            position_sigma_um=(0.07, 0.07, 0.13),
            rydberg_decay=True,
            metadata={"note": "bench"},
        )
        data = noise.to_dict()
        assert data["schema"] == "ryd-gate/noise/v1"
        assert data["runs"] == 5
        assert "n_trajectories" not in data
        assert NoiseModel.from_dict(data) == noise

    def test_from_dict_accepts_n_trajectories_alias(self):
        data = NoiseModel().to_dict()
        del data["runs"]
        data["n_trajectories"] = 7
        assert NoiseModel.from_dict(data).runs == 7

    def test_from_dict_rejects_wrong_schema(self):
        data = NoiseModel().to_dict()
        data["schema"] = "ryd-gate/device/v1"
        with pytest.raises(ValueError, match="schema mismatch"):
            NoiseModel.from_dict(data)

    def test_from_dict_uses_defaults_for_absent_fields(self):
        noise = NoiseModel.from_dict({"schema": "ryd-gate/noise/v1"})
        assert noise == NoiseModel()


class TestPhysicalKwargs:
    def test_only_the_two_decay_flags(self):
        noise = NoiseModel(rydberg_decay=True)
        assert noise.physical_kwargs() == {
            "enable_rydberg_decay": True,
            "enable_intermediate_decay": False,
        }


class TestValidateFor:
    def test_active_noise_on_non_exact_backend(self):
        noise = NoiseModel(detuning_sigma_rad_per_us=0.02)
        assert "noise.backend_unsupported" in _codes(noise.validate_for(backend="mps"))
        assert noise.validate_for(backend="exact") == []

    def test_inactive_noise_passes_everywhere(self):
        assert NoiseModel().validate_for(backend="mps") == []

    def test_decay_requires_physical_level_structure(self):
        noise = NoiseModel(rydberg_decay=True)
        assert "noise.decay_level_structure_unsupported" in _codes(
            noise.validate_for(backend="exact", level_structure="1r")
        )
        assert noise.validate_for(backend="exact", level_structure="rb87_7") == []
        assert noise.validate_for(backend="exact", level_structure="analog_3") == []

    def test_position_two_atom_only(self):
        noise = NoiseModel(position_sigma_um=0.07)
        assert "noise.position_two_atom_only" in _codes(
            noise.validate_for(backend="exact", n_atoms=3)
        )
        assert noise.validate_for(backend="exact", n_atoms=2) == []

    def test_runtime_not_stage4_fields(self):
        for kwargs in (
            {"state_prep_error": 0.01},
            {"p_false_pos": 0.01},
            {"p_false_neg": 0.01},
            {"temperature_uK": 10.0},
            {"laser_waist_um": 100.0},
        ):
            noise = NoiseModel(**kwargs)
            assert "noise.runtime_not_stage4" in _codes(
                noise.validate_for(backend="exact")
            ), kwargs
