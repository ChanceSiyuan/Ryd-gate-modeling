"""ObservableConfig: validation, serialization, MPS streaming path."""

import numpy as np
import pytest

from ryd_gate import (
    DeviceSpec,
    InteractionSpec,
    ObservableConfig,
    Pulse,
    Register,
    Sequence,
    simulate_sequence,
)


def _codes(issues):
    return [issue.code for issue in issues]


def _sequence(n_atoms=4):
    seq = Sequence(Register.chain(n_atoms, 20.0), DeviceSpec.virtual_rb87(), "1r")
    seq.declare_channel("ryd", "rydberg_global")
    seq.add(Pulse.constant(1000, float(np.pi), 0.0), "ryd")
    return seq


class TestValidation:
    def test_valid_configs(self):
        assert ObservableConfig(("sum_nr",)).validate() == []
        assert ObservableConfig(("sum_nr",), times_ns=(0, 500)).validate() == []
        assert ObservableConfig(("sum_nr",), every_ns=100).validate() == []

    def test_empty_or_invalid_names(self):
        assert "observables.names" in _codes(ObservableConfig(()).validate())
        assert "observables.names" in _codes(ObservableConfig(("",)).validate())

    def test_times_and_every_conflict(self):
        config = ObservableConfig(("sum_nr",), times_ns=(0,), every_ns=10)
        assert "observables.schedule_conflict" in _codes(config.validate())

    def test_invalid_times_and_every(self):
        assert "observables.times" in _codes(
            ObservableConfig(("a",), times_ns=(-1,)).validate()
        )
        assert "observables.every" in _codes(
            ObservableConfig(("a",), every_ns=0).validate()
        )

    def test_schedule_resolution(self):
        assert ObservableConfig(("a",), times_ns=(0, 7)).schedule_times_ns(100) == (0, 7)
        assert ObservableConfig(("a",), every_ns=500).schedule_times_ns(1000) == (0, 500, 1000)
        assert ObservableConfig(("a",)).schedule_times_ns(1000) is None


class TestSerialization:
    def test_round_trip_with_schema_tag(self):
        config = ObservableConfig(("sum_nr", "n_mean"), times_ns=(0, 500, 1000))
        data = config.to_dict()
        assert data["schema"] == "ryd-gate/observable-config/v1"
        assert ObservableConfig.from_dict(data) == config

    def test_round_trip_validates_against_frozen_schema(self):
        pytest.importorskip("jsonschema")
        from ryd_gate.core.serialization import validate_json_schema

        data = ObservableConfig(("sum_nr",), every_ns=250).to_dict()
        assert validate_json_schema(data, "observable-config") == []


class TestSimulateSequenceWiring:
    def test_invalid_config_raises_typed_error(self):
        with pytest.raises(ValueError, match="observables.names"):
            simulate_sequence(_sequence(1), observables=ObservableConfig(()))

    def test_exact_backend_ignores_streaming_schedule(self):
        config = ObservableConfig(("sum_nr",), every_ns=500)
        result = simulate_sequence(
            _sequence(1), interaction=InteractionSpec(C6=0.0), observables=config
        )
        # Exact semantics unchanged: final-state handle works, nothing streamed.
        assert "obs" not in result.raw.metadata
        assert np.isfinite(result.expectation("sum_nr"))

    def test_mps_backend_records_through_native_path(self):
        pytest.importorskip("tenpy")
        config = ObservableConfig(("sum_nr",), every_ns=500)
        result = simulate_sequence(
            _sequence(),
            backend="mps",
            interaction=InteractionSpec(C6=0.0),
            backend_options={"chi_max": 16, "dt": 2.5e-7},
            observables=config,
        )
        obs = result.raw.metadata["obs"]
        assert "sum_nr" in obs
        # every_ns=500 over a 1000 ns sequence: t = 0, 500, 1000.
        assert len(obs["sum_nr"]) == 3
        assert np.all(np.isfinite(np.asarray(obs["sum_nr"], dtype=float)))
