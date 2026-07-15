"""PEPS option-schema and evidence-record contracts (no YASTN import; PEPS21/22/23-26/35).

These exercise the pure contract layer of the PEPS backend: the exactly-ten
real-time / exactly-nine ground-state option validators and the report-only
evidence value objects, including their deep immutability and JSON serializer.
"""

import numpy as np
import pytest

from ryd_gate.backends.peps._options import validate_ground_options, validate_real_time_options
from ryd_gate.results import PEPSAmplitudeEvidence, PEPSEvidence, PEPSSampleEvidence

_RT = {
    "time_step_s": 1e-9,
    "bond_dimension": 4,
    "svd_tolerance": 1e-8,
    "ntu_max_iterations": 50,
    "ntu_iteration_tolerance": 1e-10,
    "measurement_method": "belief_propagation",
    "environment_bond_dimension": 8,
    "environment_tolerance": 1e-8,
    "environment_max_iterations": 50,
    "device": "cpu",
}
_GS = {
    "bond_dimension": 4,
    "svd_tolerance": 1e-8,
    "ntu_max_iterations": 50,
    "ntu_iteration_tolerance": 1e-10,
    "environment_bond_dimension": 8,
    "environment_tolerance": 1e-8,
    "environment_max_iterations": 50,
    "imaginary_time_schedule": ((0.1, 100), (0.01, 200)),
    "device": "cpu",
}


class TestRealTimeSchema:
    def test_ten_keys_roundtrip(self):
        opts = validate_real_time_options(dict(_RT))
        assert set(opts.parameters()) == set(_RT)
        assert opts.time_step_s == 1e-9
        assert opts.measurement_method == "belief_propagation"

    def test_missing_key_rejected(self):
        bad = {k: v for k, v in _RT.items() if k != "device"}
        with pytest.raises(ValueError, match="missing"):
            validate_real_time_options(bad)

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="unknown"):
            validate_real_time_options({**_RT, "extra": 1})

    @pytest.mark.parametrize("bad", [True, "1e-8", 1e-8 + 0j])
    def test_nonreal_tolerance_rejected(self, bad):
        with pytest.raises(ValueError, match="svd_tolerance"):
            validate_real_time_options({**_RT, "svd_tolerance": bad})

    @pytest.mark.parametrize("bad", [0.0, float("nan"), float("inf")])
    def test_nonpositive_or_nonfinite_time_step_rejected(self, bad):
        with pytest.raises(ValueError, match="time_step_s"):
            validate_real_time_options({**_RT, "time_step_s": bad})

    @pytest.mark.parametrize("key", ["svd_tolerance", "environment_tolerance"])
    def test_tolerance_at_or_above_one_rejected(self, key):
        with pytest.raises(ValueError, match=key):
            validate_real_time_options({**_RT, key: 1.0})

    def test_measurement_method_validated(self):
        with pytest.raises(ValueError, match="measurement_method"):
            validate_real_time_options({**_RT, "measurement_method": "ctm2"})

    def test_device_validated(self):
        with pytest.raises(ValueError, match="device"):
            validate_real_time_options({**_RT, "device": "gpu"})

    @pytest.mark.parametrize("bad", [True, 0, -1])
    def test_bond_dimension_rejected(self, bad):
        with pytest.raises(ValueError, match="bond_dimension"):
            validate_real_time_options({**_RT, "bond_dimension": bad})


class TestGroundSchema:
    def test_nine_keys_roundtrip(self):
        opts = validate_ground_options(dict(_GS))
        assert set(opts.parameters()) == set(_GS)
        assert opts.imaginary_time_schedule == ((0.1, 100), (0.01, 200))

    def test_time_step_key_is_unknown_here(self):
        with pytest.raises(ValueError, match="unknown"):
            validate_ground_options({**_GS, "time_step_s": 1e-9})

    def test_ntu_iteration_tolerance_above_one_accepted(self):
        # ntu_iteration_tolerance is a strictly-positive residual bound, not a 0<x<1 SVD cutoff.
        opts = validate_ground_options({**_GS, "ntu_iteration_tolerance": 5.0})
        assert opts.ntu_iteration_tolerance == 5.0

    def test_schedule_must_be_tuple_not_list(self):
        with pytest.raises(ValueError, match="tuple"):
            validate_ground_options({**_GS, "imaginary_time_schedule": [(0.1, 100)]})

    def test_schedule_dtau_must_strictly_decrease(self):
        with pytest.raises(ValueError, match="decreasing"):
            validate_ground_options({**_GS, "imaginary_time_schedule": ((0.1, 100), (0.1, 100))})

    def test_schedule_steps_must_be_positive_int(self):
        with pytest.raises(ValueError, match="max_steps"):
            validate_ground_options({**_GS, "imaginary_time_schedule": ((0.1, 0),)})


def _amp(labels=("0", "1"), err=1e-9):
    return PEPSAmplitudeEvidence(labels=labels, contraction_error=err)


def _evidence(**over):
    base = dict(
        parameters={"bond_dimension": 4, "imaginary_time_schedule": ((0.1, 100),)},
        hamiltonian_scale_rad_s=None,
        max_ntu_truncation_error=1e-6,
        cumulative_ntu_truncation_error=2e-6,
        environment_residual=3e-8,
        environment_iterations=12,
        norm_contraction_error=4e-9,
        amplitudes=(_amp(),),
        samples=(PEPSSampleEvidence(shots=100, seed=0),),
    )
    base.update(over)
    return PEPSEvidence(**base)


class TestEvidenceRecords:
    def test_amplitude_labels_canonicalized(self):
        a = PEPSAmplitudeEvidence(labels=["0", "1", "r"], contraction_error=np.float64(1e-9))
        assert a.labels == ("0", "1", "r")
        assert type(a.contraction_error) is float

    def test_amplitude_negative_error_rejected(self):
        with pytest.raises(ValueError, match="contraction_error"):
            PEPSAmplitudeEvidence(labels=("0",), contraction_error=-1.0)

    def test_sample_positive_shots_nonneg_seed(self):
        with pytest.raises(ValueError, match="shots"):
            PEPSSampleEvidence(shots=0, seed=0)
        with pytest.raises(ValueError, match="seed"):
            PEPSSampleEvidence(shots=1, seed=-1)

    def test_optional_fields_accept_none(self):
        ev = _evidence(
            hamiltonian_scale_rad_s=None,
            cumulative_ntu_truncation_error=None,
            environment_residual=None,
            environment_iterations=None,
            norm_contraction_error=None,
        )
        assert ev.cumulative_ntu_truncation_error is None
        assert ev.environment_iterations is None

    def test_hamiltonian_scale_must_be_positive_when_set(self):
        with pytest.raises(ValueError, match="hamiltonian_scale_rad_s"):
            _evidence(hamiltonian_scale_rad_s=0.0)

    def test_parameters_frozen_and_defensively_copied(self):
        src = {"bond_dimension": 4, "nested": {"a": [1, 2]}}
        ev = _evidence(parameters=src)
        # mutating the source after construction must not change the record
        src["bond_dimension"] = 999
        src["nested"]["a"].append(3)
        assert ev.parameters["bond_dimension"] == 4
        assert ev.parameters["nested"]["a"] == (1, 2)
        # the stored mapping is read-only
        with pytest.raises(TypeError):
            ev.parameters["bond_dimension"] = 1

    def test_amplitudes_samples_are_tuples(self):
        ev = _evidence(amplitudes=[_amp(), _amp(("1", "1"))], samples=[PEPSSampleEvidence(shots=10, seed=2)])
        assert isinstance(ev.amplitudes, tuple)
        assert isinstance(ev.samples, tuple)

    def test_to_dict_is_json_compatible_and_independent(self):
        import json

        ev = _evidence()
        d = ev.to_dict()
        text = json.dumps(d)  # must not raise: no tuples-as-keys, no numpy, no MappingProxy
        assert json.loads(text)["max_ntu_truncation_error"] == 1e-6
        assert d["amplitudes"][0]["labels"] == ["0", "1"]
        assert d["parameters"]["imaginary_time_schedule"] == [[0.1, 100]]
        # mutating the returned dict must not affect the record or a later to_dict()
        d["parameters"]["bond_dimension"] = -1
        d["amplitudes"].clear()
        assert ev.to_dict()["parameters"]["bond_dimension"] == 4
        assert len(ev.to_dict()["amplitudes"]) == 1

    def test_numpy_scalar_error_fields_canonicalized(self):
        ev = _evidence(max_ntu_truncation_error=np.float64(1e-6), environment_iterations=np.int64(7))
        assert type(ev.max_ntu_truncation_error) is float
        assert type(ev.environment_iterations) is int
