"""CZGateReport data contract and cz_gate_report assembly rules.

Uses a short-gate-window TO configuration: deterministic and cheap. Gate
quality is irrelevant here — physical benchmark values live in
test_cz_benchmark_pins.py.
"""

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem
from ryd_gate.analysis import gate_metrics
from ryd_gate.gates import CZGateReport, TOProtocol, cz_gate_report

X_FAST = [
    -0.6989301339711643,
    1.0296229082590798,
    0.3759232324550267,
    1.5710180991068543,
    1.4454279613697887,
    0.13,  # short gate window keeps the stiff rb87_7 solves cheap
]


def _system(**kwargs):
    return RydbergSystem.from_lattice(
        Register.chain(2, spacing_um=3.0), "rb87_7", param_set="our",
        blackmanflag=False, **kwargs,
    )


@pytest.fixture(scope="module")
def report():
    return cz_gate_report(
        _system(), TOProtocol(), X_FAST, include_error_budget=False
    )


class TestReportData:
    def test_fidelity_is_one_minus_infidelity(self, report):
        assert report.fidelity == 1.0 - report.infidelity

    def test_protocol_name_and_parameters(self, report):
        assert report.protocol == "TOProtocol"
        assert report.parameters == tuple(X_FAST)
        assert report.theta_rad == pytest.approx(X_FAST[4])

    def test_phase_error_finite_and_wrapped(self, report):
        assert np.isfinite(report.phase_error_rad)
        assert -np.pi <= report.phase_error_rad <= np.pi

    def test_residuals_present_by_default(self, report):
        assert set(report.residuals) == {"e1", "e2", "e3", "ryd", "ryd_garb"}
        assert all(np.isfinite(v) for v in report.residuals.values())


class TestSerialization:
    def test_round_trip_with_schema_tag(self, report):
        data = report.to_dict()
        assert data["schema"] == "ryd-gate/cz-gate-report/v1"
        assert "fidelity" not in data  # derived, never stored
        assert CZGateReport.from_dict(data) == report

    def test_from_dict_rejects_wrong_schema(self, report):
        data = report.to_dict()
        data["schema"] = "ryd-gate/noise/v1"
        with pytest.raises(ValueError, match="schema mismatch"):
            CZGateReport.from_dict(data)


class TestAssemblyRules:
    def test_error_budget_not_called_when_excluded(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("error_budget must not be called")

        monkeypatch.setattr(gate_metrics, "error_budget", boom)
        report = cz_gate_report(
            _system(), TOProtocol(), X_FAST, include_error_budget=False
        )
        assert report.error_budget == {}

    def test_include_residuals_false_gives_empty_map(self):
        report = cz_gate_report(
            _system(), TOProtocol(), X_FAST,
            include_error_budget=False, include_residuals=False,
        )
        assert report.residuals == {}

    def test_error_budget_matches_direct_call(self):
        system = _system(enable_rydberg_decay=True)
        report = cz_gate_report(system, TOProtocol(), X_FAST)
        direct = gate_metrics.error_budget(system, TOProtocol(), X_FAST)
        assert report.error_budget == direct

    def test_non_cz_protocol_raises(self):
        class NotCZ:
            def validate_params(self, x):
                pass

            def unpack_params(self, x, system):
                return {"t_gate": 1.0}

        with pytest.raises(ValueError, match="cz_report.protocol_not_cz"):
            cz_gate_report(None, NotCZ(), [])

    def test_infidelity_matches_average_gate_infidelity(self, report):
        direct = gate_metrics.average_gate_infidelity(
            _system(), TOProtocol(), X_FAST
        )
        assert report.infidelity == pytest.approx(float(direct), rel=1e-12)
