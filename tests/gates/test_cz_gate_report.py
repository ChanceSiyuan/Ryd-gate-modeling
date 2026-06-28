"""CZGateReport data contract and cz_gate_report assembly rules.

Uses a short-gate-window TO configuration: deterministic and cheap. Gate
quality is irrelevant here — physical benchmark values live in
test_cz_benchmark_pins.py.
"""

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem
from ryd_gate.analysis import gate_metrics
from ryd_gate.gates import CZGateReport, TOProtocol, cz_gate_report, optimize_cz_parameters

X_FAST = [
    -0.6894097925886826,
    1.040962607910546,
    0.3277877211544321,
    1.5639989822346387,
    0.6689846026179691,
    0.13,  # short gate window keeps the stiff rb87_7 solves cheap
]


def _system(**kwargs):
    return (
        RydbergSystem.set_atom_level("rb87_7", param_set="our", **kwargs)
        .set_atom_geom(Register.chain(2, spacing_um=3.0))
        .build()
    )


@pytest.fixture(scope="module")
def report():
    return cz_gate_report(
        _system(), TOProtocol(blackman=False), X_FAST, include_error_budget=False
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
            _system(), TOProtocol(blackman=False), X_FAST, include_error_budget=False
        )
        assert report.error_budget == {}

    def test_include_residuals_false_gives_empty_map(self):
        report = cz_gate_report(
            _system(), TOProtocol(blackman=False), X_FAST,
            include_error_budget=False, include_residuals=False,
        )
        assert report.residuals == {}

    def test_error_budget_matches_direct_call(self):
        system = _system(enable_rydberg_decay=True)
        report = cz_gate_report(system, TOProtocol(blackman=False), X_FAST)
        direct = gate_metrics.error_budget(system, TOProtocol(blackman=False), X_FAST)
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
            _system(), TOProtocol(blackman=False), X_FAST
        )
        assert report.infidelity == pytest.approx(float(direct), rel=1e-12)


def _coarse_proto():
    """TO protocol with a coarse step count -- keeps the optimizer-loop tests
    cheap (the theta mechanism is n_steps-independent)."""
    p = TOProtocol(blackman=False); p.n_steps = 40
    return p


class TestOptimizeCZParameters:
    def test_theta_projection_recovers_mis_set_phase(self):
        """polish=False: the 1-D theta re-fit recovers a deliberately wrong
        single-qubit Z, the dominant ill-conditioned direction under explicit |0>."""
        system = _system()
        ti = _coarse_proto().theta_index
        good_infid = float(gate_metrics.average_gate_infidelity(system, _coarse_proto(), X_FAST))

        bad = list(X_FAST)
        bad[ti] += 1.5  # detune only the single-qubit phase
        res = optimize_cz_parameters(system, _coarse_proto(), bad, polish=False)

        assert res.n_eval == 0 and len(res.x) == len(X_FAST)  # no polish, full vector
        assert res.seed_infidelity > res.theta_infidelity     # theta-only strictly improved
        # re-fitting theta is at least as good as the original (good) theta
        assert res.theta_infidelity <= good_infid + 1e-9
        assert res.infidelity == res.theta_infidelity         # polish skipped

    def test_polish_returns_improved_point(self):
        """A short polish from a near-optimal seed does not worsen the gate."""
        system = _system()
        res = optimize_cz_parameters(system, _coarse_proto(), X_FAST, maxiter=40)
        seed_infid = float(gate_metrics.average_gate_infidelity(system, _coarse_proto(), X_FAST))
        assert res.infidelity <= seed_infid + 1e-12
        assert res.n_eval > 0
