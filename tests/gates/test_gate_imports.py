"""ryd_gate.gates is a namespace over kernel objects, not a second tree."""


class TestGateNamespace:
    def test_protocol_classes_are_the_kernel_classes(self):
        from ryd_gate import gates
        from ryd_gate.core.effective_theory import lower_cz_to_effective_01r
        from ryd_gate.protocols.gate_cz import (
            ARProtocol,
            CZProtocol,
            TOProtocol,
            phase_from_chirp,
        )

        assert gates.TOProtocol is TOProtocol
        assert gates.ARProtocol is ARProtocol
        assert gates.CZProtocol is CZProtocol
        assert gates.phase_from_chirp is phase_from_chirp
        assert gates.lower_cz_to_effective_01r is lower_cz_to_effective_01r
        # AdiaProtocol / DoubleARPProtocol are fully removed.
        assert not hasattr(gates, "AdiaProtocol")
        assert not hasattr(gates, "DoubleARPProtocol")

    def test_metric_functions_are_gate_metrics_functions(self):
        from ryd_gate import gates
        from ryd_gate.analysis import gate_metrics

        assert gates.average_gate_infidelity is gate_metrics.average_gate_infidelity
        assert gates.error_budget is gate_metrics.error_budget

    def test_report_exports_live_in_gates_not_top_level(self):
        import pytest

        import ryd_gate
        from ryd_gate import gates

        assert gates.CZGateReport is not None
        assert gates.cz_gate_report is not None
        for name in ("CZGateReport", "cz_gate_report"):
            with pytest.raises(AttributeError):
                getattr(ryd_gate, name)

    def test_metric_exports_live_in_analysis_not_top_level(self):
        import pytest

        import ryd_gate
        from ryd_gate.analysis import gate_metrics

        assert gate_metrics.average_gate_infidelity is not None
        assert gate_metrics.error_budget is not None
        for name in ("average_gate_infidelity", "error_budget"):
            with pytest.raises(AttributeError):
                getattr(ryd_gate, name)

    def test_gates_all_defined(self):
        from ryd_gate import gates

        for name in gates.__all__:
            assert getattr(gates, name) is not None
