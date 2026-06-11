"""ryd_gate.gates is a namespace over kernel objects, not a second tree."""


class TestGateNamespace:
    def test_protocol_classes_are_the_kernel_classes(self):
        from ryd_gate import gates
        from ryd_gate.protocols.gate_cz_ar import ARProtocol
        from ryd_gate.protocols.gate_cz_double_arp import DoubleARPProtocol
        from ryd_gate.protocols.gate_cz_to import TOProtocol

        assert gates.TOProtocol is TOProtocol
        assert gates.ARProtocol is ARProtocol
        assert gates.DoubleARPProtocol is DoubleARPProtocol

    def test_metric_functions_are_gate_metrics_functions(self):
        from ryd_gate import gates
        from ryd_gate.analysis import gate_metrics

        assert gates.average_gate_infidelity is gate_metrics.average_gate_infidelity
        assert gates.error_budget is gate_metrics.error_budget

    def test_top_level_report_exports(self):
        from ryd_gate import CZGateReport, cz_gate_report
        from ryd_gate.gates import cz

        assert CZGateReport is cz.CZGateReport
        assert cz_gate_report is cz.cz_gate_report

    def test_top_level_metric_exports_still_lazy(self):
        from ryd_gate import average_gate_infidelity, error_budget
        from ryd_gate.analysis import gate_metrics

        assert average_gate_infidelity is gate_metrics.average_gate_infidelity
        assert error_budget is gate_metrics.error_budget

    def test_gates_all_defined(self):
        from ryd_gate import gates

        for name in gates.__all__:
            assert getattr(gates, name) is not None
