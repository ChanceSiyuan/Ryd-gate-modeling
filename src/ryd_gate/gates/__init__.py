"""Gate library: the documented CZ surface over existing kernel objects.

A domain namespace, not a second implementation tree — the protocol classes
are the kernel protocols, the metric functions are
``analysis.gate_metrics``'s, and only :class:`CZGateReport` /
:func:`cz_gate_report` are new (Stage 5).
"""

from ryd_gate.analysis.gate_metrics import average_gate_infidelity, error_budget
from ryd_gate.gates.cz import CZGateReport, cz_gate_report
from ryd_gate.protocols.gate_cz_ar import ARProtocol
from ryd_gate.protocols.gate_cz_double_arp import DoubleARPProtocol
from ryd_gate.protocols.gate_cz_to import TOProtocol

__all__ = [
    "ARProtocol",
    "CZGateReport",
    "DoubleARPProtocol",
    "TOProtocol",
    "average_gate_infidelity",
    "cz_gate_report",
    "error_budget",
]
