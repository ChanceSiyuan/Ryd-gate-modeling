"""Gate library: the documented CZ surface over existing kernel objects.

A domain namespace, not a second implementation tree — the protocol classes
are the kernel protocols, the metric functions are
``analysis.gate_metrics``'s, and only :class:`CZGateReport` /
:func:`cz_gate_report` are new (Stage 5).

``cz_gate_report`` runs the three CZ basis-state evolutions once (through the
shared overlap helper in ``analysis.gate_metrics``) and packages infidelity,
phase diagnostics, residual populations, and the decay error budget into a
serializable ``CZGateReport``. It is a data assembly step, not a simulator:
all physics goes through ``average_gate_infidelity``'s machinery and
``error_budget``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ryd_gate.analysis import gate_metrics
from ryd_gate.analysis.gate_metrics import average_gate_infidelity, error_budget
from ryd_gate.core.serialization import check_schema, json_ready, schema_tag
from ryd_gate.protocols.gate_cz import ARProtocol, DoubleARPProtocol, TOProtocol

__all__ = [
    "ARProtocol",
    "CZGateReport",
    "DoubleARPProtocol",
    "TOProtocol",
    "average_gate_infidelity",
    "cz_gate_report",
    "error_budget",
]


@dataclass(frozen=True)
class CZGateReport:
    """Deterministic CZ gate metrics for one (system, protocol, x) point.

    ``protocol`` is the protocol class name (e.g. ``"ARProtocol"``);
    ``parameters`` is the exact ``x`` vector as floats. ``fidelity`` is
    derived (``1 - infidelity``) and never stored in the payload.
    """

    protocol: str
    parameters: tuple[float, ...]
    infidelity: float
    phase_error_rad: float
    theta_rad: float
    residuals: Mapping[str, float] = field(default_factory=dict)
    error_budget: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def fidelity(self) -> float:
        return 1.0 - self.infidelity

    def to_dict(self) -> dict:
        return {
            "schema": schema_tag("cz-gate-report"),
            "protocol": self.protocol,
            "parameters": list(self.parameters),
            "infidelity": self.infidelity,
            "phase_error_rad": self.phase_error_rad,
            "theta_rad": self.theta_rad,
            "residuals": json_ready(dict(self.residuals), "report.residuals"),
            "error_budget": json_ready(
                {key: dict(val) for key, val in self.error_budget.items()},
                "report.error_budget",
            ),
            "metadata": json_ready(dict(self.metadata), "report.metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CZGateReport":
        check_schema(data, "cz-gate-report")
        return cls(
            protocol=data["protocol"],
            parameters=tuple(float(v) for v in data["parameters"]),
            infidelity=float(data["infidelity"]),
            phase_error_rad=float(data["phase_error_rad"]),
            theta_rad=float(data["theta_rad"]),
            residuals=dict(data.get("residuals", {})),
            error_budget={key: dict(val) for key, val in data.get("error_budget", {}).items()},
            metadata=dict(data.get("metadata", {})),
        )


def cz_gate_report(
    system,
    protocol,
    x: Sequence[float],
    *,
    include_error_budget: bool = True,
    include_residuals: bool = True,
    initial_states: list[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CZGateReport:
    """Build a :class:`CZGateReport` from the existing gate metrics.

    The protocol must expose ``unpack_params(...)["theta"]`` (the CZ
    single-qubit phase); otherwise ``ValueError("cz_report.protocol_not_cz")``.
    ``error_budget`` runs only when *include_error_budget* (it needs the
    decay-rate metadata of physical rb87_7 systems).
    """
    x_list = [float(v) for v in x]
    protocol.validate_params(list(x_list))
    params = protocol.unpack_params(list(x_list), system)
    if "theta" not in params:
        raise ValueError("cz_report.protocol_not_cz")

    overlaps, theta, residuals = gate_metrics._cz_overlaps(
        system, protocol, x_list, collect_residuals=include_residuals
    )
    infidelity = float(gate_metrics._nielsen_infidelity(overlaps))
    # With the Rz corrections already applied, an ideal CZ has all overlap
    # phases equal to zero; the residual conditional-phase error is their
    # second difference (equivalent to arg(ã11) - 2·arg(ã01) + arg(ã00) - π
    # on the raw overlaps, mod 2π).
    phase_error = _wrap_to_pi(
        np.angle(overlaps["a11"])
        - 2.0 * np.angle(overlaps["a01"])
        + np.angle(overlaps["a00"])
    )

    budget: dict = {}
    if include_error_budget:
        budget = gate_metrics.error_budget(
            system, protocol, x_list, initial_states=initial_states
        )

    return CZGateReport(
        protocol=type(protocol).__name__,
        parameters=tuple(x_list),
        infidelity=infidelity,
        phase_error_rad=float(phase_error),
        theta_rad=float(theta),
        residuals=dict(residuals or {}),
        error_budget=budget,
        metadata=dict(metadata or {}),
    )


def _wrap_to_pi(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)
