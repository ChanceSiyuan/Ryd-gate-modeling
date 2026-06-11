"""Benchmark pins for the CZ gate library.

Parameter sets are copied verbatim from
``scripts/notebooks/cz_gate_validation_and_errors.ipynb`` (TO dark, AR) and
``scripts/notebooks/01r_saffman_double_arp_exact.ipynb`` ("our" Double-ARP
section). The notebooks store no numeric outputs, so the reference values
below were computed once from the deterministic exact solver at Stage 5
implementation time (2026-06-11) and guard regressions from that point on.

Only the TO dark point is a high-fidelity optimum on the current kernel
(infidelity ~8e-7). The AR and Double-ARP values are deterministic *path*
pins at their documented notebook configurations: X_AR and the Saffman "our"
constants do not reproduce a high-fidelity gate under the current protocol
conventions on any system flag combination (checked over
blackmanflag x detuning_sign x param_set at pin time).
"""

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem
from ryd_gate.analysis import gate_metrics
from ryd_gate.gates import ARProtocol, DoubleARPProtocol, TOProtocol, cz_gate_report

# x = [A, omega/Omega_eff, phi0, delta/Omega_eff, theta, T/T_scale]
X_TO_DARK = [
    -0.6989301339711643,
    1.0296229082590798,
    0.3759232324550267,
    1.5710180991068543,
    1.4454279613697887,
    1.3406239758422793,
]
# x = [omega/Omega_eff, A1, phi1, A2, phi2, delta/Omega_eff, T/T_scale, theta]
X_AR = [0.85973359, 0.39146974, 0.99181418, 0.1924498, -1.17123748, -0.00826712, 1.67429728, 0.28527346]

MHZ = 2.0 * np.pi * 1e6

# Reference values: deterministic exact solver, 2026-06-11 (see module docstring).
TO_DARK_INFIDELITY = 7.803243184945e-07
AR_INFIDELITY = 5.764713261273e-01
DOUBLE_ARP_INFIDELITY = 8.705593950813e-01
RYDBERG_DECAY_BUDGET_TOTAL = 8.290118039110158e-04


def _system(**kwargs):
    return RydbergSystem.from_lattice(
        Register.chain(2, spacing_um=3.0), "rb87_7", param_set="our", **kwargs
    )


class TestBenchmarkPins:
    def test_to_dark_benchmark(self):
        report = cz_gate_report(
            _system(blackmanflag=True, detuning_sign=1),
            TOProtocol(),
            X_TO_DARK,
            include_error_budget=False,
            include_residuals=False,
        )
        # Well-optimized guard inherited from the legacy phase test.
        assert report.infidelity < 1e-4
        assert report.infidelity == pytest.approx(TO_DARK_INFIDELITY, rel=1e-3)

    def test_ar_benchmark(self):
        report = cz_gate_report(
            _system(),
            ARProtocol(),
            X_AR,
            include_error_budget=False,
            include_residuals=False,
        )
        assert report.infidelity == pytest.approx(AR_INFIDELITY, rel=1e-3)

    def test_double_arp_benchmark(self):
        protocol = DoubleARPProtocol(
            omega_max=17.0 * MHZ,
            delta_max=23.0 * MHZ,
            t_gate=0.54e-6,
            sigma=0.175 * 0.54e-6,
            n_steps=80,
            compensate_stark=True,
            stark_compensation_sign=-1.0,
        )
        report = cz_gate_report(
            _system(), protocol, [],
            include_error_budget=False, include_residuals=False,
        )
        assert report.parameters == ()
        assert report.infidelity == pytest.approx(DOUBLE_ARP_INFIDELITY, rel=1e-3)

    def test_error_budget_pin(self):
        system = _system(blackmanflag=True, detuning_sign=1, enable_rydberg_decay=True)
        budget = gate_metrics.error_budget(system, TOProtocol(), X_TO_DARK)
        assert budget["rydberg_decay"]["total"] == pytest.approx(
            RYDBERG_DECAY_BUDGET_TOTAL, rel=1e-3
        )
        components = budget["rydberg_decay"]
        assert components["total"] == pytest.approx(
            components["XYZ"] + components["AL"] + components["LG"], rel=1e-9
        )
