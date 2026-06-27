"""Benchmark pins for the CZ gate library.

The reference values below were computed from the deterministic exact solver
under the **explicit |0> model** (2026-06-26) and guard regressions from that
point on.

The dark-branch optima were re-optimized when the Rb87 |0> treatment switched
from the old second-order ``perturbative`` shift to the faithful ``explicit``
model (``scripts/reoptimize_explicit.py``). The explicit |0> changes the
single-qubit light shifts, so the perturbative-era single-qubit Z correction
(``theta``) went stale — re-fitting ``theta`` then polishing recovered every
dark-branch gate to its prior fidelity (and below).

Pin types:

- ``X_TO_DARK`` is a high-fidelity TO optimum (infidelity ~3.7e-7).
- The legacy ``X_AR`` and the Saffman "our" Double-ARP constants are
  deterministic *path* pins at their documented notebook configurations:
  those exact parameters do not reproduce a high-fidelity gate (the legacy
  ``X_AR`` sits in a non-entangling local minimum ~0.65). They guard the
  computation path, not gate quality.
- ``X_AR_LUKIN`` and ``X_AR_OUR_OPT`` are *re-optimized* high-fidelity AR
  points (lukin 1.60e-6, "our" 7.16e-6). The lukin pin test is ``slow``-marked.

The *bright* TO branch (``detuning_sign=-1``) is intentionally absent: with the
intermediate manifold at -9.1 GHz it sits only 2.3 GHz from |0> (-6.835 GHz),
so the spectator |0> leaks ~9e-3 coherently — a near-resonance floor no pulse
optimization can fix.
"""

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem
from ryd_gate.analysis import gate_metrics
from ryd_gate.gates import ARProtocol, DoubleARPProtocol, TOProtocol, cz_gate_report

# x = [A, omega/Omega_eff, phi0, delta/Omega_eff, theta, T/T_scale]
X_TO_DARK = [
    -0.6894097925886826,
    1.040962607910546,
    0.3277877211544321,
    1.5639989822346387,
    0.6689846026179691,
    1.3407418093368753,
]
# x = [omega/Omega_eff, A1, phi1, A2, phi2, delta/Omega_eff, T/T_scale, theta]
X_AR = [0.85973359, 0.39146974, 0.99181418, 0.1924498, -1.17123748, -0.00826712, 1.67429728, 0.28527346]
# Re-optimized high-fidelity AR point on the lukin system, explicit |0> model
# (scripts/reoptimize_explicit.py, 2026-06-26).
X_AR_LUKIN = [
    1.1248729881110775,
    0.6643225074894661,
    1.2040014416272569,
    0.47554852952965454,
    0.9542223492343259,
    -0.06698908370622411,
    1.4484639314193593,
    1.6238023148317802,
]
# Re-optimized high-fidelity AR point on the "our" system, explicit |0> model
# (scripts/reoptimize_explicit.py, 2026-06-26).
X_AR_OUR_OPT = [
    1.3350469419160897,
    -0.47009049836626793,
    1.030513904414609,
    -0.32029923369069635,
    5.039849219039583,
    0.9619699188096813,
    1.9974997193846544,
    1.0453182369350449,
]

MHZ = 2.0 * np.pi * 1e6

# Reference values: deterministic exact solver, explicit |0> model, 2026-06-26
# (see module docstring).
TO_DARK_INFIDELITY = 3.738562268651e-07
AR_INFIDELITY = 6.511247625966e-01
DOUBLE_ARP_INFIDELITY = 9.727856728551e-01
RYDBERG_DECAY_BUDGET_TOTAL = 8.292516768162672e-04
# Re-optimized AR points (explicit |0> model, 2026-06-26).
AR_LUKIN_INFIDELITY = 1.602054501948e-06
AR_OUR_OPT_INFIDELITY = 7.158272490759e-06


def _system(**kwargs):
    return (
        RydbergSystem.set_atom_level("rb87_7", param_set="our", **kwargs)
        .set_atom_geom(Register.chain(2, spacing_um=3.0))
        .build()
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

    @pytest.mark.slow
    def test_ar_lukin_high_fidelity_benchmark(self):
        """Re-optimized AR point on the lukin system is a genuine CZ gate.

        ~400 s exact evaluation (long gate time at this operating point);
        slow-marked. Pins both the high-fidelity guard and the exact value.
        """
        system = (
            RydbergSystem.set_atom_level("rb87_7", param_set="lukin")
            .set_atom_geom(Register.chain(2, spacing_um=3.0))
            .build()
        )
        report = cz_gate_report(
            system, ARProtocol(), X_AR_LUKIN,
            include_error_budget=False, include_residuals=False,
        )
        assert report.infidelity < 1e-3
        assert report.infidelity == pytest.approx(AR_LUKIN_INFIDELITY, rel=1e-3)

    def test_ar_our_high_fidelity_benchmark(self):
        """Multi-start re-optimization reaches a true high-fidelity AR gate on "our".

        This is the resolution of the "our dead basin" investigation: the legacy
        single start was trapped at ~0.45 (a non-entangling local minimum), but a
        multi-start search (scripts/optimize_ar_cz.py) escapes it to 9.46e-6 —
        proving the dead basin was an optimizer limitation, not a bug.
        """
        report = cz_gate_report(
            _system(), ARProtocol(), X_AR_OUR_OPT,
            include_error_budget=False, include_residuals=False,
        )
        assert report.infidelity < 1e-4
        assert report.infidelity == pytest.approx(AR_OUR_OPT_INFIDELITY, rel=1e-3)

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
