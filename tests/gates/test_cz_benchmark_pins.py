"""Benchmark pins for the CZ gate library.

Parameter sets are copied verbatim from
``scripts/notebooks/cz_gate_validation_and_errors.ipynb`` (TO dark, AR) and
``scripts/notebooks/01r_saffman_double_arp_exact.ipynb`` ("our" Double-ARP
section). The notebooks store no numeric outputs, so the reference values
below were computed once from the deterministic exact solver at Stage 5
implementation time (2026-06-11) and guard regressions from that point on.

Pin types:

- ``X_TO_DARK`` is a high-fidelity TO optimum (infidelity ~8e-7).
- The legacy ``X_AR`` and the Saffman "our" Double-ARP constants are
  deterministic *path* pins at their documented notebook configurations:
  those exact parameters do not reproduce a high-fidelity gate (the legacy
  ``X_AR`` sits in a non-entangling local minimum ~0.45-0.58). They guard the
  computation path, not gate quality.
- ``X_AR_LUKIN`` and ``X_AR_OUR_OPT`` are *re-optimized* high-fidelity AR
  points produced by ``scripts/optimize_ar_cz.py`` (2026-06-13). The legacy
  single-start Nelder-Mead got trapped in the non-entangling local minimum on
  "our"; a **multi-start** search escapes it. Result: AR reaches high fidelity
  on **both** param sets — lukin at 2.14e-4 and "our" at 9.46e-6 (fidelity
  0.99999). So the "our" dead basin was an optimizer-escape limitation (now
  fixed by multi-start), never a protocol or target bug — confirmed by
  ``scripts/diagnose_ar_target.py`` (exact 01<->10 symmetry; the stuck point
  had low leakage but a wrong conditional phase). The lukin exact evaluation
  costs ~400 s, so its pin test is ``slow``-marked.
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
# Re-optimized high-fidelity AR point on the lukin system (data/ar_opt_lukin.json,
# scripts/optimize_ar_cz.py, 2026-06-13; 303 evals from the X_AR seed).
X_AR_LUKIN = [
    1.152444475742163,
    0.5298612474274744,
    1.0463537649409957,
    0.5247328111843467,
    -1.3495589677508941,
    -0.008058653524462491,
    1.4486084769604992,
    -0.1399323673945825,
]
# Re-optimized high-fidelity AR point on the "our" system (data/ar_opt_our.json,
# scripts/optimize_ar_cz.py multi-start, 2026-06-13; escaped the legacy single-
# start's 0.45 non-entangling local minimum -> fidelity 0.99999).
X_AR_OUR_OPT = [
    1.3417376355208614,
    -0.47514206631153455,
    0.9374165061782775,
    -0.24539458137685877,
    2.755618354280231,
    1.0063163509211008,
    1.9975377136905927,
    -1.6351076243247586,
]

MHZ = 2.0 * np.pi * 1e6

# Reference values: deterministic exact solver, 2026-06-11 (see module docstring).
TO_DARK_INFIDELITY = 7.803243184945e-07
AR_INFIDELITY = 5.764713261273e-01
DOUBLE_ARP_INFIDELITY = 8.705593950813e-01
RYDBERG_DECAY_BUDGET_TOTAL = 8.290118039110158e-04
# Re-optimized AR points (verified reproducible 2026-06-13).
AR_LUKIN_INFIDELITY = 2.144749e-04
AR_OUR_OPT_INFIDELITY = 9.461682e-06


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
