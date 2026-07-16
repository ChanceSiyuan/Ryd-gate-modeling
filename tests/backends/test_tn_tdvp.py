"""MPS TDVP time-evolution tests on the post-refactor system-based seam (E22/E23).

``simulate(system, backend="mps", backend_options={time_step_s, bond_dimension,
discarded_weight_tolerance})`` -> EvolutionResult with expectation/amplitude/sample.
"""

import numpy as np
import pytest

from ryd_gate import RydbergSystem, level_structure
from ryd_gate.lattice import Register
from ryd_gate.protocols.sweep import SweepProtocol
from ryd_gate.simulate import simulate

pytest.importorskip("tenpy")

TWO_PI = 2 * np.pi
_T_GATE = 0.3e-6
_MPS_OPTS = {"time_step_s": 1e-8, "bond_dimension": 32, "discarded_weight_tolerance": 1e-10}


def _sweep(t_gate=_T_GATE, local=None):
    return SweepProtocol(
        t_gate_s=t_gate,
        omega_half_rad_s=lambda t: TWO_PI * 2.0e6 * np.sin(np.pi * t / t_gate),
        detuning_rad_s=lambda t: TWO_PI * 1.0e6,
        local_detuning_rad_s=local,
    )


def _system(preset="1r", n=4, spacing=8.0, proto=None):
    return RydbergSystem(
        level_structure=level_structure(preset),
        register=Register.chain(n, spacing_um=spacing),
        protocol=proto if proto is not None else _sweep(),
    )


def _n_r_total(system):
    obs = system.observables
    return sum(obs.n("r", i) for i in range(system.N))


class TestMPSEvolution:
    def test_endpoint_only_contract(self):
        system = _system(n=3)
        res = simulate(system, backend="mps", backend_options=_MPS_OPTS)
        np.testing.assert_array_equal(res.times, [system.t_gate])
        assert isinstance(res.amplitude(["1", "1", "1"]), complex)

    def test_observable_streaming_at_requested_times(self):
        system = _system(n=3)
        t_eval = np.linspace(0.1e-6, _T_GATE, 4)
        res = simulate(
            system, backend="mps", t_eval=t_eval,
            observables={"n_r": _n_r_total(system)}, backend_options=_MPS_OPTS,
        )
        np.testing.assert_array_equal(res.times, t_eval)
        values = res.expectation("n_r")
        assert values.shape == t_eval.shape
        assert values.dtype == np.float64
        with pytest.raises(KeyError, match="missing"):
            res.expectation("missing")

    def test_start_from_all_r_stationary_under_zero_drive(self):
        # all-|r> with zero drive: population stays in |r> (interaction is diagonal).
        system = _system(n=3, proto=SweepProtocol(
            t_gate_s=0.2e-6,
            omega_half_rad_s=lambda t: 0.0,
            detuning_rad_s=lambda t: TWO_PI * 1.0e6,
        ))
        res = simulate(
            system, ["r", "r", "r"], backend="mps",
            observables={"n_r": _n_r_total(system)}, backend_options=_MPS_OPTS,
        )
        np.testing.assert_allclose(res.expectation("n_r")[-1], 3.0, atol=1e-6)


class TestMPSOptionSchema:
    def test_missing_backend_options_is_error(self):
        with pytest.raises(ValueError, match="E23"):
            simulate(_system(n=2), backend="mps")

    def test_unknown_key_is_error(self):
        with pytest.raises(ValueError, match="unknown"):
            simulate(_system(n=2), backend="mps",
                     backend_options={**_MPS_OPTS, "svd_min": 1e-12})

    def test_missing_key_is_error(self):
        with pytest.raises(ValueError, match="missing"):
            simulate(_system(n=2), backend="mps",
                     backend_options={"time_step_s": 1e-9, "bond_dimension": 8})

    def test_nonpositive_time_step_is_error(self):
        with pytest.raises(ValueError, match="time_step_s"):
            simulate(_system(n=2), backend="mps",
                     backend_options={**_MPS_OPTS, "time_step_s": 0.0})

    def test_bool_bond_dimension_is_error(self):
        with pytest.raises(ValueError, match="bond_dimension"):
            simulate(_system(n=2), backend="mps",
                     backend_options={**_MPS_OPTS, "bond_dimension": True})

    def test_cumulative_discarded_weight_exceeded_raises(self):
        # bond_dimension=1 forces a product-state ansatz; a strong-blockade pair
        # (small spacing) entangles, so the cumulative discarded weight blows past
        # the tolerance.
        system = _system(n=4, spacing=5.0, proto=SweepProtocol(
            t_gate_s=0.3e-6,
            omega_half_rad_s=lambda t: TWO_PI * 5.0e6,
            detuning_rad_s=lambda t: 0.0,
        ))
        with pytest.raises(RuntimeError, match="discarded-weight"):
            simulate(
                system, backend="mps",
                observables={"n_r": _n_r_total(system)},
                backend_options={"time_step_s": 1e-8, "bond_dimension": 1,
                                 "discarded_weight_tolerance": 1e-6},
            )


class TestMPSPreflight:
    def test_wrong_shape_expression_is_error(self):
        with pytest.raises(ValueError, match="n_sites, local_dim"):
            simulate(_system(n=3), backend="mps", t_eval=np.array([_T_GATE]),
                     observables={"bad": _system(n=2).observables.n("r", 0)},
                     backend_options=_MPS_OPTS)


class TestMPSInitialStateAndAnchors:
    def test_mps_batch_returns_tuple_matching_single_solves(self):
        # nested initial-state list -> tuple, via the TN batch dispatch loop.
        system = _system(n=3)
        obs = {"n_r": _n_r_total(system)}
        batch = simulate(system, [["1", "1", "1"], ["r", "1", "1"]], backend="mps",
                         observables=obs, backend_options=_MPS_OPTS)
        assert isinstance(batch, tuple) and len(batch) == 2
        a = simulate(system, ["1", "1", "1"], backend="mps", observables=obs, backend_options=_MPS_OPTS)
        b = simulate(system, ["r", "1", "1"], backend="mps", observables=obs, backend_options=_MPS_OPTS)
        np.testing.assert_allclose(batch[0].expectation("n_r"), a.expectation("n_r"), atol=1e-10)
        np.testing.assert_allclose(batch[1].expectation("n_r"), b.expectation("n_r"), atol=1e-10)

    def test_plus_initial_state_on_mps(self):
        # "plus" builds the (|0>+|1>)/sqrt(2) product MPS (initial_state.py plus branch);
        # t_eval=[0.0] also exercises the record-at-start anchor branch.
        system = _system(preset="01r", n=3)
        obs = {f"n_{lvl}_{i}": system.observables.n(lvl, i) for lvl in ("0", "1", "r")
               for i in range(system.N)}
        res = simulate(system, "plus", backend="mps", t_eval=np.array([0.0]),
                       observables=obs, backend_options=_MPS_OPTS)
        np.testing.assert_array_equal(res.times, [0.0])
        for i in range(system.N):
            assert res.expectation(f"n_0_{i}")[0] == pytest.approx(0.5, abs=1e-9)
            assert res.expectation(f"n_1_{i}")[0] == pytest.approx(0.5, abs=1e-9)
            assert res.expectation(f"n_r_{i}")[0] == pytest.approx(0.0, abs=1e-9)

    def test_t0_anchor_recorded_then_evolves(self):
        # record_at_start: the t=0 anchor holds the initial-state value; a later anchor differs.
        system = _system(n=2)
        obs = {"n_r": _n_r_total(system)}
        res = simulate(system, backend="mps", t_eval=np.array([0.0, _T_GATE]),
                       observables=obs, backend_options=_MPS_OPTS)
        np.testing.assert_array_equal(res.times, [0.0, _T_GATE])
        assert res.expectation("n_r")[0] == pytest.approx(0.0, abs=1e-9)  # |1,1> has n_r=0
        assert res.expectation("n_r")[-1] > 1e-3

    def test_identity_constant_observable_on_mps(self):
        # An observable carrying a constant (factorless) identity term exercises the
        # identity_coeff branch of the MPS lowering; `1 - n_r0` here (built via the private
        # term type, since the public factory never emits a factorless term).
        from ryd_gate.core.observables import ObservableExpr, _Term

        system = _system(n=3)
        n_r0 = system.observables.n("r", 0)
        one_minus_nr0 = ObservableExpr(
            (_Term(1.0 + 0.0j, ()),) + tuple(_Term(-t.coefficient, t.factors) for t in n_r0._terms),
            n_r0._n_sites, n_r0._local_dim,
        )
        obs = {"one_minus_nr0": one_minus_nr0, "n_r0": n_r0}
        t_eval = np.array([0.15e-6, _T_GATE])
        exact = simulate(system, t_eval=t_eval, observables=obs)
        mps = simulate(system, backend="mps", t_eval=t_eval, observables=obs, backend_options=_MPS_OPTS)
        np.testing.assert_allclose(mps.expectation("one_minus_nr0"),
                                   1.0 - mps.expectation("n_r0"), atol=1e-9)
        np.testing.assert_allclose(mps.expectation("one_minus_nr0"),
                                   exact.expectation("one_minus_nr0"), atol=1e-3)


def test_exact_mps_parity_n_r():
    system = _system(preset="01r", n=4, spacing=8.0)
    obs = system.observables
    observables = {f"n_r_{i}": obs.n("r", i) for i in range(system.N)}
    t_eval = np.linspace(0.1e-6, _T_GATE, 4)
    res_exact = simulate(system, t_eval=t_eval, observables=observables)
    res_mps = simulate(system, backend="mps", t_eval=t_eval, observables=observables,
                       backend_options=_MPS_OPTS)
    for label in observables:
        np.testing.assert_allclose(
            res_mps.expectation(label), res_exact.expectation(label),
            atol=1e-3, err_msg=label,
        )
