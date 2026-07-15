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
    def test_t_eval_without_observables_is_error(self):
        with pytest.raises(ValueError, match="observables"):
            simulate(_system(n=2), backend="mps", t_eval=np.array([_T_GATE]),
                     backend_options=_MPS_OPTS)

    def test_wrong_shape_expression_is_error(self):
        with pytest.raises(ValueError, match="n_sites, local_dim"):
            simulate(_system(n=3), backend="mps", t_eval=np.array([_T_GATE]),
                     observables={"bad": _system(n=2).observables.n("r", 0)},
                     backend_options=_MPS_OPTS)

    def test_non_tn_preset_rejected(self):
        # rb87_7 presets are not TN-capable (O09).
        from ryd_gate.protocols.gate_cz import CZProtocol
        from ryd_gate.protocols.pulses import blackman_pulse

        proto = CZProtocol(
            t_gate_s=0.2e-6, intermediate_detuning_rad_s=TWO_PI * 7.8e9,
            omega_420_max_rad_s=TWO_PI * 100e6, omega_1013_max_rad_s=TWO_PI * 100e6,
            envelope_420=lambda t: blackman_pulse(t, 20e-9, 0.2e-6),
            phase_420_rad=lambda t: 0.0,
        )
        system = RydbergSystem(
            level_structure=level_structure("rb87_7_mp", magnetic_field_G=20.0),
            register=Register.chain(2, spacing_um=5.0),
            protocol=proto,
        )
        with pytest.raises(ValueError, match="TN-capable"):
            simulate(system, backend="mps", backend_options=_MPS_OPTS)


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
