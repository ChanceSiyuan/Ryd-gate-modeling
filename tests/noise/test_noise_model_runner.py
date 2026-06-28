"""NoiseModel → MonteCarloRunner configuration: exact unit mapping, no new engine."""

import numpy as np
import pytest

from ryd_gate import NoiseModel, Register, RydbergSystem, configure_monte_carlo_runner, level_structure
from ryd_gate.backends.exact import MonteCarloRunner
from ryd_gate.protocols.gate_cz import TOProtocol

X_TO = [
    -0.6894097925886826,
    1.040962607910546,
    0.3277877211544321,
    1.5639989822346387,
    0.6689846026179691,
    1.3407418093368753,
]


def _system(n_atoms=2, **kwargs):
    return (
        RydbergSystem.set_atom_level("rb87_7", param_set="our", **kwargs)
        .set_atom_geom(Register.chain(n_atoms, spacing_um=3.0))
        .build()
    )


def _runner(n_atoms=2):
    return MonteCarloRunner(_system(n_atoms).with_protocol(TOProtocol()), X_TO)


class TestConfigure:
    def test_returns_same_runner_object(self):
        runner = _runner()
        assert configure_monte_carlo_runner(runner, NoiseModel()) is runner

    def test_detuning_conversion_is_exact(self):
        runner = _runner()
        configure_monte_carlo_runner(runner, NoiseModel(detuning_sigma_rad_per_us=0.02))
        assert runner._sigma_detuning_rad == pytest.approx(0.02 * 1e6, rel=1e-12)

    def test_amplitude_and_local_rin_sigmas(self):
        runner = _runner()
        configure_monte_carlo_runner(
            runner, NoiseModel(amp_sigma=0.01, local_rin_sigma=0.03)
        )
        assert runner._sigma_amplitude == 0.01
        assert runner._sigma_local_rin == 0.03

    def test_zero_fields_do_not_enable_noise(self):
        runner = _runner()
        configure_monte_carlo_runner(runner, NoiseModel(runs=4))
        assert runner._sigma_detuning_rad is None
        assert runner._sigma_amplitude is None
        assert runner._sigma_local_rin is None
        assert runner._sigma_pos_um is None

    def test_position_scalar_expands_and_converts_um_to_m(self):
        runner = _runner()
        configure_monte_carlo_runner(runner, NoiseModel(position_sigma_um=0.07))
        # setup_position_noise takes meters and stores um internally.
        assert runner._sigma_pos_um == pytest.approx((0.07, 0.07, 0.07), rel=1e-12)

    def test_position_tuple_passes_through(self):
        runner = _runner()
        configure_monte_carlo_runner(
            runner, NoiseModel(position_sigma_um=(0.07, 0.07, 0.13))
        )
        assert runner._sigma_pos_um == pytest.approx((0.07, 0.07, 0.13), rel=1e-12)

    def test_position_requires_two_atoms(self):
        runner = _runner(n_atoms=3)
        with pytest.raises(ValueError, match="noise.position_two_atom_only"):
            configure_monte_carlo_runner(runner, NoiseModel(position_sigma_um=0.07))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"state_prep_error": 0.01},
            {"p_false_pos": 0.01},
            {"p_false_neg": 0.01},
            {"temperature_uK": 10.0},
            {"laser_waist_um": 100.0},
        ],
    )
    def test_runtime_not_stage4_fields_raise(self, kwargs):
        runner = _runner()
        with pytest.raises(ValueError, match="noise.runtime_not_stage4"):
            configure_monte_carlo_runner(runner, NoiseModel(**kwargs))

    def test_invalid_data_raises_before_mutation(self):
        runner = _runner()
        with pytest.raises(ValueError, match="noise.nonnegative"):
            configure_monte_carlo_runner(runner, NoiseModel(amp_sigma=-1.0))
        assert runner._sigma_amplitude is None


class TestDecayPhysicalKwargs:
    @pytest.mark.parametrize("preset", ["rb87_7", "analog_3"])
    def test_decay_kwargs_reproduce_manual_flags(self, preset):
        spec = level_structure(preset)
        noise = NoiseModel(rydberg_decay=True, intermediate_decay=True)
        manual = (
            RydbergSystem.set_atom_level(
                preset,
                **spec.physical_kwargs(),
                enable_rydberg_decay=True,
                enable_intermediate_decay=True,
            )
            .set_atom_geom(Register.chain(2, spacing_um=3.0))
            .build()
        )
        via_noise = (
            RydbergSystem.set_atom_level(
                preset, **spec.physical_kwargs(), **noise.physical_kwargs()
            )
            .set_atom_geom(Register.chain(2, spacing_um=3.0))
            .build()
        )
        assert via_noise.meta("enable_rydberg_decay") is True
        assert via_noise.meta("enable_intermediate_decay") is True
        h_manual = np.asarray(manual.blocks.get("H_const").matrix)
        h_via = np.asarray(via_noise.blocks.get("H_const").matrix)
        np.testing.assert_allclose(np.diag(h_via).imag, np.diag(h_manual).imag)
        assert np.any(np.diag(h_via).imag != 0.0)

    def test_no_decay_flags_keep_hermitian_diagonal(self):
        system = _system()
        h_const = np.asarray(system.blocks.get("H_const").matrix)
        assert np.all(np.diag(h_const).imag == 0.0)


class TestRunnerStatistics:
    def test_configured_runner_matches_manual_setup(self):
        from ryd_gate.backends.exact import SparseExpmBackend

        sigma_hz = 130e3
        rad_per_us = 2 * np.pi * sigma_hz / 1e6

        # Same backend and parameters on both sides; the test compares
        # configuration paths, not solver accuracy or gate quality. A short
        # gate window keeps the stiff rb87_7 evolution inside the fast suite
        # (cost scales with ||H||*t_gate, not with solver settings).
        x_fast = list(X_TO)
        x_fast[5] = 0.13
        manual = MonteCarloRunner(
            _system().with_protocol(TOProtocol(blackman=False)),
            x_fast,
            backend=SparseExpmBackend(n_steps=24),
        )
        manual.setup_detuning_noise(sigma_hz)
        manual.setup_amplitude_noise(0.01)
        result_manual = manual.run_gate_fidelity(n_shots=2, seed=11)

        configured = MonteCarloRunner(
            _system().with_protocol(TOProtocol(blackman=False)),
            x_fast,
            backend=SparseExpmBackend(n_steps=24),
        )
        configure_monte_carlo_runner(
            configured,
            NoiseModel(runs=2, detuning_sigma_rad_per_us=rad_per_us, amp_sigma=0.01),
        )
        result_configured = configured.run_gate_fidelity(n_shots=2, seed=11)

        np.testing.assert_allclose(
            result_configured.fidelities, result_manual.fidelities, rtol=1e-9
        )
        assert result_configured.n_shots == result_manual.n_shots == 2
