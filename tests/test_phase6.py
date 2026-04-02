"""Tests for Phase 6: MonteCarloRunner, observable_metrics, and Protocol.phase_420 deprecation.

Covers:
- MonteCarloRunner with Analog3LevelModel + detuning noise
- MonteCarloRunner with amplitude noise
- measure_observables against ground state
- state_overlap for known states
- norm_squared for normalized state
- Protocol subclass without phase_420 (only get_drive_coefficients needed)
"""

import numpy as np
import pytest

from ryd_gate.analysis.observable_metrics import (
    measure_observables,
    norm_squared,
    state_overlap,
)
from ryd_gate.core.models.analog_3level import Analog3LevelModel
from ryd_gate.protocols.base import Protocol
from ryd_gate.protocols.sweep import SweepProtocol
from ryd_gate.solvers.base import EvolutionResult
from ryd_gate.solvers.monte_carlo_runner import MonteCarloRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analog_model():
    return Analog3LevelModel.from_defaults()


@pytest.fixture(scope="module")
def sweep_protocol():
    return SweepProtocol()


@pytest.fixture(scope="module")
def sweep_x():
    return [-1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# MonteCarloRunner: detuning noise
# ---------------------------------------------------------------------------


class TestMonteCarloRunnerDetuningNoise:
    """MonteCarloRunner with detuning noise produces correctly shaped results."""

    def test_run_states_structure(self, analog_model, sweep_protocol, sweep_x):
        """Run 5 shots, verify nested list structure."""
        runner = MonteCarloRunner(analog_model, sweep_protocol, sweep_x)
        runner.setup_detuning_noise(sigma_detuning_hz=1e3)

        # Ground state |g,g>
        psi0 = np.zeros(9, dtype=complex)
        psi0[0] = 1.0

        results = runner.run_states([psi0], n_shots=5, seed=42)

        # Outer list has n_shots entries
        assert len(results) == 5
        # Each shot has one result per initial state
        for shot_results in results:
            assert len(shot_results) == 1
            assert isinstance(shot_results[0], EvolutionResult)
            assert shot_results[0].psi_final.shape == (9,)

    def test_norm_preserved_under_detuning_noise(self, analog_model, sweep_protocol, sweep_x):
        """Detuning noise is Hermitian -- norm should be preserved."""
        runner = MonteCarloRunner(analog_model, sweep_protocol, sweep_x)
        runner.setup_detuning_noise(sigma_detuning_hz=1e3)

        psi0 = np.zeros(9, dtype=complex)
        psi0[0] = 1.0

        results = runner.run_states([psi0], n_shots=3, seed=123)

        for shot_results in results:
            norm = np.linalg.norm(shot_results[0].psi_final)
            np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_multiple_initial_states(self, analog_model, sweep_protocol, sweep_x):
        """Run with two initial states, verify both are evolved per shot."""
        runner = MonteCarloRunner(analog_model, sweep_protocol, sweep_x)
        runner.setup_detuning_noise(sigma_detuning_hz=1e3)

        psi_gg = np.zeros(9, dtype=complex)
        psi_gg[0] = 1.0
        psi_ge = np.zeros(9, dtype=complex)
        psi_ge[1] = 1.0

        results = runner.run_states([psi_gg, psi_ge], n_shots=3, seed=99)

        assert len(results) == 3
        for shot_results in results:
            assert len(shot_results) == 2
            assert shot_results[0].psi_final.shape == (9,)
            assert shot_results[1].psi_final.shape == (9,)


# ---------------------------------------------------------------------------
# MonteCarloRunner: amplitude noise
# ---------------------------------------------------------------------------


class TestMonteCarloRunnerAmplitudeNoise:
    """MonteCarloRunner with amplitude noise applies scaling correctly."""

    def test_amplitude_noise_produces_results(self, analog_model, sweep_protocol, sweep_x):
        """Amplitude noise should not crash and should produce valid states."""
        runner = MonteCarloRunner(analog_model, sweep_protocol, sweep_x)
        runner.setup_amplitude_noise(sigma_amplitude=0.05)

        psi0 = np.zeros(9, dtype=complex)
        psi0[0] = 1.0

        results = runner.run_states([psi0], n_shots=5, seed=42)

        assert len(results) == 5
        for shot_results in results:
            assert isinstance(shot_results[0], EvolutionResult)
            # Norm should still be preserved (amplitude_scale is Hermitian scaling)
            norm = np.linalg.norm(shot_results[0].psi_final)
            np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_amplitude_noise_varies_results(self, analog_model, sweep_protocol, sweep_x):
        """Different shots should produce different final states due to noise."""
        runner = MonteCarloRunner(analog_model, sweep_protocol, sweep_x)
        runner.setup_amplitude_noise(sigma_amplitude=0.1)

        psi0 = np.zeros(9, dtype=complex)
        psi0[0] = 1.0

        results = runner.run_states([psi0], n_shots=3, seed=42)

        psi_0 = results[0][0].psi_final
        psi_1 = results[1][0].psi_final
        # With 10% amplitude noise, states should differ
        assert not np.allclose(psi_0, psi_1, atol=1e-10)


# ---------------------------------------------------------------------------
# measure_observables
# ---------------------------------------------------------------------------


class TestMeasureObservables:
    """Observable-based metric functions with Analog3LevelModel."""

    def test_measure_all_ground_state(self, analog_model):
        """Ground state |g,g>: pop_g=2, pop_e=0, pop_r=0."""
        psi0 = np.zeros(9, dtype=complex)
        psi0[0] = 1.0
        result = EvolutionResult(psi_final=psi0)

        obs = measure_observables(analog_model, result)

        np.testing.assert_allclose(obs["pop_g"], 2.0, atol=1e-12)
        np.testing.assert_allclose(obs["pop_e"], 0.0, atol=1e-12)
        np.testing.assert_allclose(obs["pop_r"], 0.0, atol=1e-12)

    def test_measure_selected_observables(self, analog_model):
        """Measure only selected observables."""
        psi0 = np.zeros(9, dtype=complex)
        psi0[0] = 1.0
        result = EvolutionResult(psi_final=psi0)

        obs = measure_observables(analog_model, result, observable_names=["pop_g", "pop_r"])

        assert set(obs.keys()) == {"pop_g", "pop_r"}
        np.testing.assert_allclose(obs["pop_g"], 2.0, atol=1e-12)
        np.testing.assert_allclose(obs["pop_r"], 0.0, atol=1e-12)

    def test_measure_per_atom_ground_state(self, analog_model):
        """Per-atom observables on |g,g>."""
        psi0 = np.zeros(9, dtype=complex)
        psi0[0] = 1.0
        result = EvolutionResult(psi_final=psi0)

        obs = measure_observables(analog_model, result, observable_names=["pop_A_g", "pop_B_g"])

        np.testing.assert_allclose(obs["pop_A_g"], 1.0, atol=1e-12)
        np.testing.assert_allclose(obs["pop_B_g"], 1.0, atol=1e-12)

    def test_measure_excited_state(self, analog_model):
        """State |g,e>: pop_g=1, pop_e=1."""
        psi_ge = np.zeros(9, dtype=complex)
        psi_ge[1] = 1.0  # |g> x |e> in 3-level: index 0*3 + 1 = 1
        result = EvolutionResult(psi_final=psi_ge)

        obs = measure_observables(analog_model, result, observable_names=["pop_g", "pop_e"])

        np.testing.assert_allclose(obs["pop_g"], 1.0, atol=1e-12)
        np.testing.assert_allclose(obs["pop_e"], 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# state_overlap
# ---------------------------------------------------------------------------


class TestStateOverlap:
    """Test state_overlap for known states."""

    def test_identical_states(self):
        psi = np.array([1.0, 0.0, 0.0], dtype=complex)
        assert np.isclose(state_overlap(psi, psi), 1.0)

    def test_orthogonal_states(self):
        psi = np.array([1.0, 0.0, 0.0], dtype=complex)
        phi = np.array([0.0, 1.0, 0.0], dtype=complex)
        assert np.isclose(state_overlap(psi, phi), 0.0)

    def test_superposition_overlap(self):
        """Equal superposition overlaps with basis state give 0.5."""
        psi = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        target = np.array([1.0, 0.0], dtype=complex)
        np.testing.assert_allclose(state_overlap(psi, target), 0.5, atol=1e-12)

    def test_phase_does_not_affect_overlap(self):
        """Global phase should not change overlap."""
        psi = np.array([1.0, 0.0], dtype=complex)
        phi = np.exp(1j * 0.7) * psi
        np.testing.assert_allclose(state_overlap(psi, phi), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# norm_squared
# ---------------------------------------------------------------------------


class TestNormSquared:
    """Test norm_squared utility."""

    def test_normalized_state(self):
        psi = np.array([1.0, 0.0, 0.0], dtype=complex)
        np.testing.assert_allclose(norm_squared(psi), 1.0, atol=1e-15)

    def test_unnormalized_state(self):
        psi = np.array([2.0, 0.0], dtype=complex)
        np.testing.assert_allclose(norm_squared(psi), 4.0, atol=1e-15)

    def test_superposition_normalized(self):
        psi = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        np.testing.assert_allclose(norm_squared(psi), 1.0, atol=1e-15)

    def test_complex_state(self):
        psi = np.array([1j, 0.0], dtype=complex)
        np.testing.assert_allclose(norm_squared(psi), 1.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Protocol subclass without phase_420
# ---------------------------------------------------------------------------


class TestProtocolWithoutPhase420:
    """Verify Protocol can be subclassed with only get_drive_coefficients."""

    def test_subclass_without_phase_420(self):
        """A protocol implementing only get_drive_coefficients (not phase_420) should work."""

        class MinimalProtocol(Protocol):
            @property
            def n_params(self) -> int:
                return 1

            def validate_params(self, x):
                if len(x) != 1:
                    raise ValueError

            def unpack_params(self, x, system):
                return {"t_gate": x[0]}

            def get_drive_coefficients(self, t, params):
                return {
                    "drive_420": np.exp(-1j * t),
                    "drive_420_dag": np.exp(1j * t),
                    "lightshift_zero": 1.0,
                }

        proto = MinimalProtocol()
        # Should not raise -- phase_420 uses the default fallback
        coeffs = proto.get_drive_coefficients(0.5, {"t_gate": 1.0})
        assert "drive_420" in coeffs

        # phase_420 default should return the drive_420 coefficient
        phase = proto.phase_420(0.5, {"t_gate": 1.0})
        np.testing.assert_allclose(phase, np.exp(-0.5j), atol=1e-12)

    def test_existing_protocols_still_work(self):
        """SweepProtocol (which has its own phase_420) should still work."""
        proto = SweepProtocol()
        params = {
            "delta_start": 1.0,
            "delta_end": 2.0,
            "t_gate": 1.0,
            "t_rise": 0.0,
            "blackmanflag": False,
            "_system_type": "atomic",
        }
        # Both paths should work
        phase = proto.phase_420(0.5, params)
        assert isinstance(phase, (complex, np.complexfloating))

        coeffs = proto.get_drive_coefficients(0.5, params)
        assert "drive_420" in coeffs
        assert "drive_420_dag" in coeffs
        assert "lightshift_zero" in coeffs
