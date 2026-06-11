"""Tests for simulate_sequence and the lazy exact result layer (results.py)."""

import numpy as np
import pytest

from ryd_gate import (
    DeviceSpec,
    InteractionSpec,
    Pulse,
    Register,
    Sequence,
    SimulationResult,
    Waveform,
    simulate_sequence,
)
from ryd_gate.ir.evolution import EvolutionResult


def _pi_pulse_sequence(n_atoms=1, spacing=20.0):
    """1r sequence with a Blackman pi-area pulse on every atom (resonant)."""
    seq = Sequence(Register.chain(n_atoms, spacing), DeviceSpec.virtual_rb87(), "1r")
    seq.declare_channel("ryd", "rydberg_global")
    seq.add(Pulse.constant_detuning(Waveform.blackman(1000, area=np.pi), 0.0), "ryd")
    return seq


class TestSimulateSequence:
    def test_returns_simulation_result_with_raw(self):
        result = simulate_sequence(_pi_pulse_sequence())
        assert isinstance(result, SimulationResult)
        assert isinstance(result.raw, EvolutionResult)
        assert result.backend == "exact"
        np.testing.assert_allclose(result.statevector(), result.raw.psi_final)

    def test_statevector_is_a_copy(self):
        result = simulate_sequence(_pi_pulse_sequence())
        vec = result.statevector()
        vec[:] = 0.0
        assert np.linalg.norm(result.raw.psi_final) > 0.9

    def test_pi_pulse_physics(self):
        """Blackman area=pi resonant pulse drives |1> -> |r| with P_r ~ 1."""
        result = simulate_sequence(_pi_pulse_sequence())
        assert result.populations("r")[0] == pytest.approx(1.0, abs=1e-3)

    def test_default_psi0_is_initial_level(self):
        """Zero-amplitude pulse: state stays in the 1r initial level |1>."""
        seq = Sequence(Register.chain(2, 20.0), DeviceSpec.virtual_rb87(), "1r")
        seq.declare_channel("ryd", "rydberg_global")
        seq.add(Pulse.constant(1000, 0.0, 0.0), "ryd")
        result = simulate_sequence(seq, interaction=InteractionSpec(C6=0.0))
        np.testing.assert_allclose(result.populations("r"), [0.0, 0.0], atol=1e-12)
        assert result.expectation("sum_n_1") == pytest.approx(2.0)

    def test_populations_match_system_expectation(self):
        result = simulate_sequence(_pi_pulse_sequence(n_atoms=2), interaction=InteractionSpec(C6=0.0))
        system, psi = result.state.system, result.state.psi
        expected = [system.expectation(f"n_r_{i}", psi) for i in range(2)]
        np.testing.assert_allclose(result.populations("r"), expected)

    def test_other_backend_not_stage3(self):
        with pytest.raises(NotImplementedError, match="backend_not_stage3"):
            simulate_sequence(_pi_pulse_sequence(), backend="peps")


class TestSampling:
    def test_counts_sum_and_reproducibility(self):
        result = simulate_sequence(_pi_pulse_sequence())
        counts = result.sample(100, basis="rydberg", seed=1)
        assert sum(counts.values()) == 100
        assert counts == result.sample(100, basis="rydberg", seed=1)
        # pi pulse: essentially all shots in |r> -> bitstring "1"
        assert counts.get("1", 0) >= 99

    def test_full_level_basis_keys(self):
        result = simulate_sequence(_pi_pulse_sequence())
        counts = result.sample(50, basis="full-level", seed=2)
        assert set(counts) <= {"1", "r"}  # single-char labels concatenate

    def test_computational_raises_on_rydberg_population(self):
        result = simulate_sequence(_pi_pulse_sequence())
        with pytest.raises(ValueError, match="noncomputational_population"):
            result.sample(10, basis="computational", seed=3)

    def test_invalid_args(self):
        result = simulate_sequence(_pi_pulse_sequence())
        with pytest.raises(ValueError):
            result.sample(0)
        with pytest.raises(ValueError, match="basis"):
            result.sample(10, basis="parity")


class TestCaching:
    def test_expectation_cached(self):
        result = simulate_sequence(_pi_pulse_sequence())
        calls = {"n": 0}
        original = result.state.expectation

        def counting(observable):
            calls["n"] += 1
            return original(observable)

        result.state.expectation = counting  # instance attribute shadows the method
        assert result.expectation("sum_nr") == result.expectation("sum_nr")
        assert calls["n"] == 1
        result.clear_cache()
        result.expectation("sum_nr")
        assert calls["n"] == 2

    def test_sample_never_cached(self):
        result = simulate_sequence(_pi_pulse_sequence())
        calls = {"n": 0}
        original = result.state.sample

        def counting(n_shots, basis="rydberg", seed=None):
            calls["n"] += 1
            return original(n_shots, basis=basis, seed=seed)

        result.state.sample = counting
        result.sample(10, seed=1)
        result.sample(10, seed=1)
        assert calls["n"] == 2
