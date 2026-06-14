"""EvolutionResult convenience accessors (reframe Phase 2).

Tests the methods in isolation by attaching a system directly; the end-to-end
wiring through ryd_gate.simulate(..., observables=[...]) is covered separately.
"""

from collections import Counter

import numpy as np
import pytest

from ryd_gate import InteractionSpec, Register, RydbergSystem, SweepProtocol, simulate
from ryd_gate.ir import EvolutionResult


def _system() -> RydbergSystem:
    return RydbergSystem.from_lattice(
        Register.chain(2),
        "1r",
        interaction=InteractionSpec(C6=0.0),
        protocol=SweepProtocol(
            t_gate=0.1, omega_half_fn=lambda t: 0.5, delta_fn=lambda t: 0.0, n_steps=10
        ),
    )


def test_final_state_alias():
    system = _system()
    psi = system.product_state(["r", "r"])
    result = EvolutionResult(psi_final=psi, system=system)
    assert result.final_state is psi


def test_expectation_precomputed_and_on_demand():
    system = _system()
    psi = system.product_state(["r", "r"])
    result = EvolutionResult(psi_final=psi, system=system, expectations={"custom": 1.23})
    assert result.expectation("custom") == 1.23           # precomputed wins
    assert np.isclose(result.expectation("sum_nr"), 2.0)  # measured on demand


def test_probabilities_normalized():
    system = _system()
    psi = system.product_state(["1", "r"])
    result = EvolutionResult(psi_final=psi, system=system)
    probs = result.probabilities()
    assert probs.shape == (system.dim,)
    assert np.isclose(probs.sum(), 1.0)


def test_sample_product_state_is_deterministic():
    system = _system()
    psi = system.product_state(["1", "r"])
    result = EvolutionResult(psi_final=psi, system=system)
    counts = result.sample(64, seed=0)
    assert isinstance(counts, Counter)
    assert sum(counts.values()) == 64
    assert counts == Counter({"1r": 64})  # |1r> has all the probability mass


def test_expectation_without_context_raises():
    result = EvolutionResult(psi_final=np.array([1.0, 0.0, 0.0, 0.0], dtype=complex))
    with pytest.raises(RuntimeError):
        result.expectation("sum_nr")


def test_sample_without_system_raises():
    result = EvolutionResult(psi_final=np.array([1.0, 0.0, 0.0, 0.0], dtype=complex))
    with pytest.raises(RuntimeError):
        result.sample(10)


def test_simulate_attaches_system_and_observables():
    system = _system()
    result = simulate(system, observables=["sum_nr"])
    assert result.system is system
    assert result.expectations is not None and "sum_nr" in result.expectations
    assert np.isclose(result.expectation("sum_nr"), result.expectations["sum_nr"])
    counts = result.sample(16, seed=0)  # works because the system/basis is attached
    assert sum(counts.values()) == 16


def test_simulate_x_optional_and_psi0_keyword():
    system = _system()
    r1 = simulate(system)                 # x omitted entirely
    r2 = simulate(system, psi0="all_1")   # psi0 by keyword, x omitted
    assert r1.final_state.shape == (system.dim,)
    assert r2.final_state.shape == (system.dim,)
