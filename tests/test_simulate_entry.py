"""Unified ryd_gate.simulate(backend=...) dispatcher."""

import numpy as np
import pytest

import ryd_gate
from ryd_gate import InteractionSpec, RydbergSystem, SweepProtocol
from ryd_gate.ir.evolution import EvolutionResult
from ryd_gate.lattice import make_chain


def _system():
    return RydbergSystem.from_lattice(
        make_chain(2),
        "1r",
        interaction=InteractionSpec(C6=0.0),
        protocol=SweepProtocol(n_steps=10),
    )


def test_simulate_exact_returns_evolution_result():
    system = _system()
    psi0 = system.ground_state()
    result = ryd_gate.simulate(system, [-1.0, 1.0, 0.1], psi0, backend="exact")
    assert isinstance(result, EvolutionResult)
    assert result.psi_final.shape == (system.dim,)
    assert np.isclose(np.linalg.norm(result.psi_final), 1.0)


def test_simulate_unknown_backend_raises():
    system = _system()
    with pytest.raises(ValueError):
        ryd_gate.simulate(system, [-1.0, 1.0, 0.1], system.ground_state(), backend="nonexistent")
