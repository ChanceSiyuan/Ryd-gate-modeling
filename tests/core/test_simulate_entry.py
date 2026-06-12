"""Unified ryd_gate.simulate(backend=...) dispatcher."""

import numpy as np
import pytest

import ryd_gate
from ryd_gate import InteractionSpec, RydbergSystem, SweepProtocol
from ryd_gate.ir import EvolutionResult
from ryd_gate.lattice import Register


def _system():
    return RydbergSystem.from_lattice(
        Register.chain(2),
        "1r",
        interaction=InteractionSpec(C6=0.0),
        protocol=SweepProtocol(
            t_gate=0.1,
            omega_half_fn=lambda t: 0.5,
            delta_fn=lambda t: 0.0,
            n_steps=10,
        ),
    )


def test_simulate_exact_returns_evolution_result():
    system = _system()
    psi0 = system.ground_state()
    result = ryd_gate.simulate(system, [], psi0, backend="exact")
    assert isinstance(result, EvolutionResult)
    assert result.psi_final.shape == (system.dim,)
    assert np.isclose(np.linalg.norm(result.psi_final), 1.0)


def test_simulate_unknown_backend_raises():
    system = _system()
    with pytest.raises(ValueError):
        ryd_gate.simulate(system, [], system.ground_state(), backend="nonexistent")
