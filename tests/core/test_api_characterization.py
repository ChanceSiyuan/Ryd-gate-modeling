"""Characterization tests pinning public-API contracts before the API reframe.

These lock behavior the reframe must preserve (the ``EvolutionResult`` field
contract, observable semantics, and the physical-parameter metadata that the
gate / addressing layers read). New ergonomics (richer result object, optional
``x``, unified ``observables=``) are tested separately as they land.

Kept deliberately small and fast: tiny ``1r`` exact solves plus metadata-only
``rb87_7`` construction.
"""

import numpy as np
import pytest

from ryd_gate import (
    InteractionSpec,
    Register,
    RydbergSystem,
    SweepProtocol,
    simulate,
)
from ryd_gate.ir import EvolutionResult


def _chain_1r(n: int = 2, t_gate: float = 0.1, n_steps: int = 10) -> RydbergSystem:
    """Minimal non-interacting 1r chain with a trivial constant sweep bound."""
    return (
        RydbergSystem.set_atom_level("1r")
        .set_atom_geom(Register.chain(n), interaction=InteractionSpec(C6=0.0))
        .set_protocol(
            SweepProtocol(
                t_gate=t_gate,
                omega_half_fn=lambda t: 0.5,
                delta_fn=lambda t: 0.0,
                n_steps=n_steps,
            )
        )
    )


def test_evolution_result_final_only_contract():
    """simulate(...) returns an EvolutionResult with a normalized final state."""
    system = _chain_1r()
    result = simulate(system, [], system.ground_state(), backend="exact_dense")
    assert isinstance(result, EvolutionResult)
    assert result.psi_final.shape == (system.dim,)
    assert np.isclose(np.linalg.norm(result.psi_final), 1.0)
    assert isinstance(result.metadata, dict)


def test_evolution_result_trajectory_contract():
    """With t_eval, states/times are populated and mutually consistent."""
    system = _chain_1r()
    t_eval = np.linspace(0.0, 0.1, 5)
    result = simulate(system, [], system.ground_state(), backend="exact_dense", t_eval=t_eval)
    times = np.asarray(result.times)
    states = np.asarray(result.states)
    assert times.ndim == 1 and times.shape[0] >= 2
    assert states.shape[0] == times.shape[0]
    assert states.shape[-1] == system.dim


def test_system_expectation_sum_nr_semantics():
    """system.expectation('sum_nr', psi) counts Rydberg occupation."""
    system = _chain_1r()
    assert np.isclose(system.expectation("sum_nr", system.product_state(["r", "r"])), 2.0)
    assert np.isclose(system.expectation("sum_nr", system.product_state(["1", "r"])), 1.0)
    assert np.isclose(system.expectation("sum_nr", system.product_state(["1", "1"])), 0.0)


def test_ground_state_and_dim_contract():
    """ground_state() is a unit vector of length local_dim**N in the 1r basis."""
    system = _chain_1r()
    psi = system.ground_state()
    assert system.N == 2
    assert system.dim == 2 ** system.N
    assert psi.shape == (system.dim,)
    assert np.isclose(np.linalg.norm(psi), 1.0)


def test_simulate_unknown_backend_raises():
    system = _chain_1r()
    with pytest.raises(ValueError):
        simulate(system, [], system.ground_state(), backend="does-not-exist")


def test_rb87_physical_metadata_present():
    """rb87_7 systems expose t_rise / Delta via metadata (pulse-shaping + static
    energies).  The Rabi scale (rabi_eff / time_scale) now lives in the CZ protocol,
    not the system, since the 420/1013 blocks are unit-normalized.
    """
    system = (
        RydbergSystem.set_atom_level("rb87_7", param_set="our")
        .set_atom_geom(Register.chain(2, spacing_um=3.0))
        .build()
    )
    assert system.meta("t_rise", None) is not None
    assert system.meta("Delta", 0.0) != 0.0
    assert system.meta("rabi_eff") is None
    assert system.meta("time_scale") is None
