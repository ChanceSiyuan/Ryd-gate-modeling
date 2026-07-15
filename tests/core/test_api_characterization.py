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
    level_structure,
    simulate,
)
from ryd_gate.ir import EvolutionResult


def _chain_1r(n: int = 2, t_gate: float = 0.1) -> RydbergSystem:
    """Minimal non-interacting 1r chain with a trivial constant sweep bound."""
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(n),
        interaction=InteractionSpec(C6=0.0),
        protocol=SweepProtocol(
            t_gate=t_gate,
            omega_half_fn=lambda t: 0.5,
            delta_fn=lambda t: 0.0,
        ),
    )


def test_evolution_result_final_only_contract():
    """simulate(...) returns an EvolutionResult with a normalized final state."""
    system = _chain_1r()
    result = simulate(system)
    assert isinstance(result, EvolutionResult)
    assert result.final_state.shape == (system.dim,)
    assert np.isclose(np.linalg.norm(result.final_state), 1.0)
    assert isinstance(result.metadata, dict)


def test_evolution_result_expectations_contract():
    """With t_eval + observables, times come back exactly and every
    expectation array is complex, time-major, shape == times.shape."""
    system = _chain_1r()
    t_eval = np.linspace(0.0, 0.1, 5)
    result = simulate(
        system, t_eval=t_eval,
        observables={"n_r": system.observables.level_sum("r")},
    )
    times = np.asarray(result.times)
    np.testing.assert_array_equal(times, t_eval)
    arr = result.expectation("n_r")
    assert arr.shape == times.shape
    assert np.iscomplexobj(arr)


def test_level_sum_expectation_semantics():
    """level_sum('r') counts Rydberg occupation on static product states."""
    from ryd_gate.core.observables import _dense_expectation

    system = _chain_1r()
    n_r = system.observables.level_sum("r")
    assert np.isclose(_dense_expectation(n_r, system.product_state(["r", "r"])), 2.0)
    assert np.isclose(_dense_expectation(n_r, system.product_state(["1", "r"])), 1.0)
    assert np.isclose(_dense_expectation(n_r, system.product_state(["1", "1"])), 0.0)


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
        simulate(system, backend="does-not-exist")


def test_rb87_physical_fields_present():
    """rb87_7 systems expose t_rise / Delta on the level structure (pulse-shaping
    + static energies).  The Rabi scale (rabi_eff / time_scale) lives in the CZ
    protocol, not the system, since the 420/1013 blocks are unit-normalized.
    """
    system = RydbergSystem(
        level_structure=level_structure("rb87_7_mp"),
        register=Register.chain(2, spacing_um=3.0),
    )
    ls = system.level_structure
    assert ls.t_rise is not None
    assert ls.Delta != 0.0
    assert ls.rabi_eff is None
    assert ls.time_scale is None
