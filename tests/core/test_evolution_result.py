"""EvolutionResult contract (Patch 5): stored-only expectations, lazy sample.

The result is a plain container: ``final_state`` (true t_gate state), ``times``
(exactly the requested t_eval; ``[t_gate]`` when None), ``expectations``
(complex, time-major, shape == times.shape), ``metadata``, ``basis``.  No
trajectory, no attached system, no on-demand measurement.
"""

from collections import Counter

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


def _system() -> RydbergSystem:
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(2),
        interaction=InteractionSpec(C6=0.0),
        protocol=SweepProtocol(
            t_gate=0.1, omega_half_fn=lambda t: 0.5, delta_fn=lambda t: 0.0
        ),
    )


def test_result_fields_and_shapes():
    system = _system()
    obs = system.observables
    result = simulate(system, observables={"n_r": obs.level_sum("r")})
    assert isinstance(result, EvolutionResult)
    assert result.final_state.shape == (system.dim,)
    np.testing.assert_array_equal(result.times, [0.1])
    arr = result.expectation("n_r")
    assert isinstance(arr, np.ndarray) and arr.shape == (1,)
    assert np.iscomplexobj(arr)
    assert result.basis is system.basis
    assert isinstance(result.metadata, dict)


def test_no_deleted_surface():
    """psi_final / states / probabilities / attached system are gone."""
    result = simulate(_system())
    for name in ("psi_final", "states", "probabilities", "system"):
        assert not hasattr(result, name)


def test_expectation_unknown_name_is_keyerror():
    system = _system()
    result = simulate(system, observables={"n_r": system.observables.level_sum("r")})
    with pytest.raises(KeyError, match="requested"):
        result.expectation("sum_nr")
    bare = simulate(system)
    with pytest.raises(KeyError):
        bare.expectation("n_r")


def test_expectations_track_times_shape():
    system = _system()
    t_eval = np.linspace(0.0, 0.1, 7)
    result = simulate(
        system, t_eval=t_eval,
        observables={"n_r": system.observables.level_sum("r")},
    )
    np.testing.assert_array_equal(result.times, t_eval)
    assert result.expectation("n_r").shape == result.times.shape == (7,)


def test_sample_is_lazy_deterministic_and_label_decoded():
    system = _system()
    result = simulate(system, ["1", "r"])
    # deterministic: same seed -> identical Counter
    a = result.sample(64, seed=0)
    b = result.sample(64, seed=0)
    assert isinstance(a, Counter)
    assert a == b
    assert sum(a.values()) == 64
    # labels decode through basis site order
    assert all(len(key) == 2 and set(key) <= {"1", "r"} for key in a)
    # different seed may differ but stays a valid distribution
    c = result.sample(64, seed=1)
    assert sum(c.values()) == 64


def test_sample_conditional_on_survival():
    """Weights renormalize by the surviving norm on each call (docstring
    semantics: conditional on survival, no loss label)."""
    basis = _system().basis
    psi = np.array([0.6, 0.0, 0.0, 0.0], dtype=complex)  # ||psi||^2 = 0.36
    result = EvolutionResult(
        final_state=psi, times=np.array([0.1]), expectations={},
        metadata={}, basis=basis,
    )
    counts = result.sample(32, seed=3)
    assert counts == Counter({"11": 32})


def test_sample_requires_dense_state_and_int_seed():
    basis = _system().basis
    result = EvolutionResult(
        final_state=object(), times=np.array([0.1]), expectations={},
        metadata={}, basis=basis,
    )
    with pytest.raises(TypeError, match="dense final state"):
        result.sample(10, seed=0)
    ok = EvolutionResult(
        final_state=np.array([1.0, 0, 0, 0], dtype=complex),
        times=np.array([0.1]), expectations={}, metadata={}, basis=basis,
    )
    with pytest.raises(TypeError, match="seed"):
        ok.sample(10, seed=None)
    with pytest.raises(ValueError, match="shots"):
        ok.sample(0, seed=0)


def test_simulate_default_initial_state_matches_explicit_labels():
    system = _system()
    r1 = simulate(system)                            # preset level ('1' on 1r)
    r2 = simulate(system, initial_state=["1", "1"])  # explicit label list
    assert r1.final_state.shape == (system.dim,)
    assert r2.final_state.shape == (system.dim,)
    assert np.allclose(r1.final_state, r2.final_state, atol=1e-12)
