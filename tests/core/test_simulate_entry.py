"""Unified ryd_gate.simulate(backend=...) dispatcher."""

import numpy as np
import pytest

import ryd_gate
from ryd_gate import InteractionSpec, RydbergSystem, SweepProtocol, level_structure
from ryd_gate.ir import EvolutionResult
from ryd_gate.lattice import Register


def _system():
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(2),
        interaction=InteractionSpec(C6=0.0),
        protocol=SweepProtocol(
            t_gate=0.1,
            omega_half_fn=lambda t: 0.5,
            delta_fn=lambda t: 0.0,
        ),
    )


def test_simulate_exact_ode_returns_evolution_result():
    system = _system()
    result = ryd_gate.simulate(system, backend="exact_ode")
    assert isinstance(result, EvolutionResult)
    assert result.final_state.shape == (system.dim,)
    assert np.isclose(np.linalg.norm(result.final_state), 1.0)


def test_simulate_default_backend_is_exact_ode():
    system = _system()
    r_default = ryd_gate.simulate(system)
    r_ode = ryd_gate.simulate(system, backend="exact_ode")
    assert np.allclose(r_default.final_state, r_ode.final_state, atol=1e-12)


def test_simulate_default_initial_state_is_preset_level():
    """initial_state=None puts every site in the preset level ('1' on 1r)."""
    system = _system()
    r_default = ryd_gate.simulate(system)
    r_labels = ryd_gate.simulate(system, ["1", "1"])
    assert np.allclose(r_default.final_state, r_labels.final_state, atol=1e-12)


def test_simulate_rejects_dense_initial_state():
    """Dense vectors go through the internal exact seam, not public simulate."""
    system = _system()
    with pytest.raises(TypeError):
        ryd_gate.simulate(system, system.ground_state())


def test_simulate_bare_exact_is_rejected():
    """The removal error names the replacement backend."""
    system = _system()
    with pytest.raises(ValueError, match="exact_ode"):
        ryd_gate.simulate(system, backend="exact")


def test_simulate_unknown_backend_raises():
    system = _system()
    with pytest.raises(ValueError):
        ryd_gate.simulate(system, backend="nonexistent")


def test_simulate_batch_honors_exact_ode():
    system = _system()
    batch = _batch_labels(system)
    batched = ryd_gate.simulate(system, batch, backend="exact_ode")
    looped = [ryd_gate.simulate(system, s, backend="exact_ode") for s in batch]
    for rb, rl in zip(batched, looped):
        assert np.allclose(rb.final_state, rl.final_state, atol=1e-8)


def test_simulate_legacy_expm_aliases_rejected():
    system = _system()
    for legacy in ("dense", "dense_expm", "sparse", "sparse_expm"):
        with pytest.raises(ValueError):
            ryd_gate.simulate(system, backend=legacy)


def _batch_labels(system):
    """Two distinct single-state label-lists valid in this system's basis."""
    lv = system.basis.local_levels
    a, b = lv[0], (lv[1] if len(lv) > 1 else lv[0])
    return [[a, a], [b, b]]


def test_simulate_batch_returns_list():
    system = _system()
    batch = _batch_labels(system)
    results = ryd_gate.simulate(system, batch)
    assert isinstance(results, list) and len(results) == len(batch)
    for r in results:
        assert isinstance(r, EvolutionResult)
        assert r.final_state.shape == (system.dim,)
        assert np.isclose(np.linalg.norm(r.final_state), 1.0)
        assert r.basis is system.basis


def test_simulate_batch_matches_per_state_loop():
    system = _system()
    batch = _batch_labels(system)
    batched = ryd_gate.simulate(system, batch)
    looped = [ryd_gate.simulate(system, s) for s in batch]
    for rb, rl in zip(batched, looped):
        assert np.allclose(rb.final_state, rl.final_state, atol=1e-10)


def test_simulate_batch_matches_simulate_states():
    from ryd_gate.backends.exact import simulate_states

    system = _system()
    batch = _batch_labels(system)
    batched = ryd_gate.simulate(system, batch)
    direct = simulate_states(system, batch)
    for rb, rd in zip(batched, direct):
        assert np.allclose(rb.final_state, rd.final_state, atol=1e-12)


def test_simulate_flat_label_list_is_single_state():
    system = _system()
    lv = system.basis.local_levels
    result = ryd_gate.simulate(system, [lv[0], lv[0]])
    assert isinstance(result, EvolutionResult)
    assert result.final_state.shape == (system.dim,)


def test_simulate_single_element_batch_returns_length_one_list():
    system = _system()
    lv = system.basis.local_levels
    results = ryd_gate.simulate(system, [[lv[0], lv[0]]])
    assert isinstance(results, list) and len(results) == 1
    assert isinstance(results[0], EvolutionResult)


def test_simulate_batch_records_observables_per_result():
    system = _system()
    batch = _batch_labels(system)
    results = ryd_gate.simulate(
        system, batch, observables={"n_r": system.observables.level_sum("r")}
    )
    for r in results:
        assert "n_r" in r.expectations
        arr = r.expectation("n_r")
        assert arr.shape == (1,) and np.iscomplexobj(arr)
