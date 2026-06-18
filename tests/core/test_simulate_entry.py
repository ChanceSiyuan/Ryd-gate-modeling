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


def test_simulate_exact_dense_returns_evolution_result():
    system = _system()
    psi0 = system.ground_state()
    result = ryd_gate.simulate(system, [], psi0, backend="exact_dense")
    assert isinstance(result, EvolutionResult)
    assert result.psi_final.shape == (system.dim,)
    assert np.isclose(np.linalg.norm(result.psi_final), 1.0)


def test_simulate_default_backend_is_exact_dense():
    system = _system()
    psi0 = system.ground_state()
    r_default = ryd_gate.simulate(system, [], psi0)
    r_dense = ryd_gate.simulate(system, [], psi0, backend="exact_dense")
    assert np.allclose(r_default.psi_final, r_dense.psi_final, atol=1e-12)


def test_simulate_bare_exact_is_rejected():
    system = _system()
    with pytest.raises(ValueError, match="exact_dense"):
        ryd_gate.simulate(system, [], system.ground_state(), backend="exact")


def test_simulate_unknown_backend_raises():
    system = _system()
    with pytest.raises(ValueError):
        ryd_gate.simulate(system, [], system.ground_state(), backend="nonexistent")


def test_simulate_exact_dense_and_sparse_agree_on_small_system():
    system = _system()
    psi0 = system.ground_state()
    r_dense = ryd_gate.simulate(system, [], psi0, backend="exact_dense")
    r_sparse = ryd_gate.simulate(system, [], psi0, backend="exact_sparse")
    assert np.allclose(r_dense.psi_final, r_sparse.psi_final, atol=1e-10)


def test_simulate_legacy_expm_aliases_rejected():
    system = _system()
    psi0 = system.ground_state()
    for legacy in ("dense", "dense_expm", "sparse", "sparse_expm"):
        with pytest.raises(ValueError):
            ryd_gate.simulate(system, [], psi0, backend=legacy)


def test_simulate_batch_honors_exact_sparse():
    system = _system()
    batch = _batch_labels(system)
    batched = ryd_gate.simulate(system, [], batch, backend="exact_sparse")
    looped = [ryd_gate.simulate(system, [], s, backend="exact_sparse") for s in batch]
    for rb, rl in zip(batched, looped):
        assert np.allclose(rb.psi_final, rl.psi_final, atol=1e-10)


def _batch_labels(system):
    """Two distinct single-state label-lists valid in this system's basis."""
    lv = system.basis.local_levels
    a, b = lv[0], (lv[1] if len(lv) > 1 else lv[0])
    return [[a, a], [b, b]]


def test_simulate_batch_returns_list():
    system = _system()
    batch = _batch_labels(system)
    results = ryd_gate.simulate(system, [], batch, backend="exact_dense")
    assert isinstance(results, list) and len(results) == len(batch)
    for r in results:
        assert isinstance(r, EvolutionResult)
        assert r.psi_final.shape == (system.dim,)
        assert np.isclose(np.linalg.norm(r.psi_final), 1.0)
        assert r.system is system


def test_simulate_batch_matches_per_state_loop():
    system = _system()
    batch = _batch_labels(system)
    batched = ryd_gate.simulate(system, [], batch, backend="exact_dense")
    looped = [ryd_gate.simulate(system, [], s, backend="exact_dense") for s in batch]
    for rb, rl in zip(batched, looped):
        assert np.allclose(rb.psi_final, rl.psi_final, atol=1e-10)


def test_simulate_batch_matches_simulate_states():
    from ryd_gate.backends.exact import simulate_states

    system = _system()
    batch = _batch_labels(system)
    batched = ryd_gate.simulate(system, [], batch, backend="exact_dense")
    direct = simulate_states(system, batch)
    for rb, rd in zip(batched, direct):
        assert np.allclose(rb.psi_final, rd.psi_final, atol=1e-12)


def test_simulate_flat_label_list_is_single_state():
    system = _system()
    lv = system.basis.local_levels
    result = ryd_gate.simulate(system, [], [lv[0], lv[0]], backend="exact_dense")
    assert isinstance(result, EvolutionResult)
    assert result.psi_final.shape == (system.dim,)


def test_simulate_single_element_batch_returns_length_one_list():
    system = _system()
    lv = system.basis.local_levels
    results = ryd_gate.simulate(system, [], [[lv[0], lv[0]]], backend="exact_dense")
    assert isinstance(results, list) and len(results) == 1
    assert isinstance(results[0], EvolutionResult)


def test_simulate_batch_attaches_observables_per_result():
    system = _system()
    batch = _batch_labels(system)
    results = ryd_gate.simulate(system, [], batch, observables=["sum_nr"], backend="exact_dense")
    for r in results:
        assert r.expectations is not None and "sum_nr" in r.expectations
        assert np.isclose(r.expectation("sum_nr"), r.expectations["sum_nr"])
