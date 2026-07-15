"""ExactODEBackend contract: format choice, terminal state, expectations, options.

Patch-5 pins: the only exact backend is the adaptive DOP853 integrator;
``hamiltonian_format`` selects storage/matvec (dense vs sparse-matvec, no
densification on the sparse path); the final state is always the true
``t_gate`` state regardless of ``t_eval``; requested-time expectations are
computed from the internally held trajectory which is then discarded (results
expose no state trajectory); batch solves share the compiled RHS but keep
per-state adaptive error control; unknown backend options, the removed expm
backend names, and invalid ``t_eval``/observables requests error loudly.
"""

import numpy as np
import pytest

import ryd_gate as rg
from ryd_gate.backends.exact import ode as ode_module
from ryd_gate.backends.exact.ode import (
    _DENSE_MEMORY_LIMIT_BYTES,
    ExactODEBackend,
    _dense_bytes_estimate,
)

MHZ = 2 * np.pi * 1e6


def _quench_system(n_side=2, spacing_um=9.0, t_gate=0.5e-6):
    return rg.RydbergSystem(
        level_structure=rg.level_structure("1r"),
        register=rg.Register.square(n_side, spacing_um=spacing_um),
        protocol=rg.TFIMQuenchProtocol(hx=2 * np.pi * 1e6, t_gate=t_gate),
    )


def _chain_system(n_atoms, t_gate=0.05e-6):
    return rg.RydbergSystem(
        level_structure=rg.level_structure("1r"),
        register=rg.Register.chain(n_atoms, spacing_um=9.0),
        protocol=rg.TFIMQuenchProtocol(hx=2 * np.pi * 1e6, t_gate=t_gate),
    )


def test_dense_and_sparse_formats_agree():
    """The two Hamiltonian storage strategies are numerically identical."""
    system = _quench_system()
    dense = rg.simulate(system, backend_options={"hamiltonian_format": "dense"})
    sparse = rg.simulate(system, backend_options={"hamiltonian_format": "sparse"})
    np.testing.assert_allclose(dense.final_state, sparse.final_state, atol=1e-10)
    assert dense.metadata["hamiltonian_format"] == "dense"
    assert sparse.metadata["hamiltonian_format"] == "sparse"


def test_final_state_is_true_terminal_state():
    """t_eval never decides the evolution endpoint: final_state is at t_gate.

    Regression for the DenseODE-era coupling bug class (gate H2): an
    endpoint-free t_eval must not change the final state, the requested times
    come back exactly (no auto-appended endpoint in public times), and each
    expectation array matches times in shape.
    """
    system = _quench_system()
    obs = system.observables
    reference = rg.simulate(system)
    t_eval = np.array([0.1e-6, 0.2e-6])
    partial = rg.simulate(
        system, t_eval=t_eval, observables={"n_r": obs.level_sum("r")}
    )
    np.testing.assert_allclose(partial.final_state, reference.final_state, atol=1e-12)
    np.testing.assert_array_equal(partial.times, t_eval)
    assert partial.expectation("n_r").shape == partial.times.shape == (2,)


def test_endpoint_only_times_and_expectations():
    """t_eval=None: times == [t_gate] (shape (1,)); expectations empty unless
    observables were requested, then shape (1,) complex."""
    system = _quench_system()
    bare = rg.simulate(system)
    np.testing.assert_array_equal(bare.times, [0.5e-6])
    assert bare.expectations == {}
    measured = rg.simulate(
        system, observables={"n_r": system.observables.level_sum("r")}
    )
    arr = measured.expectation("n_r")
    assert arr.shape == (1,) and np.iscomplexobj(arr)
    np.testing.assert_array_equal(measured.final_state, bare.final_state)


def test_analytic_rabi_via_expressions():
    """Gate H1: non-interacting 1r chain under resonant drive shows the
    analytic Rabi flop n_r(t) = sin^2(Omega t / 2) in n('r', site)."""
    omega = 2 * np.pi * 1e6
    t_gate = 1e-6
    system = rg.RydbergSystem(
        level_structure=rg.level_structure("1r"),
        register=rg.Register.chain(2),
        interaction=rg.InteractionSpec(C6=0.0),
        protocol=rg.SweepProtocol(
            t_gate=t_gate, omega_half_fn=lambda t: 0.5 * omega, delta_fn=lambda t: 0.0
        ),
    )
    obs = system.observables
    t_eval = np.linspace(0.0, t_gate, 41)
    res = rg.simulate(
        system,
        t_eval=t_eval,
        observables={"n_r_0": obs.n("r", 0), "n_r_1": obs.n("r", 1)},
    )
    expected = np.sin(0.5 * omega * t_eval) ** 2
    for label in ("n_r_0", "n_r_1"):
        arr = res.expectation(label)
        assert arr.shape == (41,)
        np.testing.assert_allclose(arr.imag, 0.0, atol=1e-10)
        np.testing.assert_allclose(arr.real, expected, atol=1e-6)


def test_complex_coherence_and_dagger():
    """Gate H3: E(ket,bra) expectations are raw complex; dagger conjugates;
    the identity expectation is the (here unit) survival norm."""
    omega = 2 * np.pi * 1e6
    t_gate = 1e-6
    from ryd_gate.protocols import DigitalAnalogProtocol

    system = rg.RydbergSystem(
        level_structure=rg.level_structure("01r"),
        register=rg.Register.chain(1),
        protocol=DigitalAnalogProtocol(t_gate=t_gate, coupling_r1=lambda t: 0.5 * omega),
    )
    obs = system.observables
    coh = obs.E("0", "1", 0)
    t_eval = np.linspace(0.0, t_gate, 5)
    res = rg.simulate(
        system,
        "plus",
        t_eval=t_eval,
        observables={"coh": coh, "coh_dag": coh.dagger(), "one": obs.identity()},
    )
    c = res.expectation("coh")
    assert np.iscomplexobj(c) and c.shape == (5,)
    # pin 3 (patch5_pins.json): resonant pi pulse on |1>->|r>, c0 static:
    # <|0><1|> = cos(pi t / t_gate) / 2, real in this frame.
    expected = 0.5 * np.cos(np.pi * t_eval / t_gate)
    np.testing.assert_allclose(c.real, expected, atol=1e-6)
    np.testing.assert_allclose(c.imag, 0.0, atol=1e-6)
    # dagger of the expression conjugates the expectation
    np.testing.assert_allclose(res.expectation("coh_dag"), np.conj(c), atol=1e-12)
    # identity = survival norm (unitary here)
    np.testing.assert_allclose(res.expectation("one").real, 1.0, atol=1e-8)


def test_auto_picks_sparse_above_memory_limit_without_densifying(monkeypatch):
    """A representative large sparse case: auto must select the matvec path and
    never call a dense conversion; the estimate is recorded in metadata."""
    n_atoms = 13  # dim 8192 -> 16 * 8192**2 = 1 GiB > the 256 MiB limit
    system = _chain_system(n_atoms)
    assert _dense_bytes_estimate(system.dim) > _DENSE_MEMORY_LIMIT_BYTES

    def _no_densify(operator):
        raise AssertionError("dense conversion on the sparse/matvec path")

    monkeypatch.setattr(ode_module, "_as_dense", _no_densify)
    result = rg.simulate(system)
    assert result.metadata["hamiltonian_format"] == "sparse"
    assert result.metadata["estimated_dense_hamiltonian_bytes"] == _dense_bytes_estimate(
        system.dim
    )
    assert np.isclose(np.linalg.norm(result.final_state), 1.0, atol=1e-6)


def test_batch_matches_separate_solves():
    """simulate() batching shares the RHS but reproduces per-state solves."""
    system = _quench_system()
    batch = rg.simulate(system, [["1", "1", "1", "1"], ["r", "1", "1", "1"]])
    single_a = rg.simulate(system, ["1", "1", "1", "1"])
    single_b = rg.simulate(system, ["r", "1", "1", "1"])
    np.testing.assert_allclose(batch[0].final_state, single_a.final_state, atol=1e-12)
    np.testing.assert_allclose(batch[1].final_state, single_b.final_state, atol=1e-12)


def test_solver_diagnostics_in_metadata():
    result = rg.simulate(_quench_system(), backend_options={"rtol": 1e-9, "atol": 1e-13})
    md = result.metadata
    assert md["solver"] == "DOP853"
    assert md["solver_success"] is True
    assert md["rtol"] == 1e-9 and md["atol"] == 1e-13
    assert md["nfev"] > 0


def test_unknown_backend_option_errors():
    with pytest.raises(ValueError, match="unknown exact backend_options"):
        rg.simulate(_quench_system(), backend_options={"n_steps": 100})


def test_removed_expm_backends_error():
    system = _quench_system()
    for name in ("exact", "exact_dense", "exact_sparse"):
        with pytest.raises(ValueError, match="has been removed"):
            rg.simulate(system, backend=name)


def test_bad_hamiltonian_format_errors():
    with pytest.raises(ValueError, match="hamiltonian_format"):
        ExactODEBackend(hamiltonian_format="dense_expm")


# ── preflight contract (gate H5) ─────────────────────────────────────────────


def test_boolean_t_eval_rejected():
    with pytest.raises(TypeError, match="t_eval"):
        rg.simulate(_quench_system(), t_eval=True)


def test_explicit_t_eval_requires_observables():
    system = _quench_system()
    with pytest.raises(ValueError, match="requires observables"):
        rg.simulate(system, t_eval=np.linspace(0.0, 0.5e-6, 5))
    with pytest.raises(ValueError, match="requires observables"):
        rg.simulate(system, t_eval=np.linspace(0.0, 0.5e-6, 5), observables={})


def test_invalid_t_eval_arrays_rejected():
    system = _quench_system()
    obs = {"n_r": system.observables.level_sum("r")}
    with pytest.raises(ValueError, match="non-empty"):
        rg.simulate(system, t_eval=np.array([]), observables=obs)
    with pytest.raises(ValueError, match="one-dimensional"):
        rg.simulate(system, t_eval=np.zeros((2, 2)), observables=obs)
    with pytest.raises(ValueError, match="strictly increasing"):
        rg.simulate(system, t_eval=np.array([0.2e-6, 0.1e-6]), observables=obs)
    with pytest.raises(ValueError, match="strictly increasing"):
        rg.simulate(system, t_eval=np.array([0.1e-6, 0.1e-6]), observables=obs)
    with pytest.raises(ValueError, match="finite"):
        rg.simulate(system, t_eval=np.array([0.0, np.nan]), observables=obs)
    with pytest.raises(ValueError, match="within"):
        rg.simulate(system, t_eval=np.array([-0.1e-6, 0.2e-6]), observables=obs)
    with pytest.raises(ValueError, match="within"):
        rg.simulate(system, t_eval=np.array([0.1e-6, 0.7e-6]), observables=obs)


def test_string_observables_rejected():
    """The string-name observable router is gone: lists and str-valued dicts fail."""
    system = _quench_system()
    with pytest.raises(TypeError, match="dict"):
        rg.simulate(system, observables=["sum_nr"])
    with pytest.raises(TypeError, match="ObservableExpr"):
        rg.simulate(system, observables={"n_r": "sum_nr"})


def test_foreign_expression_rejected():
    """Expressions built for another level structure fail preflight."""
    system = _quench_system()
    other = rg.RydbergSystem(
        level_structure=rg.level_structure("01r"),
        register=rg.Register.chain(2),
        protocol=rg.TFIMQuenchProtocol(hx=MHZ, t_gate=0.5e-6),
    )
    with pytest.raises(ValueError, match="built for"):
        rg.simulate(system, observables={"n_r": other.observables.level_sum("r")})


def test_no_trajectory_on_results():
    """The trajectory is discarded after expectations: no public states."""
    system = _quench_system()
    res = rg.simulate(
        system,
        t_eval=np.linspace(0.0, 0.5e-6, 5),
        observables={"n_r": system.observables.level_sum("r")},
    )
    assert not hasattr(res, "states")
    assert not hasattr(res, "psi_final")


def test_solve_failure_raises(monkeypatch):
    """solve_ivp failures surface as errors, not silently wrong results."""
    import scipy.integrate

    class _Failed:
        success = False
        message = "step size underflow (forced)"

    monkeypatch.setattr(
        scipy.integrate, "solve_ivp", lambda *a, **k: _Failed()
    )
    with pytest.raises(RuntimeError, match="solve_ivp failed"):
        rg.simulate(_quench_system())


def test_private_continuation_seam_matches_single_solve():
    """_simulate_from resumes a solve at t_start from a dense state: chaining
    [0, t_mid] + [t_mid, t_gate] reproduces the single [0, t_gate] solve."""
    from ryd_gate.backends.exact.compiler import ExactCompiler
    from ryd_gate.backends.exact.simulate import _simulate_from

    system = _quench_system()
    t_gate = 0.5e-6
    t_mid = 0.2e-6
    reference = rg.simulate(system)

    # first leg: integrate [0, t_mid] directly on the backend
    ir = ExactCompiler().compile(system)
    first = ExactODEBackend().evolve_many(
        ir, [system.product_state(["1", "1", "1", "1"])], t_mid
    )[0]
    resumed = _simulate_from(system, first.final_state, t_mid)
    assert resumed.metadata["solver"] == "DOP853"
    np.testing.assert_allclose(resumed.final_state, reference.final_state, atol=1e-7)
    np.testing.assert_array_equal(resumed.times, [t_gate])
