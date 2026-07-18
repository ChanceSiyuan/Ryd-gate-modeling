"""Single-start scalar-loss minimization (ADR-0015, 0018, 0020, 0022).

qoc's independence certificate: no test under ``tests/qoc`` imports ``ryd_gate``.
"""

import numpy as np
import pytest

import qoc

TARGET_W = np.array([1.0, -1.0, 3.0])
X0 = {"a": 0.0, "w": np.zeros(3)}


def quadratic(params):
    return (params["a"] - 2.0) ** 2 + float(np.sum((params["w"] - TARGET_W) ** 2))


def quadratic_gradient(params):
    return {"a": 2.0 * (params["a"] - 2.0), "w": 2.0 * (params["w"] - TARGET_W)}


@pytest.mark.parametrize("method", ["nelder-mead", "powell", "l-bfgs-b"])
def test_each_registered_method_minimizes_a_quadratic(method):
    result = qoc.minimize(quadratic, X0, method=method)
    assert result.method == method
    assert result.success
    assert result.best_loss < 1e-3
    assert isinstance(result.best_parameters["a"], float)
    assert result.best_parameters["w"].shape == (3,)
    np.testing.assert_allclose(result.best_parameters["a"], 2.0, atol=2e-2)
    np.testing.assert_allclose(result.best_parameters["w"], TARGET_W, atol=2e-2)
    assert result.n_evaluations > 0
    assert result.n_iterations > 0


def test_unknown_method_name_is_rejected():
    with pytest.raises(ValueError, match="unknown optimization method"):
        qoc.minimize(quadratic, X0, method="BFGS")


def test_named_bounds_are_enforced():
    result = qoc.minimize(quadratic, X0, method="l-bfgs-b", bounds={"a": (-np.inf, 1.0)})
    assert result.best_parameters["a"] <= 1.0 + 1e-12
    np.testing.assert_allclose(result.best_parameters["a"], 1.0, atol=1e-6)
    np.testing.assert_allclose(result.best_parameters["w"], TARGET_W, atol=1e-4)


def test_named_scales_solve_a_badly_conditioned_problem_in_physical_units():
    def loss(params):
        return (1e6 * params["b"] - 3.0) ** 2

    result = qoc.minimize(loss, {"b": 0.0}, method="l-bfgs-b", scales={"b": 1e-6})
    assert result.best_loss < 1e-10
    np.testing.assert_allclose(result.best_parameters["b"], 3e-6, rtol=1e-4)


def test_evaluation_callback_sees_every_successful_evaluation():
    events = []

    def callback(index, params, value, best):
        events.append((index, params["a"], value, best))

    result = qoc.minimize(quadratic, X0, method="nelder-mead", callback=callback)
    assert len(events) == result.n_evaluations
    assert [event[0] for event in events] == list(range(1, len(events) + 1))
    best_losses = [event[3] for event in events]
    assert all(b2 <= b1 for b1, b2 in zip(best_losses, best_losses[1:]))
    assert all(event[3] <= event[2] for event in events)


def test_non_finite_and_non_scalar_losses_fail_fast():
    with pytest.raises(ValueError, match="finite real scalar"):
        qoc.minimize(lambda params: float("nan"), {"a": 0.0}, method="nelder-mead")
    with pytest.raises(ValueError, match="finite real scalar"):
        qoc.minimize(lambda params: 1.0 + 1.0j, {"a": 0.0}, method="nelder-mead")
    with pytest.raises(ValueError, match="finite real scalar"):
        qoc.minimize(lambda params: np.zeros(2), {"a": 0.0}, method="nelder-mead")


def test_loss_exceptions_propagate_unconverted():
    def broken(params):
        raise RuntimeError("simulation exploded")

    with pytest.raises(RuntimeError, match="simulation exploded"):
        qoc.minimize(broken, {"a": 0.0}, method="powell")


def test_gradient_option_drives_lbfgsb():
    calls = {"n": 0}

    def gradient(params):
        calls["n"] += 1
        return quadratic_gradient(params)

    result = qoc.minimize(quadratic, X0, method="l-bfgs-b", options={"gradient": gradient})
    assert calls["n"] > 0
    assert result.best_loss < 1e-10
    np.testing.assert_allclose(result.best_parameters["a"], 2.0, rtol=1e-5)


def test_gradient_option_is_rejected_for_gradient_free_methods():
    with pytest.raises(ValueError, match="gradient"):
        qoc.minimize(quadratic, X0, method="nelder-mead", options={"gradient": quadratic_gradient})
