"""Discrete-adjoint GRAPE engine on hand-built toy models (ADR-0024).

qoc's independence certificate: no test under ``tests/qoc`` imports ``ryd_gate``.
All toys are dimensionless (times of order one, controls of order one).
"""

import numpy as np
import pytest

from qoc import grape

SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SZ = np.diag([1.0, -1.0]).astype(complex)
KET0 = np.array([1.0, 0.0], dtype=complex)


def transfer_objective(final_states):
    """L = 1 - |<1|psi>|^2 for the first state; costate = dL/d(conj psi) = -|1><1| psi."""
    psi = final_states[0]
    amp = psi[1]
    costate = np.array([0.0, -amp], dtype=complex)
    return 1.0 - abs(amp) ** 2, [costate]


def test_constant_drive_matches_the_analytic_rabi_value_and_gradient():
    """With H = a * sigma_x over total time T: L = cos^2(aT), dL/da = -T sin(2aT)."""
    n_slices = 7
    time_grid = np.linspace(0.0, 1.0, n_slices + 1)
    a0 = 0.4 * np.pi

    def control_map(params):
        return {"x": np.full(n_slices, params["a"])}

    def control_pullback(params, channel_gradients):
        return {"a": float(np.sum(channel_gradients["x"]))}

    value, gradient = grape.value_and_grad(
        {"a": a0},
        h0=np.zeros((2, 2), dtype=complex),
        controls={"x": SX},
        initial_states=[KET0],
        time_grid=time_grid,
        control_map=control_map,
        control_pullback=control_pullback,
        terminal_objective=transfer_objective,
    )
    np.testing.assert_allclose(value, np.cos(a0) ** 2, rtol=1e-12)
    np.testing.assert_allclose(gradient["a"], -np.sin(2.0 * a0), rtol=1e-10)


def _per_slice_setup(n_slices):
    time_grid = np.linspace(0.0, 1.0, n_slices + 1)
    params = {
        "ux": 0.8 * np.sin(np.linspace(0.3, 2.4, n_slices)),
        "uz": 0.5 * np.cos(np.linspace(-1.0, 1.7, n_slices)),
    }

    def control_map(p):
        return {"x": np.asarray(p["ux"], dtype=float), "z": np.asarray(p["uz"], dtype=float)}

    def control_pullback(p, channel_gradients):
        return {"ux": channel_gradients["x"], "uz": channel_gradients["z"]}

    kwargs = dict(
        h0=0.3 * SZ,
        controls={"x": SX, "z": SZ},
        initial_states=[KET0],
        time_grid=time_grid,
        control_map=control_map,
        control_pullback=control_pullback,
        terminal_objective=transfer_objective,
    )
    return params, kwargs


def test_gradient_matches_central_finite_differences():
    params, kwargs = _per_slice_setup(n_slices=6)
    _value, gradient = grape.value_and_grad(params, **kwargs)

    eps = 1e-6
    for name in ("ux", "uz"):
        for k in range(len(params[name])):
            plus = {key: np.array(val) for key, val in params.items()}
            minus = {key: np.array(val) for key, val in params.items()}
            plus[name][k] += eps
            minus[name][k] -= eps
            f_plus, _ = grape.value_and_grad(plus, **kwargs)
            f_minus, _ = grape.value_and_grad(minus, **kwargs)
            fd = (f_plus - f_minus) / (2.0 * eps)
            np.testing.assert_allclose(gradient[name][k], fd, rtol=1e-5, atol=1e-9)


def test_multiple_initial_states_sum_through_the_costates():
    """A two-state objective's gradient is the sum of the single-state gradients."""
    n_slices = 5
    time_grid = np.linspace(0.0, 1.0, n_slices + 1)
    ket1 = np.array([0.0, 1.0], dtype=complex)

    def control_map(p):
        return {"x": np.asarray(p["ux"], dtype=float)}

    def control_pullback(p, channel_gradients):
        return {"ux": channel_gradients["x"]}

    def single(initial, objective):
        return grape.value_and_grad(
            {"ux": np.linspace(0.2, 1.1, n_slices)},
            h0=0.2 * SZ,
            controls={"x": SX},
            initial_states=initial,
            time_grid=time_grid,
            control_map=control_map,
            control_pullback=control_pullback,
            terminal_objective=objective,
        )

    def objective_a(final_states):
        return transfer_objective(final_states)

    def objective_b(final_states):
        psi = final_states[0]
        amp = psi[0]
        return 1.0 - abs(amp) ** 2, [np.array([-amp, 0.0], dtype=complex)]

    def objective_sum(final_states):
        va, (ca,) = objective_a([final_states[0]])
        vb, (cb,) = objective_b([final_states[1]])
        return va + vb, [ca, cb]

    value_a, grad_a = single([KET0], objective_a)
    value_b, grad_b = single([ket1], objective_b)
    value_ab, grad_ab = single([KET0, ket1], objective_sum)
    np.testing.assert_allclose(value_ab, value_a + value_b, rtol=1e-12)
    np.testing.assert_allclose(grad_ab["ux"], grad_a["ux"] + grad_b["ux"], rtol=1e-10)


def test_invalid_inputs_fail_fast():
    n_slices = 3
    time_grid = np.linspace(0.0, 1.0, n_slices + 1)
    good = dict(
        h0=np.zeros((2, 2), dtype=complex),
        controls={"x": SX},
        initial_states=[KET0],
        time_grid=time_grid,
        control_map=lambda p: {"x": np.full(n_slices, p["a"])},
        control_pullback=lambda p, g: {"a": float(np.sum(g["x"]))},
        terminal_objective=transfer_objective,
    )

    with pytest.raises(ValueError, match="Hermitian"):
        grape.value_and_grad({"a": 0.1}, **{**good, "h0": np.array([[0.0, 1.0], [0.0, 0.0]])})
    with pytest.raises(ValueError, match="channels do not match"):
        grape.value_and_grad({"a": 0.1}, **{**good, "control_map": lambda p: {"y": np.zeros(n_slices)}})
    with pytest.raises(ValueError, match="must be real"):
        grape.value_and_grad({"a": 0.1}, **{**good, "control_map": lambda p: {"x": np.full(n_slices, 1.0j)}})
    with pytest.raises(ValueError, match="strictly increasing"):
        grape.value_and_grad({"a": 0.1}, **{**good, "time_grid": np.array([0.0, 0.5, 0.5, 1.0])})
    with pytest.raises(ValueError, match="one costate per initial state"):
        grape.value_and_grad({"a": 0.1}, **{**good, "terminal_objective": lambda finals: (0.0, [])})
    with pytest.raises(ValueError, match="finite real scalar"):
        grape.value_and_grad(
            {"a": 0.1}, **{**good, "terminal_objective": lambda finals: (np.nan, [np.zeros(2, dtype=complex)])}
        )
