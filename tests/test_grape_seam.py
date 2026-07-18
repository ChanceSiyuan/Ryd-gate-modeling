"""Study-level seam integration (ADR-0024): ryd_gate's bilinear export feeding
qoc's GRAPE engine and gradient method. This is deliberately the only test file
importing both packages — the study is the glue layer; neither package imports
the other."""

import numpy as np

import qoc
from qoc import grape
from ryd_gate import Register, RydbergSystem, bilinear_control_model, level_structure
from ryd_gate.protocols import SweepProtocol

T_GATE = 1e-6
N_SLICES = 16


def _model():
    system = RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(1),
        protocol=SweepProtocol(t_gate_s=T_GATE, omega_half_rad_s=lambda t: 0.0, detuning_rad_s=lambda t: 0.0),
    )
    return bilinear_control_model(system, states=[["1"], ["r"]])


def _value_and_grad_factory(h0, controls, states):
    """One-parameter study: a flat drive amplitude on the x quadrature, judged by
    the |1> -> |r> transfer loss L = 1 - |<r|psi(T)>|^2 (optimum: a*T = pi/2)."""
    time_grid = np.linspace(0.0, T_GATE, N_SLICES + 1)
    r_index = int(np.argmax(np.abs(states[("r",)])))
    zeros = np.zeros(N_SLICES)

    def control_map(params):
        return {"E[r,1]:x": np.full(N_SLICES, params["amp"]), "E[r,1]:y": zeros, "E[r,r]": zeros}

    def control_pullback(params, channel_gradients):
        return {"amp": float(np.sum(channel_gradients["E[r,1]:x"]))}

    def terminal_objective(final_states):
        psi = final_states[0]
        amp_r = psi[r_index]
        costate = np.zeros_like(psi)
        costate[r_index] = -amp_r
        return 1.0 - abs(amp_r) ** 2, [costate]

    def value_and_grad(params):
        return grape.value_and_grad(
            params,
            h0=h0,
            controls=controls,
            initial_states=[states[("1",)]],
            time_grid=time_grid,
            control_map=control_map,
            control_pullback=control_pullback,
            terminal_objective=terminal_objective,
        )

    return value_and_grad


def test_seam_gradient_matches_finite_differences():
    value_and_grad = _value_and_grad_factory(*_model())
    a0 = 0.37 * np.pi / (2.0 * T_GATE)
    _value, gradient = value_and_grad({"amp": a0})

    eps = 1e-3 * a0
    plus, _ = value_and_grad({"amp": a0 + eps})
    minus, _ = value_and_grad({"amp": a0 - eps})
    finite_difference = (plus - minus) / (2.0 * eps)
    np.testing.assert_allclose(gradient["amp"], finite_difference, rtol=1e-5)


def test_seam_optimization_reaches_the_analytic_pi_pulse():
    h0, controls, states = _model()
    value_and_grad = _value_and_grad_factory(h0, controls, states)
    a_star = np.pi / (2.0 * T_GATE)

    result = qoc.minimize(
        lambda params: value_and_grad(params)[0],
        {"amp": 0.6 * a_star},
        method="l-bfgs-b",
        bounds={"amp": (0.0, 3.0 * a_star)},
        scales={"amp": a_star},
        options={"gradient": lambda params: value_and_grad(params)[1]},
    )
    assert result.success
    assert result.best_loss < 1e-9
    np.testing.assert_allclose(result.best_parameters["amp"], a_star, rtol=1e-4)
