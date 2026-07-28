"""Tests for the direct trajectory-optimization engine (ADR-0026)."""

from __future__ import annotations

import numpy as np
import pytest

from qoc.direct import _DirectNLP


def _random_problem(seed, *, slice_sampling="midpoint", regularization=None):
    """Small random two-channel problem (D=2, K=3) plus a random point x."""
    rng = np.random.default_rng(seed)

    def herm(d):
        a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        return a + a.conj().T

    d = 2
    target = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))[0]

    def objective(u_final):
        c = np.trace(target.conj().T @ u_final)
        return 1.0 - abs(c) ** 2 / d**2, -(c / d**2) * target

    nlp = _DirectNLP(
        h0=herm(d),
        controls={"a": herm(d), "b": herm(d)},
        n_slices=3,
        dt=0.2,
        terminal_objective=objective,
        u_bounds={"a": (-2.0, 2.0), "b": (-1.0, 3.0)},
        du_bounds={"a": 5.0},
        ddu_bounds={"b": 7.0},
        fix_endpoints=True,
        regularization=regularization,
        slice_sampling=slice_sampling,
    )
    x = rng.normal(scale=0.5, size=nlp.n)
    return nlp, x


def test_layout_counts():
    nlp, _ = _random_problem(1)
    # D=2 -> M = 2*D^2 = 8; K=3; A=2.
    assert (nlp.dim, nlp.M, nlp.K, nlp.A) == (2, 8, 3, 2)
    assert nlp.n == nlp.K * nlp.M + nlp.A * (3 * nlp.K + 2)   # 24 + 22 = 46
    assert nlp.m == nlp.K * nlp.M + 2 * nlp.A * nlp.K          # 24 + 12 = 36
    assert nlp.lb.shape == nlp.ub.shape == (nlp.n,)
    # endpoint pinning: u knots 0 and K have (0, 0) bounds on every channel
    for a in range(nlp.A):
        c0 = nlp._c_off(a)
        assert nlp.lb[c0] == nlp.ub[c0] == 0.0
        assert nlp.lb[c0 + nlp.K] == nlp.ub[c0 + nlp.K] == 0.0


@pytest.mark.parametrize("slice_sampling", ["midpoint", "left"])
def test_constraint_jacobian_matches_finite_differences(slice_sampling):
    nlp, x = _random_problem(7, slice_sampling=slice_sampling)
    dense = np.zeros((nlp.m, nlp.n))
    np.add.at(dense, (nlp.jac_rows, nlp.jac_cols), nlp.jacobian(x))
    h = 1e-6
    fd = np.zeros_like(dense)
    for i in range(nlp.n):
        e = np.zeros(nlp.n)
        e[i] = h
        fd[:, i] = (nlp.constraints(x + e) - nlp.constraints(x - e)) / (2 * h)
    assert np.max(np.abs(dense - fd)) < 1e-6


def test_objective_gradient_matches_finite_differences():
    nlp, x = _random_problem(11, regularization={"u": 0.3, "du": 0.2, "ddu": 0.1})
    grad = nlp.gradient(x)
    h = 1e-6
    for i in range(nlp.n):
        e = np.zeros(nlp.n)
        e[i] = h
        fd = (nlp.objective(x + e) - nlp.objective(x - e)) / (2 * h)
        assert abs(grad[i] - fd) < 1e-6


def test_rollout_initial_point_is_feasible():
    nlp, _ = _random_problem(3)
    x0 = nlp.initial_point(
        {"a": np.array([0.0, 0.5, 0.4, 0.0]), "b": np.zeros(4)}, None
    )
    assert np.max(np.abs(nlp.constraints(x0))) < 1e-12
    assert np.all(x0 >= nlp.lb) and np.all(x0 <= nlp.ub)


def test_initial_unitaries_override_is_used_verbatim():
    nlp, _ = _random_problem(9)
    stack = np.stack([np.eye(2, dtype=complex)] * nlp.K)
    x0 = nlp.initial_point(None, stack)
    for k in range(1, nlp.K + 1):
        off = nlp._u_off(k)
        re = x0[off : off + 4].reshape(2, 2)
        im = x0[off + 4 : off + 8].reshape(2, 2)
        np.testing.assert_allclose(re + 1j * im, np.eye(2), atol=1e-14)
    # identity stack under a nonzero drift violates the defect constraints
    assert np.max(np.abs(nlp.constraints(x0))) > 1e-6


def test_terminal_objective_contract_is_validated():
    nlp, x = _random_problem(5)
    nlp.terminal_objective = lambda u: 1.0  # not a pair
    with pytest.raises(TypeError, match="pair"):
        nlp.objective(x)
    nlp.terminal_objective = lambda u: (np.inf, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        nlp.objective(x)
    nlp.terminal_objective = lambda u: (0.0, np.zeros(3))
    with pytest.raises(ValueError, match="gradient"):
        nlp.objective(x)


def test_validation_errors():
    ident = np.eye(2)

    def obj(u):
        return 0.0, np.zeros((2, 2))

    def build(**kw):
        base = dict(
            h0=ident,
            controls={"a": ident},
            n_slices=3,
            dt=0.2,
            terminal_objective=obj,
            u_bounds={"a": (-1.0, 1.0)},
        )
        base.update(kw)
        return _DirectNLP(**base)

    build()  # baseline constructs cleanly
    with pytest.raises(ValueError, match="Hermitian"):
        build(h0=np.array([[0.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="n_slices"):
        build(n_slices=1)
    with pytest.raises(ValueError, match="dt"):
        build(dt=0.0)
    with pytest.raises(ValueError, match="u_bounds"):
        build(u_bounds={"a": (0.5, 1.0)})  # violates lo <= 0 <= hi
    with pytest.raises(ValueError, match="u_bounds"):
        build(u_bounds={"b": (-1.0, 1.0)})  # channel mismatch
    with pytest.raises(ValueError, match="du_bounds"):
        build(du_bounds={"a": -1.0})
    with pytest.raises(ValueError, match="slice_sampling"):
        build(slice_sampling="right")
    with pytest.raises(ValueError, match="regularization"):
        build(regularization={"jerk": 1.0})
    with pytest.raises(ValueError, match="initial_controls"):
        build().initial_point({"zz": np.zeros(4)}, None)
    with pytest.raises(ValueError, match="initial_unitaries"):
        build().initial_point(None, np.zeros((2, 2, 2)))
