"""ObservableExpr algebra, lowering, and dense evaluation (rewritten API).

Pure unit tests of the immutable scalar expression algebra exposed as
``system.observables`` — exactly two primitives ``E(ket, bra, site)`` and
``n(level, site)`` plus the closed set ``+``, ``-``, unary ``-``, scalar ``*``,
``@``, ``.dagger()`` and the Python ``sum()`` seed (O11/O13). The private
lowered form and the private dense evaluator (raw complex ``<psi|O|psi>``, no
norm division) are pinned here because both backends build on them.
"""

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.core.model import BasisSpec
from ryd_gate.core.observables import (
    ObservableExpr,
    ObservableFactory,
    _dense_expectation,
)
from ryd_gate.core.states import dense_product_state
from ryd_gate.protocols import SweepProtocol

# ── helpers ──────────────────────────────────────────────────────────────────


def _idle_sweep(t_gate=1.0):
    """A do-nothing sweep (compatible with 1r/01r), only needed to bind a system."""
    return SweepProtocol(
        t_gate_s=t_gate,
        omega_half_rad_s=lambda t: 0.0,
        detuning_rad_s=lambda t: 0.0,
    )


def _system(name="01r", n=2):
    return RydbergSystem(
        level_structure=level_structure(name),
        register=Register.chain(n),
        protocol=_idle_sweep(),
    )


def _factory(levels, n):
    """A bare ObservableFactory + BasisSpec (no ARC, no protocol)."""
    basis = BasisSpec(
        site_labels=tuple(str(i) for i in range(n)),
        local_levels=levels,
        local_dim=len(levels),
        total_dim=len(levels) ** n,
    )
    return ObservableFactory(basis), basis


# ── factory surface & immutability ───────────────────────────────────────────


def test_factory_is_bound_and_read_only():
    obs = _system().observables
    assert isinstance(obs, ObservableFactory)
    with pytest.raises(AttributeError):
        obs.anything = 1
    expr = obs.n("r", 0)
    assert isinstance(expr, ObservableExpr)
    with pytest.raises(AttributeError):
        expr._terms = ()


def test_deleted_factories_raise_attributeerror():
    obs = _system().observables
    for name in (
        "level_sum",
        "weighted_level_sum",
        "product_projector",
        "identity",
        "site_populations",
    ):
        assert not hasattr(obs, name)
        with pytest.raises(AttributeError):
            getattr(obs, name)


def test_unknown_levels_and_sites_rejected():
    obs, _ = _factory(("0", "1", "r"), 2)
    with pytest.raises(ValueError, match="Level 'x'"):
        obs.n("x", 0)
    with pytest.raises(ValueError, match="site"):
        obs.n("r", 5)


# ── primitive values (E/n) via the private dense evaluator ───────────────────


def test_projector_and_manual_level_sum_values():
    obs, basis = _factory(("0", "1", "r"), 2)
    psi = dense_product_state(["1", "r"], basis)
    assert _dense_expectation(obs.n("r", 0), psi) == pytest.approx(0.0)
    assert _dense_expectation(obs.n("r", 1), psi) == pytest.approx(1.0)
    n_r = sum(obs.n("r", i) for i in range(2))
    n_1 = sum(obs.n("1", i) for i in range(2))
    assert _dense_expectation(n_r, psi) == pytest.approx(1.0)
    assert _dense_expectation(n_1, psi) == pytest.approx(1.0)
    weighted = 2.0 * obs.n("r", 0) + (-3.0) * obs.n("r", 1)
    assert _dense_expectation(weighted, psi) == pytest.approx(-3.0)


def test_product_projector_from_algebra():
    obs, basis = _factory(("0", "1", "r"), 2)
    psi = dense_product_state(["1", "r"], basis)
    assert _dense_expectation(obs.n("1", 0) @ obs.n("r", 1), psi) == pytest.approx(1.0)
    assert _dense_expectation(obs.n("r", 0) @ obs.n("1", 1), psi) == pytest.approx(0.0)


def test_transition_expectation_is_raw_complex():
    obs, basis = _factory(("0", "1", "r"), 1)
    psi = (
        dense_product_state(["0"], basis)
        + np.exp(1j * 0.7) * dense_product_state(["1"], basis)
    ) / np.sqrt(2)
    val = _dense_expectation(obs.E("0", "1", 0), psi)
    # <psi| |0><1| |psi> = conj(c0) * c1
    assert val == pytest.approx(0.5 * np.exp(1j * 0.7))
    dag = _dense_expectation(obs.E("0", "1", 0).dagger(), psi)
    assert dag == pytest.approx(np.conj(val))


# ── closed algebra ───────────────────────────────────────────────────────────


def test_algebra_add_sub_scale_neg_product():
    obs, basis = _factory(("0", "1", "r"), 2)
    psi = dense_product_state(["1", "r"], basis)
    total = obs.n("r", 0) + obs.n("r", 1)
    assert _dense_expectation(total, psi) == pytest.approx(1.0)
    diff = sum(obs.n("r", i) for i in range(2)) - sum(obs.n("1", i) for i in range(2))
    assert _dense_expectation(diff, psi) == pytest.approx(0.0)
    scaled = 2.5 * obs.n("r", 1)
    assert _dense_expectation(scaled, psi) == pytest.approx(2.5)
    assert _dense_expectation(-scaled, psi) == pytest.approx(-2.5)
    pair = obs.n("1", 0) @ obs.n("r", 1)
    assert _dense_expectation(pair, psi) == pytest.approx(1.0)
    assert _dense_expectation(obs.n("r", 0) @ obs.n("r", 1), psi) == pytest.approx(0.0)


def test_same_site_product_multiplies_matrices():
    obs, basis = _factory(("0", "1", "r"), 1)
    psi = (dense_product_state(["0"], basis) + dense_product_state(["1"], basis)) / np.sqrt(2)
    # |0><1| @ |1><0| = |0><0|
    composed = obs.E("0", "1", 0) @ obs.E("1", "0", 0)
    assert _dense_expectation(composed, psi) == pytest.approx(
        _dense_expectation(obs.n("0", 0), psi)
    )
    # |0><1| @ |0><1| = 0
    squared = obs.E("0", "1", 0) @ obs.E("0", "1", 0)
    assert _dense_expectation(squared, psi) == pytest.approx(0.0)


def test_bell_projector_from_algebra():
    """|B><B| for B = (|01> + |10>)/sqrt(2) is expressible in the closed set."""
    obs, basis = _factory(("0", "1", "r"), 2)
    proj = 0.5 * (
        (obs.n("0", 0) @ obs.n("1", 1))
        + (obs.n("1", 0) @ obs.n("0", 1))
        + (obs.E("0", "1", 0) @ obs.E("1", "0", 1))
        + (obs.E("1", "0", 0) @ obs.E("0", "1", 1))
    )
    bell = (dense_product_state(["0", "1"], basis) + dense_product_state(["1", "0"], basis)) / np.sqrt(2)
    ortho = (dense_product_state(["0", "1"], basis) - dense_product_state(["1", "0"], basis)) / np.sqrt(2)
    assert _dense_expectation(proj, bell) == pytest.approx(1.0)
    assert _dense_expectation(proj, ortho) == pytest.approx(0.0, abs=1e-12)


def test_sum_builtin_seed_and_nonzero_seed_rejected():
    obs, basis = _factory(("0", "1", "r"), 2)
    total = sum(obs.n("r", i) for i in range(2))  # 0 + expr + expr
    assert isinstance(total, ObservableExpr)
    psi = dense_product_state(["r", "r"], basis)
    assert _dense_expectation(total, psi) == pytest.approx(2.0)
    with pytest.raises(TypeError):
        2 + obs.n("r", 0)  # a nonzero number is not an identity


# ── type / shape guards ──────────────────────────────────────────────────────


def test_scalar_star_rejects_operator_product_and_foreign_types():
    obs, _ = _factory(("0", "1", "r"), 2)
    with pytest.raises(TypeError, match="scalar"):
        obs.n("r", 0) * obs.n("r", 1)
    with pytest.raises(TypeError, match="ObservableExpr"):
        obs.n("r", 0) + np.eye(3)


def test_cross_system_combination_rejected():
    a, _ = _factory(("0", "1", "r"), 2)
    b, _ = _factory(("1", "r"), 2)
    with pytest.raises(ValueError, match="differently shaped"):
        a.n("r", 0) + b.n("r", 0)


def test_batch_evaluation_time_major():
    obs, basis = _factory(("0", "1", "r"), 2)
    total = obs.n("r", 0) + obs.n("r", 1)
    states = np.stack(
        [dense_product_state(["1", "r"], basis), dense_product_state(["r", "r"], basis)]
    )
    vals = _dense_expectation(total, states)
    assert vals.shape == (2,)
    np.testing.assert_allclose(vals, [1.0, 2.0])


def test_dimension_mismatch_rejected():
    obs, _ = _factory(("0", "1", "r"), 2)  # dim = 9
    with pytest.raises(ValueError, match="does not match"):
        _dense_expectation(obs.n("r", 0) + obs.n("r", 1), np.zeros(4, dtype=complex))


# ── Hermiticity is enforced at simulate time, not in the algebra ─────────────


def test_non_hermitian_observable_rejected_at_simulate():
    system = _system("1r", 1)
    obs = system.observables
    with pytest.raises(ValueError, match="Hermitian"):
        simulate(system, observables={"c": obs.E("1", "r", 0)})


def test_hermitian_coherence_accepted_at_simulate():
    system = _system("1r", 1)
    obs = system.observables
    A = obs.E("1", "r", 0)
    result = simulate(system, observables={"x": A + A.dagger()})
    assert result.expectation("x").dtype == np.float64
