"""ObservableExpr algebra, lowering, and dense evaluation (Patch 5).

Pure unit tests of the immutable scalar expression algebra exposed as
``system.observables`` — no evolution.  The private lowered form (sum of
coefficient x per-site local-matrix products) and the private dense evaluator
(raw complex ``<psi|O|psi>``, no norm division) are pinned here because both
backends build on them.
"""

import numpy as np
import pytest

from ryd_gate import InteractionSpec, Register, RydbergSystem, level_structure
from ryd_gate.core.observables import (
    ObservableExpr,
    ObservableFactory,
    _dense_expectation,
)


def _system(name="01r", n=2):
    return RydbergSystem(
        level_structure=level_structure(name),
        register=Register.chain(n),
        interaction=InteractionSpec(C6=0.0),
    )


def _psi(system, labels):
    return system.product_state(labels)


def test_factory_is_bound_and_read_only():
    system = _system()
    obs = system.observables
    assert isinstance(obs, ObservableFactory)
    with pytest.raises(AttributeError):
        obs.anything = 1
    expr = obs.n("r", 0)
    assert isinstance(expr, ObservableExpr)
    with pytest.raises(AttributeError):
        expr._terms = ()


def test_unknown_levels_and_sites_rejected():
    obs = _system().observables
    with pytest.raises(ValueError, match="Level 'x'"):
        obs.n("x", 0)
    with pytest.raises(ValueError, match="site"):
        obs.n("r", 5)
    with pytest.raises(ValueError, match="weights"):
        obs.weighted_level_sum("r", [1.0])
    with pytest.raises(ValueError, match="labels"):
        obs.product_projector(["0"])


def test_projector_and_level_sum_values():
    system = _system()
    obs = system.observables
    psi = _psi(system, ["1", "r"])
    assert _dense_expectation(obs.n("r", 0), psi) == pytest.approx(0.0)
    assert _dense_expectation(obs.n("r", 1), psi) == pytest.approx(1.0)
    assert _dense_expectation(obs.level_sum("r"), psi) == pytest.approx(1.0)
    assert _dense_expectation(obs.level_sum("1"), psi) == pytest.approx(1.0)
    assert _dense_expectation(
        obs.weighted_level_sum("r", [2.0, -3.0]), psi
    ) == pytest.approx(-3.0)


def test_product_projector_and_identity():
    system = _system()
    obs = system.observables
    psi = _psi(system, ["1", "r"])
    assert _dense_expectation(obs.product_projector(["1", "r"]), psi) == pytest.approx(1.0)
    assert _dense_expectation(obs.product_projector(["r", "1"]), psi) == pytest.approx(0.0)
    # identity = raw survival norm, no normalization
    assert _dense_expectation(obs.identity(), 0.5 * psi) == pytest.approx(0.25)


def test_transition_expectation_is_raw_complex():
    system = _system("01r", n=1)
    obs = system.observables
    psi = (
        _psi(system, ["0"]) + np.exp(1j * 0.7) * _psi(system, ["1"])
    ) / np.sqrt(2)
    val = _dense_expectation(obs.E("0", "1", 0), psi)
    # <psi| |0><1| |psi> = conj(c0) * c1
    assert val == pytest.approx(0.5 * np.exp(1j * 0.7))
    dag = _dense_expectation(obs.E("0", "1", 0).dagger(), psi)
    assert dag == pytest.approx(np.conj(val))


def test_algebra_add_sub_scale_product():
    system = _system()
    obs = system.observables
    psi = _psi(system, ["1", "r"])
    total = obs.n("r", 0) + obs.n("r", 1)
    assert _dense_expectation(total, psi) == pytest.approx(
        _dense_expectation(obs.level_sum("r"), psi)
    )
    diff = obs.level_sum("r") - obs.level_sum("1")
    assert _dense_expectation(diff, psi) == pytest.approx(0.0)
    scaled = 2.5 * obs.n("r", 1)
    assert _dense_expectation(scaled, psi) == pytest.approx(2.5)
    assert _dense_expectation(-scaled, psi) == pytest.approx(-2.5)
    # operator product: pair correlation projector
    pair = obs.n("1", 0) @ obs.n("r", 1)
    assert _dense_expectation(pair, psi) == pytest.approx(1.0)
    assert _dense_expectation(obs.n("r", 0) @ obs.n("r", 1), psi) == pytest.approx(0.0)


def test_same_site_product_multiplies_matrices():
    system = _system("01r", n=1)
    obs = system.observables
    psi = (_psi(system, ["0"]) + _psi(system, ["1"])) / np.sqrt(2)
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
    system = _system()
    obs = system.observables
    proj = 0.5 * (
        (obs.n("0", 0) @ obs.n("1", 1))
        + (obs.n("1", 0) @ obs.n("0", 1))
        + (obs.E("0", "1", 0) @ obs.E("1", "0", 1))
        + (obs.E("1", "0", 0) @ obs.E("0", "1", 1))
    )
    bell = (_psi(system, ["0", "1"]) + _psi(system, ["1", "0"])) / np.sqrt(2)
    ortho = (_psi(system, ["0", "1"]) - _psi(system, ["1", "0"])) / np.sqrt(2)
    assert _dense_expectation(proj, bell) == pytest.approx(1.0)
    assert _dense_expectation(proj, ortho) == pytest.approx(0.0, abs=1e-12)


def test_scalar_star_rejects_operator_product_and_foreign_types():
    obs = _system().observables
    with pytest.raises(TypeError, match="scalar"):
        obs.n("r", 0) * obs.n("r", 1)
    with pytest.raises(TypeError, match="ObservableExpr"):
        obs.n("r", 0) + np.eye(3)


def test_cross_system_combination_rejected():
    a = _system("01r", n=2).observables
    b = _system("1r", n=2).observables
    with pytest.raises(ValueError, match="differently shaped"):
        a.n("r", 0) + b.n("r", 0)


def test_batch_evaluation_time_major():
    system = _system()
    obs = system.observables
    states = np.stack([_psi(system, ["1", "r"]), _psi(system, ["r", "r"])])
    vals = _dense_expectation(obs.level_sum("r"), states)
    assert vals.shape == (2,)
    np.testing.assert_allclose(vals, [1.0, 2.0])


def test_dimension_mismatch_rejected():
    obs = _system("01r", n=2).observables
    with pytest.raises(ValueError, match="does not match"):
        _dense_expectation(obs.level_sum("r"), np.zeros(4, dtype=complex))


def test_site_populations_profile_is_plain_dict():
    system = _system()
    profile = system.observables.site_populations("r")
    assert sorted(profile) == ["n_r_0", "n_r_1"]
    psi = _psi(system, ["1", "r"])
    assert _dense_expectation(profile["n_r_1"], psi) == pytest.approx(1.0)
