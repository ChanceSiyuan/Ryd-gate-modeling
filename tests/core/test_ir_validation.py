"""Strict preflight rules of ``ir.validate_evolution_request`` (E06/O02/O06).

Pins the untested branches of the shared measurement-request validator: t_gate
positivity, the observables-dict type/label/value guards, and every explicit
``t_eval`` structural rule (boolean, shape, finiteness, monotonicity, lower
bound, and the ``np.isclose`` grace at the upper bound).
"""

import numpy as np
import pytest

from ryd_gate.core.model import BasisSpec
from ryd_gate.core.observables import ObservableFactory
from ryd_gate.ir import validate_evolution_request


def _obs():
    """A bare Hermitian ObservableExpr dict on a 1-site {1, r} basis (no ARC)."""
    basis = BasisSpec(site_labels=("0",), local_levels=("1", "r"), local_dim=2, total_dim=2)
    return {"n_r": ObservableFactory(basis).n("r", 0)}


# ── t_gate ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("t_gate", [0.0, -1.0])
def test_nonpositive_t_gate_rejected(t_gate):
    with pytest.raises(ValueError, match="t_gate must be positive"):
        validate_evolution_request(t_gate, None, None)


# ── observables dict guards ──────────────────────────────────────────────────


def test_observables_not_a_dict_rejected():
    with pytest.raises(TypeError, match="dict"):
        validate_evolution_request(1.0, None, [1, 2, 3])


def test_non_string_observable_label_rejected():
    (expr,) = _obs().values()
    with pytest.raises(TypeError, match="labels must be strings"):
        validate_evolution_request(1.0, None, {0: expr})


def test_non_observableexpr_value_rejected():
    with pytest.raises(TypeError, match="ObservableExpr"):
        validate_evolution_request(1.0, None, {"bad": np.eye(2)})


# ── explicit t_eval structural rules ─────────────────────────────────────────


def test_boolean_t_eval_rejected():
    with pytest.raises(TypeError, match="boolean"):
        validate_evolution_request(1.0, True, _obs())


def test_non_1d_t_eval_rejected():
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_evolution_request(1.0, [[0.0, 0.5]], _obs())


def test_non_finite_t_eval_rejected():
    with pytest.raises(ValueError, match="finite"):
        validate_evolution_request(1.0, [0.0, np.inf], _obs())


@pytest.mark.parametrize("t_eval", [[0.5, 0.5], [0.5, 0.2]])
def test_non_increasing_t_eval_rejected(t_eval):
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_evolution_request(1.0, t_eval, _obs())


def test_negative_first_time_rejected():
    with pytest.raises(ValueError, match=r"\[0, t_gate\]"):
        validate_evolution_request(1.0, [-0.1, 0.5], _obs())


def test_upper_bound_isclose_grace_accepted():
    t_gate = 1.0
    t_eval = [0.0, t_gate * (1.0 + 1e-13)]  # just above t_gate, within rtol=1e-12
    times, obs = validate_evolution_request(t_gate, t_eval, _obs())
    assert times[-1] == pytest.approx(t_gate, rel=1e-12)
    assert "n_r" in obs
