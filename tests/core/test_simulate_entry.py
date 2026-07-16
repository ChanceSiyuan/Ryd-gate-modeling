"""Unified ``ryd_gate.simulate`` dispatcher & input semantics (rewritten API).

``simulate(system, initial_state=None, *, backend="exact_ode", t_eval=None,
observables=None, backend_options=None)``. ``initial_state=None`` -> ``['1']*N``
(E27); ``"plus"`` -> ``(|0>+|1>)/sqrt2`` per site; a flat label list is one state;
a nested list is a batch (returns a ``tuple``). Explicit ``t_eval`` is strict
(E06) and requires observables; unknown ``backend_options`` keys raise.
"""

from itertools import product

import numpy as np
import pytest

import ryd_gate
from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.protocols import SweepProtocol
from ryd_gate.results import EvolutionResult

# ── helpers ──────────────────────────────────────────────────────────────────


def _driven_system(n=2, omega_half=np.pi / 2, t_gate=1.0):
    """Constant |1>-|r> drive with the pair interaction disabled outright.

    ``interaction_cutoff_um=0.0`` drops the (fast) Rydberg pair phase so the
    rad/s-scale toy drive isn't a stiff multi-scale ODE that DOP853 must
    integrate over the 1 s gate (cf. test_rydberg_system_model.py:99).
    """
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(n, spacing_um=30.0),
        protocol=SweepProtocol(
            t_gate_s=t_gate,
            omega_half_rad_s=lambda t: omega_half,
            detuning_rad_s=lambda t: 0.0,
        ),
        interaction_cutoff_um=0.0,
    )


def _idle_01r(n=1):
    return RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(n),
        protocol=SweepProtocol(
            t_gate_s=1.0,
            omega_half_rad_s=lambda t: 0.0,
            detuning_rad_s=lambda t: 0.0,
        ),
    )


def _amps(result, levels, n):
    return {labels: result.amplitude(list(labels)) for labels in product(levels, repeat=n)}


def _assert_amps_close(r_a, r_b, levels, n, atol=1e-8):
    a, b = _amps(r_a, levels, n), _amps(r_b, levels, n)
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=atol)


# ── backend dispatch ─────────────────────────────────────────────────────────


def test_default_backend_is_exact_ode():
    system = _driven_system()
    r_default = ryd_gate.simulate(system)
    r_ode = ryd_gate.simulate(system, backend="exact_ode")
    assert isinstance(r_default, EvolutionResult)
    _assert_amps_close(r_default, r_ode, ("1", "r"), 2)


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        ryd_gate.simulate(_driven_system(), backend="nonexistent")


def test_backend_name_is_case_insensitive():
    """Backend dispatch lowercases the name (simulate.py:42)."""
    system = _driven_system()
    r_upper = ryd_gate.simulate(system, backend="EXACT_ODE")
    r_lower = ryd_gate.simulate(system, backend="exact_ode")
    assert isinstance(r_upper, EvolutionResult)
    _assert_amps_close(r_upper, r_lower, ("1", "r"), 2)


# ── initial-state semantics ──────────────────────────────────────────────────


def test_default_initial_state_is_all_ones():
    """initial_state=None seeds every site in |1> (E27)."""
    system = _driven_system()
    r_default = ryd_gate.simulate(system)
    r_labels = ryd_gate.simulate(system, ["1", "1"])
    _assert_amps_close(r_default, r_labels, ("1", "r"), 2)
    # and it is genuinely different from a different seed state
    r_other = ryd_gate.simulate(system, ["r", "r"])
    assert r_default.amplitude(["1", "1"]) != pytest.approx(r_other.amplitude(["1", "1"]))


def test_plus_initial_state():
    system = _idle_01r(n=1)  # H = 0, so 'plus' is preserved
    result = ryd_gate.simulate(system, "plus")
    assert result.amplitude(["0"]) == pytest.approx(1.0 / np.sqrt(2))
    assert result.amplitude(["1"]) == pytest.approx(1.0 / np.sqrt(2))
    assert result.amplitude(["r"]) == pytest.approx(0.0)


def test_single_element_nested_batch_returns_length_one_tuple():
    results = ryd_gate.simulate(_driven_system(), [["1", "1"]])
    assert isinstance(results, tuple) and len(results) == 1
    assert isinstance(results[0], EvolutionResult)


# ── backend options & t_eval strictness ──────────────────────────────────────


def test_empty_t_eval_rejected():
    system = _driven_system()
    with pytest.raises(ValueError):
        ryd_gate.simulate(
            system, t_eval=[], observables={"n_r": system.observables.n("r", 0)}
        )


def test_t_eval_out_of_range_rejected():
    system = _driven_system()  # t_gate = 1.0
    with pytest.raises(ValueError):
        ryd_gate.simulate(
            system, t_eval=[2.0], observables={"n_r": system.observables.n("r", 0)}
        )
