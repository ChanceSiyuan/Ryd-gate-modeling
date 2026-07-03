"""Tests for the 297 four-level CZ gate metrics (``*_297`` functions).

ARC-backed for the end-to-end cases; the Nielsen-formula unit tests are pure
numpy.
"""

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem
from ryd_gate.analysis.gate_metrics import (
    _nielsen_infidelity,
    _nielsen_infidelity_297,
    average_gate_infidelity_297,
    error_budget_297,
    population_evolution_297,
    project_theta_297,
)
from ryd_gate.protocols import Direct297CZProtocol, Direct297TOProtocol

_POWER_W = 1e-3
_AREA_UM2 = 100.0


def _two_atom_system(**level_kwargs):
    level_kwargs.setdefault("magnetic_field_G", 100.0)
    return RydbergSystem.set_atom_level("rb87_297_clock_4", **level_kwargs).set_atom_geom(
        Register.chain(2, spacing_um=3.0)
    )


def _flat_cz(t_gate=0.4e-6):
    return Direct297CZProtocol(
        t_gate=t_gate,
        A_297=lambda s: 1.0,
        power_at_atoms_w=_POWER_W,
        beam_area_um2=_AREA_UM2,
    )


def test_formula_is_zero_for_perfect_corrected_overlaps():
    ov = {"a00": 1 + 0j, "a01": 1 + 0j, "a10": 1 + 0j, "a11": 1 + 0j}
    assert _nielsen_infidelity_297(ov) == pytest.approx(0.0, abs=1e-12)


def test_formula_reduces_to_seven_level_when_symmetric():
    # For a01 == a10 the four-state formula must match the three-state
    # seven-level Nielsen formula exactly.
    a00 = 0.99 * np.exp(0.02j)
    a01 = 0.97 * np.exp(-0.05j)
    a11 = 0.95 * np.exp(0.11j)
    four = {"a00": a00, "a01": a01, "a10": a01, "a11": a11}
    three = {"a00": a00, "a01": a01, "a11": a11}
    assert _nielsen_infidelity_297(four) == pytest.approx(_nielsen_infidelity(three))


def test_rejects_non_297_system():
    system = RydbergSystem.set_atom_level("01r").set_atom_geom(
        Register.chain(2, spacing_um=3.0)
    )
    with pytest.raises(ValueError, match="rb87_297_clock_4"):
        average_gate_infidelity_297(system, _flat_cz(), [])


def test_smoke_two_atom_infidelity():
    system = _two_atom_system(enable_rydberg_decay=False)
    proto = Direct297TOProtocol(_POWER_W, _AREA_UM2)
    x = [0.0, 0.0, 0.0, 0.0, 0.0, 7.6]
    infidelity, residuals = average_gate_infidelity_297(
        system, proto, x, return_residuals=True
    )
    assert np.isfinite(infidelity)
    assert 0.0 <= infidelity <= 1.0
    assert set(residuals) == {"r", "r_garb", "logical_loss"}
    assert all(0.0 <= v <= 2.0 for v in residuals.values())


def test_project_theta_improves_or_matches_deliberately_wrong_seed():
    system = _two_atom_system(enable_rydberg_decay=False)
    proto = Direct297TOProtocol(_POWER_W, _AREA_UM2)
    x = [0.0, 0.0, 0.0, 0.0, 0.9, 7.6]  # deliberately offset theta

    proj = project_theta_297(system, proto, x)

    assert proj.seed_theta == 0.9
    assert proj.seed_infidelity == pytest.approx(
        average_gate_infidelity_297(system, proto, x)
    )
    assert proj.infidelity <= proj.seed_infidelity + 1e-15
    # Only theta moves: same length, all non-theta entries preserved exactly.
    assert len(proj.x) == len(x)
    assert proj.x[:4] == x[:4] and proj.x[5] == x[5]
    assert proj.x[4] == proj.theta
    # The projected x reproduces the projected score through the public metric.
    assert average_gate_infidelity_297(system, proto, proj.x) == pytest.approx(
        proj.infidelity, abs=1e-12
    )


def test_project_theta_requires_theta_index():
    system = _two_atom_system(enable_rydberg_decay=False)
    with pytest.raises(ValueError, match="theta_index"):
        project_theta_297(system, _flat_cz(), [])


def test_project_theta_rejects_non_297_system():
    system = RydbergSystem.set_atom_level("01r").set_atom_geom(
        Register.chain(2, spacing_um=3.0)
    )
    proto = Direct297TOProtocol(_POWER_W, _AREA_UM2)
    with pytest.raises(ValueError, match="rb87_297_clock_4"):
        project_theta_297(system, proto, [0.0, 0.0, 0.0, 0.0, 0.0, 7.6])


def test_budget_near_zero_without_drive():
    # Zero envelope -> nothing ever reaches the Rydberg levels, so every decay
    # integral and residual vanishes.
    system = _two_atom_system(enable_rydberg_decay=False)
    proto = Direct297CZProtocol(
        t_gate=0.4e-6,
        A_297=lambda s: 0.0,
        power_at_atoms_w=_POWER_W,
        beam_area_um2=_AREA_UM2,
    )
    budget = error_budget_297(system, proto, [])
    assert set(budget) == {
        "p_ryd_decay", "p_target_ryd_decay", "p_garb_decay",
        "p_ryd_residual", "p_garb_residual", "p_total",
    }
    assert all(abs(v) < 1e-12 for v in budget.values())


def test_p_ryd_decay_matches_gamma_integral():
    system = _two_atom_system(enable_rydberg_decay=True)
    proto = _flat_cz()
    budget = error_budget_297(system, proto, [], initial_states=["01"])

    pops = population_evolution_297(system, proto, [], "01")
    gamma = system.meta("ryd_state_decay_rate")
    expected = gamma * np.trapezoid(pops["ryd"] + pops["ryd_garb"], pops["t_list"])
    assert budget["p_ryd_decay"] == pytest.approx(expected, rel=1e-3)
    assert budget["p_target_ryd_decay"] + budget["p_garb_decay"] == pytest.approx(
        budget["p_ryd_decay"]
    )
    assert budget["p_total"] == pytest.approx(
        budget["p_ryd_decay"] + budget["p_ryd_residual"] + budget["p_garb_residual"]
    )
    assert budget["p_ryd_decay"] > 0.0
