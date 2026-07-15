"""Tests for the unified, protocol-bound RydbergSystem: interaction resolution,
observables, and end-to-end exact simulation on the effective 1r / 01r presets.

The public system surface no longer exposes ``dim`` / ``basis`` /
``static_hamiltonian_terms`` / ``interaction_pairs`` / ``product_state``; the
resolved pair interaction is inspected through the private ``_static_terms``
(the compiler seam these core tests are allowed to reach into).
"""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.core.observables import ObservableExpr
from ryd_gate.core.operators import ChannelResolvedPairSpec, RydbergPairInteractionSpec, materialize_sparse_operator
from ryd_gate.core.states import dense_product_state
from ryd_gate.protocols import DigitalAnalogProtocol, SweepProtocol


def _sweep(t_gate_s=0.1, omega=1.0, delta=0.0):
    return SweepProtocol(
        t_gate_s=t_gate_s,
        omega_half_rad_s=lambda t: 0.5 * omega,
        detuning_rad_s=lambda t: delta,
    )


def _build(name, register, *, protocol=None, interaction_cutoff_um=None):
    return RydbergSystem(
        level_structure=level_structure(name),
        register=register,
        protocol=protocol if protocol is not None else _sweep(),
        interaction_cutoff_um=interaction_cutoff_um,
    )


def _pair_spec(system):
    return next(t.operator for t in system._static_terms if t.name == "H_pair")


def _total_population(system):
    """Sum of all level projectors over all sites == N * ||psi||^2 (norm probe)."""
    obs = system.observables
    return sum(
        obs.n(level, i)
        for i in range(system.N)
        for level in system.level_structure.levels
    )


# ── interaction resolution ───────────────────────────────────────────────────


def test_1r_lattice_interaction_and_observables():
    system = _build("1r", Register.rectangle(2, 2))

    assert system.level_structure.levels == ("1", "r")
    assert system.N == 4
    assert isinstance(system.observables.n("r", 0), ObservableExpr)
    assert isinstance(_pair_spec(system), RydbergPairInteractionSpec)


def test_all_pair_vdw_is_default():
    system = _build("1r", Register.chain(4, spacing_um=4.0))
    assert len(_pair_spec(system).pairs) == 6  # C(4, 2)


def test_interaction_cutoff_truncates_pairs():
    # rectangle(2,2, spacing=6): 4 edge pairs at r=6, 2 diagonal pairs at r=6*sqrt2.
    all_pairs = _build("1r", Register.rectangle(2, 2, spacing_um=6.0))
    nn_pairs = _build(
        "1r", Register.rectangle(2, 2, spacing_um=6.0), interaction_cutoff_um=6.0
    )
    assert len(_pair_spec(all_pairs).pairs) == 6
    assert len(_pair_spec(nn_pairs).pairs) == 4


def test_single_atom_has_no_pairs():
    system = _build("1r", Register.chain(1))
    assert _pair_spec(system).pairs == ()


def test_vdw_energy_on_double_rydberg_state():
    system = _build("1r", Register.chain(2, spacing_um=4.0))
    spec = _pair_spec(system)
    V = spec.pairs[0][2]

    H_pair = materialize_sparse_operator(spec, system._basis).toarray()
    psi = dense_product_state(["r", "r"], system._basis)
    energy = np.real(np.vdot(psi, H_pair @ psi))
    assert np.isclose(energy, V)


# ── end-to-end exact simulation ──────────────────────────────────────────────


def test_sweep_simulation_conserves_population():
    # Norm-conservation check on the bare drive dynamics: disable the (GHz-scale)
    # Rydberg pair term so the rad/s-scale toy drive isn't a stiff multi-scale ODE.
    system = _build(
        "1r", Register.chain(2), protocol=_sweep(omega=1.0, delta=0.0),
        interaction_cutoff_um=0.0,
    )
    obs = system.observables
    result = simulate(
        system,
        observables={
            "total": _total_population(system),
            "n_r": sum(obs.n("r", i) for i in range(system.N)),
        },
    )
    # closed Hermitian evolution -> unit norm -> total population == N.
    assert result.expectation("total")[-1] == pytest.approx(system.N)
    n_r = result.expectation("n_r")[-1]
    assert 0.0 <= n_r <= system.N


def test_t_eval_array_returns_requested_times_exactly():
    system = _build("1r", Register.chain(1), protocol=_sweep())
    t_eval = np.array([0.0, 0.05, 0.1])
    result = simulate(
        system,
        t_eval=t_eval,
        observables={"n_r": system.observables.n("r", 0)},
    )
    np.testing.assert_array_equal(result.times, t_eval)
    assert result.expectation("n_r").shape == (len(t_eval),)


def test_01r_digital_analog_simulation():
    protocol = DigitalAnalogProtocol(t_gate_s=0.1, coupling_r1_rad_s=lambda t: 0.5)
    system = _build(
        "01r", Register.chain(2, spacing_um=4.0), protocol=protocol,
        interaction_cutoff_um=0.0,
    )
    result = simulate(system, ["1", "1"], observables={"total": _total_population(system)})

    assert system.level_structure.levels == ("0", "1", "r")
    assert result.expectation("total")[-1] == pytest.approx(system.N)


# ── level-structure presets ──────────────────────────────────────────────────


def test_level_structure_presets():
    assert level_structure("1r").levels == ("1", "r")
    assert level_structure("01r").levels == ("0", "1", "r")
    with pytest.raises(ValueError, match="Unknown level-structure"):
        level_structure("1er")


@pytest.mark.parametrize("removed", ["analog_3", "ger", "01"])
def test_removed_presets_raise(removed):
    with pytest.raises(ValueError, match="Unknown level-structure"):
        level_structure(removed)


# ── rb87 seven-level manifold construction (ARC-backed; slow) ────────────────


@pytest.mark.slow
def test_rb87_7_lattice_constructs_channel_resolved_free_mp_model():
    from ryd_gate.protocols import CZProtocol

    cz = CZProtocol(
        t_gate_s=1e-6,
        intermediate_detuning_rad_s=2 * np.pi * 7e9,
        omega_420_max_rad_s=2 * np.pi * 400e6,
        omega_1013_max_rad_s=2 * np.pi * 180e6,
        envelope_420=lambda t: 1.0,
        phase_420_rad=lambda t: 0.0,
    )
    system = RydbergSystem(
        level_structure=level_structure("rb87_7_mp"),
        register=Register.chain(2, spacing_um=3.0),
        protocol=cz,
    )
    assert system.level_structure.levels == ("0", "1", "e1", "e2", "e3", "r", "r_garb")
    # rb87_7 uses the isotropic S-state pair interaction (not channel-resolved).
    spec = _pair_spec(system)
    assert isinstance(spec, RydbergPairInteractionSpec)
    assert not isinstance(spec, ChannelResolvedPairSpec)
    assert len(spec.pairs) == 1
