"""Contract tests for the explicit, protocol-bound RydbergSystem constructor.

``RydbergSystem(level_structure=..., register=..., protocol=...,
interaction_cutoff_um=None)`` resolves the level structure up front via
``level_structure(name, **physical_kwargs)`` (an immutable preset), takes an
explicit 2D register, and binds a *mandatory* protocol. The public surface is
exactly ``level_structure``, ``register``, ``protocol``, ``interaction_cutoff_um``,
``N``, ``t_gate``, ``observables``, ``with_protocol`` and ``ground_state``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.core.observables import ObservableExpr, ObservableFactory
from ryd_gate.protocols import CZProtocol, DigitalAnalogProtocol, SweepProtocol


def _sweep(t_gate_s=0.1, omega=1.0, delta=0.0):
    return SweepProtocol(
        t_gate_s=t_gate_s,
        omega_half_rad_s=lambda t: 0.5 * omega,
        detuning_rad_s=lambda t: delta,
    )


def _digital_analog(t_gate_s=0.1):
    return DigitalAnalogProtocol(t_gate_s=t_gate_s, coupling_r1_rad_s=lambda t: 0.5)


def _cz():
    """A fully specified CZ protocol (compatible only with rb87_7_*)."""
    return CZProtocol(
        t_gate_s=1e-6,
        intermediate_detuning_rad_s=2 * np.pi * 7e9,
        omega_420_max_rad_s=2 * np.pi * 400e6,
        omega_1013_max_rad_s=2 * np.pi * 180e6,
        envelope_420=lambda t: 1.0,
        phase_420_rad=lambda t: 0.0,
    )


# ── happy paths ──────────────────────────────────────────────────────────────


def test_1r_chain_builds_usable_system():
    proto = _sweep()
    register = Register.chain(3, spacing_um=5.0)
    system = RydbergSystem(
        level_structure=level_structure("1r"), register=register, protocol=proto
    )

    assert system.N == 3
    assert system.register is register
    assert system.protocol is proto
    assert system.interaction_cutoff_um is None
    assert system.t_gate == pytest.approx(0.1)
    assert system.level_structure.levels == ("1", "r")


def test_single_atom_register_is_a_complete_system():
    system = RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(1),
        protocol=_sweep(),
    )
    assert system.N == 1


def test_01r_observables_are_the_bound_expression_factory():
    system = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(2, spacing_um=4.0),
        protocol=_digital_analog(),
    )
    assert system.level_structure.levels == ("0", "1", "r")
    assert isinstance(system.observables, ObservableFactory)
    assert isinstance(system.observables.n("r", 0), ObservableExpr)


def test_interaction_cutoff_is_stored():
    system = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.rectangle(2, 2, spacing_um=6.0),
        protocol=_digital_analog(),
        interaction_cutoff_um=6.0,
    )
    assert system.interaction_cutoff_um == pytest.approx(6.0)


# ── protocol is mandatory ────────────────────────────────────────────────────


def test_protocol_is_required():
    with pytest.raises(TypeError):
        RydbergSystem(
            level_structure=level_structure("1r"), register=Register.chain(2)
        )
    with pytest.raises(TypeError, match="protocol is mandatory"):
        RydbergSystem(
            level_structure=level_structure("1r"),
            register=Register.chain(2),
            protocol=None,
        )


# ── argument-type and validation ─────────────────────────────────────────────


def test_wrong_argument_types_raise():
    reg = Register.chain(2, spacing_um=5.0)
    with pytest.raises(TypeError, match="level_structure"):
        RydbergSystem(level_structure="1r", register=reg, protocol=_sweep())
    with pytest.raises(TypeError, match="register"):
        RydbergSystem(
            level_structure=level_structure("1r"),
            register=[(0.0, 0.0), (4.0, 0.0)],
            protocol=_sweep(),
        )


def test_negative_interaction_cutoff_raises():
    with pytest.raises(ValueError, match="interaction_cutoff_um"):
        RydbergSystem(
            level_structure=level_structure("1r"),
            register=Register.chain(2),
            protocol=_sweep(),
            interaction_cutoff_um=-1.0,
        )
    with pytest.raises(ValueError, match="interaction_cutoff_um"):
        RydbergSystem(
            level_structure=level_structure("1r"),
            register=Register.chain(2),
            protocol=_sweep(),
            interaction_cutoff_um=np.nan,
        )


# ── protocol / preset compatibility ──────────────────────────────────────────


def test_incompatible_protocol_rejected():
    # CZProtocol drives the rb87_7 lasers; it is not compatible with 1r.
    with pytest.raises(ValueError, match="not compatible"):
        RydbergSystem(
            level_structure=level_structure("1r"),
            register=Register.chain(2),
            protocol=_cz(),
        )


# ── with_protocol ────────────────────────────────────────────────────────────


def test_with_protocol_returns_new_bound_system():
    first = _sweep(delta=0.0)
    second = _sweep(delta=1.0)
    system = RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(2, spacing_um=4.0),
        protocol=first,
    )
    rebound = system.with_protocol(second)

    assert isinstance(rebound, RydbergSystem)
    assert rebound is not system
    assert rebound.protocol is second
    assert system.protocol is first  # receiver untouched


def test_with_protocol_rejects_incompatible():
    system = RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(2),
        protocol=_sweep(),
    )
    with pytest.raises(ValueError, match="not compatible"):
        system.with_protocol(_cz())


# ── rb87 seven-level manifold construction (ARC-backed; slow) ────────────────


@pytest.mark.slow
@pytest.mark.parametrize("tag", ["rb87_7_mp", "rb87_7_pm"])
def test_rb87_7_manifold_binds_cz_protocol(tag):
    system = RydbergSystem(
        level_structure=level_structure(tag),
        register=Register.chain(2, spacing_um=3.0),
        protocol=_cz(),
    )
    assert system.N == 2
    assert system.level_structure.levels == ("0", "1", "e1", "e2", "e3", "r", "r_garb")
    assert system.t_gate == pytest.approx(1e-6)
