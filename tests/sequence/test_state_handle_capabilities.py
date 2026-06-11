"""Capability-aware result handles (Stage 3)."""

import math

import pytest

from ryd_gate import DeviceSpec, Pulse, Register, Sequence, Waveform, simulate_sequence
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.ir.evolution import EvolutionResult
from ryd_gate.results import (
    MPSStateHandle,
    SimulationResult,
    StateMaterializationError,
    UnsupportedResultQuery,
    UnsupportedStateHandle,
)


def _sequence():
    seq = Sequence(Register.chain(1, 20.0), DeviceSpec.virtual_rb87(), "1r")
    seq.declare_channel("ryd", "rydberg_global")
    seq.add(Pulse.constant_detuning(Waveform.blackman(1000, area=math.pi), 0.0), "ryd")
    return seq


def test_exact_result_capabilities_and_metadata():
    result = simulate_sequence(_sequence())
    assert result.state.kind == "statevector"
    assert result.capabilities >= {"expectation", "sampling", "statevector"}
    assert result.state.n_atoms == 1
    assert result.state.local_levels == ("1", "r")
    assert result.state.atom_ids == ("q0",)


def test_unsupported_handle_refuses_every_query():
    handle = UnsupportedStateHandle("peps", "peps.state_handle_not_implemented")
    assert handle.kind == "unsupported"
    assert handle.capabilities == frozenset()

    with pytest.raises(UnsupportedResultQuery, match="peps.state_handle_not_implemented"):
        handle.expectation("sum_nr")
    with pytest.raises(UnsupportedResultQuery, match="peps.state_handle_not_implemented"):
        handle.populations("r")
    with pytest.raises(UnsupportedResultQuery, match="peps.state_handle_not_implemented"):
        handle.sample(10)
    with pytest.raises(UnsupportedResultQuery, match="peps.state_handle_not_implemented"):
        handle.statevector(max_dim=4)


def test_mps_statevector_materialization_guards():
    spec = create_tn_lattice_spec(2, 1, level_structure="1r")
    handle = MPSStateHandle(object(), spec, ("q0", "q1"))

    with pytest.raises(StateMaterializationError, match="mps.statevector_requires_max_dim"):
        handle.statevector()
    with pytest.raises(StateMaterializationError, match="mps.statevector_too_large"):
        handle.statevector(max_dim=3)


def test_mps_unsupported_observable_raises_typed_error():
    spec = create_tn_lattice_spec(2, 1, level_structure="1r")
    handle = MPSStateHandle(object(), spec, ("q0", "q1"))

    with pytest.raises(UnsupportedResultQuery, match="mps.observable_unsupported"):
        handle.expectation("total_energy")


def test_simulation_result_capabilities_mirror_state():
    state = UnsupportedStateHandle("peps", "peps.state_handle_not_implemented")
    result = SimulationResult(
        raw=EvolutionResult(psi_final=None),
        state=state,
        backend="peps",
        sequence=_sequence(),
    )

    assert result.capabilities == state.capabilities
    assert result.capabilities == frozenset()
