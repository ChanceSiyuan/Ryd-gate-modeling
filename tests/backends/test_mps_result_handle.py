"""Stage 3 MPS-native result handles."""

import numpy as np
import pytest

pytest.importorskip("tenpy")

from ryd_gate import (
    DeviceSpec,
    InteractionSpec,
    Pulse,
    Register,
    Sequence,
    simulate_sequence,
)
from ryd_gate.backends.tenpy_mps.backends import measure_mps_observable
from ryd_gate.backends.tenpy_mps.observables import measure_mean_rydberg, measure_site_occupations
from ryd_gate.backends.tenpy_mps.state import product_state_mps
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.results import StateMaterializationError, UnsupportedResultQuery


def _constant_pi_sequence(n_atoms=4):
    seq = Sequence(Register.chain(n_atoms, 20.0), DeviceSpec.virtual_rb87(), "1r")
    seq.declare_channel("ryd", "rydberg_global")
    seq.add(Pulse.constant(1000, float(np.pi), 0.0), "ryd")
    return seq


def _mps_options():
    return {"chi_max": 16, "dt": 2.5e-7}


def _assert_same_state_up_to_phase(actual, expected, *, atol=1e-2):
    actual = np.asarray(actual, dtype=complex)
    expected = np.asarray(expected, dtype=complex)
    idx = int(np.argmax(np.abs(expected)))
    phase = 1.0 if abs(expected[idx]) < 1e-12 else actual[idx] / expected[idx]
    np.testing.assert_allclose(actual, phase * expected, atol=atol)


def test_simulate_sequence_mps_returns_native_result():
    result = simulate_sequence(
        _constant_pi_sequence(),
        backend="mps",
        interaction=InteractionSpec(C6=0.0),
        backend_options=_mps_options(),
    )

    assert result.backend == "mps"
    assert result.state.kind == "mps"
    assert result.raw.metadata["state_handle_kind"] == "mps"
    assert "expectation" in result.capabilities


def test_mps_sum_nr_matches_exact():
    seq = _constant_pi_sequence()
    exact = simulate_sequence(seq, interaction=InteractionSpec(C6=0.0))
    mps = simulate_sequence(
        seq,
        backend="mps",
        interaction=InteractionSpec(C6=0.0),
        backend_options=_mps_options(),
    )

    assert mps.expectation("sum_nr") == pytest.approx(exact.expectation("sum_nr"), abs=2e-2)


def test_mps_atom_id_observable_and_unsupported_name():
    result = simulate_sequence(
        _constant_pi_sequence(),
        backend="mps",
        interaction=InteractionSpec(C6=0.0),
        backend_options=_mps_options(),
    )

    assert result.expectation("n_r_q0") == pytest.approx(result.expectation("n_r_0"), abs=1e-12)
    with pytest.raises(UnsupportedResultQuery, match="mps.observable_unsupported"):
        result.expectation("made_up")


def test_mps_statevector_materializes_for_tiny_system_and_matches_exact():
    seq = _constant_pi_sequence()
    exact = simulate_sequence(seq, interaction=InteractionSpec(C6=0.0))
    mps = simulate_sequence(
        seq,
        backend="mps",
        interaction=InteractionSpec(C6=0.0),
        backend_options=_mps_options(),
    )

    _assert_same_state_up_to_phase(mps.statevector(max_dim=4096), exact.statevector(), atol=2e-2)
    with pytest.raises(StateMaterializationError, match="mps.statevector_too_large"):
        mps.statevector(max_dim=8)


def test_mps_level_structure_gate_fails_before_compile():
    seq = Sequence(Register.chain(1, 20.0), DeviceSpec.virtual_rb87(), "rb87_7")

    with pytest.raises(ValueError, match="level_structure.backend_unsupported"):
        simulate_sequence(seq, backend="mps")


def test_measure_mps_observable_matches_existing_helpers():
    spec = create_tn_lattice_spec(2, 1, level_structure="1r")
    psi = product_state_mps(spec, "af1")

    assert measure_mps_observable(psi, spec, "n_mean") == pytest.approx(
        measure_mean_rydberg(psi, spec)
    )
    np.testing.assert_allclose(
        measure_mps_observable(psi, spec, "n_i"),
        measure_site_occupations(psi, spec),
    )
