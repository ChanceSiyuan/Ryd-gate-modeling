"""EvolutionResult contract (rewritten API).

The result exposes exactly ``times`` and the three physical readouts
``expectation(name)`` (real float64 array), ``amplitude(level_labels)`` (complex)
and ``sample(shots=, seed=)`` (``Counter[tuple[str, ...]]``). The backend-native
final state, ``metadata``, ``basis`` and ``probabilities`` are gone.
"""

from collections import Counter

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.protocols import SweepProtocol
from ryd_gate.results import EvolutionResult

# ── helpers ──────────────────────────────────────────────────────────────────


def _rabi_system(omega_half=np.pi / 2, t_gate=1.0, n=1):
    """Constant-drive |1>-|r> Rabi. omega_half=pi/2, t_gate=1 => a single-atom pi pulse."""
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(n, spacing_um=30.0),
        protocol=SweepProtocol(
            t_gate_s=t_gate,
            omega_half_rad_s=lambda t: omega_half,
            detuning_rad_s=lambda t: 0.0,
        ),
    )


def _idle_system(n=2):
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(n),
        protocol=SweepProtocol(
            t_gate_s=1.0,
            omega_half_rad_s=lambda t: 0.0,
            detuning_rad_s=lambda t: 0.0,
        ),
    )


# ── surface ──────────────────────────────────────────────────────────────────


def test_times_default_is_t_gate_and_readonly():
    result = simulate(_idle_system())
    assert isinstance(result, EvolutionResult)
    np.testing.assert_array_equal(result.times, [1.0])
    assert result.times.flags.writeable is False


def test_no_deleted_surface():
    result = simulate(_idle_system())
    for name in (
        "final_state",
        "psi_final",
        "states",
        "probabilities",
        "metadata",
        "basis",
        "expectations",
    ):
        assert not hasattr(result, name)


# ── expectation() ────────────────────────────────────────────────────────────


def test_expectation_is_real_float64():
    system = _rabi_system()  # pi pulse: population fully in |r> at t_gate
    result = simulate(system, observables={"n_r": system.observables.n("r", 0)})
    arr = result.expectation("n_r")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1,)
    assert arr.dtype == np.float64
    assert arr[0] == pytest.approx(1.0, abs=1e-5)


def test_expectation_unknown_name_is_keyerror():
    system = _idle_system()
    result = simulate(system, observables={"n_r": system.observables.n("r", 0)})
    with pytest.raises(KeyError, match="request"):
        result.expectation("sum_nr")
    bare = simulate(system)
    with pytest.raises(KeyError):
        bare.expectation("n_r")


def test_expectation_tracks_times_shape():
    system = _rabi_system()
    t_eval = np.linspace(0.0, 1.0, 7)
    result = simulate(
        system,
        t_eval=t_eval,
        observables={"n_r": system.observables.n("r", 0)},
    )
    np.testing.assert_array_equal(result.times, t_eval)
    assert result.expectation("n_r").shape == result.times.shape == (7,)


# ── amplitude() ──────────────────────────────────────────────────────────────


def test_amplitude_pi_pulse_transfers_to_rydberg():
    result = simulate(_rabi_system())
    assert abs(result.amplitude(["r"])) ** 2 == pytest.approx(1.0, abs=1e-5)
    assert abs(result.amplitude(["1"])) ** 2 == pytest.approx(0.0, abs=1e-5)


def test_amplitude_returns_complex():
    result = simulate(_idle_system())
    amp = result.amplitude(["1", "1"])
    assert isinstance(amp, complex)
    assert abs(amp) == pytest.approx(1.0)


# ── sample() ─────────────────────────────────────────────────────────────────


def test_sample_is_lazy_deterministic_and_tuple_keyed():
    result = simulate(_rabi_system())  # final state ~ |r>
    a = result.sample(shots=64, seed=0)
    b = result.sample(shots=64, seed=0)
    assert isinstance(a, Counter)
    assert a == b
    assert sum(a.values()) == 64
    assert all(isinstance(key, tuple) and all(isinstance(s, str) for s in key) for key in a)
    assert a == Counter({("r",): 64})


def test_sample_requires_keyword_ints():
    result = simulate(_idle_system())
    with pytest.raises(TypeError):
        result.sample(64, 0)  # keyword-only
    with pytest.raises(ValueError, match="shots"):
        result.sample(shots=0, seed=0)
    with pytest.raises(TypeError, match="seed"):
        result.sample(shots=8, seed=None)


# ── batch returns a tuple of results ─────────────────────────────────────────


def test_batch_returns_tuple_of_results():
    system = _rabi_system(n=1)
    results = simulate(system, [["1"], ["r"]])
    assert isinstance(results, tuple)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, EvolutionResult)
    # first started in |1> (pi pulse -> |r>), second started in |r| (pi pulse -> |1>)
    assert abs(results[0].amplitude(["r"])) ** 2 == pytest.approx(1.0, abs=1e-5)
    assert abs(results[1].amplitude(["1"])) ** 2 == pytest.approx(1.0, abs=1e-5)
