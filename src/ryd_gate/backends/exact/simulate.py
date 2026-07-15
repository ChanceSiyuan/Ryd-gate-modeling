"""Exact state-vector simulation entry point (adaptive DOP853, ODE-only)."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ryd_gate.core.observables import _dense_expectation
from ryd_gate.core.states import dense_plus_state, dense_product_state, product_index
from ryd_gate.ir import validate_evolution_request
from ryd_gate.results import EvolutionResult

from .compiler import compile_exact
from .ode import evolve_states, validated_exact_options

_IMAG_TOL = 1e-6


def simulate(system, state, *, t_eval=None, observables=None, backend_options=None):
    """Evolve one initial state (label list or ``"plus"``); returns an EvolutionResult."""
    return simulate_states(
        system, [state], t_eval=t_eval, observables=observables, backend_options=backend_options,
    )[0]


def simulate_states(system, states, *, t_eval=None, observables=None, backend_options=None):
    """Evolve a batch of initial states under one shared compilation (E10 tuple)."""
    opts = validated_exact_options(backend_options)
    t_gate = system.t_gate
    times_pub, obs = validate_evolution_request(t_gate, t_eval, observables)
    ham, t_gate = compile_exact(system, hamiltonian_format=opts["hamiltonian_format"])

    eval_times = np.array([t_gate]) if times_pub is None else np.asarray(times_pub, dtype=float)
    solver_eval = np.unique(np.concatenate([eval_times, [t_gate]]))
    final_col = int(np.argmin(np.abs(solver_eval - t_gate)))
    eval_cols = [int(np.argmin(np.abs(solver_eval - t))) for t in eval_times]

    psi0_list = [_initial_dense(system, s) for s in states]
    ys = evolve_states(ham, t_gate, psi0_list, solver_eval, rtol=opts["rtol"], atol=opts["atol"])

    results = []
    for y in ys:
        states_at_eval = y[:, eval_cols].T  # (n_eval, dim)
        expectations = {
            name: _real_expectation(_dense_expectation(expr, states_at_eval), name)
            for name, expr in obs.items()
        }
        reader = _DenseReader(y[:, final_col], system._basis)
        results.append(EvolutionResult(times=eval_times, expectations=expectations, reader=reader))
    return tuple(results)


def _initial_dense(system, state) -> np.ndarray:
    if isinstance(state, str) and state == "plus":
        return dense_plus_state(system._basis)
    return dense_product_state(state, system._basis)


def _real_expectation(vals, name: str) -> np.ndarray:
    v = np.asarray(vals, dtype=complex).reshape(-1)
    scale = max(1.0, float(np.max(np.abs(v)))) if v.size else 1.0
    if np.any(np.abs(v.imag) > _IMAG_TOL * scale):
        raise ValueError(
            f"observable {name!r} produced a non-real expectation (max imag "
            f"{float(np.max(np.abs(v.imag))):.2e}); observables must be Hermitian."
        )
    return v.real.astype(float)


class _DenseReader:
    """Private final-state reader for amplitude/sample (dense exact state)."""

    __slots__ = ("_state", "_basis")

    def __init__(self, state: np.ndarray, basis) -> None:
        self._state = np.asarray(state, dtype=complex)
        self._basis = basis

    def amplitude(self, labels) -> complex:
        _validate_labels(labels, self._basis)
        return complex(self._state[product_index(labels, self._basis)])

    def sample(self, shots: int, seed: int) -> Counter:
        weights = np.abs(self._state) ** 2
        total = float(weights.sum())
        if not total > 0.0:
            raise ValueError("cannot sample from a zero-norm final state.")
        probs = weights / total
        rng = np.random.default_rng(seed)
        draws = rng.choice(probs.size, size=shots, p=probs)
        d, n, levels = self._basis.local_dim, self._basis.n_sites, self._basis.local_levels
        counts: Counter = Counter()
        for idx in draws:
            counts[tuple(levels[(int(idx) // d ** (n - 1 - i)) % d] for i in range(n))] += 1
        return counts


def _validate_labels(labels, basis) -> None:
    if len(labels) != basis.n_sites:
        raise ValueError(f"amplitude() needs {basis.n_sites} per-site labels, got {len(labels)}.")
    for label in labels:
        if label not in basis.local_levels:
            raise ValueError(f"unknown level label {label!r}; levels are {basis.local_levels}.")
