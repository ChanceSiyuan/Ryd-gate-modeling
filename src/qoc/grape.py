"""Discrete-adjoint GRAPE engine over one bilinear control model (ADR-0024).

The engine consumes only plain arrays and mappings — a Hermitian drift matrix,
one Hermitian operator per real control channel, and initial state vectors —
plus two caller-supplied callbacks. It neither knows nor asks which physical
package produced them, and it performs no physical validation.

Conventions:

- Piecewise-constant slice propagation with ``H_k = h0 + sum_a u[a][k] * controls[a]``
  and ``U_k = expm(-1j * H_k * dt_k)`` on the slices of ``time_grid``.
- ``control_map(parameters) -> {channel: (K,) real array}`` supplies the
  per-slice channel values; ``control_pullback(parameters, channel_gradients)``
  receives ``{channel: (K,) dL/du array}`` and returns the named parameter
  gradient. Every nonlinearity between parameters and channels lives in this
  pair.
- ``terminal_objective(final_states) -> (value, costates)`` where
  ``costates[j] = dL/d(conj(psi_j(T)))`` so that ``dL = 2 Re sum_j <costate_j | d psi_j>``.
- The returned gradient is the discrete adjoint gradient: exact to machine
  precision for the slice propagation actually evaluated (forward states,
  backward costates through the same slice propagators, exact Frechet
  derivatives of each slice exponential).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.linalg import expm, expm_frechet


def _require_hermitian(name: str, value: object, *, dim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix; got shape {arr.shape}.")
    if dim is not None and arr.shape[0] != dim:
        raise ValueError(f"{name} must have shape ({dim}, {dim}); got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    tol = 1e-10 * max(1.0, float(np.max(np.abs(arr))))
    if not np.allclose(arr, arr.conj().T, rtol=0.0, atol=tol):
        raise ValueError(f"{name} must be Hermitian.")
    return arr


def _require_vector(name: str, value: object, dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=complex)
    if arr.shape != (dim,):
        raise ValueError(f"{name} must be a vector of shape ({dim},); got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _require_channel_values(u: object, channels: tuple[str, ...], n_slices: int) -> dict[str, np.ndarray]:
    if not isinstance(u, Mapping):
        raise TypeError(f"control_map must return a mapping from channel names to (K,) arrays; got {type(u).__name__}.")
    if set(u.keys()) != set(channels):
        missing = sorted(set(channels) - set(u.keys()))
        extra = sorted(set(u.keys()) - set(channels))
        raise ValueError(f"control_map channels do not match the model: missing {missing}, unexpected {extra}.")
    out: dict[str, np.ndarray] = {}
    for name in channels:
        arr = np.asarray(u[name])
        if np.iscomplexobj(arr):
            raise ValueError(f"control values for channel {name!r} must be real.")
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (n_slices,):
            raise ValueError(f"control values for channel {name!r} must have shape ({n_slices},); got {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"control values for channel {name!r} must be finite.")
        out[name] = arr
    return out


def _forward_pass(
    parameters: Mapping[str, Any],
    *,
    h0: np.ndarray,
    controls: Mapping[str, np.ndarray],
    initial_states: Sequence[np.ndarray],
    time_grid: np.ndarray,
    control_map: Callable[[Mapping[str, Any]], Mapping[str, np.ndarray]],
    terminal_objective: Callable[[list[np.ndarray]], tuple[float, Sequence[np.ndarray]]],
    keep_slices: bool,
) -> dict[str, Any]:
    """Shared validated forward propagation; evaluates the terminal objective."""
    drift = _require_hermitian("h0", h0)
    dim = drift.shape[0]

    if not isinstance(controls, Mapping) or len(controls) == 0:
        raise ValueError("controls must be a non-empty mapping from channel names to Hermitian operators.")
    channels = tuple(str(name) for name in controls.keys())
    operators = {name: _require_hermitian(f"controls[{name!r}]", op, dim=dim) for name, op in controls.items()}

    times = np.asarray(time_grid, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError(f"time_grid must be a one-dimensional array of at least two times; got shape {times.shape}.")
    if not np.all(np.isfinite(times)):
        raise ValueError("time_grid must contain only finite times.")
    steps = np.diff(times)
    if np.any(steps <= 0.0):
        raise ValueError("time_grid must be strictly increasing.")
    n_slices = int(steps.size)

    if len(initial_states) == 0:
        raise ValueError("initial_states must contain at least one state vector.")
    columns = [_require_vector(f"initial_states[{j}]", state, dim) for j, state in enumerate(initial_states)]
    n_states = len(columns)
    psi = np.column_stack(columns)

    u = _require_channel_values(control_map(parameters), channels, n_slices)

    generators: list[np.ndarray] = []
    propagators: list[np.ndarray] = []
    forward: list[np.ndarray] = [psi]
    for k in range(n_slices):
        h_k = drift.copy()
        for name in channels:
            h_k += u[name][k] * operators[name]
        x_k = -1j * steps[k] * h_k
        u_k = expm(x_k)
        psi = u_k @ psi
        if keep_slices:
            generators.append(x_k)
            propagators.append(u_k)
            forward.append(psi)

    final_states = [np.array(psi[:, j]) for j in range(n_states)]
    objective = terminal_objective(final_states)
    try:
        raw_value, raw_costates = objective
    except (TypeError, ValueError):
        raise TypeError("terminal_objective must return a (value, costates) pair.") from None
    if isinstance(raw_value, (bool, np.bool_)) or np.iscomplexobj(raw_value) or np.ndim(raw_value) != 0:
        raise ValueError(f"terminal_objective must return one finite real scalar value; got {raw_value!r}.")
    value = float(raw_value)
    if not np.isfinite(value):
        raise ValueError(f"terminal_objective must return one finite real scalar value; got {value!r}.")
    if len(raw_costates) != n_states:
        raise ValueError(f"terminal_objective must return one costate per initial state; got {len(raw_costates)}.")

    return {
        "dim": dim,
        "channels": channels,
        "operators": operators,
        "steps": steps,
        "n_slices": n_slices,
        "generators": generators,
        "propagators": propagators,
        "forward": forward,
        "value": value,
        "raw_costates": raw_costates,
    }


def value(
    parameters: Mapping[str, Any],
    *,
    h0: np.ndarray,
    controls: Mapping[str, np.ndarray],
    initial_states: Sequence[np.ndarray],
    time_grid: np.ndarray,
    control_map: Callable[[Mapping[str, Any]], Mapping[str, np.ndarray]],
    terminal_objective: Callable[[list[np.ndarray]], tuple[float, Sequence[np.ndarray]]],
) -> float:
    """Evaluate one candidate's scalar value only (no backward pass).

    Accepts the same ``terminal_objective`` contract as :func:`value_and_grad`;
    the returned costates are ignored.
    """
    state = _forward_pass(
        parameters,
        h0=h0,
        controls=controls,
        initial_states=initial_states,
        time_grid=time_grid,
        control_map=control_map,
        terminal_objective=terminal_objective,
        keep_slices=False,
    )
    return float(state["value"])


def value_and_grad(
    parameters: Mapping[str, Any],
    *,
    h0: np.ndarray,
    controls: Mapping[str, np.ndarray],
    initial_states: Sequence[np.ndarray],
    time_grid: np.ndarray,
    control_map: Callable[[Mapping[str, Any]], Mapping[str, np.ndarray]],
    control_pullback: Callable[[Mapping[str, Any], dict[str, np.ndarray]], Mapping[str, Any]],
    terminal_objective: Callable[[list[np.ndarray]], tuple[float, Sequence[np.ndarray]]],
) -> tuple[float, dict[str, Any]]:
    """Evaluate one candidate's scalar value and named parameter gradient.

    Returns ``(value, gradient)`` where ``gradient`` is the named mapping
    produced by ``control_pullback``.
    """
    state = _forward_pass(
        parameters,
        h0=h0,
        controls=controls,
        initial_states=initial_states,
        time_grid=time_grid,
        control_map=control_map,
        terminal_objective=terminal_objective,
        keep_slices=True,
    )
    dim = state["dim"]
    channels = state["channels"]
    operators = state["operators"]
    steps = state["steps"]
    n_slices = state["n_slices"]
    generators = state["generators"]
    propagators = state["propagators"]
    forward = state["forward"]
    value = state["value"]
    chi = np.column_stack(
        [_require_vector(f"costates[{j}]", c, dim) for j, c in enumerate(state["raw_costates"])]
    )

    # Backward pass: costates through the same slice propagators, exact
    # Frechet derivative of each slice exponential per channel.
    channel_gradients = {name: np.zeros(n_slices) for name in channels}
    for k in range(n_slices - 1, -1, -1):
        for name in channels:
            frechet = expm_frechet(generators[k], -1j * steps[k] * operators[name], compute_expm=False)
            channel_gradients[name][k] = 2.0 * float(np.real(np.vdot(chi, frechet @ forward[k])))
        chi = propagators[k].conj().T @ chi

    gradient = control_pullback(parameters, channel_gradients)
    if not isinstance(gradient, Mapping):
        raise TypeError(f"control_pullback must return a named gradient mapping; got {type(gradient).__name__}.")
    return value, dict(gradient)
