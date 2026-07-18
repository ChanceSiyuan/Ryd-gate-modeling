"""Single-start scalar-loss minimization (ADR-0015, 0016, 0017, 0018, 0020, 0021, 0022, 0023).

``minimize`` is numerical parameter optimization only: it packs the named
candidate into private flat coordinates, runs one registered solver, and
returns the best candidate with the original names and shapes. Study
orchestration (multistart, continuation, validation) stays with the caller,
and no loss evaluation is ever cached or deduplicated.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import scipy.optimize

from ._parameters import ParameterMapping, ParameterPacker
from ._result import OptimizationResult

_SCIPY_METHODS = {
    "nelder-mead": "Nelder-Mead",
    "powell": "Powell",
    "l-bfgs-b": "L-BFGS-B",
}
_GRADIENT_METHODS = {"l-bfgs-b"}


def _check_scalar_loss(value: object) -> float:
    """Fail fast unless ``value`` is one finite real scalar (ADR-0020)."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"the scalar loss must return one finite real scalar; got a boolean {value!r}.")
    if isinstance(value, complex) or np.iscomplexobj(value):
        raise ValueError(f"the scalar loss must return one finite real scalar; got a complex value {value!r}.")
    if np.ndim(value) != 0:
        raise ValueError(f"the scalar loss must return one finite real scalar; got shape {np.shape(value)}.")
    out = float(value)  # type: ignore[arg-type]
    if not math.isfinite(out):
        raise ValueError(
            f"the scalar loss must return one finite real scalar; got {out!r}. "
            "Convert a recognized undesirable candidate into an explicit finite penalty inside the loss."
        )
    return out


def minimize(
    loss: Callable[[Mapping[str, Any]], float],
    x0: ParameterMapping,
    *,
    method: str,
    bounds: Mapping[str, tuple] | None = None,
    scales: Mapping[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
    callback: Callable[[int, dict, float, float], None] | None = None,
) -> OptimizationResult:
    """Run one single-start solve of ``loss`` from the named candidate ``x0``.

    ``method`` is a stable lower-case name selecting one registered solver.
    ``options`` is that method's option mapping, forwarded to the solver, with
    two reserved keys: ``"gradient"`` (gradient methods only) supplies a
    callable mapping named parameters to a named gradient mapping with the
    same names and shapes, and ``"iteration_callback"`` supplies a callable
    invoked with the named parameters after each accepted solver iteration
    (line-search evaluations are not iterations). ``callback(index,
    parameters, loss, best_loss)`` is invoked after each successful finite
    loss evaluation.
    """
    if not isinstance(method, str) or method not in _SCIPY_METHODS:
        raise ValueError(f"unknown optimization method {method!r}; registered methods: {sorted(_SCIPY_METHODS)}.")
    if not callable(loss):
        raise TypeError(f"loss must be callable; got {type(loss).__name__}.")
    if callback is not None and not callable(callback):
        raise TypeError(f"callback must be callable or None; got {type(callback).__name__}.")

    packer = ParameterPacker(x0)
    scale = packer.pack_scales(scales)
    z0 = packer.pack(x0) / scale

    packed_bounds = packer.pack_bounds(bounds)
    scipy_bounds = None
    if packed_bounds is not None:
        scipy_bounds = [(lo / s, hi / s) for (lo, hi), s in zip(packed_bounds, scale)]

    solver_options = dict(options) if options is not None else {}
    gradient = solver_options.pop("gradient", None)
    if gradient is not None:
        if method not in _GRADIENT_METHODS:
            raise ValueError(f"options['gradient'] is only supported by methods {sorted(_GRADIENT_METHODS)}.")
        if not callable(gradient):
            raise TypeError(f"options['gradient'] must be callable; got {type(gradient).__name__}.")
    iteration_callback = solver_options.pop("iteration_callback", None)
    if iteration_callback is not None and not callable(iteration_callback):
        raise TypeError(f"options['iteration_callback'] must be callable; got {type(iteration_callback).__name__}.")

    state: dict[str, Any] = {"n_evaluations": 0, "best_loss": math.inf, "best_z": None}

    def fun(z: np.ndarray) -> float:
        params = packer.unpack(np.asarray(z, dtype=float) * scale)
        value = _check_scalar_loss(loss(params))
        state["n_evaluations"] += 1
        if value < state["best_loss"]:
            state["best_loss"] = value
            state["best_z"] = np.array(z, dtype=float)
        if callback is not None:
            callback(state["n_evaluations"], params, value, state["best_loss"])
        return value

    jac = None
    if gradient is not None:

        def jac(z: np.ndarray) -> np.ndarray:
            params = packer.unpack(np.asarray(z, dtype=float) * scale)
            named = gradient(params)
            packed = packer.pack(named, what="gradient")
            return packed * scale  # chain rule: d/d(x/s) = s * d/dx

    scipy_callback = None
    if iteration_callback is not None:

        def scipy_callback(zk: np.ndarray) -> None:
            iteration_callback(packer.unpack(np.asarray(zk, dtype=float) * scale))

    res = scipy.optimize.minimize(
        fun,
        z0,
        method=_SCIPY_METHODS[method],
        jac=jac,
        bounds=scipy_bounds,
        callback=scipy_callback,
        options=solver_options or None,
    )

    best_z = state["best_z"] if state["best_z"] is not None else np.asarray(res.x, dtype=float)
    best_loss = state["best_loss"] if math.isfinite(state["best_loss"]) else float(res.fun)
    return OptimizationResult(
        best_parameters=packer.unpack(np.asarray(best_z, dtype=float) * scale),
        best_loss=float(best_loss),
        success=bool(res.success),
        message=str(res.message),
        method=method,
        n_iterations=int(getattr(res, "nit", 0) or 0),
        n_evaluations=int(state["n_evaluations"]),
    )
