"""Shared strict preflight for evolution requests (all backends).

The canonical protocol→backend lowering lives in ``ryd_gate.core.lowering``; the
result types live in ``ryd_gate.results``. This module holds only the strict
``t_eval`` / observables validation every backend runs before evolving.
"""

from __future__ import annotations

import numpy as np


def validate_evolution_request(t_gate: float, t_eval, observables):
    """Validate the measurement request; return ``(t_eval copy or None, obs dict)``.

    Rules (E06/O02/O06):

    - ``t_eval=None`` -> measure only at ``t_gate`` (public ``times == [t_gate]``).
    - explicit ``t_eval``: 1-D, non-empty, finite, strictly increasing, within
      ``[0, t_gate]``; returned verbatim (no auto-appended endpoint). Boolean
      ``t_eval`` is a ``TypeError``; ``[]`` is a ``ValueError`` (use ``None``).
    - explicit ``t_eval`` with no observables is a ``ValueError``.
    - ``observables`` is ``None`` or ``dict[str, ObservableExpr]``; every final
      expression must be Hermitian (O06/O12).
    """
    from ryd_gate.core.observables import ObservableExpr, _is_hermitian

    t_gate = float(t_gate)
    if not t_gate > 0.0:
        raise ValueError(f"t_gate must be positive; got {t_gate!r}.")

    if observables is None:
        obs: dict = {}
    elif isinstance(observables, dict):
        for label, expr in observables.items():
            if not isinstance(label, str):
                raise TypeError(f"observable labels must be strings; got {type(label).__name__}.")
            if not isinstance(expr, ObservableExpr):
                raise TypeError(
                    f"observables[{label!r}] must be an ObservableExpr built from "
                    f"system.observables; got {type(expr).__name__}."
                )
            if not _is_hermitian(expr):
                raise ValueError(
                    f"observables[{label!r}] is not Hermitian; expectations must be "
                    "real. Split a non-Hermitian coherence into Hermitian X/Y parts."
                )
        obs = dict(observables)
    else:
        raise TypeError(
            f"observables must be a dict[str, ObservableExpr] or None; got {type(observables).__name__}."
        )

    if t_eval is None:
        return None, obs
    if isinstance(t_eval, (bool, np.bool_)):
        raise TypeError("t_eval must be an array of times or None (boolean t_eval is invalid).")
    times = np.array(t_eval, dtype=float, copy=True)
    if times.ndim != 1:
        raise ValueError("t_eval must be a one-dimensional array of times.")
    if times.size == 0:
        raise ValueError("explicit t_eval must be non-empty; use t_eval=None for an endpoint-only run.")
    if not np.all(np.isfinite(times)):
        raise ValueError("t_eval must contain only finite times.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("t_eval must be strictly increasing (no duplicates).")
    if times[0] < 0.0:
        raise ValueError(f"t_eval must lie within [0, t_gate]; got t_eval[0]={times[0]}.")
    if times[-1] > t_gate and not np.isclose(times[-1], t_gate, rtol=1e-12, atol=0.0):
        raise ValueError(f"t_eval must lie within [0, t_gate={t_gate}]; got t_eval[-1]={times[-1]}.")
    if not obs:
        raise ValueError(
            "explicit t_eval requires observables (nothing to record otherwise); "
            "pass observables={label: expr} or drop t_eval."
        )
    return times, obs
