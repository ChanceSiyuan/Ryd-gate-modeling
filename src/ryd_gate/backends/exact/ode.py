"""Fixed-solver (DOP853) exact ODE driver and its strict options (E04/E05/E11)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

_DEFAULTS = {"hamiltonian_format": "auto", "rtol": 1e-8, "atol": 1e-12}


def validated_exact_options(backend_options: dict | None) -> dict:
    """Validate exact-backend options; return the resolved schema (E04/E11)."""
    opts = dict(_DEFAULTS)
    if backend_options:
        unknown = set(backend_options) - set(_DEFAULTS)
        if unknown:
            raise ValueError(
                f"unknown exact backend option(s): {sorted(unknown)}. "
                f"Allowed: {sorted(_DEFAULTS)}."
            )
        opts.update(backend_options)
    if opts["hamiltonian_format"] not in ("auto", "dense", "sparse"):
        raise ValueError(
            f"hamiltonian_format must be 'auto', 'dense' or 'sparse'; got {opts['hamiltonian_format']!r}."
        )
    for key in ("rtol", "atol"):
        v = float(opts[key])
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(f"{key} must be finite and strictly positive; got {opts[key]!r}.")
        opts[key] = v
    return opts


def evolve_states(ham, t_gate: float, psi0_list, solver_eval, *, rtol: float, atol: float):
    """Integrate each initial state to ``t_gate`` (DOP853), recording ``solver_eval``.

    Returns a list of ``(dim, len(solver_eval))`` complex arrays, one per input
    state. The compiled ``ham`` is shared across the batch.
    """

    def rhs(t, y):
        return -1j * ham.apply(t, y)

    out = []
    for psi0 in psi0_list:
        sol = solve_ivp(
            rhs, (0.0, float(t_gate)), np.asarray(psi0, dtype=complex),
            method="DOP853", t_eval=solver_eval, rtol=rtol, atol=atol, dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(f"exact ODE solver failed: {sol.message}")
        out.append(sol.y)
    return out
