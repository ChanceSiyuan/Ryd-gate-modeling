"""Observable-based metric functions using SystemModel.observables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ryd_gate.core.system_model import SystemModel
    from ryd_gate.solvers.base import EvolutionResult


def measure_observables(
    model: SystemModel,
    result: EvolutionResult,
    observable_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute expectation values of observables on the final state.

    Parameters
    ----------
    model : SystemModel
        System model with observable registry.
    result : EvolutionResult
        Evolution result containing psi_final.
    observable_names : list of str, optional
        Which observables to measure. None measures all.

    Returns
    -------
    dict mapping observable names to expectation values.
    """
    if observable_names is None:
        return model.observables.measure_all(result.psi_final)
    return {
        name: model.observables.measure(name, result.psi_final)
        for name in observable_names
    }


def measure_trajectory(
    model: SystemModel,
    states: np.ndarray,
    observable_names: list[str],
) -> dict[str, np.ndarray]:
    """Measure observables at each time step of a state trajectory.

    Parameters
    ----------
    model : SystemModel
        System model with observable registry.
    states : ndarray, shape (dim, n_t) or (n_t, dim)
        State trajectory. Column-major ``(dim, n_t)`` from
        ``DenseODEBackend``, row-major ``(n_t, dim)`` from
        ``SparseExpmBackend``.
    observable_names : list of str
        Observable names registered in ``model.observables``.

    Returns
    -------
    dict mapping observable names to ndarray of shape (n_t,).
    """
    # Detect layout: (dim, n_t) vs (n_t, dim)
    if states.shape[0] == model.basis.total_dim and states.ndim == 2:
        get_psi = lambda k: states[:, k]
        n_t = states.shape[1]
    else:
        get_psi = lambda k: states[k]
        n_t = states.shape[0]

    result = {name: np.empty(n_t) for name in observable_names}
    for k in range(n_t):
        psi = get_psi(k)
        for name in observable_names:
            result[name][k] = model.observables.measure(name, psi)
    return result


def state_overlap(psi: np.ndarray, target: np.ndarray) -> float:
    """Compute |<target|psi>|^2."""
    return float(abs(np.vdot(target, psi)) ** 2)


def norm_squared(psi: np.ndarray) -> float:
    """Compute <psi|psi> (useful for non-Hermitian evolution with decay)."""
    return float(np.real(np.vdot(psi, psi)))
