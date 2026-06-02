"""High-level TN simulation entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.backends.base import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.protocols.sweep import SweepProtocol


def simulate_tn(
    spec,
    protocol: SweepProtocol,
    x: list[float],
    initial_state: str | np.ndarray | object = "all_ground",
    method: str = "tdvp",
    t_eval: np.ndarray | None = None,
    observables: list[str] | None = None,
    backend_options: dict | None = None,
) -> EvolutionResult:
    """High-level TN simulation entry point.

    Parameters
    ----------
    spec : TNLatticeSpec
        Tensor-network lattice specification.
    protocol : SweepProtocol
        Sweep protocol describing drive coefficients and local
        addressing.
    x : [delta_start, delta_end, t_sweep]
        Protocol parameters.
    initial_state : str, ndarray, or tenpy MPS
        Initial state. Strings: ``"all_ground"``, ``"af1"``,
        ``"af2"``. Arrays: per-site occupation (0/1).
    method : str
        ``"tdvp"`` for time evolution, ``"dmrg"`` for ground state.
    t_eval : ndarray or None
        Times at which to record observables (only for TDVP).
    observables : list of str or None
        Observable names to record (only for TDVP).
    backend_options : dict or None
        Options passed to the backend constructor
        (e.g., ``{"chi_max": 512, "dt": 0.1}``).

    Returns
    -------
    EvolutionResult
    """
    from .lattice_spec import TNLatticeSpec

    if not isinstance(spec, TNLatticeSpec):
        raise TypeError("simulate_tn() requires a TNLatticeSpec.")

    opts = backend_options or {}

    if method == "dmrg":
        from .backends import TenpyDMRGBackend
        from .backends import _TNProtocolContext, _pin_deltas_from_params

        backend = TenpyDMRGBackend(**opts)
        params = protocol.unpack_params(x, _TNProtocolContext(spec))
        pin_deltas = _pin_deltas_from_params(params, spec.N)
        # For DMRG, use delta_end as the target detuning
        Delta = params["delta_end"] if "delta_end" in params else x[1] if len(x) > 1 else x[0]
        return backend.find_ground_state(
            spec, Delta, pin_deltas=pin_deltas,
            initial_state=initial_state,
        )

    elif method == "tdvp":
        from .backends import TenpyTDVPBackend
        from .state import product_state_mps

        backend = TenpyTDVPBackend(**opts)
        # Build initial MPS if not already one
        if not hasattr(initial_state, "expectation_value"):
            psi0 = product_state_mps(spec, initial_state)
        else:
            psi0 = initial_state

        return backend.evolve(
            spec, protocol, x, psi0,
            t_eval=t_eval,
            observables=observables,
        )

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'dmrg' or 'tdvp'.")
