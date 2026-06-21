"""Unified simulation entry point.

``ryd_gate.simulate(system, x, psi0, backend=...)`` is a thin dispatcher over the
two real engines:

- exact state-vector — :func:`ryd_gate.backends.exact.simulate`
  (``backend`` in ``{"exact_dense", "exact_sparse"}``; bare ``"exact"`` was removed)
- tensor-network — :func:`ryd_gate.backends.tn_common.simulate_tn`
  (``backend`` in ``{"mps", "peps"}``)

For tensor-network backends the system's geometry and bound protocol are lowered
to a TN lattice spec automatically; ``backend_options`` and other engine kwargs
are forwarded unchanged.
"""

from __future__ import annotations

import numpy as np

from ryd_gate.ir import EvolutionResult

# The exact state-vector engine must be selected explicitly: ``exact_dense`` forces a
# dense per-step ``expm``, ``exact_sparse`` forces ``expm_multiply``. The old auto key
# ``"exact"`` has been removed -- callers choose the solver themselves. The map doubles
# as the membership test (``key in _EXACT_KINDS``) and the routing-key -> kind lookup.
_EXACT_KINDS = {"exact_dense": "dense", "exact_sparse": "sparse"}


def _is_state_batch(psi0) -> bool:
    """True iff ``psi0`` is a batch of states (a list of per-atom label-lists).

    The single allowed batch form is a list/tuple whose entries are themselves
    label-lists, e.g. ``[["1","1"], ["0","0"]]``. A flat label-list like
    ``["0","0"]`` is a single product state, not a batch.
    """
    return (
        isinstance(psi0, (list, tuple))
        and len(psi0) > 0
        and isinstance(psi0[0], (list, tuple))
    )


def _forced_expm_kind(backend_key: str) -> str:
    """Map a public exact routing key to a forced expm kind (``dense``/``sparse``)."""
    try:
        return _EXACT_KINDS[backend_key]
    except KeyError:
        raise ValueError(f"Unknown exact backend {backend_key!r}.") from None


def _resolve_exact_solver_backend(system, backend_key: str, backend_options):
    """Return a concrete :class:`SolverBackend` for the forced exact routing key."""
    kind = _forced_expm_kind(backend_key)
    from ryd_gate.backends._options import as_backend_options
    from ryd_gate.backends.exact.simulate import make_forced_expm_backend, resolve_n_steps

    opts = as_backend_options(backend_options)
    return make_forced_expm_backend(kind, n_steps=resolve_n_steps(system, opts))


def simulate(
    system,
    x=(),
    psi0="all_ground",
    *,
    backend: str = "exact_dense",
    observables: list[str] | None = None,
    **kwargs,
) -> EvolutionResult:
    """Compile a protocol-bound Rydberg system and evolve it with ``backend``.

    Parameters
    ----------
    system
        A :class:`~ryd_gate.core.system.RydbergSystem` with a protocol bound.
    x
        Protocol parameter vector. Optional (defaults to empty): only the
        CZ-gate protocols (``TOProtocol`` / ``ARProtocol``) take parameters;
        schedule-on-the-protocol cases (sweep, TFIM, digital-analog) need none.
    psi0
        Initial state (e.g. ``"all_ground"``, an occupation list, or a backend
        state object). Forwarded to the selected engine. A **list of per-atom
        label-lists** (e.g. ``[["1","1"], ["0","0"]]``) is treated as a *batch* of
        initial states evolved under the same compiled protocol; the call then
        returns a ``list[EvolutionResult]`` (one per state). On the exact backend
        the compilation -- and, for the dense-``expm`` ladder models, each step's
        propagator -- is shared across the batch (see
        :func:`ryd_gate.backends.exact.simulate_states`); tensor-network backends
        loop per state (correct, but no per-step speedup). A flat label-list such
        as ``["0","0"]`` remains a single product state.
    backend
        Exact state-vector backends (the solver must be chosen explicitly; the old
        auto key ``"exact"`` has been removed):

        * ``"exact_dense"`` (default) — :class:`~ryd_gate.backends.exact.dense_expm.DenseExpmBackend`
        * ``"exact_sparse"`` — :class:`~ryd_gate.backends.exact.sparse_expm.SparseExpmBackend`

        Tensor-network names (``"mps"``, ``"peps"``) are also accepted.
    observables
        Optional names of registered observables to evaluate. They are exposed
        on the result via ``result.expectations`` / ``result.expectation(name)``
        (final-state values on the exact backend; per-time series on the
        tensor-network backends, which measure during evolution).
    **kwargs
        Engine-specific options forwarded verbatim (e.g. ``t_eval``,
        ``backend_options``, ``method``).

    Returns
    -------
    EvolutionResult
        The measuring ``system`` is attached, so ``result.expectation(...)``,
        ``result.sample(...)``, and ``result.final_state`` work directly. For a
        batched ``psi0`` (list of label-lists) a ``list[EvolutionResult]`` is
        returned instead, one per initial state.
    """
    key = backend.lower()
    if key == "exact":
        raise ValueError(
            "backend='exact' has been removed; choose the exact solver explicitly "
            "with backend='exact_dense' or backend='exact_sparse'."
        )
    if _is_state_batch(psi0):
        if key in _EXACT_KINDS:
            from ryd_gate.backends.exact import simulate_states as simulate_exact_states

            solver = _resolve_exact_solver_backend(
                system, key, kwargs.get("backend_options"),
            )
            results = simulate_exact_states(
                system, list(psi0), x,
                t_eval=kwargs.get("t_eval"),
                backend_options=kwargs.get("backend_options"),
                backend=solver,
            )
            return [_attach_measurement(r, system, observables) for r in results]
        # Tensor-network backends gain no per-step speedup from batching (per-state
        # evolution dominates); loop, reusing the normal single-state path per state.
        return [
            simulate(system, x, s, backend=backend, observables=observables, **kwargs)
            for s in psi0
        ]

    if key in _EXACT_KINDS:
        from ryd_gate.backends.exact import simulate as simulate_exact

        solver = _resolve_exact_solver_backend(
            system, key, kwargs.get("backend_options"),
        )
        result = simulate_exact(system, x, psi0, backend=solver, **kwargs)
        return _attach_measurement(result, system, observables)

    # TN dispatch is not terminal: ``key`` selects the engine inside simulate_tn,
    # so the name is forwarded for downstream routing/normalization.
    from ryd_gate.backends.tn_common import simulate_tn
    from ryd_gate.backends.tn_common.compiler import tn_lattice_spec_from_system

    if system.protocol is None:
        raise ValueError(
            "Tensor-network simulation requires a protocol bound to the system. "
            "Construct with `protocol=...` or call `.with_protocol(...)`."
        )
    spec = tn_lattice_spec_from_system(system)
    result = simulate_tn(
        spec, system.protocol, x, initial_state=psi0, backend=key,
        observables=observables, **kwargs,
    )
    return _attach_measurement(result, system, observables)


def _attach_measurement(result: EvolutionResult, system, observables) -> EvolutionResult:
    """Attach the measuring system and surface requested observables.

    Tensor-network backends record requested observables in ``metadata["obs"]``
    (per-time arrays); the exact backend returns a dense final state, so the
    observables are evaluated here. Both land in ``result.expectations`` for a
    single, backend-agnostic readout.
    """
    result.system = system
    obs = result.metadata.get("obs") if isinstance(result.metadata, dict) else None
    if obs:
        result.expectations = dict(obs)
    if observables and isinstance(result.psi_final, np.ndarray):
        eager = {name: system.expectation(name, result.psi_final) for name in observables}
        result.expectations = {**(result.expectations or {}), **eager}
    return result
