"""Unified simulation entry point.

``ryd_gate.simulate(system, initial_state=None, ...)`` is a thin dispatcher
over the two real engines:

- exact state-vector — :func:`ryd_gate.backends.exact.simulate`
  (``backend="exact_ode"``, the adaptive DOP853 integrator)
- tensor-network — ``ryd_gate.backends.tn_common`` (``backend`` in
  ``{"mps", "peps"}``)

The system's bound protocol is fully specified; its normalized quantities
resolve against the system at compile time (the resolved duration is
``system.t_gate``).  For tensor-network backends the system's geometry and
bound protocol are lowered to a TN lattice spec automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryd_gate.ir import EvolutionResult, validate_evolution_request

if TYPE_CHECKING:
    from ryd_gate.core.observables import ObservableExpr

# The only exact engine is the adaptive ODE integrator; the old piecewise-expm
# backends (exact_dense/exact_sparse) are gone. Storage/matvec strategy is a
# backend option: backend_options={"hamiltonian_format": "auto"|"dense"|"sparse"}.
_EXACT_BACKENDS = frozenset({"exact_ode"})
_REMOVED_EXACT = frozenset({"exact", "exact_dense", "exact_sparse"})


def simulate(
    system,
    initial_state=None,
    *,
    backend: str = "exact_ode",
    t_eval=None,
    observables: "dict[str, ObservableExpr] | None" = None,
    backend_options: dict | None = None,
) -> EvolutionResult:
    """Compile a protocol-bound Rydberg system and evolve it with ``backend``.

    Parameters
    ----------
    system
        A :class:`~ryd_gate.core.system.RydbergSystem` with a fully specified
        protocol bound.  The protocol decides the gate duration
        (``system.t_gate``).
    initial_state
        ``None`` (default) starts every site in the preset's initial level;
        ``"plus"`` starts every site in ``(|0>+|1>)/sqrt(2)``; a flat sequence
        of per-site level labels (e.g. ``["0", "1"]``) is one product state; a
        nested sequence of label sequences (e.g. ``[["0","0"], ["0","1"]]``)
        is a *batch* of product states evolved under the same compiled
        protocol, returning a ``list[EvolutionResult]``.  Arbitrary dense
        vectors and backend-native states are not accepted here — research
        scripts needing them use the explicit backend seams.
    backend
        ``"exact_ode"`` (default) or a tensor-network name (``"mps"``,
        ``"peps"``).
    t_eval
        Measurement times only — never the evolution endpoint.  ``None``
        records at ``t_gate`` (result ``times == [t_gate]``); an explicit array
        must be 1-D, non-empty, finite, strictly increasing, within
        ``[0, t_gate]``, and requires ``observables``.  The requested times
        come back exactly as ``result.times``.
    observables
        ``dict[str, ObservableExpr]`` built from ``system.observables``
        (e.g. ``{"n_r": system.observables.level_sum("r")}``); each entry
        becomes a complex array in ``result.expectations`` with one raw
        ``<psi|O|psi>`` value per entry of ``result.times``.
    backend_options
        Engine-specific numerical options: for ``exact_ode``
        ``{"hamiltonian_format": "auto"|"dense"|"sparse", "rtol": ...,
        "atol": ...}`` (unknown keys error); ``{"dt": ...}`` and truncation
        options for the TN engines.

    Returns
    -------
    EvolutionResult or list[EvolutionResult]
        One result, or one per batch entry for a nested ``initial_state``.
    """
    from ryd_gate.core.states import normalize_initial_state

    key = backend.lower()
    if key in _REMOVED_EXACT:
        raise ValueError(
            f"backend={backend!r} has been removed; the only exact backend is "
            "'exact_ode' (dense/sparse Hamiltonian storage is a backend option: "
            "backend_options={'hamiltonian_format': 'auto'|'dense'|'sparse'})."
        )

    kind, state = normalize_initial_state(initial_state, system.basis.n_sites)

    if kind == "batch":
        if key in _EXACT_BACKENDS:
            from ryd_gate.backends.exact import simulate_states as simulate_exact_states

            return simulate_exact_states(
                system, state,
                t_eval=t_eval,
                observables=observables,
                backend_options=backend_options,
            )
        # Tensor-network backends gain no per-step speedup from batching (per-state
        # evolution dominates); loop, reusing the normal single-state path per state.
        return [
            simulate(
                system, s, backend=backend, t_eval=t_eval,
                observables=observables, backend_options=backend_options,
            )
            for s in state
        ]

    if key in _EXACT_BACKENDS:
        from ryd_gate.backends.exact import simulate as simulate_exact

        return simulate_exact(
            system, state, t_eval=t_eval, observables=observables,
            backend_options=backend_options,
        )

    # TN dispatch is not terminal: ``key`` selects the engine inside simulate_tn,
    # so the name is forwarded for downstream routing/normalization.
    from ryd_gate.backends.tn_common.compiler import tn_lattice_spec_from_system
    from ryd_gate.backends.tn_common.simulate import simulate_tn

    if system.protocol is None:
        raise ValueError(
            "Tensor-network simulation requires a protocol bound to the system. "
            "Construct with `protocol=...` or call `.with_protocol(...)`."
        )
    # Unified preflight (strict t_eval + dict-of-expression observables) runs
    # before any TN evolution; the engines receive the validated dict.
    validate_evolution_request(system.t_gate, t_eval, observables)
    spec = tn_lattice_spec_from_system(system)
    return simulate_tn(
        spec, system.protocol, initial_state=state, backend=key,
        observables=observables, t_eval=t_eval, backend_options=backend_options,
    )
