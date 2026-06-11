"""Unified simulation entry point.

``ryd_gate.simulate(system, x, psi0, backend=...)`` is a thin dispatcher over the
two real engines:

- exact state-vector — :func:`ryd_gate.backends.exact.simulate`
  (``backend`` in ``{"exact", "sparse", "sparse_expm"}``)
- tensor-network — :func:`ryd_gate.backends.tn_common.simulate_tn`
  (``backend`` in ``{"mps", "peps", "gputn", "pepskit"}``)

For tensor-network backends the system's geometry and bound protocol are lowered
to a TN lattice spec automatically; ``backend_options`` and other engine kwargs
are forwarded unchanged.
"""

from __future__ import annotations

from ryd_gate.ir.evolution import EvolutionResult

_EXACT_BACKENDS = {"exact", "sparse", "sparse_expm"}


def simulate(system, x, psi0="all_ground", *, backend: str = "exact", **kwargs) -> EvolutionResult:
    """Compile a protocol-bound Rydberg system and evolve it with ``backend``.

    Parameters
    ----------
    system
        A :class:`~ryd_gate.core.system.RydbergSystem` with a protocol bound.
    x
        Protocol parameter vector.
    psi0
        Initial state (e.g. ``"all_ground"``, an occupation list, or a backend
        state object). Forwarded to the selected engine.
    backend
        ``"exact"`` (default) for exact state-vector evolution, or a
        tensor-network backend name. Unknown names raise ``ValueError``.
    **kwargs
        Engine-specific options forwarded verbatim (e.g. ``t_eval``,
        ``backend_options``, ``method``, ``observables``).
    """
    key = backend.lower()
    if key in _EXACT_BACKENDS:
        # Exact dispatch is terminal: the exact engine has a single solver path,
        # so the name is consumed here rather than forwarded. (Its own ``backend``
        # arg is a SolverBackend object / auto-select, not a routing key.)
        from ryd_gate.backends.exact import simulate as simulate_exact

        return simulate_exact(system, x, psi0, **kwargs)

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
    return simulate_tn(spec, system.protocol, x, initial_state=psi0, backend=key, **kwargs)


def simulate_sequence(
    sequence,
    *,
    backend: str = "exact",
    psi0=None,
    interaction=None,
    observables=None,
    **kwargs,
):
    """Compile a :class:`~ryd_gate.sequence.Sequence` and run it.

    Parameters
    ----------
    sequence
        A built :class:`~ryd_gate.sequence.Sequence`.
    backend
        ``"exact"`` or ``"mps"`` in Stage 3. Other backend names still raise
        ``NotImplementedError`` through the sequence path.
    psi0
        Initial state. ``None`` (default) prepares every atom in the level
        structure's ``initial_level_or_default()``; strings and arrays are
        forwarded to the engine unchanged.
    interaction
        Optional :class:`~ryd_gate.core.level_structures.InteractionSpec`.
    observables
        Optional :class:`~ryd_gate.observables.ObservableConfig`. On
        ``backend="mps"`` the names/times are lowered onto the existing
        TeNPy streaming measurement path (recorded values land in
        ``result.raw.metadata["obs"]``). The exact backend ignores the
        streaming schedule (Stage 6 rule: final-state handle semantics
        are unchanged).
    **kwargs
        Engine options forwarded verbatim to :func:`simulate`
        (e.g. ``t_eval``, ``n_steps``).

    Returns
    -------
    SimulationResult
        Lazy result wrapper; the kernel ``EvolutionResult`` stays at ``.raw``.
    """
    from ryd_gate.protocols.sequence_protocol import compile_sequence_to_system
    from ryd_gate.results import (
        ExactStateHandle,
        MPSStateHandle,
        SimulationResult,
        UnsupportedResultQuery,
        UnsupportedStateHandle,
    )

    key = backend.lower()
    if key not in {"exact", "mps"}:
        raise NotImplementedError(
            f"simulate_sequence.backend_not_stage3: backend {backend!r} is not available "
            "through the sequence path in Stage 3."
        )
    if not sequence.level_structure.supports_backend(key):
        raise ValueError("level_structure.backend_unsupported")
    if observables is not None:
        from ryd_gate.core.validation import raise_for_errors

        raise_for_errors(observables.validate())
        if key == "mps":
            import numpy as np

            kwargs.setdefault("observables", list(observables.names))
            schedule = observables.schedule_times_ns(sequence.duration_ns)
            if schedule is not None:
                kwargs.setdefault(
                    "t_eval", np.asarray([t * 1e-9 for t in schedule], dtype=float)
                )
    system = compile_sequence_to_system(sequence, interaction=interaction)
    if psi0 is None:
        level = sequence.level_structure.initial_level_or_default()
        psi0 = (
            system.product_state([level] * sequence.register.n_atoms)
            if key == "exact"
            else [level] * sequence.register.n_atoms
        )
    raw = simulate(system, [], psi0, backend=key, **kwargs)
    raw.metadata.setdefault(
        "state_handle_kind",
        "statevector" if key == "exact" else "unsupported",
    )

    if raw.metadata.get("state_handle_kind") == "statevector":
        state = ExactStateHandle(
            psi=raw.psi_final,
            system=system,
            register=sequence.register,
            level_structure=sequence.level_structure,
        )
    elif raw.metadata.get("state_handle_kind") == "mps":
        native_state = raw.metadata.get("native_state", raw.psi_final)
        if native_state is None:
            raise UnsupportedResultQuery("mps.native_state_missing")
        spec = raw.metadata.get("tn_spec")
        if spec is None:
            raise UnsupportedResultQuery("mps.spec_missing")
        state = MPSStateHandle(
            mps=native_state,
            spec=spec,
            register_ids=tuple(sequence.register.ids),
            metadata=dict(raw.metadata),
        )
    else:
        state = UnsupportedStateHandle(
            backend=key,
            reason_code=_unsupported_state_reason(key),
            n_atoms=sequence.register.n_atoms,
            local_levels=tuple(sequence.level_structure.levels),
            atom_ids=tuple(sequence.register.ids),
        )
    return SimulationResult(raw=raw, state=state, backend=key, sequence=sequence)


def _unsupported_state_reason(backend: str) -> str:
    if backend == "gputn":
        return "gputn.state_handle_not_implemented"
    if backend == "peps":
        return "peps.state_handle_not_implemented"
    if backend == "mps":
        return "mps.state_handle_not_implemented"
    return f"{backend}.state_handle_not_implemented"
