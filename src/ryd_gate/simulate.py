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

import numpy as np

from ryd_gate.ir import EvolutionResult

_EXACT_BACKENDS = {"exact", "sparse", "sparse_expm"}


def simulate(
    system,
    x=(),
    psi0="all_ground",
    *,
    backend: str = "exact",
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
        state object). Forwarded to the selected engine.
    backend
        ``"exact"`` (default) for exact state-vector evolution, or a
        tensor-network backend name. Unknown names raise ``ValueError``.
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
        ``result.sample(...)``, and ``result.final_state`` work directly.
    """
    key = backend.lower()
    if key in _EXACT_BACKENDS:
        # Exact dispatch is terminal: the exact engine has a single solver path,
        # so the name is consumed here rather than forwarded. (Its own ``backend``
        # arg is a SolverBackend object / auto-select, not a routing key.)
        from ryd_gate.backends.exact import simulate as simulate_exact

        result = simulate_exact(system, x, psi0, **kwargs)
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

