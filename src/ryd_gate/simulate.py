"""Unified simulation entry point.

``ryd_gate.simulate(system, x, psi0, backend=...)`` is a thin dispatcher over the
two real engines:

- exact state-vector — :func:`ryd_gate.backends.exact.simulate`
  (``backend`` in ``{"exact", "sparse", "sparse_expm"}``)
- tensor-network — :func:`ryd_gate.backends.tn_common.simulate_tn`
  (``backend`` in ``{"tn", "tenpy", "mps", "gputn", "itensors", "ttn", "2dtn",
  "peps", "nqs"}``)

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
        from ryd_gate.backends.exact import simulate as simulate_exact

        return simulate_exact(system, x, psi0, **kwargs)

    from ryd_gate.backends.tn_common import simulate_tn
    from ryd_gate.backends.tn_common.compiler import tn_lattice_spec_from_system

    if system.protocol is None:
        raise ValueError(
            "Tensor-network simulation requires a protocol bound to the system. "
            "Construct with `protocol=...` or call `.with_protocol(...)`."
        )
    spec = tn_lattice_spec_from_system(system)
    return simulate_tn(spec, system.protocol, x, initial_state=psi0, backend=key, **kwargs)
