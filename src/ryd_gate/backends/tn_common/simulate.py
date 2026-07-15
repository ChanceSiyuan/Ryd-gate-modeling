"""Tensor-network simulation entry point (the single TN seam).

``simulate_tn(system, initial_state, *, backend, t_eval, observables,
backend_options)`` consumes a fully constructed :class:`RydbergSystem` — exactly
like the exact backend — runs the shared preflight (E06/O06 via
``validate_evolution_request``, O09 preset capability, E26 observable
capability), lowers the system to :class:`TNTerms`, and dispatches to the MPS or
PEPS engine.  Both return an :class:`~ryd_gate.results.EvolutionResult` with a
lazy state reader (ER04/ER09).
"""

from __future__ import annotations

import numpy as np

from ryd_gate.ir import validate_evolution_request
from ryd_gate.results import EvolutionResult

from ._expressions import preflight_tn_observables
from .compiler import compile_tn_terms

_IMAG_TOL = 1e-6


def simulate_tn(
    system,
    initial_state,
    *,
    backend: str,
    t_eval=None,
    observables=None,
    backend_options=None,
) -> EvolutionResult:
    """Evolve a protocol-bound system with the MPS or PEPS engine."""
    t_gate = system.t_gate
    times_pub, obs = validate_evolution_request(t_gate, t_eval, observables)
    realization = getattr(system, "_realization", None)
    terms = compile_tn_terms(system, realization=realization)  # O09: rejects non-1r/01r
    out_times = np.array([terms.t_gate]) if times_pub is None else np.asarray(times_pub, dtype=float)

    key = str(backend).lower()
    if key == "mps":
        from ryd_gate.backends.tenpy_mps.backends import evolve_mps, validate_mps_options

        opts = validate_mps_options(backend_options)
        preflight_tn_observables(
            obs, n_sites=terms.n_sites, local_dim=terms.local_dim,
            backend="mps", max_term_sites=None,
        )
        out_times, complex_expect, reader = evolve_mps(terms, initial_state, out_times, obs, opts)
        expectations = {name: _real_expectation(vals, name) for name, vals in complex_expect.items()}
        return EvolutionResult(times=out_times, expectations=expectations, reader=reader)
    if key == "peps":
        return _simulate_peps(system, initial_state, terms, out_times, obs, backend_options)
    if key == "graph_peps":
        return _simulate_graph_peps(system, initial_state, terms, out_times, obs, backend_options)
    raise ValueError(  # pragma: no cover
        f"unknown TN backend {backend!r}; use 'mps', 'peps' or 'graph_peps'."
    )


def _simulate_peps(system, initial_state, terms, out_times, obs, backend_options) -> EvolutionResult:
    """Real-time PEPS: dependency-free preflight (PEPS §5.3), then lazy YASTN engine."""
    from ryd_gate.backends.peps import (
        evolve_peps,
        peps_lattice_spec,
        validate_and_map_pairs,
        validate_real_time_options,
    )
    from ryd_gate.backends.tn_common.initial_state import initial_local_amplitudes

    opts = validate_real_time_options(backend_options)          # (2) exact ten-key options
    lattice_spec = peps_lattice_spec(system.register)           # (3) provenance/layout
    amps = initial_local_amplitudes(terms, initial_state)       # (5) validated (N, d) amplitudes
    max_sites = 1 if opts.measurement_method == "belief_propagation" else 2
    preflight_tn_observables(                                    # (6) observable capability
        obs, n_sites=terms.n_sites, local_dim=terms.local_dim,
        backend="peps", max_term_sites=max_sites,
    )
    pair_bonds = validate_and_map_pairs(lattice_spec, terms)     # (7) compiled pair/topology
    out_times, expectations, reader = evolve_peps(              # (8/9) lazy YASTN import + run
        terms, lattice_spec, pair_bonds, amps, out_times, obs, opts,
    )
    # PEPS returns already validity-checked real expectation arrays (PEPS §9.3): do not
    # re-apply the shared MPS 1e-6 converter.
    return EvolutionResult(times=out_times, expectations=expectations, reader=reader)


def _simulate_graph_peps(system, initial_state, terms, out_times, obs, backend_options) -> EvolutionResult:
    """Real-time graph-PEPS: dependency-free preflight, then lazy quimb engine."""
    from ryd_gate.backends.graph_tn import build_graph, evolve_graph_tn, validate_real_time_options
    from ryd_gate.backends.tn_common.initial_state import initial_local_amplitudes

    opts = validate_real_time_options(backend_options)          # exact six-key options
    amps = initial_local_amplitudes(terms, initial_state)       # validated (N, d) amplitudes
    preflight_tn_observables(                                    # shape check (any arity supported)
        obs, n_sites=terms.n_sites, local_dim=terms.local_dim,
        backend="graph_peps", max_term_sites=None,
    )
    graph = build_graph(terms)                                  # arbitrary interaction graph
    out_times, complex_expect, reader = evolve_graph_tn(        # lazy quimb import + run
        terms, graph, amps, out_times, obs, opts,
    )
    expectations = {name: _real_expectation(vals, name) for name, vals in complex_expect.items()}
    return EvolutionResult(times=out_times, expectations=expectations, reader=reader)


def _real_expectation(vals, name: str) -> np.ndarray:
    """Real ``float64`` expectation with an imaginary-part tolerance check (O06)."""
    v = np.asarray(vals, dtype=complex).reshape(-1)
    scale = max(1.0, float(np.max(np.abs(v)))) if v.size else 1.0
    if np.any(np.abs(v.imag) > _IMAG_TOL * scale):
        raise ValueError(
            f"observable {name!r} produced a non-real expectation (max imag "
            f"{float(np.max(np.abs(v.imag))):.2e}); observables must be Hermitian."
        )
    return v.real.astype(float)
