"""quimb loading, product state, simple-update evolution, measurement, orchestration.

This is the only module that imports ``quimb`` — and only lazily, inside
``_load_quimb`` (no module-scope import), so the dependency-free preflight in the
dispatcher runs and every geometry/option/topology error surfaces even without
quimb installed.

Real time steps a time-dependent Hamiltonian manually with a symmetric second-order
Trotter simple-update sweep (quimb forbids ``TEBDGen`` for real time); imaginary
time freezes the drive at ``at`` and runs ``SimpleUpdateGen`` over the annealing
schedule. Both act on the arbitrary interaction graph and contract observables via
exact / cluster / belief-propagation environments.
"""

from __future__ import annotations

import importlib.util

import numpy as np

from ryd_gate.backends.graph_tn._options import GraphTNError
from ryd_gate.backends.graph_tn._readout import _GroundStateReader, _RealTimeReader
from ryd_gate.backends.tn_common.compiler import plan_segments


def _load_quimb(device: str):
    """Import ``quimb.tensor`` lazily; reject GPU (no clean quimb-on-GPU path here)."""
    if device == "cuda":
        raise GraphTNError(
            "backend='graph_peps' does not support device='cuda'; the graph simple-update "
            "runs on the quimb NumPy backend. Use device='cpu' (or the YASTN backend='peps' "
            "for CUDA)."
        )
    if importlib.util.find_spec("quimb") is None:
        raise GraphTNError("backend='graph_peps' requires quimb (pip install quimb).")
    import quimb.tensor as qtn

    return qtn


def _nn_operator(terms) -> np.ndarray:
    """Rydberg pair operator ``kron(n_R, n_R)`` as a flat ``(d*d, d*d)`` matrix."""
    pr = np.asarray(terms.rydberg_projector(), dtype=complex)
    return np.kron(pr, pr)


def _product_psi(qtn, graph, terms, amps):
    """Bond-dim-1 graph tensor network overwritten with per-site amplitude vectors."""
    d = terms.local_dim
    psi = qtn.TN_from_edges_rand(list(graph.edges), D=1, phys_dim=d, seed=0, dtype="complex128")
    for i in range(graph.n_sites):
        t = psi[psi.site_tag(i)]
        data = np.zeros(t.shape, dtype=complex)
        ax = t.inds.index(psi.site_ind(i))
        sl = [0] * t.ndim
        sl[ax] = slice(None)
        data[tuple(sl)] = np.asarray(amps[i], dtype=complex)
        t.modify(data=data)
    return psi


def _local_ham(terms, t: float) -> np.ndarray:
    h = np.asarray(terms.local_hamiltonians(float(t)), dtype=complex)
    if not np.all(np.isfinite(h)):
        raise GraphTNError(f"local Hamiltonian at t={t} has a non-finite matrix element.")
    return h


def _build_ham(qtn, graph, h_local, nn):
    """LocalHamGen for the graph: per-site ``h_local[i]`` merged into ``V_ij * kron(n_R,n_R)``."""
    h2 = {e: graph.couplings[e] * nn for e in graph.edges}
    h1 = {i: h_local[i] for i in range(graph.n_sites)}
    return qtn.LocalHamGen(H2=h2, H1=h1)


# ── observable measurement ───────────────────────────────────────────────────


def _measure(psi, obs_exprs, terms, method: str, max_distance: int) -> dict[str, complex]:
    """Complex expectation of each observable expression on ``psi``."""
    d = terms.local_dim
    out: dict[str, complex] = {}
    for label, expr in obs_exprs.items():
        const = 0.0 + 0.0j
        acc: dict[tuple[int, ...], np.ndarray] = {}
        for term in expr._terms:
            factors = term.factors
            if not factors:
                const += complex(term.coefficient)
                continue
            sites = tuple(int(s) for s, _ in factors)
            op = np.asarray(factors[0][1], dtype=complex)
            for _, m in factors[1:]:
                op = np.kron(op, np.asarray(m, dtype=complex))
            op = complex(term.coefficient) * op
            acc[sites] = acc[sites] + op if sites in acc else op
        quimb_terms = {
            key: (op.reshape((d,) * (2 * len(key))) if len(key) > 1 else op)
            for key, op in acc.items()
        }
        out[label] = const + _contract_terms(psi, quimb_terms, method, max_distance)
    return out


def _contract_terms(psi, quimb_terms, method: str, max_distance: int) -> complex:
    if not quimb_terms:
        return 0.0 + 0.0j
    kw = dict(normalized=True, return_all=True)
    if method == "exact":
        res = psi.compute_local_expectation_exact(quimb_terms, **kw)
    elif method == "cluster":
        res = psi.compute_local_expectation_cluster(quimb_terms, max_distance=max_distance, **kw)
    elif method == "bp":
        res = psi.compute_local_expectation_simple(quimb_terms, max_distance=max_distance, **kw)
    else:  # pragma: no cover - options validation forbids this
        raise GraphTNError(f"unknown measurement_method {method!r}.")
    total = 0.0 + 0.0j
    for val in res.values():
        v = val[0] if isinstance(val, tuple) else val
        total += complex(np.asarray(v).reshape(()))
    return total


# ── real-time evolution ──────────────────────────────────────────────────────


def evolve_graph_tn(terms, graph, amps, out_times, obs_exprs, options):
    """Time-dependent graph-PEPS real-time evolution; return ``(out_times, expect, reader)``."""
    qtn = _load_quimb(options.device)
    nn = _nn_operator(terms)
    psi = _product_psi(qtn, graph, terms, amps)
    gauges: dict = {}
    psi.gauge_all_simple_(max_iterations=1, gauges=gauges)

    records: dict[str, list[complex]] = {label: [] for label in obs_exprs}

    def measure_now():
        if not obs_exprs:
            return
        snap = psi.copy()
        snap.gauge_simple_insert(gauges)
        for label, value in _measure(
            snap, obs_exprs, terms, options.measurement_method, options.cluster_max_distance
        ).items():
            records[label].append(value)

    record_at_start, segments = plan_segments(terms.t_gate, out_times, options.time_step_s)
    if record_at_start:
        measure_now()
    for segment in segments:
        for k in range(segment.n_sub):
            t_mid = segment.t0 + (k + 0.5) * segment.dt_sub
            ham = _build_ham(qtn, graph, _local_ham(terms, t_mid), nn)
            ordering = ham.get_auto_ordering("sort")
            sweep = list(ordering) + list(reversed(ordering))  # symmetric Strang step
            x = -1j * segment.dt_sub / 2.0
            for where in sweep:
                gate = ham.get_gate_expm(where, x)
                psi.gate_simple_(
                    gate, where, gauges=gauges,
                    max_bond=options.bond_dimension, cutoff=options.svd_cutoff,
                )
            psi.normalize_simple(gauges)
        if segment.record:
            measure_now()

    final = psi.copy()
    final.gauge_simple_insert(gauges)
    reader = _RealTimeReader(final, terms)
    expectations = {label: np.asarray(vals, dtype=complex) for label, vals in records.items()}
    return out_times, expectations, reader


# ── imaginary-time ground state ──────────────────────────────────────────────


def solve_graph_tn_ground(terms, graph, at, amps, obs_exprs, options):
    """Imaginary-time graph-PEPS ground state; return ``(expectations, reader)``."""
    qtn = _load_quimb(options.device)
    nn = _nn_operator(terms)
    h_local = _local_ham(terms, float(at))
    ham = _build_ham(qtn, graph, h_local, nn)

    psi0 = _product_psi(qtn, graph, terms, amps)
    su = qtn.SimpleUpdateGen(
        psi0, ham, D=options.bond_dimension, imag=True,
        compute_energy_final=False, progbar=False, gate_opts={"cutoff": options.svd_cutoff},
    )
    for tau, steps in options.imaginary_time_schedule:
        su.tau = float(tau)
        su.evolve(int(steps), progbar=False)
    psi = su.get_state()

    energy = _ground_energy(psi, terms, graph, h_local, nn, options)
    expectations: dict[str, float] = {"energy": energy}
    if obs_exprs:
        measured = _measure(
            psi, obs_exprs, terms, options.measurement_method, options.cluster_max_distance
        )
        for label, value in measured.items():
            expectations[label] = _real_scalar(value, label)

    reader = _GroundStateReader(psi, terms)
    return expectations, reader


def _ground_energy(psi, terms, graph, h_local, nn, options) -> float:
    """``<psi|H|psi>`` (rad/s) from the frozen local + pair terms via the chosen method."""
    d = terms.local_dim
    quimb_terms: dict[tuple[int, ...], np.ndarray] = {(i,): h_local[i] for i in range(graph.n_sites)}
    for e in graph.edges:
        quimb_terms[e] = (graph.couplings[e] * nn).reshape(d, d, d, d)
    value = _contract_terms(psi, quimb_terms, options.measurement_method, options.cluster_max_distance)
    return _real_scalar(value, "energy")


def _real_scalar(value, name: str) -> float:
    v = complex(value)
    scale = max(1.0, abs(v))
    if abs(v.imag) > 1e-6 * scale:
        raise GraphTNError(
            f"observable {name!r} produced a non-real expectation (imag {v.imag:.2e}); "
            "observables must be Hermitian."
        )
    return float(v.real)
