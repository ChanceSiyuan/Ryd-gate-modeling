"""Many-body ground-state solver for a fully constructed ``1r`` system (E12-E21).

``system.ground_state(at, method, initial_state, observables, method_options)``
freezes the Hamiltonian at time ``at`` (via the canonical drive lowering) and
finds a ground-state representative with DMRG (E19) or PEPS imaginary-time (E20).
Both return the same :class:`~ryd_gate.results.GroundStateResult`: the reserved
``expectation("energy")`` plus the requested Hermitian observables, a lazy
``amplitude(labels, phase_reference=...)`` (the global phase is gauge, fixed by
the reference), and ``sample(shots, seed)``.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ryd_gate.backends.tn_common._expressions import preflight_tn_observables
from ryd_gate.backends.tn_common.compiler import compile_tn_terms
from ryd_gate.backends.tn_common.initial_state import validate_labels
from ryd_gate.results import GroundStateResult

_DMRG_KEYS = (
    "bond_dimension", "discarded_weight_tolerance", "relative_energy_tolerance",
    "entropy_tolerance", "max_sweeps",
)
_IMAG_TOL = 1e-6
_PHASE_REF_TOL = 1e-9


def solve_ground_state(system, *, at, method, initial_state, observables=None, method_options):
    """Ground state of a ``1r`` system frozen at time ``at`` (E12-E21)."""
    if system.level_structure.name != "1r":
        raise ValueError(
            f"ground_state() requires a '1r' system (E12); got "
            f"{system.level_structure.name!r}."
        )
    t_gate = system.t_gate
    at = float(at)
    if not np.isfinite(at) or at < 0.0 or at > t_gate * (1 + 1e-12):
        raise ValueError(f"`at` must be finite and within [0, t_gate={t_gate}]; got {at!r}.")
    if method not in ("dmrg", "peps_imaginary_time", "graph_peps_imaginary_time"):
        raise ValueError(
            f"method must be 'dmrg', 'peps_imaginary_time' or 'graph_peps_imaginary_time' "
            f"(E14); got {method!r}."
        )
    obs = _validate_ground_observables(observables)
    if initial_state is None or isinstance(initial_state, str):
        raise TypeError("ground_state initial_state must be an explicit level-label list (E17).")

    terms = compile_tn_terms(system)  # 1r is TN-capable
    labels = validate_labels(terms, initial_state)

    if method == "dmrg":
        preflight_tn_observables(obs, n_sites=terms.n_sites, local_dim=terms.local_dim,
                                 backend="mps", max_term_sites=None)
        return _solve_dmrg(terms, at, labels, obs, method_options)
    if method == "graph_peps_imaginary_time":
        return _solve_graph_peps_ground(terms, at, labels, obs, method_options)
    # PEPS imaginary time: dependency-free preflight (PEPS §5.3) then lazy YASTN engine.
    from ryd_gate.backends.peps import (
        peps_lattice_spec,
        solve_peps_ground_state,
        validate_and_map_pairs,
        validate_ground_options,
    )
    from ryd_gate.backends.tn_common.initial_state import initial_local_amplitudes

    opts = validate_ground_options(method_options)          # nine-key options
    lattice_spec = peps_lattice_spec(system.register)       # Register provenance/layout
    amps = initial_local_amplitudes(terms, labels)          # explicit product-state labels
    # private CTM measurement supports up to 2-site terms (E26).
    preflight_tn_observables(obs, n_sites=terms.n_sites, local_dim=terms.local_dim,
                             backend="peps", max_term_sites=2)
    pair_bonds = validate_and_map_pairs(lattice_spec, terms)  # compiled pair/topology
    expectations, reader = solve_peps_ground_state(
        terms, lattice_spec, pair_bonds, at, amps, obs, opts,
    )
    return GroundStateResult(expectations=expectations, reader=reader)


def _solve_graph_peps_ground(terms, at, labels, obs, method_options):
    """Arbitrary-geometry graph-PEPS imaginary-time ground state (lazy quimb engine)."""
    from ryd_gate.backends.graph_tn import build_graph, solve_graph_tn_ground, validate_ground_options
    from ryd_gate.backends.tn_common.initial_state import initial_local_amplitudes

    opts = validate_ground_options(method_options)              # exact six-key options
    amps = initial_local_amplitudes(terms, labels)             # explicit product-state labels
    preflight_tn_observables(obs, n_sites=terms.n_sites, local_dim=terms.local_dim,
                             backend="graph_peps", max_term_sites=None)
    graph = build_graph(terms)                                 # arbitrary interaction graph
    expectations, reader = solve_graph_tn_ground(terms, graph, at, amps, obs, opts)
    return GroundStateResult(expectations=expectations, reader=reader)


def _validate_ground_observables(observables) -> dict:
    from ryd_gate.core.observables import ObservableExpr, _is_hermitian

    if observables is None:
        return {}
    if not isinstance(observables, dict):
        raise TypeError(f"observables must be a dict or None; got {type(observables).__name__}.")
    if "energy" in observables:
        raise ValueError("'energy' is a reserved GroundStateResult observable name (E21).")
    for label, expr in observables.items():
        if not isinstance(expr, ObservableExpr):
            raise TypeError(f"observables[{label!r}] must be an ObservableExpr.")
        if not _is_hermitian(expr):
            raise ValueError(f"observables[{label!r}] is not Hermitian; expectations must be real.")
    return dict(observables)


def _validate_options(method_options, keys: tuple[str, ...], method: str) -> dict:
    if not isinstance(method_options, dict) or not method_options:
        raise ValueError(f"method={method!r} requires method_options with exactly {list(keys)}.")
    have = set(method_options)
    missing = set(keys) - have
    unknown = have - set(keys)
    if missing or unknown:
        raise ValueError(
            f"method={method!r} method_options must be exactly {list(keys)}; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}."
        )
    return dict(method_options)


def _pos_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return int(value)


def _pos_tol(value, name: str, *, below_one: bool = False) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0 or (below_one and v >= 1.0):
        raise ValueError(f"{name} must be finite, > 0{' and < 1' if below_one else ''}; got {value!r}.")
    return v


# ── DMRG (E19) ───────────────────────────────────────────────────────────────


def _solve_dmrg(terms, at, labels, obs, method_options):
    from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
    from tenpy.tools import optimization

    from ryd_gate.backends.tenpy_mps.model import build_mps_model, build_tn_site
    from ryd_gate.backends.tenpy_mps.observables import lower_observables
    from ryd_gate.backends.tenpy_mps.state import build_initial_mps

    o = _validate_options(method_options, _DMRG_KEYS, "dmrg")
    bond = _pos_int(o["bond_dimension"], "bond_dimension")
    max_sweeps = _pos_int(o["max_sweeps"], "max_sweeps")
    dwt = _pos_tol(o["discarded_weight_tolerance"], "discarded_weight_tolerance", below_one=True)
    rel_e_tol = _pos_tol(o["relative_energy_tolerance"], "relative_energy_tolerance")
    ent_tol = _pos_tol(o["entropy_tolerance"], "entropy_tolerance")

    site = build_tn_site(terms)
    psi = build_initial_mps(terms, labels, site)
    model = build_mps_model(terms, site, at)
    dmrg_params = {
        "trunc_params": {"chi_max": bond, "svd_min": 1e-14},
        "max_sweeps": max_sweeps, "min_sweeps": 2,
        "max_E_err": rel_e_tol, "max_S_err": ent_tol, "mixer": True,
    }
    from tenpy.tools.misc import TenpyInconsistencyError

    with optimization.temporary_level(3):
        eng = TwoSiteDMRGEngine(psi, model, dmrg_params)
        try:
            energy, psi = eng.run()
        except TenpyInconsistencyError as exc:
            # TeNPy's own truncation-consistency check also signals non-convergence.
            raise RuntimeError(
                f"DMRG did not meet the convergence criteria (E19): {exc}. "
                "Increase bond_dimension or max_sweeps."
            ) from exc

    stats = eng.sweep_stats
    rel_e = abs(stats["Delta_E"][-1]) / max(1.0, abs(stats["E"][-1]))
    d_s = abs(stats["Delta_S"][-1])
    trunc = float(stats["max_trunc_err"][-1])
    if not (rel_e <= rel_e_tol and d_s <= ent_tol and trunc <= dwt):
        raise RuntimeError(
            f"DMRG reached {eng.sweeps} sweeps without meeting all convergence "
            f"criteria (E19): relative_energy_change={rel_e:.2e} (<= {rel_e_tol:.0e}), "
            f"entropy_change={d_s:.2e} (<= {ent_tol:.0e}), discarded_weight={trunc:.2e} "
            f"(<= {dwt:.0e}); increase bond_dimension or max_sweeps."
        )

    plan = lower_observables(obs, terms) if obs else None
    with optimization.temporary_level(3):
        measured = plan.measure(psi) if plan is not None else {}
    expectations = {"energy": float(np.real(energy))}
    for label, value in measured.items():
        expectations[label] = _real_scalar(value, label)
    reader = _MPSGroundStateReader(psi, terms, site)
    return GroundStateResult(expectations=expectations, reader=reader)


def _real_scalar(value, name: str) -> float:
    v = complex(value)
    scale = max(1.0, abs(v))
    if abs(v.imag) > _IMAG_TOL * scale:
        raise ValueError(
            f"observable {name!r} produced a non-real expectation (imag {v.imag:.2e}); "
            "observables must be Hermitian."
        )
    return float(v.real)


class _MPSGroundStateReader:
    """Lazy amplitude/sample over a DMRG ground-state MPS (E15)."""

    __slots__ = ("_psi", "_terms", "_site")

    def __init__(self, psi, terms, site) -> None:
        self._psi = psi
        self._terms = terms
        self._site = site

    def amplitude(self, labels, phase_reference) -> complex:
        from ryd_gate.backends.tenpy_mps.state import build_initial_mps

        labels = validate_labels(self._terms, labels)
        reference = validate_labels(self._terms, phase_reference)
        ref_bra = build_initial_mps(self._terms, reference, self._site)
        ref_amp = complex(ref_bra.overlap(self._psi))
        if abs(ref_amp) < _PHASE_REF_TOL:
            raise ValueError(
                f"phase_reference amplitude is numerically zero ({abs(ref_amp):.2e}); "
                "choose a reference basis state with nonzero amplitude (E15)."
            )
        phase = ref_amp / abs(ref_amp)
        bra = build_initial_mps(self._terms, labels, self._site)
        return complex(bra.overlap(self._psi)) / phase

    def sample(self, shots: int, seed: int) -> Counter:
        rng = np.random.default_rng(seed)
        levels = self._terms.levels
        counts: Counter = Counter()
        for _ in range(int(shots)):
            sigmas, _amp = self._psi.sample_measurements(rng=rng)
            counts[tuple(levels[int(s)] for s in sigmas)] += 1
        return counts
