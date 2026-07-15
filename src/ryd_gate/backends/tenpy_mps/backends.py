"""MPS TDVP time evolution (E22/E23) driven by :class:`TNTerms`.

Two-site TDVP with a piecewise-constant Hamiltonian rebuilt at each step's
midpoint.  The requested ``t_eval`` anchors and ``system.t_gate`` are true step
boundaries (E06).  The cumulative discarded Schmidt weight is checked against
``discarded_weight_tolerance`` and raises if exceeded (E23); Krylov tolerance,
per-step SVD cutoff and canonicalization are private implementation.
"""

from __future__ import annotations

import numpy as np

from ryd_gate.backends.tn_common.compiler import plan_segments

from .model import build_mps_model, build_tn_site
from .observables import lower_observables
from .state import MPSReader, build_initial_mps

_MPS_OPTION_KEYS = ("time_step_s", "bond_dimension", "discarded_weight_tolerance")
# Private per-singular-value cutoff; the public knob is the cumulative
# discarded-weight tolerance plus the bond-dimension hard cap (E23).
_SVD_MIN = 1e-14


def validate_mps_options(backend_options) -> dict:
    """Validate the exactly-three mandatory MPS TDVP options (E23)."""
    if not isinstance(backend_options, dict) or not backend_options:
        raise ValueError(
            "backend='mps' requires backend_options with exactly the keys "
            f"{list(_MPS_OPTION_KEYS)} (E23)."
        )
    keys = set(backend_options)
    missing = set(_MPS_OPTION_KEYS) - keys
    unknown = keys - set(_MPS_OPTION_KEYS)
    if missing or unknown:
        raise ValueError(
            f"backend='mps' backend_options must be exactly {list(_MPS_OPTION_KEYS)} "
            f"(E23); missing={sorted(missing)}, unknown={sorted(unknown)}."
        )
    time_step_s = float(backend_options["time_step_s"])
    if not np.isfinite(time_step_s) or time_step_s <= 0.0:
        raise ValueError(f"time_step_s must be finite and strictly positive; got {time_step_s!r}.")
    bond = backend_options["bond_dimension"]
    if isinstance(bond, bool) or not isinstance(bond, (int, np.integer)) or bond < 1:
        raise ValueError(f"bond_dimension must be a positive integer; got {bond!r}.")
    dwt = float(backend_options["discarded_weight_tolerance"])
    if not np.isfinite(dwt) or dwt <= 0.0 or dwt >= 1.0:
        raise ValueError(
            f"discarded_weight_tolerance must be finite, > 0 and < 1; got {dwt!r}."
        )
    return {"time_step_s": time_step_s, "bond_dimension": int(bond), "discarded_weight_tolerance": dwt}


def evolve_mps(terms, initial_state, out_times, obs_exprs, opts):
    """Run TDVP; return ``(out_times, complex_expectations, reader)``."""
    from tenpy.algorithms.tdvp import TwoSiteTDVPEngine
    from tenpy.tools import optimization

    site = build_tn_site(terms)
    psi = build_initial_mps(terms, initial_state, site)
    plan = lower_observables(obs_exprs, terms) if obs_exprs else None
    records: dict[str, list[complex]] = {label: [] for label in obs_exprs}

    def record() -> None:
        if plan is None:
            return
        for label, value in plan.measure(psi).items():
            records[label].append(value)

    record_at_start, segments = plan_segments(terms.t_gate, out_times, opts["time_step_s"])
    tdvp_params = {"trunc_params": {"chi_max": opts["bond_dimension"], "svd_min": _SVD_MIN}}
    tolerance = opts["discarded_weight_tolerance"]

    if record_at_start:
        record()
    cumulative_discarded = 0.0
    # The tensors are trivially charged; skipping TeNPy's per-op sanity checks
    # (optimization level 3) removes the dominant Python overhead on these small,
    # dense blocks without affecting the numerics.
    with optimization.temporary_level(3):
        for segment in segments:
            for k in range(segment.n_sub):
                t_mid = segment.t0 + (k + 0.5) * segment.dt_sub
                model = build_mps_model(terms, site, t_mid)
                eng = TwoSiteTDVPEngine(psi, model, tdvp_params)
                eng.evolve(1, segment.dt_sub)
                # trunc_err_list holds the per-bond discarded Schmidt weight of the
                # last evolve (the accumulated ``trunc_err`` stays empty on this path).
                cumulative_discarded += float(sum(eng.trunc_err_list))
                if cumulative_discarded > tolerance:
                    raise RuntimeError(
                        "MPS TDVP exceeded the cumulative discarded-weight tolerance "
                        f"({cumulative_discarded:.3e} > {tolerance:.3e}); reduce time_step_s "
                        "or increase bond_dimension (E23)."
                    )
            if segment.record:
                record()

    expectations = {label: np.asarray(values, dtype=complex) for label, values in records.items()}
    reader = MPSReader(psi, terms, site)
    return out_times, expectations, reader
