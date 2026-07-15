"""BP/CTM environments: fixed construction, report-only summaries, measurement, energy.

No module-scope YASTN import (handles arrive from the engine). Environments are
never gated on ``out.converged`` (PEPS02): a finite high residual and cap
exhaustion are reportable estimates, not failures. Only mathematically-meaningless
summaries (a wrong sweep count, a post-first-sweep nonfinite/negative CTM residual,
an out-of-range ``max_D``) or non-real Hermitian expectations fail.
"""

from __future__ import annotations

from dataclasses import dataclass

from ryd_gate.backends.peps._numerics import PEPSError, finite_nonneg, real_scalar


@dataclass(frozen=True, slots=True)
class _EnvSummary:
    """Report-only environment summary: ``residual`` may be ``None`` (PEPS-structural)."""

    residual: float | None
    iterations: int


# ── fixed construction (PEPS §9.1) ───────────────────────────────────────────


def build_bp_environment(h, psi, opts):
    env = h.fpeps.EnvBP(psi)
    out = env.iterate_(max_sweeps=opts.environment_max_iterations, diff_tol=opts.environment_tolerance)
    iterations = _sweeps(out, opts.environment_max_iterations, "belief-propagation")
    residual = finite_nonneg(out.max_diff, "belief-propagation residual (max_diff)")
    return env, _EnvSummary(residual=residual, iterations=iterations)


def build_ctm_environment(h, psi, opts):
    env = h.fpeps.EnvCTM(psi, init="eye")
    out = env.iterate_(
        {"D_total": opts.environment_bond_dimension, "tol": opts.environment_tolerance},
        moves="hv",
        method="2x2 corner",
        max_sweeps=opts.environment_max_iterations,
        corner_tol=opts.environment_tolerance,
    )
    iterations = _sweeps(out, opts.environment_max_iterations, "CTM")
    max_d = out.max_D
    if isinstance(max_d, bool) or not isinstance(max_d, int) or not (1 <= max_d <= opts.environment_bond_dimension):
        raise PEPSError(
            f"CTM max_D={max_d!r} is not a positive integer within environment_bond_dimension "
            f"({opts.environment_bond_dimension})."
        )
    residual = _ctm_residual(out.max_dsv, iterations)
    return env, _EnvSummary(residual=residual, iterations=iterations)


def measurement_environment(h, psi, opts):
    """Build the requested BP or CTM measurement environment (PEPS §9.1)."""
    if opts.measurement_method == "belief_propagation":
        return build_bp_environment(h, psi, opts)
    return build_ctm_environment(h, psi, opts)


def _sweeps(out, cap: int, what: str) -> int:
    sweeps = out.sweeps
    if isinstance(sweeps, bool) or not isinstance(sweeps, int) or not (1 <= sweeps <= cap):
        raise PEPSError(f"{what} returned an invalid sweep count {sweeps!r} (cap {cap}).")
    return sweeps


def _ctm_residual(max_dsv, iterations: int) -> float | None:
    """CTM residual: one structural first-sweep NaN → None; later nonfinite/negative fails."""
    v = float(max_dsv)
    if v != v:  # NaN
        if iterations == 1:
            return None
        raise PEPSError("CTM residual (max_dsv) is NaN after two or more sweeps.")
    return finite_nonneg(v, "CTM residual (max_dsv)")


# ── observable lowering + measurement (PEPS §9.3) ────────────────────────────


def lower_observables(exprs, ops, site_to_coord, method: str):
    """Lower each ObservableExpr into identity/single/pair contributions, validating site count.

    BP terms may act on at most one distinct site; CTM at most two. Rejection happens
    here (before evolution), never as a silent BP→CTM fallback.
    """
    max_sites = 1 if method == "belief_propagation" else 2
    cache: dict[bytes, object] = {}

    def wrap(matrix):
        import numpy as np

        arr = np.asarray(matrix, dtype=complex)
        key = arr.tobytes()
        t = cache.get(key)
        if t is None:
            t = ops.matrix(arr)
            cache[key] = t
        return t

    lowered: dict[str, tuple] = {}
    for label, expr in exprs.items():
        identity_coeff = 0.0 + 0.0j
        singles: list[tuple] = []
        pairs: list[tuple] = []
        for term in expr._terms:
            factors = term.factors
            sites = {int(site) for site, _ in factors}
            if len(sites) > max_sites:
                raise PEPSError(
                    f"observable {label!r} has a term on {len(sites)} distinct sites; "
                    f"measurement_method={method!r} supports at most {max_sites}. "
                    "Use measurement_method='ctm' for two-site correlators."
                )
            if not factors:
                identity_coeff += complex(term.coefficient)
            elif len(factors) == 1:
                site, matrix = factors[0]
                singles.append((complex(term.coefficient), wrap(matrix), site_to_coord[int(site)]))
            else:
                (si, mi), (sj, mj) = factors[0], factors[1]
                pairs.append(
                    (complex(term.coefficient), wrap(mi), site_to_coord[int(si)], wrap(mj), site_to_coord[int(sj)])
                )
        lowered[label] = (identity_coeff, singles, pairs)
    return lowered


def measure(env, lowered) -> dict[str, float]:
    """Measure each lowered observable, converting the complete Hermitian sum to a real scalar."""
    single_cache: dict[int, dict] = {}
    out: dict[str, float] = {}
    for label, (identity_coeff, singles, pairs) in lowered.items():
        value = identity_coeff
        for coeff, tensor, coord in singles:
            key = id(tensor)
            if key not in single_cache:
                single_cache[key] = env.measure_1site(tensor)
            value += coeff * complex(single_cache[key][coord])
        for coeff, tensor_i, coord_i, tensor_j, coord_j in pairs:
            value += coeff * complex(env.measure_nsite(tensor_i, tensor_j, sites=[coord_i, coord_j]))
        out[label] = real_scalar(value, f"observable {label!r}")
    return out


def ground_energy(env, ops, site_to_coord, h_local, pair_bonds) -> complex:
    """Complex <H> of the frozen unscaled Hamiltonian via one CTM (caller applies validity)."""
    total = 0.0 + 0.0j
    for i, coord in enumerate(site_to_coord):
        total += complex(env.measure_1site(ops.matrix(h_local[i]))[coord])
    nR = ops.rydberg()
    for coord_i, coord_j, V in pair_bonds:
        total += V * complex(env.measure_nsite(nR, nR, sites=[coord_i, coord_j]))
    return total
