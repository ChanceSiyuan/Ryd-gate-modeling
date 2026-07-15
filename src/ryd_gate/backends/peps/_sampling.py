"""Validated sequential-conditional PEPS sampling (belief propagation + CTM).

No module-scope YASTN import: the YASTN/MPS/fpeps handles arrive at call time from
the engine (``h.yastn``/``h.mps``/``h.fpeps``), so the dependency-free preflight in
the dispatcher still runs on machines without YASTN.

The two single-sample conditional loops are ported from YASTN commit
``30b1d8bb4dc691a25bf6394b061c564128ede8e0`` (Apache-2.0): the belief-propagation
loop from ``envs/_env_bp.py::EnvBP.sample`` and the CTM boundary-MPS loops from
``envs/_env_window.py::_sample_one_columns`` / ``_sample_one_rows``. Only the
necessary single-sample bodies are reproduced. Unlike the pinned samplers, this
adapter (a) owns its RNG — one ``np.random.default_rng(seed)`` scalar per
conditional site, never the backend/global RNG — and (b) validates every complex
candidate weight before taking its real part (PEPS §10/§13), so a non-real or
materially-negative conditional weight raises instead of being silently ``.real``-ed.

Traversal orientation matches amplitude/norm (``_boundary._orientation``): columns
first (``dirn='v'``) when ``Nx <= Ny``, rows first (``dirn='h'``) otherwise, always
low-to-high. CTM sampling uses ``normalize=True`` boundary updates (the OPPOSITE of
amplitude/norm contraction) because only conditional probability ratios matter and
the global complex factor cancels. The CTM sampler requires a genuinely 2D grid
(both ``Nx >= 2`` and ``Ny >= 2``): on a 1-wide strip the transverse boundary MPS is
degenerate, so it raises rather than risk a stalled zipper/compression sweep; use
belief propagation for a chain. Ground state reuses the retained final CTM; it is
never mutated (the transfer MPO is rebuilt per layer from fresh tensors). Real-time
sampling builds the selected BP/CTM environment lazily from ``psi`` and never falls
back BP->CTM.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

from ryd_gate.backends.peps._boundary import _orientation
from ryd_gate.backends.peps._numerics import _VALIDITY_RTOL, PEPSError

# fpeps rank-5 site-tensor leg order: (top, left, bottom, right, phys[fused]).
_PHYS_AX = 4


@dataclass(slots=True)
class _BPLocal:
    """Local BP message record (copied ``tR/lR/bR/rR``); avoids importing EnvBP_local."""

    tR: object
    lR: object
    bR: object
    rR: object


def sample(h, ops, psi, terms, site_to_coord, controls, options, ground_env, shots, seed):
    """Draw ``shots`` validated conditional samples; return ``Counter[tuple[str, ...]]``."""
    nx_total, ny_total = int(psi.Nx), int(psi.Ny)
    dirn, _ = _orientation((nx_total, ny_total))
    levels = tuple(terms.levels)
    rng = np.random.default_rng(int(seed))
    method = getattr(options, "measurement_method", None)

    try:
        if ground_env is not None:
            draw = _make_ctm_sampler(h, ops, levels, controls, ground_env, nx_total, ny_total, dirn)
        elif method == "belief_propagation":
            env = h.fpeps.EnvBP(psi)
            env.iterate_(max_sweeps=controls.env_max_iterations, diff_tol=controls.env_tolerance)
            draw = _make_bp_sampler(h, ops, levels, env, nx_total, ny_total, dirn)
        elif method == "ctm":
            env = h.fpeps.EnvCTM(psi, init="eye")
            env.iterate_(
                {"D_total": controls.env_bond_dimension, "tol": controls.env_tolerance},
                moves="hv", method="2x2 corner",
                max_sweeps=controls.env_max_iterations, corner_tol=controls.env_tolerance,
            )
            draw = _make_ctm_sampler(h, ops, levels, controls, env, nx_total, ny_total, dirn)
        else:
            raise PEPSError(f"PEPS sampling received an unsupported measurement_method {method!r}.")

        counts: collections.Counter = collections.Counter()
        n_sites = int(terms.n_sites)
        for _ in range(int(shots)):
            outcome = draw(rng)
            counts[tuple(outcome[site_to_coord[i]] for i in range(n_sites))] += 1
    except h.yastn.YastnError as exc:  # pinned YASTN tensor error; do not broaden
        raise PEPSError(f"YASTN sampling contraction failed: {exc}") from exc
    return counts


# ── shared candidate-weight validation + selection (PEPS §10/§13 step 4) ──────


def _scalar(value) -> complex:
    """Bring a device scalar to a host Python ``complex`` (weights/RNG branching only)."""
    return complex(value.item() if hasattr(value, "item") else value)


def _select(weights, u: float) -> tuple[int, float]:
    """Validate complex candidate weights (``terms.levels`` order), return (index, p_selected)."""
    d = len(weights)
    w = np.empty(d, dtype=float)
    for i, z in enumerate(weights):
        zr, zi = float(z.real), float(z.imag)
        if not (np.isfinite(zr) and np.isfinite(zi)):
            raise PEPSError(f"sampling candidate weight {i} is not finite: {z!r}.")
        slack = _VALIDITY_RTOL * max(1.0, abs(z))  # per-candidate; never widened by a neighbour
        if abs(zi) > slack:
            raise PEPSError(
                f"sampling candidate weight {i} has imaginary part {zi:.3e} exceeding slack {slack:.3e}."
            )
        if zr < 0.0:
            if zr < -slack:
                raise PEPSError(
                    f"sampling candidate weight {i} is negative beyond slack ({zr:.3e} < -{slack:.3e})."
                )
            zr = 0.0
        w[i] = zr
    total = float(w.sum())
    if not (np.isfinite(total) and total > 0.0):
        raise PEPSError(f"sampling candidate weights do not sum to a finite positive total: {total!r}.")
    p = w / total
    p_sum = float(p.sum())
    if not (np.isfinite(p_sum) and p_sum > 0.0):
        raise PEPSError(f"sampling probabilities do not sum to a finite positive value: {p_sum!r}.")
    p = p / p_sum
    cdf = np.cumsum(p)
    if not np.all(np.isfinite(cdf)) or bool(np.any(np.diff(cdf) < 0.0)):
        raise PEPSError("sampling CDF is not finite and nondecreasing.")
    cdf[-1] = 1.0
    index = min(int(np.searchsorted(cdf, u, side="right")), d - 1)
    if not p[index] > 0.0:  # guards u == 0 selecting a leading zero-probability candidate
        raise PEPSError("selected sampling candidate has non-positive probability.")
    return index, float(p[index])


# ── belief-propagation adapter (port of EnvBP.sample; YASTN 30b1d8b, Apache-2.0) ──


def _match_ancilla(h, ket, op):
    """Tensor a rank-2 physical operator with the trivial ancilla identity of the fused ket leg.

    Local port of YASTN ``_gates_auxiliary.match_ancilla`` (rank-2 branch; Apache-2.0):
    the PEPS physical leg is product-fused with an ancilla, so the projector must be
    tensored with identity on that leg before it can match the fused leg.
    """
    leg = ket.get_legs(axes=-1)
    if not leg.is_fused():
        return op
    _, anc = leg.unfuse_leg()
    one = h.yastn.eye(config=ket.config, legs=[anc, anc.conj()], isdiag=False)
    gnew = h.yastn.tensordot(op, one, axes=((), ()))
    return gnew.fuse_legs(axes=((0, 2), (1, 3)))


def _normalized_message(h, R):
    """Normalize a downstream BP message; its QR ``R.norm()`` must be finite and positive."""
    rn = float(R.norm())
    if not (np.isfinite(rn) and rn > 0.0):
        raise PEPSError(f"belief-propagation message QR norm is not finite/positive: {rn!r}.")
    return R / rn


def _make_bp_sampler(h, ops, levels, env, nx_total, ny_total, dirn):
    yastn = h.yastn
    coords = [(nx, ny) for nx in range(nx_total) for ny in range(ny_total)]
    order = (
        [(nx, ny) for ny in range(ny_total) for nx in range(nx_total)] if dirn == "v"
        else [(nx, ny) for nx in range(nx_total) for ny in range(ny_total)]
    )
    kets = {coord: env.psi[coord].ket for coord in coords}
    matched = {coord: [_match_ancilla(h, ket, ops.projector(lvl)) for lvl in levels]
               for coord, ket in kets.items()}

    def draw(rng):
        # Independent local copies of the converged messages; only downstream tR/lR are overwritten.
        msgs = {coord: _BPLocal(env[coord].tR, env[coord].lR, env[coord].bR, env[coord].rR)
                for coord in coords}
        outcome = {}
        for coord in order:
            nx, ny = coord
            ket = kets[coord]
            m = msgs[coord]
            # ── ported single-sample body (EnvBP.sample; Apache-2.0) ──
            atlbr = yastn.ncon([ket, m.tR, m.lR, m.bR, m.rR],
                               [(1, 2, 3, 4, -4), (-0, 1), (-1, 2), (-2, 3), (-3, 4)])
            weights = []
            for proj in matched[coord]:
                atmp = yastn.tensordot(atlbr, proj, axes=(_PHYS_AX, 1))
                weights.append(_scalar(yastn.vdot(atlbr, atmp)))
            k, p_sel = _select(weights, rng.random())
            outcome[coord] = levels[k]
            ketp = yastn.tensordot(ket, matched[coord][k], axes=(_PHYS_AX, 1)) / p_sel
            if nx + 1 < nx_total:
                tmp = yastn.ncon([ketp, m.tR, m.lR, m.rR],
                                 [(1, 2, -2, 3, -4), (-0, 1), (-1, 2), (-3, 3)])
                _, r = tmp.qr(axes=((0, 1, 3, 4), 2), sQ=tmp.s[2])
                msgs[nx + 1, ny].tR = _normalized_message(h, r)
            if ny + 1 < ny_total:
                tmp = yastn.ncon([ketp, m.tR, m.lR, m.bR],
                                 [(1, 2, 3, -3, -4), (-0, 1), (-1, 2), (-2, 3)])
                _, r = tmp.qr(axes=((0, 1, 2, 4), 3), sQ=tmp.s[3])
                msgs[nx, ny + 1].lR = _normalized_message(h, r)
        return outcome

    return draw


# ── CTM adapter (port of _sample_one_columns/_sample_one_rows; YASTN 30b1d8b, Apache-2.0) ──


def _make_ctm_sampler(h, ops, levels, controls, env_ctm, nx_total, ny_total, dirn):
    if nx_total < 2 or ny_total < 2:
        # The CTM boundary-MPS sampler needs a genuinely 2D grid: on a 1-wide strip the
        # transverse boundary MPS is degenerate and the zipper/compression sweep can stall.
        raise PEPSError(
            f"CTM sampling requires a 2D grid (got Nx={nx_total}, Ny={ny_total}); for a chain use "
            "measurement_method='belief_propagation' (real time) or backend='mps'/'exact_ode' (1D)."
        )
    mps = h.mps
    env_win = h.fpeps.EnvWindow(env_ctm, (0, nx_total), (0, ny_total))
    offset = 1  # EnvWindow prepends one CTM boundary tensor
    opts_svd = {"D_total": controls.env_bond_dimension, "tol": controls.env_tolerance}
    opts_var = {"method": "1site", "overlap_tol": controls.env_tolerance,
                "max_sweeps": controls.env_max_iterations, "normalize": True}
    projs = [ops.projector(lvl) for lvl in levels]

    def _sample_layer(rng, menv, tm, base, inner_range, transverse, is_column):
        """Condition one boundary layer (a column when is_column else a row) low-to-high."""
        outcome = {}
        for pos, idx in enumerate(inner_range, start=offset):
            coord = (idx, transverse) if is_column else (transverse, idx)
            weights = []
            for proj in projs:
                tm[pos].set_operator_(proj)  # DoublePepsTensor set_operator_ matches the ancilla
                weights.append(_scalar(menv.measure(bd=(pos - 1, pos + 1))))
            k, p_sel = _select(weights, rng.random())
            outcome[coord] = levels[k]
            tm[pos].set_operator_(projs[k] / p_sel)
            if idx + 1 < base:
                menv.update_env_(pos, to="last")
        return outcome

    def draw(rng):
        outcome = {}
        if dirn == "v":  # columns first: Nx <= Ny
            vec = env_win[0, "l"]
            for ny in range(ny_total):
                vecc = env_win[ny, "r"].conj()
                tm = env_win[ny, "v"]
                menv = mps.Env(vecc, [tm, vec])
                menv.setup_(to=offset)
                outcome.update(_sample_layer(rng, menv, tm, nx_total, range(nx_total), ny, True))
                if ny + 1 < ny_total:
                    vec_new = mps.zipper(tm, vec, opts_svd=opts_svd, normalize=True)
                    mps.compression_(vec_new, (tm, vec), **opts_var)
                    vec = vec_new
        else:  # rows first: Nx > Ny
            vec = env_win[0, "t"]
            for nx in range(nx_total):
                vecc = env_win[nx, "b"].conj()
                tm = env_win[nx, "h"]
                menv = mps.Env(vecc, [tm, vec])
                menv.setup_(to=offset)
                outcome.update(_sample_layer(rng, menv, tm, ny_total, range(ny_total), nx, False))
                if nx + 1 < nx_total:
                    vec_new = mps.zipper(tm, vec, opts_svd=opts_svd, normalize=True)
                    mps.compression_(vec_new, (tm, vec), **opts_var)
                    vec = vec_new
        return outcome

    return draw
