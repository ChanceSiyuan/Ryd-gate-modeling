"""Shared solver machinery: quintic envelope, block-max DOP853, batched kernel.

The block-max DOP853 subclass reproduces the installed SciPy DOP853 error estimate
independently per (point, logical input) block so tolerance is enforced per block.
``integrate_batch`` owns the generic original-frame propagation — segmented
restarts, per-column global-phase shifts, atom-swap reconstruction and t_eval
trajectory sampling — and takes an injected ``rhs_factory`` so the two-drive+Stark
(vs single-drive) Hamiltonian difference lives only in each script's factory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import scipy.integrate
from scipy.integrate._ivp.rk import DOP853 as _ScipyDOP853

# The two-atom CZ logical inputs, in the canonical basis-index order; "10" is the
# atom-swap image of "01".  Shared by both sweep scripts.
LOGICAL_INPUTS = ("00", "01", "10", "11")


# ── Analytic pulse: quintic smoothstep envelope and its integral ─────────────


def quintic(u):
    """Quintic smoothstep q(u) = 10 u^3 - 15 u^4 + 6 u^5 on [0, 1] (clipped)."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def quintic_antideriv(u):
    """Q(u) = integral_0^u q(v) dv = 2.5 u^4 - 3 u^5 + u^6 on [0, 1] (clipped)."""
    u = np.clip(u, 0.0, 1.0)
    return u ** 4 * (2.5 + u * (-3.0 + u))


def envelope(s, ramp: float = 0.15):
    """Power envelope E(s) on s in [0, 1]; scalar or ndarray."""
    s = np.clip(s, 0.0, 1.0)
    return np.where(
        s < ramp,
        quintic(s / ramp),
        np.where(s > 1.0 - ramp, quintic((1.0 - s) / ramp), 1.0),
    )


def envelope_integral(s, ramp: float = 0.15):
    """J(s) = integral_0^s E(v) dv, closed form; scalar or ndarray."""
    s = np.clip(s, 0.0, 1.0)
    rise = ramp * quintic_antideriv(s / ramp)
    mid = s - 0.5 * ramp
    fall = 1.0 - ramp - ramp * quintic_antideriv((1.0 - s) / ramp)
    return np.where(s < ramp, rise, np.where(s > 1.0 - ramp, fall, mid))


# ── Block-max DOP853 solver ──────────────────────────────────────────────────
#
# scipy's DOP853 controls a single RMS error norm over the whole flattened vector,
# which would let one inaccurate point/state hide inside a large batch.  The
# subclass below reproduces the installed SciPy estimate *independently* for every
# 49-component (point, logical input) block and returns the maximum, so tolerance
# is enforced per block.  The private seam (E3/E5 + _estimate_error_norm) is pinned
# by ``verify_scipy_error_norm`` at startup and by unit tests.


class BlockMaxDOP853(_ScipyDOP853):
    """DOP853 whose error norm is the max of per-block SciPy DOP853 norms."""

    block_size: int = 49  # overridden per solve via a dynamic subclass

    def _estimate_error_norm(self, K, h, scale):
        err5 = np.dot(K.T, self.E5) / scale
        err3 = np.dot(K.T, self.E3) / scale
        e5 = err5.reshape(-1, self.block_size)
        e3 = err3.reshape(-1, self.block_size)
        n5 = np.einsum("ij,ij->i", e5.real, e5.real) + np.einsum("ij,ij->i", e5.imag, e5.imag)
        n3 = np.einsum("ij,ij->i", e3.real, e3.real) + np.einsum("ij,ij->i", e3.imag, e3.imag)
        denom = n5 + 0.01 * n3
        out = np.zeros_like(n5)
        mask = denom > 0.0
        out[mask] = np.abs(h) * n5[mask] / np.sqrt(denom[mask] * self.block_size)
        return float(out.max())


def make_block_solver_class(block_size: int) -> type:
    """A BlockMaxDOP853 subclass bound to ``block_size`` (usable with solve_ivp)."""
    return type(f"BlockMaxDOP853_{block_size}", (BlockMaxDOP853,), {"block_size": block_size})


def verify_scipy_error_norm(n_blocks: int = 3, block: int = 49, seed: int = 12345) -> float:
    """Worst *relative* deviation of the custom norm from installed SciPy, random inputs.

    Verifies (1) the single-block custom norm reproduces the installed SciPy DOP853
    ``_estimate_error_norm`` and (2) the multi-block norm equals the max of the
    per-block SciPy norms.  Pins the private E3/E5/_estimate_error_norm seam.
    """
    rng = np.random.default_rng(seed)
    n_stages = _ScipyDOP853.n_stages

    class _Probe:
        E3 = _ScipyDOP853.E3
        E5 = _ScipyDOP853.E5

    def _custom(n: int, K, h, scale) -> float:
        solver = make_block_solver_class(block)(
            lambda t, y: 0 * y, 0.0, np.zeros(n, complex), 1.0)
        return BlockMaxDOP853._estimate_error_norm(solver, K, h, scale)

    def _rel(a: float, b: float) -> float:
        return abs(a - b) / max(abs(a), abs(b), 1e-300)

    worst = 0.0
    for _ in range(20):
        h = float(rng.uniform(1e-12, 1e-6))
        k1 = (rng.standard_normal((n_stages + 1, block))
              + 1j * rng.standard_normal((n_stages + 1, block)))
        scale1 = rng.uniform(1e-12, 1e-6, size=block)
        ref = _ScipyDOP853._estimate_error_norm(_Probe(), k1, h, scale1)
        worst = max(worst, _rel(ref, _custom(block, k1, h, scale1)))

        km = (rng.standard_normal((n_stages + 1, n_blocks * block))
              + 1j * rng.standard_normal((n_stages + 1, n_blocks * block)))
        scalem = rng.uniform(1e-12, 1e-6, size=n_blocks * block)
        per_block = max(
            _ScipyDOP853._estimate_error_norm(
                _Probe(), km[:, b * block:(b + 1) * block], h,
                scalem[b * block:(b + 1) * block])
            for b in range(n_blocks)
        )
        worst = max(worst, _rel(per_block, _custom(n_blocks * block, km, h, scalem)))
    return worst


# ── Batched original-frame integration kernel ────────────────────────────────


@dataclass
class BatchResult:
    """Terminal states and diagnostics for one integrated batch."""

    psi_final: np.ndarray        # (n_points, 4, dim) complex128, original frame
    leakage: np.ndarray          # (n_points, 4) direct nonlogical population
    max_leakage: np.ndarray      # (n_points,)
    worst_input: list[str]       # per point, the argmax logical input
    return_prob: np.ndarray      # (n_points, 4) |<s|psi_s>|^2
    norm_err: np.ndarray         # (n_points, 4) | ||psi||^2 - 1 |
    nfev: int
    used_swap: bool
    times: np.ndarray | None = None      # trajectory sample times (if requested)
    states: np.ndarray | None = None     # (n_times, n_points, 4, dim) (if requested)


def _integrate_segments(
    rhs: Callable,
    y0: np.ndarray,
    t_gate: float,
    ramp: float,
    rtol: float,
    atol: float,
    block_size: int,
    t_eval: np.ndarray | None,
):
    """Adaptive DOP853 over [0, rT], [rT, (1-r)T], [(1-r)T, T] with state carry-over.

    Returns ``(y_final, nfev, times, states)``; trajectory arrays are None unless
    ``t_eval`` is given (then states has shape (n_times, len(y0))).
    """
    cls = make_block_solver_class(block_size)
    breakpoints = [0.0, ramp * t_gate, (1.0 - ramp) * t_gate, t_gate]
    y = y0
    nfev = 0
    times_out: list[np.ndarray] = []
    states_out: list[np.ndarray] = []
    for t0, t1 in zip(breakpoints[:-1], breakpoints[1:]):
        if t_eval is None:
            solver = cls(rhs, t0, y, t1, rtol=rtol, atol=atol)
            while solver.status == "running":
                solver.step()
            if solver.status != "finished":
                raise RuntimeError(f"DOP853 failed on segment [{t0:g}, {t1:g}]")
            y = solver.y
            nfev += solver.nfev
        else:
            mask = (t_eval >= t0) & (t_eval < t1) if t1 < t_gate else \
                   (t_eval >= t0) & (t_eval <= t1)
            seg_eval = np.unique(np.concatenate([t_eval[mask], [t1]]))
            sol = scipy.integrate.solve_ivp(
                rhs, (t0, t1), y, method=cls, rtol=rtol, atol=atol, t_eval=seg_eval)
            if not sol.success:
                raise RuntimeError(f"DOP853 failed on segment [{t0:g}, {t1:g}]: {sol.message}")
            keep = np.isin(sol.t, t_eval[mask])
            times_out.append(sol.t[keep])
            states_out.append(sol.y.T[keep])
            y = sol.y[:, -1]
            nfev += sol.nfev
    if t_eval is None:
        return y, nfev, None, None
    times = np.concatenate(times_out) if times_out else np.empty(0)
    states = np.concatenate(states_out, axis=0) if states_out else None
    return y, nfev, times, states


def integrate_batch(
    ops,
    t_gate: float,
    point_params: dict[str, np.ndarray],
    state_labels,
    *,
    rhs_factory: Callable,
    dim: int,
    rtol: float,
    atol: float,
    ramp: float = 0.15,
    use_shifts: bool = True,
    segmented: bool = True,
    t_eval: np.ndarray | None = None,
    initial_indices: Sequence[int] | None = None,
    reverse_time: bool = False,
) -> BatchResult:
    """Propagate all logical inputs of the batch's panel points together.

    ``point_params`` maps names to equal-length 1-D per-point arrays; they are
    expanded (point-major) to per-column arrays and handed, with the per-column
    global-phase ``shift``, to ``rhs_factory(ops, cols, t_gate, ramp)`` which
    returns the flattened complex128 RHS.  Columns are the logical inputs of
    ``state_labels`` — when it omits "10" that column is reconstructed by the
    atom-swap permutation.  Each column is solved with its bare logical diagonal
    energy subtracted (H - c_s I) and the exact global phase exp(-i c_s T) restored.

    ``initial_indices`` overrides the initial basis state of each column (one index
    per state label), bypassing the atom-swap reconstruction: the returned columns
    are then exactly ``state_labels``, and the leakage/return_prob/worst_input
    diagnostics are taken over those columns.  ``reverse_time`` marks a leg
    integrated in tau = T - t, whose solved variable is chi(tau) = exp(-i c tau)
    psi(T - tau) — the same shift subtraction, but the global phase is restored
    with the conjugate factor.  Both exist for the backward adjoint leg of the
    297 filter-function pass, which propagates the nonlogical basis states from T
    back to 0; leaving the shift in place is what keeps that leg as cheap as the
    forward one (its dominant component is the static one).
    """
    point_params = {k: np.asarray(v, dtype=float) for k, v in point_params.items()}
    shapes = {v.shape for v in point_params.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 1:
        raise ValueError("point_params arrays must be equal-length 1-D arrays")
    n_points = next(iter(shapes))[0]

    state_labels = tuple(state_labels)
    state_cols = {s: i for i, s in enumerate(state_labels)}
    n_states = len(state_labels)
    n_cols = n_points * n_states

    col_of_point = np.repeat(np.arange(n_points), n_states)
    cols = {name: arr[col_of_point] for name, arr in point_params.items()}

    logical_of_state = (
        {s: int(i) for s, i in zip(state_labels, initial_indices)}
        if initial_indices is not None
        else {s: ops.logical_indices[LOGICAL_INPUTS.index(s)] for s in state_labels}
    )
    col_logical_idx = np.asarray([logical_of_state[s] for s in state_labels] * n_points)
    shifts = ops.h_static_diag[col_logical_idx] if use_shifts else np.zeros(n_cols)
    cols["shift"] = shifts

    rhs = rhs_factory(ops, cols, t_gate, ramp)

    y0 = np.zeros((n_cols, dim), dtype=np.complex128)
    for p in range(n_points):
        for j, s in enumerate(state_labels):
            y0[p * n_states + j, logical_of_state[s]] = 1.0

    if segmented:
        y_fin, nfev, times, traj = _integrate_segments(
            rhs, y0.ravel(), t_gate, ramp, rtol, atol, dim, t_eval)
    else:
        cls = make_block_solver_class(dim)
        solver = cls(rhs, 0.0, y0.ravel(), t_gate, rtol=rtol, atol=atol)
        while solver.status == "running":
            solver.step()
        if solver.status != "finished":
            raise RuntimeError("DOP853 failed (unsegmented)")
        y_fin, nfev, times, traj = solver.y, solver.nfev, None, None

    phase = 1j if reverse_time else -1j

    def assemble(y_flat: np.ndarray, t_at: float) -> np.ndarray:
        """(n_cols*dim,) chi at ``t_at`` -> (n_points, 4 | n_states, dim) restored psi."""
        chi = y_flat.reshape(n_cols, dim) * np.exp(phase * shifts * t_at)[:, None]
        if initial_indices is not None:   # columns are the given states, as given
            return chi.reshape(n_points, n_states, dim)
        psi = np.empty((n_points, 4, dim), dtype=np.complex128)
        for p in range(n_points):
            for j, s in enumerate(LOGICAL_INPUTS):
                if s in state_cols:
                    psi[p, j] = chi[p * n_states + state_cols[s]]
                else:  # s == "10", reconstructed from 01 by the atom swap
                    psi[p, j] = chi[p * n_states + state_cols["01"]][ops.swap_perm]
        return psi

    psi_final = assemble(y_fin, t_gate)

    nonlogical = np.setdiff1d(np.arange(dim), ops.logical_indices)
    pops = np.abs(psi_final) ** 2
    leakage = pops[:, :, nonlogical].sum(axis=2)
    max_leakage = leakage.max(axis=1)
    out_labels = tuple(state_labels) if initial_indices is not None else LOGICAL_INPUTS
    out_indices = ([logical_of_state[s] for s in out_labels]
                   if initial_indices is not None else ops.logical_indices)
    worst_input = [out_labels[int(np.argmax(leakage[p]))] for p in range(n_points)]
    return_prob = np.stack(
        [pops[:, j, out_indices[j]] for j in range(len(out_labels))], axis=1)
    norm_err = np.abs(pops.sum(axis=2) - 1.0)

    states = None
    if traj is not None:
        states = np.stack(
            [assemble(traj[i], float(times[i])) for i in range(traj.shape[0])], axis=0)

    return BatchResult(
        psi_final=psi_final, leakage=leakage, max_leakage=max_leakage,
        worst_input=worst_input, return_prob=return_prob, norm_err=norm_err,
        nfev=int(nfev), used_swap=bool(tuple(state_labels) != LOGICAL_INPUTS),
        times=times, states=states,
    )
