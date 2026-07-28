"""Direct trajectory optimization over one bilinear control model (ADR-0026).

The engine solves the knot-point NLP of direct quantum optimal control: the
slice propagators GRAPE only rolls forward become explicit decision variables,
and the Schrodinger dynamics becomes per-slice defect constraints:

    minimize   terminal_objective(U_K) + quadratic control regularizers
    over       isovec(U_k) (k = 1..K) and, per channel, u / du / ddu knots
    subject to isovec(U_k) - isovec(expm(-i H_k dt) U_{k-1}) = 0   (defects)
               u_{k+1} = u_k + du_k dt,  du_{k+1} = du_k + ddu_k dt (chain)
               plain variable bounds on u, du, ddu (+ endpoint pinning)

with U_0 = I fixed and H_k = h0 + sum_a s_a(k) * controls[a], where s_a(k)
samples the piecewise-linear control defined by the knot values at the slice
midpoint (default) or left knot.

Conventions:

- ``isovec(U) = concat(vec(Re U), vec(Im U))`` with row-major (C-order) vec.
- ``terminal_objective(U_N) -> (value, G)`` with ``G = dL/d(conj(U_N))`` so
  that ``dL = 2 Re Tr(G^H dU_N)`` — the matrix analogue of the grape costate
  convention.
- Controls are internally nondimensionalized per channel by
  ``s_a = max(|lo|, |hi|)``; du and ddu scale by ``s_a/dt`` and ``s_a/dt^2``,
  which turns the chain constraints into coefficient-one rows.
- First derivatives are exact and sparse (``expm_frechet`` per slice and
  channel); IPOPT runs with a limited-memory Hessian approximation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, expm_frechet

from .grape import _require_hermitian

# stencil entries are (knot offset relative to the slice index k, weight)
_SAMPLING_STENCILS = {"midpoint": ((-1, 0.5), (0, 0.5)), "left": ((-1, 1.0),)}


def _isovec(u: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ravel(u.real), np.ravel(u.imag)])


def _left_mult_real(e: np.ndarray) -> np.ndarray:
    """Real 2D^2 x 2D^2 matrix of ``U -> E @ U`` acting on isovec(U)."""
    kr = np.kron(e, np.eye(e.shape[0]))
    return np.block([[kr.real, -kr.imag], [kr.imag, kr.real]])


class _DirectNLP:
    """Solver-independent NLP assembly for the direct method.

    Variable layout (flat vector of length ``n``): first ``isovec(U_k)`` for
    k = 1..K (``M = 2 dim^2`` reals each), then per channel ``a`` one
    contiguous block of ``u`` knots 0..K, ``du`` knots 0..K, ``ddu`` knots
    0..K-1 — all in scaled units. Constraint layout (length ``m``): K defect
    blocks of M rows, then per channel K u-chain rows and K du-chain rows.
    """

    def __init__(
        self,
        *,
        h0,
        controls,
        n_slices,
        dt,
        terminal_objective,
        u_bounds,
        du_bounds=None,
        ddu_bounds=None,
        fix_endpoints=True,
        regularization=None,
        slice_sampling="midpoint",
    ):
        self.drift = _require_hermitian("h0", h0)
        self.dim = self.drift.shape[0]
        if not isinstance(controls, Mapping) or len(controls) == 0:
            raise ValueError(
                "controls must be a non-empty mapping from channel names to Hermitian operators."
            )
        self.channels = tuple(str(name) for name in controls.keys())
        self.ops = [
            _require_hermitian(f"controls[{name!r}]", op, dim=self.dim)
            for name, op in controls.items()
        ]
        if isinstance(n_slices, bool) or int(n_slices) != n_slices or int(n_slices) < 2:
            raise ValueError(f"n_slices must be an integer >= 2; got {n_slices!r}.")
        self.K = int(n_slices)
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"dt must be finite and positive; got {dt!r}.")
        if not callable(terminal_objective):
            raise TypeError("terminal_objective must be callable.")
        self.terminal_objective = terminal_objective
        if slice_sampling not in _SAMPLING_STENCILS:
            raise ValueError(
                f"slice_sampling must be one of {sorted(_SAMPLING_STENCILS)}; got {slice_sampling!r}."
            )
        self.stencil = _SAMPLING_STENCILS[slice_sampling]

        if not isinstance(u_bounds, Mapping) or set(u_bounds.keys()) != set(self.channels):
            raise ValueError("u_bounds must provide exactly one (lo, hi) pair per control channel.")
        self.u_lo, self.u_hi, self.scales = [], [], []
        for name in self.channels:
            lo, hi = (float(v) for v in u_bounds[name])
            if not (np.isfinite(lo) and np.isfinite(hi)) or not (lo <= 0.0 <= hi) or lo == hi:
                raise ValueError(
                    f"u_bounds[{name!r}] must be finite with lo <= 0 <= hi and lo < hi; got ({lo}, {hi})."
                )
            self.u_lo.append(lo)
            self.u_hi.append(hi)
            self.scales.append(max(abs(lo), abs(hi)))
        self.du_max = self._abs_bounds("du_bounds", du_bounds)
        self.ddu_max = self._abs_bounds("ddu_bounds", ddu_bounds)

        reg = dict(regularization) if regularization is not None else {}
        unknown = set(reg.keys()) - {"u", "du", "ddu"}
        if unknown:
            raise ValueError(
                f"regularization keys must be within {{'u', 'du', 'ddu'}}; got {sorted(unknown)}."
            )
        self.reg = tuple(float(reg.get(key, 0.0)) for key in ("u", "du", "ddu"))
        if any(not np.isfinite(w) or w < 0.0 for w in self.reg):
            raise ValueError("regularization weights must be finite and non-negative.")

        self.M = 2 * self.dim * self.dim
        self.A = len(self.channels)
        self.n = self.K * self.M + self.A * (3 * self.K + 2)
        self.m = self.K * self.M + 2 * self.A * self.K
        self.fix_endpoints = bool(fix_endpoints)
        self._memo_key = None

        self.lb = np.full(self.n, -np.inf)
        self.ub = np.full(self.n, np.inf)
        for a in range(self.A):
            s = self.scales[a]
            c0 = self._c_off(a)
            self.lb[c0 : c0 + self.K + 1] = self.u_lo[a] / s
            self.ub[c0 : c0 + self.K + 1] = self.u_hi[a] / s
            if self.fix_endpoints:
                for j in (0, self.K):
                    self.lb[c0 + j] = 0.0
                    self.ub[c0 + j] = 0.0
            if self.du_max[a] is not None:
                b = self.du_max[a] * self.dt / s
                self.lb[c0 + self.K + 1 : c0 + 2 * (self.K + 1)] = -b
                self.ub[c0 + self.K + 1 : c0 + 2 * (self.K + 1)] = b
            if self.ddu_max[a] is not None:
                b = self.ddu_max[a] * self.dt**2 / s
                self.lb[c0 + 2 * (self.K + 1) : c0 + 3 * self.K + 2] = -b
                self.ub[c0 + 2 * (self.K + 1) : c0 + 3 * self.K + 2] = b

        rows, cols = [], []
        for k in range(1, self.K + 1):
            r0 = (k - 1) * self.M
            rows.extend(range(r0, r0 + self.M))
            cols.extend(range(self._u_off(k), self._u_off(k) + self.M))
            if k >= 2:
                for i in range(self.M):
                    rows.extend([r0 + i] * self.M)
                    cols.extend(range(self._u_off(k - 1), self._u_off(k - 1) + self.M))
            for a in range(self.A):
                for off, _w in self.stencil:
                    rows.extend(range(r0, r0 + self.M))
                    cols.extend([self._iu(a, k + off)] * self.M)
        base = self.K * self.M
        for a in range(self.A):
            for k in range(self.K):
                rows.extend([base] * 3)
                cols.extend([self._iu(a, k + 1), self._iu(a, k), self._idu(a, k)])
                base += 1
            for k in range(self.K):
                rows.extend([base] * 3)
                cols.extend([self._idu(a, k + 1), self._idu(a, k), self._iddu(a, k)])
                base += 1
        self.jac_rows = np.asarray(rows, dtype=np.int64)
        self.jac_cols = np.asarray(cols, dtype=np.int64)

    # ── variable layout ──────────────────────────────────────────────────

    def _u_off(self, k):  # isovec(U_k) start, k = 1..K
        return (k - 1) * self.M

    def _c_off(self, a):  # channel block start
        return self.K * self.M + a * (3 * self.K + 2)

    def _iu(self, a, j):  # u knot j = 0..K
        return self._c_off(a) + j

    def _idu(self, a, j):  # du knot j = 0..K
        return self._c_off(a) + (self.K + 1) + j

    def _iddu(self, a, j):  # ddu knot j = 0..K-1
        return self._c_off(a) + 2 * (self.K + 1) + j

    # ── validation helpers ───────────────────────────────────────────────

    def _abs_bounds(self, name, bounds):
        if bounds is None:
            return [None] * len(self.channels)
        if not isinstance(bounds, Mapping) or set(bounds.keys()) - set(self.channels):
            raise ValueError(f"{name} keys must be a subset of the control channels.")
        out = []
        for channel in self.channels:
            if channel not in bounds or bounds[channel] is None:
                out.append(None)
                continue
            value = float(bounds[channel])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name}[{channel!r}] must be finite and positive; got {bounds[channel]!r}."
                )
            out.append(value)
        return out

    def _terminal(self, u_final):
        out = self.terminal_objective(np.array(u_final))
        try:
            raw_value, raw_g = out
        except (TypeError, ValueError):
            raise TypeError("terminal_objective must return a (value, gradient) pair.") from None
        if isinstance(raw_value, (bool, np.bool_)) or np.iscomplexobj(raw_value) or np.ndim(raw_value) != 0:
            raise ValueError(
                f"terminal_objective must return one finite real scalar value; got {raw_value!r}."
            )
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(
                f"terminal_objective must return one finite real scalar value; got {value!r}."
            )
        g = np.asarray(raw_g, dtype=complex)
        if g.shape != (self.dim, self.dim) or not np.all(np.isfinite(g)):
            raise ValueError(
                f"terminal_objective gradient must be a finite ({self.dim}, {self.dim}) matrix."
            )
        return value, g

    # ── evaluation ───────────────────────────────────────────────────────

    def _slice_value(self, x, k, a):
        """Physical control sample entering H for slice k (1..K)."""
        return self.scales[a] * sum(w * x[self._iu(a, k + off)] for off, w in self.stencil)

    def _forward(self, x):
        x = np.ascontiguousarray(x, dtype=float)
        key = x.tobytes()
        if self._memo_key == key:
            return self._memo
        d = self.dim
        unitaries = [np.eye(d, dtype=complex)]
        for k in range(1, self.K + 1):
            off = self._u_off(k)
            re = x[off : off + d * d].reshape(d, d)
            im = x[off + d * d : off + self.M].reshape(d, d)
            unitaries.append(re + 1j * im)
        gens, props = [], []
        for k in range(1, self.K + 1):
            h_k = self.drift.copy()
            for a in range(self.A):
                h_k = h_k + self._slice_value(x, k, a) * self.ops[a]
            gen = -1j * self.dt * h_k
            gens.append(gen)
            props.append(expm(gen))
        self._memo_key = key
        self._memo = {"U": unitaries, "gens": gens, "props": props}
        return self._memo

    def constraints(self, x):
        x = np.ascontiguousarray(x, dtype=float)
        fwd = self._forward(x)
        c = np.empty(self.m)
        for k in range(1, self.K + 1):
            r0 = (k - 1) * self.M
            c[r0 : r0 + self.M] = _isovec(fwd["U"][k]) - _isovec(fwd["props"][k - 1] @ fwd["U"][k - 1])
        base = self.K * self.M
        for a in range(self.A):
            for k in range(self.K):
                c[base] = x[self._iu(a, k + 1)] - x[self._iu(a, k)] - x[self._idu(a, k)]
                base += 1
            for k in range(self.K):
                c[base] = x[self._idu(a, k + 1)] - x[self._idu(a, k)] - x[self._iddu(a, k)]
                base += 1
        return c

    def jacobian(self, x):
        x = np.ascontiguousarray(x, dtype=float)
        fwd = self._forward(x)
        vals = []
        for k in range(1, self.K + 1):
            vals.append(np.ones(self.M))
            if k >= 2:
                vals.append((-_left_mult_real(fwd["props"][k - 1])).ravel())
            for a in range(self.A):
                frechet = expm_frechet(
                    fwd["gens"][k - 1], -1j * self.dt * self.ops[a], compute_expm=False
                )
                col_core = -_isovec(frechet @ fwd["U"][k - 1])
                for _off, w in self.stencil:
                    vals.append(col_core * (self.scales[a] * w))
        vals.append(np.tile([1.0, -1.0, -1.0], 2 * self.A * self.K))
        return np.concatenate(vals)

    def objective(self, x):
        x = np.ascontiguousarray(x, dtype=float)
        value, _ = self._terminal(self._forward(x)["U"][self.K])
        w_u, w_du, w_ddu = self.reg
        for a in range(self.A):
            c0 = self._c_off(a)
            u = x[c0 : c0 + self.K + 1]
            du = x[c0 + self.K + 1 : c0 + 2 * (self.K + 1)]
            ddu = x[c0 + 2 * (self.K + 1) : c0 + 3 * self.K + 2]
            value += w_u * float(u @ u) + w_du * float(du @ du) + w_ddu * float(ddu @ ddu)
        return value

    def gradient(self, x):
        x = np.ascontiguousarray(x, dtype=float)
        _, g = self._terminal(self._forward(x)["U"][self.K])
        grad = np.zeros(self.n)
        off = self._u_off(self.K)
        grad[off : off + self.dim * self.dim] = 2.0 * np.ravel(g.real)
        grad[off + self.dim * self.dim : off + self.M] = 2.0 * np.ravel(g.imag)
        w_u, w_du, w_ddu = self.reg
        for a in range(self.A):
            c0 = self._c_off(a)
            grad[c0 : c0 + self.K + 1] += 2.0 * w_u * x[c0 : c0 + self.K + 1]
            grad[c0 + self.K + 1 : c0 + 2 * (self.K + 1)] += (
                2.0 * w_du * x[c0 + self.K + 1 : c0 + 2 * (self.K + 1)]
            )
            grad[c0 + 2 * (self.K + 1) : c0 + 3 * self.K + 2] += (
                2.0 * w_ddu * x[c0 + 2 * (self.K + 1) : c0 + 3 * self.K + 2]
            )
        return grad

    # ── initial point ────────────────────────────────────────────────────

    def initial_point(self, initial_controls, initial_unitaries):
        if initial_controls is not None:
            extra = set(initial_controls.keys()) - set(self.channels)
            if extra:
                raise ValueError(f"initial_controls contains unknown channels {sorted(extra)}.")
        x0 = np.zeros(self.n)
        for a, name in enumerate(self.channels):
            u = np.zeros(self.K + 1)
            if initial_controls is not None and name in initial_controls:
                u = np.asarray(initial_controls[name], dtype=float)
                if u.shape != (self.K + 1,) or not np.all(np.isfinite(u)):
                    raise ValueError(
                        f"initial_controls[{name!r}] must be a finite ({self.K + 1},) array."
                    )
            uhat = u / self.scales[a]
            duhat = np.zeros(self.K + 1)
            duhat[:-1] = np.diff(uhat)
            dduhat = np.diff(duhat)
            c0 = self._c_off(a)
            x0[c0 : c0 + self.K + 1] = uhat
            x0[c0 + self.K + 1 : c0 + 2 * (self.K + 1)] = duhat
            x0[c0 + 2 * (self.K + 1) : c0 + 3 * self.K + 2] = dduhat
        if initial_unitaries is not None:
            u_arr = np.asarray(initial_unitaries, dtype=complex)
            if u_arr.shape != (self.K, self.dim, self.dim) or not np.all(np.isfinite(u_arr)):
                raise ValueError(
                    f"initial_unitaries must be a finite ({self.K}, {self.dim}, {self.dim}) array."
                )
            mats = list(u_arr)
        else:
            mats = []
            u_mat = np.eye(self.dim, dtype=complex)
            for k in range(1, self.K + 1):
                h_k = self.drift.copy()
                for a in range(self.A):
                    h_k = h_k + self._slice_value(x0, k, a) * self.ops[a]
                u_mat = expm(-1j * self.dt * h_k) @ u_mat
                mats.append(u_mat)
        for k in range(1, self.K + 1):
            x0[self._u_off(k) : self._u_off(k) + self.M] = _isovec(mats[k - 1])
        return np.clip(x0, self.lb, self.ub)


_DEFAULT_IPOPT_OPTIONS = {
    "hessian_approximation": "limited-memory",
    "tol": 1e-8,
    "print_level": 0,
    "sb": "yes",
    "mu_strategy": "adaptive",
}

# IPOPT return statuses accepted as convergence: 0 = Solve_Succeeded,
# 1 = Solved_To_Acceptable_Level.
_ACCEPTED_IPOPT_STATUS = (0, 1)


@dataclass(frozen=True)
class DirectResult:
    """Solution of one direct trajectory-optimization solve.

    ``accepted`` requires both a converged IPOPT status and a recomputed
    maximum defect residual within ``feasibility_tol``; a converged-but-
    infeasible run never reports ``accepted=True``. ``objective`` is the
    terminal value only (regularizers excluded); ``controls``/``du``/``ddu``
    are per-channel knot arrays in physical units.
    """

    controls: dict[str, np.ndarray]
    du: dict[str, np.ndarray]
    ddu: dict[str, np.ndarray]
    unitaries: np.ndarray
    objective: float
    max_defect: float
    ipopt_status: int
    ipopt_message: str
    n_iter: int
    accepted: bool


def optimize(
    h0,
    controls,
    *,
    n_slices,
    dt,
    terminal_objective,
    u_bounds,
    du_bounds=None,
    ddu_bounds=None,
    fix_endpoints=True,
    regularization=None,
    slice_sampling="midpoint",
    initial_controls=None,
    initial_unitaries=None,
    maxiter=1000,
    feasibility_tol=1e-8,
    ipopt_options=None,
):
    """Solve the direct-method NLP with IPOPT and return a :class:`DirectResult`."""
    try:
        import cyipopt
    except ImportError as exc:
        raise ImportError(
            "qoc.direct.optimize requires the optional cyipopt dependency; "
            "install it with `uv sync --extra qoc-direct`."
        ) from exc

    nlp = _DirectNLP(
        h0=h0,
        controls=controls,
        n_slices=n_slices,
        dt=dt,
        terminal_objective=terminal_objective,
        u_bounds=u_bounds,
        du_bounds=du_bounds,
        ddu_bounds=ddu_bounds,
        fix_endpoints=fix_endpoints,
        regularization=regularization,
        slice_sampling=slice_sampling,
    )
    x0 = nlp.initial_point(initial_controls, initial_unitaries)

    iterations = {"n": 0}

    class _Callbacks:
        def objective(self, x):
            return nlp.objective(x)

        def gradient(self, x):
            return nlp.gradient(x)

        def constraints(self, x):
            return nlp.constraints(x)

        def jacobian(self, x):
            return nlp.jacobian(x)

        def jacobianstructure(self):
            return nlp.jac_rows, nlp.jac_cols

        def intermediate(self, alg_mod, iter_count, *args):
            iterations["n"] = int(iter_count)

    problem = cyipopt.Problem(
        n=nlp.n,
        m=nlp.m,
        problem_obj=_Callbacks(),
        lb=nlp.lb,
        ub=nlp.ub,
        cl=np.zeros(nlp.m),
        cu=np.zeros(nlp.m),
    )
    merged = dict(_DEFAULT_IPOPT_OPTIONS)
    merged["max_iter"] = int(maxiter)
    merged["constr_viol_tol"] = float(feasibility_tol)
    if ipopt_options:
        merged.update(ipopt_options)
    for name, value in merged.items():
        problem.add_option(name, value)

    x_opt, info = problem.solve(x0)
    x_opt = np.ascontiguousarray(x_opt, dtype=float)

    max_defect = float(np.max(np.abs(nlp.constraints(x_opt)[: nlp.K * nlp.M])))
    fwd = nlp._forward(x_opt)
    unitaries = np.stack(fwd["U"])
    value, _ = nlp._terminal(fwd["U"][nlp.K])
    controls_out, du_out, ddu_out = {}, {}, {}
    for a, name in enumerate(nlp.channels):
        c0 = nlp._c_off(a)
        s = nlp.scales[a]
        controls_out[name] = s * x_opt[c0 : c0 + nlp.K + 1]
        du_out[name] = (s / nlp.dt) * x_opt[c0 + nlp.K + 1 : c0 + 2 * (nlp.K + 1)]
        ddu_out[name] = (s / nlp.dt**2) * x_opt[c0 + 2 * (nlp.K + 1) : c0 + 3 * nlp.K + 2]

    status = int(info["status"])
    message = info["status_msg"]
    if isinstance(message, bytes):
        message = message.decode(errors="replace")
    return DirectResult(
        controls=controls_out,
        du=du_out,
        ddu=ddu_out,
        unitaries=unitaries,
        objective=float(value),
        max_defect=max_defect,
        ipopt_status=status,
        ipopt_message=str(message),
        n_iter=iterations["n"],
        accepted=(status in _ACCEPTED_IPOPT_STATUS) and (max_defect <= float(feasibility_tol)),
    )
