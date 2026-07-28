# qoc.direct Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the direct trajectory-optimization engine of
`docs/superpowers/specs/2026-07-28-direct-qoc-zxz-design.md` as `qoc.direct`,
solved by IPOPT via the optional `cyipopt` dependency.

**Architecture:** One new module `src/qoc/direct.py` with two layers: a
solver-independent pure-numpy NLP assembly class `_DirectNLP` (variables,
bounds, constraints, exact sparse Jacobian, objective/gradient) and a thin
public `optimize(...)` driver that feeds it to IPOPT through `cyipopt` and
returns a plain `DirectResult`. Mirrors the array-only discipline of
`src/qoc/grape.py` (ADR-0024); decision recorded as ADR-0026.

**Tech Stack:** numpy, scipy (`expm`, `expm_frechet`), cyipopt>=1.4 (optional
extra), pytest.

## Global Constraints

- **Scope: qoc only** (user directive 2026-07-28). The ONLY files that may
  change: `src/qoc/direct.py` (new), `src/qoc/__init__.py`,
  `tests/qoc/test_direct.py` (new), `pyproject.toml`
  (`[project.optional-dependencies]` only),
  `docs/adr/0026-solve-direct-trajectory-optimization-in-qoc-with-ipopt.md`
  (new), and one status-line edit in
  `docs/superpowers/specs/2026-07-28-direct-qoc-zxz-design.md`. Nothing in
  `src/ryd_gate/`, `scripts/`, or any compat-lock test may be touched.
- `qoc` never imports `ryd_gate` or any physics package; `import cyipopt`
  happens only inside `optimize` (module import stays dependency-free).
- Conventions fixed by the spec: `isovec(U) = concat(vec(Re U), vec(Im U))`
  with row-major (C-order) vec; `terminal_objective(U_N) -> (value, G)` with
  `G = dL/d(conj(U_N))` so `dL = 2 Re Tr(G^H dU_N)`; controls internally
  nondimensionalized per channel by `s_a = max(|lo|,|hi|)`; `accepted`
  requires IPOPT status in {0, 1} AND recomputed `max_defect <=
  feasibility_tol` (default 1e-8).
- This repo is an sshfs mount of the DGX. **File edits** use local paths
  under `/home/chance/dgx/Ryd-gate-modeling/`; **all git and pytest commands
  run over ssh** (git on the mount hangs):
  `ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && <cmd>'`.
- Work directly on `main`. NEVER `git push`.
- Commit messages end with
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

## File Structure

- `src/qoc/direct.py` — new: `_DirectNLP` (assembly) + `DirectResult` +
  `optimize` (cyipopt driver). One responsibility: the direct-method NLP.
- `tests/qoc/test_direct.py` — new: derivative/validation tests (Task 1,
  cyipopt-free) + solver tests (Task 2).
- `pyproject.toml` — add `qoc-direct` extra; add `cyipopt>=1.4` to `dev`.
- `docs/adr/0026-...md`, `src/qoc/__init__.py`, spec status line — Task 3.

---

### Task 1: `_DirectNLP` assembly with exact derivatives

**Files:**
- Create: `src/qoc/direct.py`
- Create: `tests/qoc/test_direct.py`

**Interfaces:**
- Consumes: `_require_hermitian` from `src/qoc/grape.py` (private
  cross-import inside the package, intentional DRY).
- Produces (used by Task 2): class `_DirectNLP` with constructor keywords
  `h0, controls, n_slices, dt, terminal_objective, u_bounds, du_bounds,
  ddu_bounds, fix_endpoints, regularization, slice_sampling`; attributes
  `n, m, K, M, A, dim, dt, channels, scales, lb, ub, jac_rows, jac_cols`;
  methods `objective(x)`, `gradient(x)`, `constraints(x)`, `jacobian(x)`,
  `initial_point(initial_controls, initial_unitaries)`, `_c_off(a)`,
  `_u_off(k)`, `_forward(x)`, `_terminal(u_final)`; module helpers
  `_isovec(u)`, `_left_mult_real(e)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/qoc/test_direct.py`:

```python
"""Tests for the direct trajectory-optimization engine (ADR-0026)."""

from __future__ import annotations

import numpy as np
import pytest

from qoc.direct import _DirectNLP


def _random_problem(seed, *, slice_sampling="midpoint", regularization=None):
    """Small random two-channel problem (D=2, K=3) plus a random point x."""
    rng = np.random.default_rng(seed)

    def herm(d):
        a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        return a + a.conj().T

    d = 2
    target = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))[0]

    def objective(u_final):
        c = np.trace(target.conj().T @ u_final)
        return 1.0 - abs(c) ** 2 / d**2, -(c / d**2) * target

    nlp = _DirectNLP(
        h0=herm(d),
        controls={"a": herm(d), "b": herm(d)},
        n_slices=3,
        dt=0.2,
        terminal_objective=objective,
        u_bounds={"a": (-2.0, 2.0), "b": (-1.0, 3.0)},
        du_bounds={"a": 5.0},
        ddu_bounds={"b": 7.0},
        fix_endpoints=True,
        regularization=regularization,
        slice_sampling=slice_sampling,
    )
    x = rng.normal(scale=0.5, size=nlp.n)
    return nlp, x


def test_layout_counts():
    nlp, _ = _random_problem(1)
    # D=2 -> M = 2*D^2 = 8; K=3; A=2.
    assert (nlp.dim, nlp.M, nlp.K, nlp.A) == (2, 8, 3, 2)
    assert nlp.n == nlp.K * nlp.M + nlp.A * (3 * nlp.K + 2)   # 24 + 22 = 46
    assert nlp.m == nlp.K * nlp.M + 2 * nlp.A * nlp.K          # 24 + 12 = 36
    assert nlp.lb.shape == nlp.ub.shape == (nlp.n,)
    # endpoint pinning: u knots 0 and K have (0, 0) bounds on every channel
    for a in range(nlp.A):
        c0 = nlp._c_off(a)
        assert nlp.lb[c0] == nlp.ub[c0] == 0.0
        assert nlp.lb[c0 + nlp.K] == nlp.ub[c0 + nlp.K] == 0.0


@pytest.mark.parametrize("slice_sampling", ["midpoint", "left"])
def test_constraint_jacobian_matches_finite_differences(slice_sampling):
    nlp, x = _random_problem(7, slice_sampling=slice_sampling)
    dense = np.zeros((nlp.m, nlp.n))
    np.add.at(dense, (nlp.jac_rows, nlp.jac_cols), nlp.jacobian(x))
    h = 1e-6
    fd = np.zeros_like(dense)
    for i in range(nlp.n):
        e = np.zeros(nlp.n)
        e[i] = h
        fd[:, i] = (nlp.constraints(x + e) - nlp.constraints(x - e)) / (2 * h)
    assert np.max(np.abs(dense - fd)) < 1e-6


def test_objective_gradient_matches_finite_differences():
    nlp, x = _random_problem(11, regularization={"u": 0.3, "du": 0.2, "ddu": 0.1})
    grad = nlp.gradient(x)
    h = 1e-6
    for i in range(nlp.n):
        e = np.zeros(nlp.n)
        e[i] = h
        fd = (nlp.objective(x + e) - nlp.objective(x - e)) / (2 * h)
        assert abs(grad[i] - fd) < 1e-6


def test_rollout_initial_point_is_feasible():
    nlp, _ = _random_problem(3)
    x0 = nlp.initial_point(
        {"a": np.array([0.0, 0.5, 0.4, 0.0]), "b": np.zeros(4)}, None
    )
    assert np.max(np.abs(nlp.constraints(x0))) < 1e-12
    assert np.all(x0 >= nlp.lb) and np.all(x0 <= nlp.ub)


def test_initial_unitaries_override_is_used_verbatim():
    nlp, _ = _random_problem(9)
    stack = np.stack([np.eye(2, dtype=complex)] * nlp.K)
    x0 = nlp.initial_point(None, stack)
    for k in range(1, nlp.K + 1):
        off = nlp._u_off(k)
        re = x0[off : off + 4].reshape(2, 2)
        im = x0[off + 4 : off + 8].reshape(2, 2)
        np.testing.assert_allclose(re + 1j * im, np.eye(2), atol=1e-14)
    # identity stack under a nonzero drift violates the defect constraints
    assert np.max(np.abs(nlp.constraints(x0))) > 1e-6


def test_terminal_objective_contract_is_validated():
    nlp, x = _random_problem(5)
    nlp.terminal_objective = lambda u: 1.0  # not a pair
    with pytest.raises(TypeError, match="pair"):
        nlp.objective(x)
    nlp.terminal_objective = lambda u: (np.inf, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        nlp.objective(x)
    nlp.terminal_objective = lambda u: (0.0, np.zeros(3))
    with pytest.raises(ValueError, match="gradient"):
        nlp.objective(x)


def test_validation_errors():
    ident = np.eye(2)

    def obj(u):
        return 0.0, np.zeros((2, 2))

    def build(**kw):
        base = dict(
            h0=ident,
            controls={"a": ident},
            n_slices=3,
            dt=0.2,
            terminal_objective=obj,
            u_bounds={"a": (-1.0, 1.0)},
        )
        base.update(kw)
        return _DirectNLP(**base)

    build()  # baseline constructs cleanly
    with pytest.raises(ValueError, match="Hermitian"):
        build(h0=np.array([[0.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="n_slices"):
        build(n_slices=1)
    with pytest.raises(ValueError, match="dt"):
        build(dt=0.0)
    with pytest.raises(ValueError, match="u_bounds"):
        build(u_bounds={"a": (0.5, 1.0)})  # violates lo <= 0 <= hi
    with pytest.raises(ValueError, match="u_bounds"):
        build(u_bounds={"b": (-1.0, 1.0)})  # channel mismatch
    with pytest.raises(ValueError, match="du_bounds"):
        build(du_bounds={"a": -1.0})
    with pytest.raises(ValueError, match="slice_sampling"):
        build(slice_sampling="right")
    with pytest.raises(ValueError, match="regularization"):
        build(regularization={"jerk": 1.0})
    with pytest.raises(ValueError, match="initial_controls"):
        build().initial_point({"zz": np.zeros(4)}, None)
    with pytest.raises(ValueError, match="initial_unitaries"):
        build().initial_point(None, np.zeros((2, 2, 2)))
```

- [ ] **Step 2: Run tests to verify they fail on import**

Run:
`ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run --extra dev pytest tests/qoc/test_direct.py -q'`
Expected: collection error — `ModuleNotFoundError: No module named 'qoc.direct'`.

- [ ] **Step 3: Implement `_DirectNLP`**

Create `src/qoc/direct.py`:

```python
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
```

- [ ] **Step 4: Run the Task 1 tests to verify they pass**

Run:
`ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run --extra dev pytest tests/qoc/test_direct.py -q'`
Expected: all PASS (8 tests, ~seconds).

- [ ] **Step 5: Run the rest of the qoc tests (no regressions)**

Run:
`ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run --extra dev pytest tests/qoc -q'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && git add src/qoc/direct.py tests/qoc/test_direct.py && git commit -m "Add the direct-method NLP assembly with exact sparse derivatives

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"'
```

---

### Task 2: `optimize` driver, `DirectResult`, cyipopt dependency

**Files:**
- Modify: `src/qoc/direct.py` (append after `_DirectNLP`)
- Modify: `tests/qoc/test_direct.py` (append)
- Modify: `pyproject.toml` (`[project.optional-dependencies]` block only)

**Interfaces:**
- Consumes: `_DirectNLP` exactly as produced by Task 1.
- Produces: `qoc.direct.optimize(h0, controls, *, n_slices, dt,
  terminal_objective, u_bounds, du_bounds=None, ddu_bounds=None,
  fix_endpoints=True, regularization=None, slice_sampling="midpoint",
  initial_controls=None, initial_unitaries=None, maxiter=1000,
  feasibility_tol=1e-8, ipopt_options=None) -> DirectResult` and the frozen
  dataclass `DirectResult(controls, du, ddu, unitaries, objective,
  max_defect, ipopt_status, ipopt_message, n_iter, accepted)`.

- [ ] **Step 1: Add cyipopt to pyproject**

In `pyproject.toml`, add `"cyipopt>=1.4",` as the last entry of the `dev`
list, and insert a new extra directly after the `dev` block:

```toml
# IPOPT-backed direct trajectory optimization (qoc.direct.optimize).
qoc-direct = [
    "cyipopt>=1.4",
]
```

Verify the environment resolves:
`ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run --extra dev --extra qoc-direct python -c "import cyipopt; print(cyipopt.__version__)"'`
Expected: a version >= 1.4 prints.

- [ ] **Step 2: Write the failing solver tests**

Append to `tests/qoc/test_direct.py`:

```python
from scipy.linalg import expm as _expm

from qoc.direct import DirectResult, optimize

_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SY = np.array([[0.0, -1.0j], [1.0j, 0.0]])
_SZ = np.diag([1.0, -1.0]).astype(complex)


def _infidelity(target):
    d = target.shape[0]

    def objective(u_final):
        c = np.trace(target.conj().T @ u_final)
        return 1.0 - abs(c) ** 2 / d**2, -(c / d**2) * target

    return objective


def _solve(**overrides):
    kwargs = dict(
        h0=0.1 * _SZ,
        controls={"x": _SX, "y": _SY},
        n_slices=8,
        dt=0.25,
        terminal_objective=_infidelity(_expm(-1j * (np.pi / 2) * _SX)),
        u_bounds={"x": (-2.0, 2.0), "y": (-2.0, 2.0)},
        maxiter=500,
    )
    kwargs.update(overrides)
    return optimize(**kwargs)


def test_known_synthesis_converges():
    result = _solve(
        initial_controls={"x": 1.3 * np.sin(np.linspace(0.0, np.pi, 9)), "y": np.zeros(9)}
    )
    assert isinstance(result, DirectResult)
    assert result.accepted
    assert result.objective < 1e-3
    assert result.max_defect <= 1e-8
    assert result.unitaries.shape == (9, 2, 2)
    np.testing.assert_allclose(result.unitaries[0], np.eye(2), atol=1e-14)
    for name in ("x", "y"):
        assert result.controls[name].shape == (9,)
        assert result.du[name].shape == (9,)
        assert result.ddu[name].shape == (8,)


def test_infeasible_identity_start_converges():
    result = _solve(initial_unitaries=np.stack([np.eye(2, dtype=complex)] * 8))
    assert result.accepted
    assert result.objective < 1e-3


def test_bounds_and_endpoints_respected():
    result = _solve(
        du_bounds={"x": 4.0, "y": 4.0},
        ddu_bounds={"x": 40.0, "y": 40.0},
        initial_controls={"x": 1.3 * np.sin(np.linspace(0.0, np.pi, 9)), "y": np.zeros(9)},
    )
    assert result.accepted
    for name in ("x", "y"):
        u = result.controls[name]
        assert abs(u[0]) < 1e-9 and abs(u[-1]) < 1e-9
        assert np.all(u >= -2.0 - 1e-9) and np.all(u <= 2.0 + 1e-9)
        assert np.max(np.abs(result.du[name])) <= 4.0 + 1e-6
        assert np.max(np.abs(result.ddu[name])) <= 40.0 + 1e-4


def test_unconverged_run_reports_not_accepted():
    result = _solve(maxiter=1)
    assert not result.accepted
```

- [ ] **Step 3: Run tests to verify the new ones fail**

Run:
`ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run --extra dev --extra qoc-direct pytest tests/qoc/test_direct.py -q'`
Expected: the whole file errors at collection with
`ImportError: cannot import name 'DirectResult' from 'qoc.direct'` (the
appended top-level import fails before any test runs — that is the expected
failing state).

- [ ] **Step 4: Implement `DirectResult` and `optimize`**

Append to `src/qoc/direct.py`:

```python
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
```

- [ ] **Step 5: Run the full test file**

Run:
`ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run --extra dev --extra qoc-direct pytest tests/qoc/test_direct.py -q'`
Expected: all 12 tests PASS (the three synthesis solves take seconds each).
If `test_known_synthesis_converges` fails on the fidelity threshold, inspect
`result.ipopt_message` and `result.objective` first — a locally optimal but
imperfect pulse means the bounds/objective wiring is fine but the start is
poor; loosen the start (e.g. amplitude 1.0) rather than the assertion.

- [ ] **Step 6: Commit**

```bash
ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && git add src/qoc/direct.py tests/qoc/test_direct.py pyproject.toml uv.lock && git commit -m "Add qoc.direct.optimize: IPOPT-backed direct trajectory optimization

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"'
```

(If `uv.lock` was not modified by the resolve step, commit without it.)

---

### Task 3: ADR-0026, public export, spec scope note

**Files:**
- Create: `docs/adr/0026-solve-direct-trajectory-optimization-in-qoc-with-ipopt.md`
- Modify: `src/qoc/__init__.py`
- Modify: `docs/superpowers/specs/2026-07-28-direct-qoc-zxz-design.md` (status line only)

**Interfaces:**
- Consumes: `qoc.direct` module as produced by Tasks 1–2.
- Produces: `qoc.direct` reachable as a public submodule (`import qoc;
  qoc.direct.optimize`), matching how `qoc.grape` is exposed.

- [ ] **Step 1: Write the ADR**

Create `docs/adr/0026-solve-direct-trajectory-optimization-in-qoc-with-ipopt.md`:

```markdown
# Solve direct trajectory optimization in qoc with IPOPT

GRAPE (ADR-0024) compresses all time slices into one endpoint sensitivity
and can honor hardware limits only through penalties. The direct method
reserved by ADR-0007 — intermediate states as decision variables plus local
dynamics constraints — is implemented in `qoc.direct` over the same exported
bilinear control model arrays as GRAPE: knot unitaries `isovec(U_k)` and
per-channel control chains `u -> du -> ddu` are the decision variables, the
slice dynamics `U_k = expm(-i H(u) dt) U_{k-1}` are per-slice defect
constraints with exact `expm_frechet` Jacobians, and every hardware limit
(amplitude, slew rate, curvature, endpoint pinning) is a plain variable
bound. The double-integrator chain makes piecewise-linear waveforms and
their derivative bounds native instead of penalized, and the optimizer may
start from and traverse infeasible state trajectories.

The nonlinear program is solved by IPOPT through the optional `cyipopt`
dependency (extra `qoc-direct`), imported lazily inside `optimize` so the
base package stays dependency-free. The terminal objective remains a
caller-supplied callback in the grape costate convention (`G =
dL/d(conj(U_N))`); `qoc` still never imports a physics package. Durations
stay fixed within one solve (ADR-0005); this knot-point formulation was
chosen because it extends to free knot durations without restructuring.

## Considered options

- scipy-only solvers (trust-constr, hand-rolled augmented Lagrangian) —
  rejected: at the target scale (~10^4 variables with as many equality
  constraints) their convergence is the dominant project risk, and an
  augmented-Lagrangian schedule would become our code to tune and maintain.
- Piccolo.jl — rejected: introduces a Julia toolchain into a Python
  repository and bypasses the qoc package boundary entirely.
- Defect constraints on state vectors instead of unitaries — deferred: the
  first consumer scores the full propagator; a state-trajectory variant can
  be added beside this one when an ensemble objective needs it.
```

- [ ] **Step 2: Export the submodule**

In `src/qoc/__init__.py`:

- change `from . import grape` to `from . import direct, grape`
- change the `__all__` list to
  `["OptimizationResult", "direct", "grape", "minimize"]`
- append one paragraph to the module docstring, after the `qoc.grape`
  paragraph:

```python
"""
``qoc.direct`` solves the same bilinear control model as a direct
trajectory-optimization NLP (ADR-0026): knot unitaries and control chains
are decision variables, slice dynamics are defect constraints, and IPOPT
(optional ``cyipopt`` dependency) solves the program.
"""
```

(merge the text into the existing docstring — do not add a second docstring
literal).

- [ ] **Step 3: Record the descope in the spec**

In `docs/superpowers/specs/2026-07-28-direct-qoc-zxz-design.md`, extend the
status line to:

```markdown
Date: 2026-07-28. Status: approved (solver route A: cyipopt/IPOPT).
Scope 2026-07-28 (user): implementation covers the qoc-side deliverables
only (engine, cyipopt extra, ADR-0026, engine tests); the ZXZ study
(deliverable 3 and §4) is deferred and kept here as reference wiring.
```

- [ ] **Step 4: Run the qoc suite and the documentation-structure test**

Run:
`ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && LC_ALL=C.UTF-8 uv run --extra dev --extra qoc-direct pytest tests/qoc tests/test_documentation_structure.py -q'`
Expected: PASS (documentation test guards ADR formatting/registration; if it
requires ADRs to be listed in an index, follow the failure message).

- [ ] **Step 5: Verify the public import path**

Run:
`ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run --extra dev python -c "import qoc; print(qoc.direct.DirectResult.__name__, callable(qoc.direct.optimize))"'`
Expected: `DirectResult True` (`cyipopt` is imported only inside `optimize`
by construction, so `import qoc` never requires it).

- [ ] **Step 6: Commit**

```bash
ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && git add docs/adr/0026-solve-direct-trajectory-optimization-in-qoc-with-ipopt.md src/qoc/__init__.py docs/superpowers/specs/2026-07-28-direct-qoc-zxz-design.md && git commit -m "Register qoc.direct: ADR-0026, public export, spec scope note

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"'
```

---

## Final verification (controller, after all tasks)

Run the fast remote suite:
`ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && LC_ALL=C.UTF-8 uv run --extra dev --extra qoc-direct pytest -q -m "not slow"'`
Expected: PASS with no new failures relative to `main` before this plan.
