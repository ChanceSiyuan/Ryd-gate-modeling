# Direct Quantum Optimal Control (qoc.direct) + ZXZ Synthesis Reproduction — Design

Date: 2026-07-28. Status: approved (solver route A: cyipopt/IPOPT).
Scope 2026-07-28 (user): implementation covers the qoc-side deliverables
only (engine, cyipopt extra, ADR-0026, engine tests); the ZXZ study
(deliverable 3 and §4) is deferred and kept here as reference wiring.

## Goal

Implement the "direct quantum optimal control" method of arXiv:2508.19075 (the
`new.tex` manuscript in this repo, Sec. "Direct Quantum Optimal Control" +
trajectory-optimization appendix) as a second engine in the `qoc` package,
beside `qoc.grape`, and reproduce the paper's ZXZ-synthesis benchmark: smooth
global Ω(t)/Δ(t) pulses on a 3-atom Rydberg chain that synthesize
`exp(-i·0.8·Z₁X₂Z₃)` under Aquila hardware constraints, at the paper's
fidelity level (~0.89 at T=1.2 µs, ~0.94 at T=3.6 µs), beating GRAPE under
identical constraints.

## Background

- The paper's method (appendix, Piccolo.jl formulation): knot-point variables
  `z_k = (isovec(U_k), u_k, u̇_k, ü_k)`, Schrödinger dynamics as per-step
  defect constraints, control smoothness via the double-integrator chain
  `u → u̇ → ü` with bounds, terminal unitary-fidelity objective, solved as a
  sparse NLP with IPOPT.
- Repo seams already in place: ADR-0007 reserves the name "direct" for exactly
  this structure (intermediate states as optimization variables + local
  dynamics constraints); ADR-0015 anticipates "a separate future interface for
  local dynamics constraints and Jacobians"; ADR-0024's
  `ryd_gate.bilinear_control_model` exports `H(u) = h0 + Σ_a u_a·C_a` as plain
  arrays — the exact form the defect constraints need. ADR-0005: durations
  stay fixed inside one solve (free-time is explicitly out of scope here).

## Deliverables

1. `src/qoc/direct.py` — the direct trajectory-optimization engine.
2. `docs/adr/0026-*.md` — ADR recording the decision (direct engine in `qoc`
   over the exported bilinear model, solved by IPOPT via the optional
   `cyipopt` dependency).
3. `scripts/zxz_direct_qoc.py` — the ZXZ reproduction study (direct pulses +
   GRAPE comparison + exact-ODE validation), results under
   `results/zxz_direct_qoc/`.
4. Tests: `tests/test_qoc_direct.py` (engine, physics-free) and study-side
   pins inside the study test file `tests/test_zxz_direct_qoc.py`.

---

## 1. Engine: `src/qoc/direct.py`

Same discipline as `qoc/grape.py`: consumes only plain arrays, mappings, and
callbacks; never imports a physics package; fail-fast input validation
mirroring grape's `_require_hermitian` / `_require_vector` helpers.

### Public API

```python
result = qoc.direct.optimize(
    h0,                     # (D,D) Hermitian drift
    controls,               # mapping {channel: (D,D) Hermitian}; iteration order = channel order
    *,
    n_slices,               # K >= 2 propagation slices; K+1 control knots (0..K)
    dt,                     # fixed slice duration (same time unit as 1/energy unit of h0)
    terminal_objective,     # callable U_N (D,D) -> (value, dL_dconjU)  [see contract]
    u_bounds,               # {channel: (lo, hi)} finite bounds, lo <= 0 <= hi required
    du_bounds=None,         # {channel: max_abs} slew bound on u̇; None entry/absent = unbounded
    ddu_bounds=None,        # {channel: max_abs} curvature bound on ü; None/absent = unbounded
    fix_endpoints=True,     # u at knots 0 and K constrained to 0
    regularization=None,    # {"u": w_u, "du": w_du, "ddu": w_ddu} quadratic costs, default all 0
    slice_sampling="midpoint",  # "midpoint": H(ū_k), ū_k=(u_k+u_{k+1})/2 ; "left": H(u_k)
    initial_controls=None,  # {channel: (K+1,)} knot values; default zeros
    initial_unitaries=None, # (K, D, D) for U_1..U_K; default forward rollout of initial_controls
    maxiter=1000,
    feasibility_tol=1e-8,   # acceptance gate on max defect residual
    ipopt_options=None,     # mapping passed through to IPOPT verbatim (may override defaults)
)
```

### NLP formulation

Knots `0..K`, slices `1..K` (slice k evolves knot k-1 → knot k). `U_0 = I` is
fixed (not a variable).

Decision variables, packed in this order into one flat vector:

1. `isovec(U_k)` for k = 1..K — `isovec(U) = concat(vec_r(Re U), vec_r(Im U))`
   with row-major (C-order) `vec_r`; 2D² reals per knot.
2. Per channel a (in `controls` iteration order): `u_a` at knots 0..K,
   then `u̇_a` at knots 0..K, then `ü_a` at knots 0..K-1.

Equality constraints:

- Defect, slice k = 1..K:
  `isovec(U_k) - isovec(expm(-i·H_k·dt) · U_{k-1}) = 0`, with
  `H_k = h0 + Σ_a s_a(k)·C_a` where `s_a(k)` is `ū_{a,k-1} = (u_{a,k-1}+u_{a,k})/2`
  for midpoint sampling (default; 2nd-order accurate for the piecewise-linear
  waveform the knot values define) or `u_{a,k-1}` for left sampling.
- Control chain, k = 0..K-1: `u_{k+1} - u_k - u̇_k·dt = 0` and
  `u̇_{k+1} - u̇_k - ü_k·dt = 0` (per channel; linear).
- Endpoints (when `fix_endpoints`): `u_{a,0} = u_{a,K} = 0`, implemented as
  variable bounds (0, 0).

Inequality handling: all `u_bounds` / `du_bounds` / `ddu_bounds` are plain
variable bounds — IPOPT-native, no constraint rows.

Objective:
`L = terminal_value(U_K) + Σ_a Σ_k [ w_u·û_{a,k}² + w_du·û̇_{a,k}² + w_ddu·ü̂_{a,k}² ]`
where `û, û̇, ü̂` are the scaled variables (see Scaling), so the weights are
dimensionless; all default to 0.

### Terminal objective contract (mirrors grape's costate convention)

`terminal_objective(U_N) -> (value, G)` with `G = dL/d(conj(U_N))` so that
`dL = 2·Re Tr(G† dU_N)`. The engine converts `G` to the gradient with respect
to `isovec(U_K)`. The engine never interprets the objective; unitary fidelity
lives in the study.

### Scaling

Internally every control variable is nondimensionalized: `û_a = u_a / s_a`
with `s_a = max(|lo_a|, |hi_a|)` (finite by the `u_bounds` requirement);
`u̇, ü` scale by `s_a/dt` and `s_a/dt²`. `isovec(U)` entries are already O(1).
IPOPT additionally runs its default `gradient-based` NLP scaling. Results are
reported back in physical units.

### Derivatives

Exact first derivatives, assembled sparse (block-banded):

- Defect w.r.t. `isovec(U_{k-1})`: the real 2D²×2D² isomorphism of left
  multiplication by `E_k = expm(-i·H_k·dt)`; w.r.t. `isovec(U_k)`: identity.
- Defect w.r.t. the knot values entering `s_a(k)`: `expm_frechet` of
  `(-i·H_k·dt, -i·C_a·dt·∂s_a/∂u)` applied to `U_{k-1}` (∂s/∂u = 1/2, 1/2 for
  midpoint; 1 for left) — same machinery as `qoc/grape.py`.
- Chain constraints: constant sparse blocks.
- Objective gradient w.r.t. `isovec(U_K)` from `G`; regularizer gradients are
  diagonal.

Jacobian sparsity structure is precomputed once (COO row/col arrays) and only
values are refilled per evaluation. Second derivatives are NOT provided; IPOPT
runs with `hessian_approximation = "limited-memory"`.

### Solver

`import cyipopt` happens inside `optimize` only (module import stays
dependency-free). Default IPOPT options:
`{"hessian_approximation": "limited-memory", "max_iter": maxiter,
"tol": 1e-8, "constr_viol_tol": feasibility_tol, "print_level": 0,
"sb": "yes", "mu_strategy": "adaptive"}`, shallow-merged under
`ipopt_options`.

### Result contract

`DirectResult` dataclass (plain data, backend-independent in the ADR-0019
spirit):

- `controls`, `du`, `ddu`: `{channel: (K+1,)}` (`ddu` is `(K,)`), physical units;
- `unitaries`: `(K+1, D, D)` including `U_0 = I`;
- `objective`: float terminal value at the solution;
- `max_defect`: max abs defect residual at the solution (recomputed by the
  engine, not read from IPOPT);
- `ipopt_status: int`, `ipopt_message: str`, `n_iter: int`;
- `accepted: bool` — IPOPT success/acceptable status AND
  `max_defect <= feasibility_tol`. A converged-but-infeasible run never
  reports `accepted=True`.

Failure modes: invalid inputs raise `ValueError`/`TypeError` immediately
(shape, hermiticity, finiteness, bound sanity, missing/extra channels in any
per-channel mapping); a cleanly finishing but unsuccessful IPOPT run returns a
result with `accepted=False` (no raise); a missing `cyipopt` raises
`ImportError` with the install hint (`uv sync --extra qoc-direct`).

## 2. Dependency

New optional extra `qoc-direct = ["cyipopt>=1.4"]` in `pyproject.toml`
(no PyPI wheels — the sdist links against a system IPOPT via pkg-config; on the
DGX: conda-forge ipopt=3.14.11 at ~/opt/ipopt), also appended to the `dev` extra so the
remote test suite exercises the engine unconditionally (no skip markers).
Installed on the DGX via the usual `uv run --extra dev ...` flow.

## 3. ADR-0026

`docs/adr/0026-solve-direct-trajectory-optimization-in-qoc-with-ipopt.md`,
one page in the house style: direct engine lives in `qoc` over the same
exported bilinear arrays as GRAPE (ADR-0024), with knot unitaries as decision
variables and per-slice defect constraints (ADR-0007 naming discipline);
double-integrator control chain for smoothness/slew; solved by IPOPT through
the optional `cyipopt` dependency; fixed `dt`/`K` per ADR-0005 (free-time
noted as the future extension this formulation was chosen to permit).
Considered options recorded: pure-scipy trust-constr / hand-rolled augmented
Lagrangian (rejected: convergence risk at ~10⁴ variables), Piccolo.jl
(rejected: Julia toolchain, bypasses the qoc seam).

## 4. Study: `scripts/zxz_direct_qoc.py`

### System (simpler than the approved sketch — noted change)

The design review draft reached the 8-dim problem by slicing the 27-dim
`01r`/`DigitalAnalogProtocol` system. The repo already has a leaner exact
match, adopted here: preset `"1r"` (levels {1, r} per atom → D = 2³ = 8
directly, no slicing) with `SweepProtocol`, which is precisely the paper's
model: `H[r,1](t) = Ω(t)/2` (global), `H[r,r](t) = -Δ(t)`.

- `Register.chain(3, spacing_um=8.9)`; `level_structure("1r", ryd_level=70)`
  — ARC's isotropic 70S C6 (≈2π×862.7 GHz·µm⁶) is the value the paper/Aquila
  quote (5.42×10⁻²⁴ rad·m⁶/s). Sanity assert in the study:
  `V_NN/2π = C6/(2π·8.9⁶) ≈ 1.74 MHz` within 2%.
- `bilinear_control_model(system)` export → channels `E[r,1]:x`, `E[r,1]:y`,
  `E[r,r]`. The study passes only `E[r,1]:x` (value = Ω/2, phase 0) and
  `E[r,r]` (value = -Δ) to the engine; dropping `E[r,1]:y` pins the drive
  phase to 0.
- Nondimensionalization: the study scales the exported matrices from rad/s to
  rad/µs (×10⁻⁶) and works in µs throughout (both direct and GRAPE), matching
  the paper's natural units so its penalty weights carry over and the NLP is
  well-scaled.

### Target and conventions

- `U_target = expm(-i·0.8·Z₁X₂Z₃)` on the 8-dim basis, with
  `Z = |1⟩⟨1| - |r⟩⟨r|` (paper's Z = 1 - 2n_r) and `X = |1⟩⟨r| + |r⟩⟨1|`,
  built in the exact basis ordering of the bilinear export (pinned by test).
- Objective: `L = 1 - |Tr(U_target† U_N)|² / D²`; gradient callback
  `G = -(Tr(U_target† U_N)/D²)·U_target`.

### Hardware-constraint constants (Aquila, in rad/µs and µs)

- `dt = 0.05`; Pulse 1: `T = 1.2` (K = 24); Pulse 2: `T = 3.6` (K = 72).
- Ω/2π ∈ [0, 2.4 MHz] → channel `u_Ω = Ω/2 ∈ [0, 7.5398]` rad/µs;
  Ω slew ≤ 250 rad/µs² → `du_Ω ≤ 125` rad/µs².
- Δ/2π ∈ [-20, 20 MHz] → `u_Δ = -Δ ∈ [-125.66, 125.66]` rad/µs;
  Δ slew ≤ 2500 rad/µs² → `du_Δ ≤ 2500` rad/µs².
- `ddu` default = `du_max/dt` per channel (loosest finite-curvature bound;
  tightening it is the study's smoothness knob and is recorded per run).
- Endpoints fixed to zero (paper convention).

### Direct runs

Per duration (Pulse 1, Pulse 2): 8 restarts from smooth random feasible
starts — interior knot values drawn uniform in the bounds, endpoints 0,
smoothed by one window-3 moving-average pass; `u̇, ü` from finite differences;
unitaries from the forward rollout — keep the best accepted result. Seeds
fixed (`numpy.random.default_rng(seed)` with recorded seeds 0..7).
Acceptance: best unitary fidelity ≥ 0.85 (Pulse 1) and ≥ 0.90 (Pulse 2) —
paper values 0.894 / 0.945 minus reproduction slack; report the achieved
numbers alongside.

### GRAPE comparison (existing `qoc.grape`, ADR-0024/0025 wiring)

- Parameters: interior knot values per channel (23 for K=24), piecewise-linear
  waveform; `control_map` averages adjacent knots onto ZOH slices (midpoint
  sampling, consistent with the direct engine's default), pullback distributes
  accordingly.
- Loss (paper's): `1 - |Tr|²/D² + λ·Σ g_k + r·⟨(d²u/dt²)²⟩` with quadratic
  bound-violation penalties `g_k` on values and first differences, λ = 100,
  r ∈ {0, 1e-8, 1e-7, 1e-6}; optimizer `qoc.minimize(..., method="l-bfgs-b")`
  with the caller gradient routed per ADR-0025 (the physics gradient from
  `grape.value_and_grad`, penalty gradients added by the study). The optimizer
  differs from the paper's Adam; the comparison is about the landscape, and
  L-BFGS-B is the repo seam — stated in the study header.
- 100 seeds per r at T = 1.2 µs (D=8: cheap). Per-seed final fidelities +
  best pulses → npz.
- Comparison claim to reproduce: direct's best beats the GRAPE distribution
  median across all r by a wide margin (paper Fig. 3b).

### Independent validation (cz_grape_e2e_validation discipline)

Replay the winning pulses through `simulate(..., backend="exact_ode")` on the
same 1r system with `SweepProtocol`:

1. ZOH replay at the engine's slice values → discrete↔continuous parity gate:
   |F_ode - F_discrete| < 1e-3.
2. Piecewise-linear replay (the physical waveform) → reported as the
   experiment-faithful fidelity next to the knot-model fidelity.

### Outputs

`results/zxz_direct_qoc/`: one npz per direct run (controls, fidelity,
IPOPT diagnostics, seed), one npz for the GRAPE distribution, and the
Fig.-3-style comparison figure (violin + pulse traces) as png+pdf. Flat
line-by-line script, npz-first so plots replay without recompute (notebook
migration is a later step, per the user's notebook conventions).

## 5. Tests

`tests/test_qoc_direct.py` (physics-free):

- Defect + chain constraint Jacobians vs central finite differences on random
  Hermitian models (D=2, K=3, two channels, both slice samplings; tol 1e-6).
- Terminal-gradient plumbing: engine gradient of the objective w.r.t.
  `isovec(U_K)` vs finite differences through a quadratic test objective.
- Known-solvable synthesis: single qubit, `h0 = 0.1·Z`, one X-drive channel,
  target `expm(-i·π/2·X)` reachable within bounds → `accepted=True`,
  fidelity > 0.999.
- Bounds respected at the solution (u within bounds, endpoints 0, |u̇|, |ü|
  within bounds when set).
- Infeasible start: `initial_unitaries = identity stack` (violates defects)
  still converges on the known-solvable problem.
- Validation errors: non-Hermitian input, channel mismatch between `controls`
  and `u_bounds`, infinite bound, K < 2.

`tests/test_zxz_direct_qoc.py` (study pins, fast):

- Basis/sign pin: build the study system, constant controls
  (Ω/2 = 1 rad/µs, Δ = 0.7 rad/µs) for 0.1 µs; propagate the sliced 8-dim
  bilinear model and match `simulate` state fidelity to 1e-9 (constant-control
  parity, ADR-0024 style, now for `SweepProtocol`/1r).
- `Z₁X₂Z₃` construction matches an explicitly hand-written 8×8 matrix.
- V_NN sanity: ARC C6 at n=70 gives V_NN/2π within 2% of 1.74 MHz
  (slow-marked if ARC download is needed; uses the memoized params pattern).

No changes to any sweeplib compat locks (untouched subsystems).

## Acceptance criteria (project level)

1. Engine test file green locally and on the DGX suite.
2. Study produces accepted direct pulses with fidelity ≥ 0.85 (T=1.2 µs) and
   ≥ 0.90 (T=3.6 µs); exact-ODE parity gate passes.
3. GRAPE comparison npz shows the direct result above the GRAPE median for
   every r.
4. ADR-0026 committed; `uv sync` with the new extra works on the DGX.

## Out of scope (recorded)

Free-time optimization (Δt as variable, ADR-0005 future), state-trajectory
variant, matrix-free augmented-Lagrangian solver, the 8-atom experimental
pulse family (τ = 0.1..0.8 continuation), neural-network gate-family
interpolation, exposing the engine to any non-bilinear Hamiltonian structure.

## Runbook notes

All runs/tests on the DGX (`ssh chance@100.106.69.117`, `uv run --extra dev
--extra qoc-direct ...`); ARC-touching steps use `HOME=/tmp/arc297home`; the
study itself is single-process (no fork-safety concerns beyond the memoized
ARC params already in place).
