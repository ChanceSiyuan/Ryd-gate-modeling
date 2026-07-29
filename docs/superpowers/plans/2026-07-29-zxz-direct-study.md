# ZXZ Direct-QOC Study (Fig. 3 Reproduction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the deferred ZXZ reproduction study of
`docs/superpowers/specs/2026-07-28-direct-qoc-zxz-design.md` §3–§5:
`scripts/zxz_direct_qoc.py` (direct pulses + GRAPE comparison + exact-ODE
validation + Fig.-3-style plots) and its test pins. The compute campaign
itself runs after the tasks, driven by the controller.

**Architecture:** One flat study script with argparse subcommands
(`model-check`, `direct`, `grape`, `validate`, `plot`), npz-first so plots
replay without recompute. Physics enters ONLY through existing exports
(`SweepProtocol`, `level_structure("1r")`, `Register.chain`,
`bilinear_control_model`, `simulate`); optimization only through
`qoc.direct.optimize` and `qoc.grape` + `qoc.minimize`. Working units:
**rad/µs and µs** everywhere in the study (SI conversion only at the
`simulate` replay boundary).

**Tech Stack:** numpy, scipy, matplotlib, qoc (direct+grape), ryd_gate,
concurrent.futures (process pool over seeds).

## Global Constraints

- Only these files may change: `scripts/zxz_direct_qoc.py` (new),
  `tests/test_zxz_direct_qoc.py` (new). `results/zxz_direct_qoc/` is created
  at runtime and never committed by tasks. No edits to `src/`, `pyproject`,
  or any existing test.
- The checkout at `/home/chance/dgx/Ryd-gate-modeling` is an sshfs mount of
  the DGX. **Edit files via local paths**; **run ALL git/pytest/python
  commands over ssh**:
  `ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && <cmd>'`
  (git on the mount hangs). Any command that builds the model touches ARC →
  prefix the remote command with `HOME=/tmp/arc297home` (sqlite isolation).
- Work on `main`; NEVER push. Commit trailer:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- ARC-touching tests carry `@pytest.mark.slow` (repo convention); pure-math
  pins stay fast.
- Physical constants are FIXED by the spec (§3, Aquila, rad/µs & µs):
  `DT_US = 0.05`; durations 1.2 µs (K=24) and 3.6 µs (K=72);
  Ω/2π ∈ [0, 2.4 MHz] → channel `u_Ω = Ω/2 ∈ [0, 2π·1.2]`; Ω slew 250
  rad/µs² → channel slew 125; Δ/2π ∈ [−20, 20] → channel `u_Δ = −Δ ∈
  [−2π·20, 2π·20]`, slew 2500; `ddu` default = channel slew / DT_US;
  endpoints fixed 0; target `U_target = expm(−i·0.8·Z₁X₂Z₃)`,
  Z = |1⟩⟨1| − |r⟩⟨r|; fidelity `F = |Tr(U_target† U)|²/64`.
- System: `Register.chain(3, spacing_um=8.9)`,
  `level_structure("1r", ryd_level=70)`, `SweepProtocol` reference with zero
  drives, `bilinear_control_model(system, states=<all 8 product labels>)`;
  study channels are `E[r,1]:x` (= Ω/2) and `E[r,r]` (= −Δ); `E[r,1]:y` is
  dropped (drive phase pinned to 0). Convert the exported `h0` from rad/s to
  rad/µs (×1e-6); control OPERATORS are unchanged (the channel VALUE carries
  the rad/µs unit).

---

## File Structure

- `scripts/zxz_direct_qoc.py` — constants; `build_model()`; `build_zxz(index)`
  + `build_target(index)`; `unitary_infidelity`; direct/grape worker
  functions (top-level, picklable); replay helpers; five subcommands.
- `tests/test_zxz_direct_qoc.py` — fast ZXZ-matrix pin; slow ARC pin
  (V_NN sign+magnitude, basis bijectivity, constant-control parity vs
  `simulate`).

---

### Task 1: Model, target, objective, pins (`model-check`)

**Files:**
- Create: `scripts/zxz_direct_qoc.py`
- Create: `tests/test_zxz_direct_qoc.py`

**Interfaces:**
- Produces (later tasks rely on): module-level `TAU, RYD_LEVEL=70,
  SPACING_UM=8.9, N_ATOMS=3, DT_US=0.05, U_OMEGA_MAX, DELTA_MAX,
  SLEW_U_OMEGA=125.0, SLEW_U_DELTA=2500.0, TAU_JEFF=0.8,
  DURATIONS={"pulse1": 1.2, "pulse2": 3.6}, CH_OM="E[r,1]:x",
  CH_DE="E[r,r]", RESULTS_DIR=Path("results/zxz_direct_qoc"), LABELS`;
  functions `build_model() -> dict(h0, controls, index, labels)`,
  `build_zxz(index) -> (8,8) ndarray`, `build_target(index) -> (8,8)`,
  `unitary_infidelity(u_final, target) -> (float, (8,8) ndarray)`,
  `fidelity(u, target) -> float`, and a `main()` argparse scaffold with the
  `model-check` subcommand registered.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_zxz_direct_qoc.py`:

```python
"""Pins for the ZXZ direct-QOC study (spec 2026-07-28 §4-§5)."""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from zxz_direct_qoc import (  # noqa: E402
    DT_US,
    TAU,
    TAU_JEFF,
    build_model,
    build_target,
    build_zxz,
    fidelity,
    unitary_infidelity,
)

_Z = np.diag([1.0, -1.0]).astype(complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def _lex_index():
    labels = [tuple(p) for p in product("1r", repeat=3)]
    return {lab: i for i, lab in enumerate(labels)}


def test_zxz_matrix_pin():
    # site 0 most significant in the lexicographic ordering -> Z (x) X (x) Z
    index = _lex_index()
    expected = np.kron(np.kron(_Z, _X), _Z)
    np.testing.assert_allclose(build_zxz(index), expected, atol=1e-14)
    target = build_target(index)
    np.testing.assert_allclose(target, expm(-1j * TAU_JEFF * expected), atol=1e-12)


def test_unitary_infidelity_gradient_convention():
    rng = np.random.default_rng(4)
    target = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))[0]
    u = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))[0]
    value, g = unitary_infidelity(u, target)
    assert value == pytest.approx(1.0 - fidelity(u, target))
    h = 1e-7
    for i, j in ((0, 0), (2, 5)):
        e = np.zeros((8, 8), dtype=complex)
        e[i, j] = h
        fd_re = (unitary_infidelity(u + e, target)[0] - unitary_infidelity(u - e, target)[0]) / (2 * h)
        fd_im = (unitary_infidelity(u + 1j * e, target)[0] - unitary_infidelity(u - 1j * e, target)[0]) / (2 * h)
        assert abs(2.0 * g[i, j].real - fd_re) < 1e-6
        assert abs(2.0 * g[i, j].imag - fd_im) < 1e-6


@pytest.mark.slow
def test_model_pins_arc():
    from ryd_gate import Register, RydbergSystem, level_structure, simulate
    from ryd_gate.protocols import SweepProtocol

    model = build_model()
    index = model["index"]
    # bijective basis mapping
    assert sorted(index.values()) == list(range(8))
    # NN vdW: +C6/8.9^6 with ARC 70S C6 ~ 2pi x 862.7 GHz um^6 -> ~2pi x 1.736 MHz
    i_rr1 = index[("r", "r", "1")]
    i_r1r = index[("r", "1", "r")]
    v_nn = model["h0"][i_rr1, i_rr1].real
    v_nnn = model["h0"][i_r1r, i_r1r].real
    assert v_nn > 0.0, "repulsive S-state vdW expected; sign convention broke"
    assert abs(v_nn / TAU - 1.736) < 0.04
    assert abs(v_nnn - v_nn / 64.0) < 1e-6 * v_nn
    # constant-control parity: discrete chain vs exact_ode (ADR-0024 style)
    u_om, u_de, t_us = 1.0, 0.7, 0.1
    h = model["h0"] + u_om * model["controls"]["E[r,1]:x"] + u_de * model["controls"]["E[r,r]"]
    psi0 = np.zeros(8, dtype=complex)
    psi0[index[("1", "1", "1")]] = 1.0
    psi_disc = expm(-1j * t_us * h) @ psi0
    protocol = SweepProtocol(
        t_gate_s=t_us * 1e-6,
        omega_half_rad_s=lambda t: u_om * 1e6,
        detuning_rad_s=lambda t: -u_de * 1e6,
    )
    system = RydbergSystem(
        level_structure=level_structure("1r", ryd_level=70),
        register=Register.chain(3, spacing_um=8.9),
        protocol=protocol,
    )
    result = simulate(system, ["1", "1", "1"], backend="exact_ode")
    psi_ode = np.array(
        [result.amplitude(list(lab)) for lab, _ in sorted(index.items(), key=lambda kv: kv[1])]
    )
    overlap = abs(np.vdot(psi_disc, psi_ode))
    assert overlap > 1.0 - 1e-9
```

- [ ] **Step 2: Run tests to verify they fail on import**

Run:
`ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && HOME=/tmp/arc297home uv run --extra dev --extra qoc-direct pytest tests/test_zxz_direct_qoc.py -q'`
Expected: collection error — `ModuleNotFoundError: No module named 'zxz_direct_qoc'`.

- [ ] **Step 3: Implement the study module core**

Create `scripts/zxz_direct_qoc.py`:

```python
"""ZXZ synthesis study (arXiv:2508.19075 Fig. 3 reproduction; spec 2026-07-28 §3-§4).

Direct quantum optimal control (qoc.direct/IPOPT) vs GRAPE (qoc.grape +
qoc.minimize) on the 3-atom 1r analog chain: synthesize
U_target = expm(-i*0.8*Z1 X2 Z3) under Aquila hardware constraints.

Units: rad/us and us everywhere; SI (rad/s, s) only at the simulate boundary.
Subcommands: model-check | direct | grape | validate | plot. All artifacts go
to results/zxz_direct_qoc/ as npz so plotting replays without recompute.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
from scipy.linalg import expm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

TAU = 2.0 * np.pi
RYD_LEVEL = 70
SPACING_UM = 8.9
N_ATOMS = 3
DT_US = 0.05
U_OMEGA_MAX = TAU * 2.4 / 2.0        # channel value = Omega/2      (rad/us)
DELTA_MAX = TAU * 20.0               # |channel value| = |Delta|    (rad/us)
SLEW_U_OMEGA = 250.0 / 2.0           # Omega slew 250 rad/us^2 -> channel
SLEW_U_DELTA = 2500.0
TAU_JEFF = 0.8
DURATIONS = {"pulse1": 1.2, "pulse2": 3.6}
CH_OM = "E[r,1]:x"
CH_DE = "E[r,r]"
LABELS = [tuple(p) for p in product("1r", repeat=N_ATOMS)]
RESULTS_DIR = REPO_ROOT / "results" / "zxz_direct_qoc"


def build_model():
    """8-dim bilinear model (rad/us) of the 3-atom 1r chain + basis index map."""
    from ryd_gate import Register, RydbergSystem, bilinear_control_model, level_structure
    from ryd_gate.protocols import SweepProtocol

    reference = SweepProtocol(
        t_gate_s=1e-6,
        omega_half_rad_s=lambda t: 0.0,
        detuning_rad_s=lambda t: 0.0,
    )
    system = RydbergSystem(
        level_structure=level_structure("1r", ryd_level=RYD_LEVEL),
        register=Register.chain(N_ATOMS, spacing_um=SPACING_UM),
        protocol=reference,
    )
    h0_si, channels, states = bilinear_control_model(system, states=[list(l) for l in LABELS])
    index = {}
    for lab in LABELS:
        vec = states[lab]
        idx = int(np.argmax(np.abs(vec)))
        if abs(abs(vec[idx]) - 1.0) > 1e-12:
            raise RuntimeError(f"product state {lab} is not a coordinate vector")
        index[lab] = idx
    return {
        "h0": np.asarray(h0_si, dtype=complex) * 1e-6,   # rad/s -> rad/us
        "controls": {CH_OM: np.asarray(channels[CH_OM]), CH_DE: np.asarray(channels[CH_DE])},
        "index": index,
        "labels": LABELS,
    }


def build_zxz(index):
    """Z1 X2 Z3 in the basis ordering given by index (Z = |1><1| - |r><r|)."""
    op = np.zeros((8, 8), dtype=complex)
    for lab, col in index.items():
        flipped = (lab[0], "r" if lab[1] == "1" else "1", lab[2])
        sign = (1.0 if lab[0] == "1" else -1.0) * (1.0 if lab[2] == "1" else -1.0)
        op[index[flipped], col] = sign
    return op


def build_target(index):
    return expm(-1j * TAU_JEFF * build_zxz(index))


def fidelity(u, target):
    return float(abs(np.trace(target.conj().T @ u)) ** 2) / target.shape[0] ** 2


def unitary_infidelity(u_final, target):
    """(1 - F, G) with G = dL/d(conj U) per the qoc costate convention."""
    d2 = target.shape[0] ** 2
    c = np.trace(target.conj().T @ u_final)
    return 1.0 - float(abs(c) ** 2) / d2, -(c / d2) * target


U_BOUNDS = {CH_OM: (0.0, U_OMEGA_MAX), CH_DE: (-DELTA_MAX, DELTA_MAX)}
DU_BOUNDS = {CH_OM: SLEW_U_OMEGA, CH_DE: SLEW_U_DELTA}
DDU_BOUNDS = {CH_OM: SLEW_U_OMEGA / DT_US, CH_DE: SLEW_U_DELTA / DT_US}


def cmd_model_check(args):
    model = build_model()
    index = model["index"]
    v_nn = model["h0"][index[("r", "r", "1")], index[("r", "r", "1")]].real
    target = build_target(index)
    print(f"basis order: {[lab for lab, _ in sorted(index.items(), key=lambda kv: kv[1])]}")
    print(f"V_NN/2pi = {v_nn / TAU:.4f} MHz (expect ~ +1.736)")
    print(f"F(identity) = {fidelity(np.eye(8, dtype=complex), target):.6f}")
    print(f"channels: {list(model['controls'])}, dim = {model['h0'].shape[0]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("model-check").set_defaults(func=cmd_model_check)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run (same command as Step 2). Expected: 2 fast tests PASS, slow test
deselected by default config? No — this file is run directly, so the slow
test RUNS here (ARC, ~1-2 min on first C6 evaluation): expect **3 passed**.
If `v_nn` fails the sign or magnitude assert, STOP and report the measured
value (do not adjust the constant — the spec pins +1.736; a sign flip means
the repo C6 convention differs and the controller must rule).

- [ ] **Step 5: Run `model-check`**

Run:
`ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && HOME=/tmp/arc297home uv run --extra dev --extra qoc-direct python scripts/zxz_direct_qoc.py model-check'`
Expected: prints basis order, `V_NN/2pi ~ 1.73-1.74 MHz`, `F(identity) =
0.485542` (ZXZ has eigenvalues ±1 with multiplicity 4, so Tr U_target =
8·cos(0.8) = 5.5738 and F = 5.5738²/64 = 0.4855), channels list.

- [ ] **Step 6: ruff + commit**

`ssh ... 'uv run --extra dev ruff check scripts/zxz_direct_qoc.py tests/test_zxz_direct_qoc.py'` → clean, then:

```bash
ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && git add scripts/zxz_direct_qoc.py tests/test_zxz_direct_qoc.py && git commit -m "Add ZXZ study core: model wiring, target, objective, pins

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"'
```

---

### Task 2: `direct` subcommand (IPOPT pulse search)

**Files:**
- Modify: `scripts/zxz_direct_qoc.py`

**Interfaces:**
- Consumes: Task 1 constants/functions verbatim.
- Produces: `run_direct_seed(seed, tag, duration_us, maxiter, model_arrays)`
  (top-level worker) and subcommand
  `direct --tag {pulse1,pulse2,all} --seeds N --workers W --maxiter M`;
  npz per run `RESULTS_DIR/direct_<tag>_seed<seed>.npz` with fields
  `u_omega, u_delta, du_omega, du_delta, ddu_omega, ddu_delta, fidelity,
  objective, max_defect, accepted, ipopt_status, n_iter, seed, K, dt_us,
  duration_us`; summary `RESULTS_DIR/direct_<tag>_summary.json` (per-seed
  fidelity/accepted/n_iter + best seed).

- [ ] **Step 1: Implement the worker and subcommand**

Append to `scripts/zxz_direct_qoc.py` (before `main`), and register the
subcommand inside `main` (`p = sub.add_parser("direct"); p.add_argument("--tag", default="all"); p.add_argument("--seeds", type=int, default=8); p.add_argument("--workers", type=int, default=8); p.add_argument("--maxiter", type=int, default=4000); p.set_defaults(func=cmd_direct)`):

```python
def _smooth_random_knots(rng, k, lo, hi):
    u = rng.uniform(lo, hi, k + 1)
    u[0] = u[-1] = 0.0
    u = np.convolve(u, np.ones(3) / 3.0, mode="same")
    u[0] = u[-1] = 0.0
    return np.clip(u, lo, hi)


def run_direct_seed(seed, tag, duration_us, maxiter, model_arrays):
    """One IPOPT solve from one smooth random start. Top-level: picklable."""
    from qoc import direct

    h0, ops_om, ops_de, target = model_arrays
    controls = {CH_OM: ops_om, CH_DE: ops_de}
    k = int(round(duration_us / DT_US))
    rng = np.random.default_rng(seed)
    initial = {
        CH_OM: _smooth_random_knots(rng, k, 0.0, U_OMEGA_MAX),
        CH_DE: _smooth_random_knots(rng, k, -DELTA_MAX, DELTA_MAX),
    }
    result = direct.optimize(
        h0,
        controls,
        n_slices=k,
        dt=DT_US,
        terminal_objective=partial(unitary_infidelity, target=target),
        u_bounds=U_BOUNDS,
        du_bounds=DU_BOUNDS,
        ddu_bounds=DDU_BOUNDS,
        initial_controls=initial,
        maxiter=maxiter,
    )
    out = RESULTS_DIR / f"direct_{tag}_seed{seed}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        u_omega=result.controls[CH_OM],
        u_delta=result.controls[CH_DE],
        du_omega=result.du[CH_OM],
        du_delta=result.du[CH_DE],
        ddu_omega=result.ddu[CH_OM],
        ddu_delta=result.ddu[CH_DE],
        fidelity=1.0 - result.objective,
        objective=result.objective,
        max_defect=result.max_defect,
        accepted=result.accepted,
        ipopt_status=result.ipopt_status,
        n_iter=result.n_iter,
        seed=seed,
        K=k,
        dt_us=DT_US,
        duration_us=duration_us,
    )
    return {
        "seed": seed,
        "fidelity": 1.0 - result.objective,
        "accepted": bool(result.accepted),
        "ipopt_status": int(result.ipopt_status),
        "n_iter": int(result.n_iter),
    }


def cmd_direct(args):
    model = build_model()
    target = build_target(model["index"])
    model_arrays = (model["h0"], model["controls"][CH_OM], model["controls"][CH_DE], target)
    tags = list(DURATIONS) if args.tag == "all" else [args.tag]
    for tag in tags:
        duration = DURATIONS[tag]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(run_direct_seed, seed, tag, duration, args.maxiter, model_arrays)
                for seed in range(args.seeds)
            ]
            for fut in futures:
                row = fut.result()
                rows.append(row)
                print(f"[{tag}] seed {row['seed']}: F={row['fidelity']:.4f} "
                      f"accepted={row['accepted']} status={row['ipopt_status']} it={row['n_iter']}")
        accepted = [r for r in rows if r["accepted"]]
        best = max(accepted or rows, key=lambda r: r["fidelity"])
        summary = {"tag": tag, "duration_us": duration, "runs": rows, "best": best}
        (RESULTS_DIR / f"direct_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[{tag}] best: seed {best['seed']} F={best['fidelity']:.4f} "
              f"({len(accepted)}/{len(rows)} accepted)")
```

- [ ] **Step 2: Smoke run (1 seed, small maxiter, pulse1)**

Run:
`ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && HOME=/tmp/arc297home OMP_NUM_THREADS=1 uv run --extra dev --extra qoc-direct python scripts/zxz_direct_qoc.py direct --tag pulse1 --seeds 1 --workers 1 --maxiter 300'`
Expected: one `[pulse1] seed 0: F=...` line (any F; accepted may be False at
maxiter 300 — that is fine for smoke), `direct_pulse1_seed0.npz` +
`direct_pulse1_summary.json` created. Report the smoke F and wall time.

- [ ] **Step 3: ruff + commit**

`ruff check scripts/zxz_direct_qoc.py` clean, then commit
`"Add ZXZ study direct subcommand (parallel IPOPT seeds)"` with trailer.
Do NOT commit anything under `results/`.

---

### Task 3: `grape` subcommand (penalized L-BFGS-B baseline)

**Files:**
- Modify: `scripts/zxz_direct_qoc.py`

**Interfaces:**
- Consumes: Task 1 core; `qoc.grape.value_and_grad`, `qoc.minimize`
  (ADR-0025 caller gradient via `options={"gradient": ...}`).
- Produces: `run_grape_seed(seed, r_weight, k, model_arrays, maxiter)`
  (top-level worker) and subcommand
  `grape --duration-us 1.2 --seeds N --workers W --maxiter M --r-values a,b,c`;
  one npz `RESULTS_DIR/grape_T<duration>.npz` with fields
  `fidelities (R,S), final_losses (R,S), max_violation (R,S),
  u_omega (R,S,K+1), u_delta (R,S,K+1), r_values (R,), seeds (S,),
  K, dt_us, duration_us, lambda_penalty`.

GRAPE setup (paper appendix, mapped to our units): parameters are the
interior knot values (K−1 per channel, endpoints fixed 0) of a
piecewise-linear waveform; slices take midpoint values (consistent with the
direct engine); loss = unitary infidelity + λ·(bound violations² + slew
violations²) + r·mean(second differences²), λ = 100, penalties only — NO
hard bounds (paper's GRAPE has no bound-native optimizer, that is the point
of the comparison). Init: u_Ω ~ U(0, U_OMEGA_MAX), u_Δ ~ U(−2π·5, 2π·5)
(paper's b = 5 MHz init spread).

- [ ] **Step 1: Implement worker + subcommand**

Append (and register in `main`:
`p = sub.add_parser("grape"); p.add_argument("--duration-us", type=float, default=1.2); p.add_argument("--seeds", type=int, default=100); p.add_argument("--workers", type=int, default=16); p.add_argument("--maxiter", type=int, default=1500); p.add_argument("--r-values", default="0,1e-8,1e-7,1e-6"); p.set_defaults(func=cmd_grape)`):

```python
def _knots_from_params(named, k):
    om = np.concatenate([[0.0], np.asarray(named["omega"], dtype=float), [0.0]])
    de = np.concatenate([[0.0], np.asarray(named["delta"], dtype=float), [0.0]])
    return om, de


def _penalty_and_grad(knots, lo, hi, slew_max):
    """lambda-weighted quadratic penalties on values and knot slopes."""
    value = 0.0
    grad = np.zeros_like(knots)
    over = np.maximum(0.0, knots - hi)
    under = np.maximum(0.0, lo - knots)
    value += float(over @ over + under @ under)
    grad += 2.0 * over - 2.0 * under
    slopes = np.diff(knots) / DT_US
    excess = np.maximum(0.0, np.abs(slopes) - slew_max)
    value += float(excess @ excess)
    d_slope = 2.0 * excess * np.sign(slopes) / DT_US
    grad[1:] += d_slope
    grad[:-1] -= d_slope
    return value, grad


def _smoothness_and_grad(knots):
    """mean of squared second differences (rad/us^3 units)."""
    d2 = (knots[:-2] - 2.0 * knots[1:-1] + knots[2:]) / DT_US**2
    n = max(d2.size, 1)
    value = float(d2 @ d2) / n
    grad = np.zeros_like(knots)
    coeff = 2.0 * d2 / (n * DT_US**2)
    grad[:-2] += coeff
    grad[1:-1] += -2.0 * coeff
    grad[2:] += coeff
    return value, grad


def run_grape_seed(seed, r_weight, k, model_arrays, maxiter):
    """One penalized L-BFGS-B GRAPE run. Top-level: picklable."""
    import qoc
    from qoc import grape

    h0, ops_om, ops_de, target = model_arrays
    controls = {CH_OM: ops_om, CH_DE: ops_de}
    time_grid = np.linspace(0.0, k * DT_US, k + 1)
    basis_states = [np.eye(8, dtype=complex)[:, j] for j in range(8)]
    lam = 100.0

    def terminal_objective(final_states):
        u = np.column_stack(final_states)
        value, g = unitary_infidelity(u, target)
        return value, [np.array(g[:, j]) for j in range(8)]

    def control_map(named):
        om, de = _knots_from_params(named, k)
        return {CH_OM: 0.5 * (om[:-1] + om[1:]), CH_DE: 0.5 * (de[:-1] + de[1:])}

    def control_pullback(named, channel_gradients):
        out = {}
        for name, key in ((CH_OM, "omega"), (CH_DE, "delta")):
            g = np.asarray(channel_gradients[name], dtype=float)
            knots = np.zeros(k + 1)
            knots[:-1] += 0.5 * g
            knots[1:] += 0.5 * g
            out[key] = knots[1:-1]
        return out

    def full_loss_and_grad(named):
        fid_value, fid_grad = grape.value_and_grad(
            named, h0=h0, controls=controls, initial_states=basis_states,
            time_grid=time_grid, control_map=control_map,
            control_pullback=control_pullback, terminal_objective=terminal_objective,
        )
        value = fid_value
        grads = {key: np.asarray(fid_grad[key], dtype=float).copy() for key in ("omega", "delta")}
        om, de = _knots_from_params(named, k)
        for knots, key, lo, hi, slew in (
            (om, "omega", 0.0, U_OMEGA_MAX, SLEW_U_OMEGA),
            (de, "delta", -DELTA_MAX, DELTA_MAX, SLEW_U_DELTA),
        ):
            p_val, p_grad = _penalty_and_grad(knots, lo, hi, slew)
            s_val, s_grad = _smoothness_and_grad(knots)
            value += lam * p_val + r_weight * s_val
            grads[key] += lam * p_grad[1:-1] + r_weight * s_grad[1:-1]
        return value, grads

    cache = {}

    def loss(named):
        key = (np.asarray(named["omega"]).tobytes(), np.asarray(named["delta"]).tobytes())
        if key not in cache:
            cache.clear()
            cache[key] = full_loss_and_grad(named)
        return cache[key][0]

    def gradient(named):
        loss(named)
        key = (np.asarray(named["omega"]).tobytes(), np.asarray(named["delta"]).tobytes())
        return cache[key][1]

    rng = np.random.default_rng(seed)
    x0 = {
        "omega": rng.uniform(0.0, U_OMEGA_MAX, k - 1),
        "delta": rng.uniform(-TAU * 5.0, TAU * 5.0, k - 1),
    }
    result = qoc.minimize(
        loss, x0, method="l-bfgs-b",
        scales={"omega": U_OMEGA_MAX, "delta": DELTA_MAX},
        options={"gradient": gradient, "maxiter": maxiter},
    )
    om, de = _knots_from_params(result.best_parameters, k)
    fid_value = grape.value(
        result.best_parameters, h0=h0, controls=controls, initial_states=basis_states,
        time_grid=time_grid, control_map=control_map, terminal_objective=terminal_objective,
    )
    viol = max(
        float(np.max(np.maximum(0.0, om - U_OMEGA_MAX))),
        float(np.max(np.maximum(0.0, -om))),
        float(np.max(np.maximum(0.0, np.abs(de) - DELTA_MAX))),
        float(np.max(np.maximum(0.0, np.abs(np.diff(om)) / DT_US - SLEW_U_OMEGA))),
        float(np.max(np.maximum(0.0, np.abs(np.diff(de)) / DT_US - SLEW_U_DELTA))),
    )
    return {
        "seed": seed, "fidelity": 1.0 - fid_value, "final_loss": result.best_loss,
        "max_violation": viol, "u_omega": om, "u_delta": de,
    }


def cmd_grape(args):
    model = build_model()
    target = build_target(model["index"])
    model_arrays = (model["h0"], model["controls"][CH_OM], model["controls"][CH_DE], target)
    k = int(round(args.duration_us / DT_US))
    r_values = [float(v) for v in args.r_values.split(",")]
    seeds = list(range(args.seeds))
    fids = np.zeros((len(r_values), len(seeds)))
    losses = np.zeros_like(fids)
    viols = np.zeros_like(fids)
    om_all = np.zeros((len(r_values), len(seeds), k + 1))
    de_all = np.zeros_like(om_all)
    for i, r_weight in enumerate(r_values):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(run_grape_seed, seed, r_weight, k, model_arrays, args.maxiter)
                for seed in seeds
            ]
            for j, fut in enumerate(futures):
                row = fut.result()
                fids[i, j] = row["fidelity"]
                losses[i, j] = row["final_loss"]
                viols[i, j] = row["max_violation"]
                om_all[i, j] = row["u_omega"]
                de_all[i, j] = row["u_delta"]
        print(f"[grape r={r_weight:g}] median F={np.median(fids[i]):.4f} "
              f"best F={np.max(fids[i]):.4f} max_viol={np.max(viols[i]):.3g}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        RESULTS_DIR / f"grape_T{args.duration_us}.npz",
        fidelities=fids, final_losses=losses, max_violation=viols,
        u_omega=om_all, u_delta=de_all, r_values=np.asarray(r_values),
        seeds=np.asarray(seeds), K=k, dt_us=DT_US, duration_us=args.duration_us,
        lambda_penalty=100.0,
    )
```

- [ ] **Step 2: Smoke run (2 seeds × 1 r)**

Run:
`ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && HOME=/tmp/arc297home OMP_NUM_THREADS=1 uv run --extra dev --extra qoc-direct python scripts/zxz_direct_qoc.py grape --duration-us 1.2 --seeds 2 --workers 2 --maxiter 200 --r-values 1e-7'`
Expected: one `[grape r=1e-07] median F=...` line, `grape_T1.2.npz` written.
Report smoke F values and wall time per solve.

- [ ] **Step 3: ruff + commit**

Commit `"Add ZXZ study grape subcommand (penalized L-BFGS-B baseline)"`.

---

### Task 4: `validate` + `plot` subcommands

**Files:**
- Modify: `scripts/zxz_direct_qoc.py`

**Interfaces:**
- Consumes: npz layouts from Tasks 2-3.
- Produces: subcommands
  `validate --tag pulse1 --seed <best>` (exact-ODE replay; writes
  `RESULTS_DIR/validate_<tag>.npz` with `f_discrete, f_ode_zoh,
  f_ode_linear, gate_pass, seed`) and
  `plot --duration-us 1.2` (writes `RESULTS_DIR/plots/fig3b_violin.png/.pdf`
  and `fig3c_pulses.png/.pdf`).

- [ ] **Step 1: Implement `validate`**

Append (register: `p = sub.add_parser("validate"); p.add_argument("--tag", required=True); p.add_argument("--seed", type=int, default=None); p.set_defaults(func=cmd_validate)`):

```python
def _replay_unitary(model, u_om_knots, u_de_knots, k, mode):
    """Exact-ODE propagator of the knot waveform (mode: 'zoh'|'linear')."""
    from ryd_gate import Register, RydbergSystem, level_structure, simulate
    from ryd_gate.protocols import SweepProtocol

    t_total_us = k * DT_US
    knots_t = np.arange(k + 1) * DT_US
    mid_om = 0.5 * (u_om_knots[:-1] + u_om_knots[1:])
    mid_de = 0.5 * (u_de_knots[:-1] + u_de_knots[1:])

    def value_at(t_s, knots, mids):
        t_us = min(max(t_s * 1e6, 0.0), t_total_us - 1e-12)
        if mode == "zoh":
            return float(mids[int(t_us / DT_US)])
        return float(np.interp(t_us, knots_t, knots))

    protocol = SweepProtocol(
        t_gate_s=t_total_us * 1e-6,
        omega_half_rad_s=lambda t: value_at(t, u_om_knots, mid_om) * 1e6,
        detuning_rad_s=lambda t: -value_at(t, u_de_knots, mid_de) * 1e6,
    )
    system = RydbergSystem(
        level_structure=level_structure("1r", ryd_level=RYD_LEVEL),
        register=Register.chain(N_ATOMS, spacing_um=SPACING_UM),
        protocol=protocol,
    )
    index = model["index"]
    ordered = [lab for lab, _ in sorted(index.items(), key=lambda kv: kv[1])]
    results = simulate(system, [list(lab) for lab in ordered], backend="exact_ode")
    u_ode = np.zeros((8, 8), dtype=complex)
    for j, res in enumerate(results):
        for i, lab in enumerate(ordered):
            u_ode[i, j] = res.amplitude(list(lab))
    return u_ode


def cmd_validate(args):
    model = build_model()
    target = build_target(model["index"])
    if args.seed is None:
        summary = json.loads((RESULTS_DIR / f"direct_{args.tag}_summary.json").read_text())
        args.seed = int(summary["best"]["seed"])
    data = np.load(RESULTS_DIR / f"direct_{args.tag}_seed{args.seed}.npz")
    k = int(data["K"])
    f_disc = float(data["fidelity"])
    u_zoh = _replay_unitary(model, data["u_omega"], data["u_delta"], k, "zoh")
    u_lin = _replay_unitary(model, data["u_omega"], data["u_delta"], k, "linear")
    f_zoh = fidelity(u_zoh, target)
    f_lin = fidelity(u_lin, target)
    gate_pass = bool(abs(f_zoh - f_disc) < 1e-3)
    np.savez(
        RESULTS_DIR / f"validate_{args.tag}.npz",
        f_discrete=f_disc, f_ode_zoh=f_zoh, f_ode_linear=f_lin,
        gate_pass=gate_pass, seed=args.seed,
    )
    print(f"[{args.tag} seed {args.seed}] F_discrete={f_disc:.5f} "
          f"F_ode_zoh={f_zoh:.5f} (gate {'PASS' if gate_pass else 'FAIL'}) "
          f"F_ode_linear={f_lin:.5f}")
```

- [ ] **Step 2: Implement `plot`**

Append (register: `p = sub.add_parser("plot"); p.add_argument("--duration-us", type=float, default=1.2); p.set_defaults(func=cmd_plot)`):

```python
def cmd_plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = RESULTS_DIR / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    g = np.load(RESULTS_DIR / f"grape_T{args.duration_us}.npz")
    fids, r_values = g["fidelities"], g["r_values"]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    parts = ax.violinplot([fids[i] for i in range(len(r_values))],
                          positions=np.arange(len(r_values)), showmedians=True)
    ax.set_xticks(np.arange(len(r_values)))
    ax.set_xticklabels([f"r={v:g}" for v in r_values])
    colors = {"pulse1": "tab:red", "pulse2": "tab:purple"}
    for tag, color in colors.items():
        path = RESULTS_DIR / f"direct_{tag}_summary.json"
        if path.exists():
            best = json.loads(path.read_text())["best"]
            ax.axhline(best["fidelity"], color=color, ls="--", lw=1.2,
                       label=f"direct {tag} (T={DURATIONS[tag]} us): F={best['fidelity']:.3f}")
    ax.set_ylabel("unitary fidelity")
    ax.set_xlabel(f"GRAPE regularization (T={args.duration_us} us, "
                  f"{fids.shape[1]} seeds)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("ZXZ synthesis: direct method vs GRAPE (Fig. 3b style)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plots / f"fig3b_violin.{ext}", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.5), sharex=True)
    best1 = json.loads((RESULTS_DIR / "direct_pulse1_summary.json").read_text())["best"]
    d = np.load(RESULTS_DIR / f"direct_pulse1_seed{best1['seed']}.npz")
    t = np.arange(int(d["K"]) + 1) * DT_US
    axes[0].plot(t, 2.0 * d["u_omega"] / TAU, label="direct pulse1")
    axes[1].plot(t, -d["u_delta"] / TAU, label="direct pulse1")
    i_best = np.unravel_index(np.argmax(fids), fids.shape)
    tg = np.arange(int(g["K"]) + 1) * DT_US
    axes[0].plot(tg, 2.0 * g["u_omega"][i_best] / TAU, alpha=0.7,
                 label=f"best GRAPE (r={r_values[i_best[0]]:g})")
    axes[1].plot(tg, -g["u_delta"][i_best] / TAU, alpha=0.7,
                 label=f"best GRAPE (r={r_values[i_best[0]]:g})")
    axes[0].set_ylabel("Omega/2pi (MHz)")
    axes[1].set_ylabel("Delta/2pi (MHz)")
    axes[1].set_xlabel("t (us)")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Optimized pulses (Fig. 3c style)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plots / f"fig3c_pulses.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {plots}/fig3b_violin.png|pdf and fig3c_pulses.png|pdf")
```

- [ ] **Step 3: Smoke verify**

Using the Task 2/3 smoke artifacts:
`... python scripts/zxz_direct_qoc.py validate --tag pulse1 --seed 0` →
prints the three fidelities (gate may FAIL on an unconverged smoke pulse —
report, don't fix), and
`... python scripts/zxz_direct_qoc.py plot --duration-us 1.2` → both plot
files exist. (Prefix both with `HOME=/tmp/arc297home`.)

- [ ] **Step 4: ruff + full test file + commit**

ruff clean; `pytest tests/test_zxz_direct_qoc.py -q` still 3 passed;
commit `"Add ZXZ study validate and plot subcommands"`.

---

## Campaign (controller-run, after all tasks)

1. Full direct: `direct --tag all --seeds 8 --workers 8 --maxiter 4000`
   (nohup, `HOME=/tmp/arc297home OMP_NUM_THREADS=1`).
2. GRAPE r-ladder calibration: 8 seeds per r on `{0,1e-8,1e-7,1e-6}` — the
   ladder must span irregular→over-smoothed pulses; shift decades if not.
3. Full GRAPE: `grape --duration-us 1.2 --seeds 100 --workers 16` with the
   calibrated ladder.
4. `validate --tag pulse1` / `--tag pulse2`; acceptance: best accepted
   direct F ≥ 0.85 (pulse1) / ≥ 0.90 (pulse2); ZOH parity gate passes;
   direct best > GRAPE median for every r.
5. `plot`; report Fig. 3b/3c to the user.
