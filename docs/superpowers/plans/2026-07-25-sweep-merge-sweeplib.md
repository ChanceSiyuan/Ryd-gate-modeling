# Single-Pass Sweep + sweeplib Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the coherent+scatter passes into one trajectory-sampled `run` pass in both sweep scripts and extract their duplicated machinery into `scripts/sweeplib/`, under byte-identical hash/store compatibility with the two live new-era stores.

**Architecture:** Compat-lock tests land first and gate every task. The ode script is refactored in three steps (core extraction → store/runner extraction → merge + deletions), then plotting/CLI extraction, then the 297 script is rebased onto sweeplib (after the in-flight 297 smoke commit), then the new campaign driver deploys. The batched DOP853 kernel stays custom; direct `ryd_gate` calls are used for all model construction and reference solves.

**Tech Stack:** Python, numpy/scipy DOP853, `ryd_gate` src API, pytest importlib script loading, ssh remote execution on the DGX.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-25-sweep-merge-sweeplib-design.md` (hash literals live there and in Task 1's tests).
- Hash locks: ode a3.0 physics `d66867f0f1f9404203933778250f859bb672e6c3081d266b2acaa734b0d06f3c`, pulse `671a54574d9ad674f211086f11a20174d0734427c6dd2077d4ff4635752d4f3e`, model `2b0a443017c9769519b0e493bd5372d379eee95df8b7da0e0cfb658d508817df`; 297 a3.0 physics `a6653e742bd4592e499a56f7586c50f743049db8b97a38f5a297aa696e7897ca`, pulse `7e8bb1b09a93508ab3fe0f17c1659ca29804ba7bcdf48d54e88cc5ed336de4e9`, model `17dfcb524e1ddcadaaacf833d019b93e8f4edfb57f90c32fcbd028516f925cc3`. A red lock test is ALWAYS an implementation bug — never change a lock value.
- Serialized key field names frozen: `delta_idx` (ode), `n_idx` (297); chunk/scatter NPZ schemas frozen.
- Driver-locked invocations must keep parsing: `pilot --spacing-um A --workers auto --batch-size auto`, `run --spacing-um A --workers auto --batch-size auto --target-level 13`, `scatter --spacing-um A --level 13 --workers auto --batch-size auto`, `export/plot --spacing-um A [--metric M] [--no-veil]`.
- Deletions: pilot `--bench-workers` benchmark; `run` budget/reserve-hours + P90 ladder + `--target-level auto` (explicit choices 4/7/13/25, default 13). `audit` stays.
- Repo is an sshfs mount of DGX `chance@100.106.69.117:~/Ryd-gate-modeling`; edit locally, run ALL pytest/git remotely (`ssh … < /dev/null`); ARC-touching runs use `HOME=/tmp/arc297home OMP_NUM_THREADS=1 nice -n 15`. Never kill DGX processes; the a4.0 drain and the 297 smoke agent's throttled jobs may still be finishing.
- Commit per task, message ends `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; no push.

---

### Task 1: Compat-lock test harness

**Files:** Create `tests/test_sweep_compat_locks.py`.

**Interfaces:** Produces the gate file every later task must keep green. Loads both scripts via importlib (module names `max_leakage_ode_sweep` / `max_leakage_297_sweep`, aliases `mlso`/`mls297`).

- [ ] Write tests: (1) `mlso.ScanConfig().physics_hash()` == ode literal; `mlso.pulse_hash()` == ode literal; same pair for `mls297` with 297 literals. (2) Real-store round-trips, `pytest.mark.skipif` when the store dir is absent: for each of `results/max_leakage_ode/a3.0` and `results/max_leakage_297/a3.0`, `store.load_manifest()` parses, `load_records(manifest, include_states=False)` returns >0 ok rows, `load_scatter_records` parses (ode a3.0 must have >0, 297 may be 0), and the manifest hashes equal the literals. (3) `cmd_status` smoke on each live store (capsys asserts "records:"). (4) `@pytest.mark.slow` per script: `warm_and_build(ScanConfig())` model_hash == literal (ode builds 8 Δe panels ~minutes; 297 builds 8 n panels).
- [ ] Run fast locks green against CURRENT code (pre-refactor baseline), run the two slow locks once (remote, ARC-isolated env). Expected: all pass with zero source changes.
- [ ] Commit `"tests: compat locks for live sweep stores (hashes, schemas, status)"`.

### Task 2: sweeplib core — axes + solver

**Files:** Create `scripts/sweeplib/__init__.py`, `scripts/sweeplib/axes.py`, `scripts/sweeplib/solver.py`; Modify `scripts/max_leakage_ode_sweep.py`; Create `tests/test_sweeplib.py`.

**Interfaces (produced):**
- `axes.py`: `LEVEL_DENS/LEVEL_SIZES/LEVEL_FROM_SIZE`, `canon_coord`, `coord_value_mhz`, `axis_coords`, `axis_values_mhz`, and `make_pointkey_type(panel_field: str, id_prefix: str, omega_anchors, dsweep_anchors, panel_len: int, n_t: int)` returning `(PointKey, make_key, panel_keys, all_panels, all_keys)` where the dataclass's first field is NAMED `panel_field` (use `dataclasses.make_dataclass`) and `.id()`/`.panel`/`.omega_mhz()`/`.dsweep_mhz()`/`.level()` match the current per-script semantics.
- `solver.py`: `quintic`, `quintic_antideriv`, `envelope`, `envelope_integral`, `BlockMaxDOP853`, `make_block_solver_class(dim)`, `verify_scipy_error_norm`, `integrate_batch(ops, t_gate, point_params: dict[str, np.ndarray], state_labels, *, rhs_factory, dim, rtol, atol, ramp, use_shifts, segmented, t_eval)` where `rhs_factory(ops, cols: dict[str, np.ndarray], t_gate, ramp) -> Callable[[float, np.ndarray], np.ndarray]` is supplied by the script (cols = per-column expanded params). Result container class moves here unchanged (attribute names preserved).
- ode script: deletes its local copies, imports from sweeplib, builds its PointKey via the factory with `panel_field="delta_idx"`, `id_prefix="d"`, and supplies `rhs_factory` implementing the two-drive+Stark RHS (exact current math).
- [ ] TDD: move the axes/solver tests from `tests/test_max_leakage_ode_sweep.py` into `tests/test_sweeplib.py` (parameterized over both key configurations), keep script-specific assertions in place; run ode fast file + compat locks + sweeplib file green.
- [ ] Commit `"sweeplib: axes + solver core; ode script consumes"`.

### Task 3: sweeplib store + runner

**Files:** Create `scripts/sweeplib/store.py`, `scripts/sweeplib/runner.py`; Modify `scripts/max_leakage_ode_sweep.py`; Test `tests/test_sweeplib.py`.

**Interfaces:** `Store(root, *, key_type, key_fields, provenance_columns)` with the current chunk/scatter read/write/guard/flock API; `Runner`, `CostModel`, `set_worker_context`, worker entry taking the script's `rhs_factory` + scatter tables; gammas ALWAYS `dict[int, dict[str, float]]` keyed by panel row (ode passes the same dict per row — serialized formats unchanged, verified by compat locks).
- [ ] Move store/runner tests (mini-store fixtures parameterized per script config); ode + locks + sweeplib green.
- [ ] Commit `"sweeplib: store + runner; ode script consumes"`.

### Task 4: ode single-pass merge + machinery deletions

**Files:** Modify `scripts/max_leakage_ode_sweep.py`, `scripts/sweeplib/runner.py`; Test `tests/test_max_leakage_ode_sweep.py`, `tests/test_sweeplib.py`.

- [ ] `run` batches always integrate with `t_eval = cfg.n_eval_trajectory` samples; worker computes `scattering_integrals` and returns terminal + scatter payloads; the completion path writes the coherent chunk AND the scatter records in one step (skip scatter-write for keys already in the scatter series — resume safety).
- [ ] Trajectory-equivalence gate becomes a store-level setup step shared by run/scatter: executed when `reports/scatter_gate.json` is absent or not ok, recorded, else skipped.
- [ ] `cmd_scatter` becomes pure backfill (unchanged CLI; now typically a no-op).
- [ ] Delete `--bench-workers` machinery and the budget/reserve-hours P90 ladder; `--target-level` explicit, default "13"; locked driver invocations still parse (test asserts).
- [ ] Tests: synthetic merged-run on a mini store asserts one `run` produces BOTH series for the computed keys and that a subsequent `cmd_scatter` reports 0 missing; parser deletions test; compat locks green (schemas unchanged).
- [ ] Commit `"max_leakage_ode_sweep: single-pass run writes both series; drop bench/ETA ladder"`.

### Task 5: sweeplib plotting + cli; ode slimming

**Files:** Create `scripts/sweeplib/plotting.py`, `scripts/sweeplib/cli.py`; Modify `scripts/max_leakage_ode_sweep.py`; Test `tests/test_sweeplib.py`, `tests/test_max_leakage_ode_sweep.py`.

- [ ] Extract interpolation/veil/credibility floor/8×9 renderer (row-labeller + metric list parameterized) and the shared parser scaffold; ode script ends as constants + config + pulse + model layer (direct src calls) + rhs_factory + CLI wiring. Target ≤900 lines; report the final count.
- [ ] Move plot/CLI tests to sweeplib where shared; ode plot smoke (six metrics incl. total_error joins) + full fast repo suite green.
- [ ] Commit `"sweeplib: plotting + cli; ode script slimmed"`.

### Task 6: 297 script onto sweeplib + merge

**Precondition:** the in-flight 297 smoke agent has committed (check `git log -- scripts/max_leakage_297_sweep.py` for its end-to-end smoke commit before dispatching; if not landed, hold this task).

**Files:** Modify `scripts/max_leakage_297_sweep.py`; Test `tests/test_max_leakage_297_sweep.py`, `tests/test_sweeplib.py`.

- [ ] Rebase the 297 script onto sweeplib exactly as the ode script (panel_field="n_idx", id_prefix="n", single-drive rhs_factory, per-n gammas, five metrics), apply the same merge + deletions; dedup its test file against `tests/test_sweeplib.py`.
- [ ] Compat locks green (297 hashes/store); restricted merged-run smoke on the live 297 store: `run --target-level 4 --panels 1,1 --workers 4` (ARC-isolated, nice) must RESUME the store (hash gate accepts), compute the missing level-0 nodes and write both series; then `export` + `plot --metric total_error` render.
- [ ] Commit `"max_leakage_297_sweep: onto sweeplib, single-pass run"`.

### Task 7: campaign driver deployment

**Precondition:** final whole-branch review (dispatched by the controller after Task 6) has passed and fixes, if any, are merged.

**Files:** Create `~/spacing_family_run2.sh` on the DGX (not in repo).

- [ ] Verify the old a4.0 drain finished (no `max_leakage_ode_sweep.py` processes; store unlocked) and a3.0 (ode) has exports + six plots.
- [ ] New driver: for a in 4 5 7 10: `pilot --spacing-um $a --workers auto --batch-size auto` → `run --spacing-um $a --workers auto --batch-size auto --target-level 13` → `scatter --spacing-um $a --level 13 --workers auto --batch-size auto` (backfill safety; near-no-op) → `export` → six `plot` metrics `--no-veil`; then the 297 campaign on `results/max_leakage_297/a3.0`: `run --target-level 13 --workers auto --batch-size auto` → `scatter --level 13` → `export` → five plots `--no-veil`. Every stage runs with `HOME=/tmp/arc297home OMP_NUM_THREADS=1` (ARC sqlite isolation; harmless for non-ARC stages). `set -uo pipefail`, `|| exit 1` per stage, nohup launch, log `~/spacing_family_run2.log`.
- [ ] Launch; verify: a4.0 `run` RESUMES (log shows existing points skipped, hash gate silent), merged pass writes scatter records during run (scatter dir grows during the run stage), healthy worker utilization.
- [ ] Report launch state + measured merged-pass s/point.

---

## Final review

After Task 6 (before Task 7): dispatch the whole-branch code review over the full range from the pre-Task-1 BASE; triage ledger Minors there; fix wave as one subagent if findings.
