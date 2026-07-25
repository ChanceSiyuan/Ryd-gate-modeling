# Single-pass sweep + sweeplib refactor — design

Date: 2026-07-25
Status: approved in conversation (merge both scripts; delete pilot benchmark
and ETA ladder, keep audit; shared package at `scripts/sweeplib/`; remaining
un-scanned data is produced by the new code).

## Goal

1. **Single-pass acquisition**: `run` integrates each batch with trajectory
   sampling and writes BOTH series — coherent chunks (terminal states) and
   scatter records (Γ·∫⟨n⟩dt) — in one pass, in
   `scripts/max_leakage_ode_sweep.py` and `scripts/max_leakage_297_sweep.py`.
   Measured on a3.0: coherent-only ≈ 28 core-s/pt, scatter ≈ 110 core-s/pt →
   merged ≈ scatter-only cost, ~19% saved per spacing.
2. **Readability**: extract the duplicated machinery of the two ~3.1k-line
   scripts into a shared package `scripts/sweeplib/`; the scripts shrink to
   locked constants + pulse formulas + a model layer of direct `ryd_gate`
   (src) calls + an injected RHS factory + CLI (~600–800 lines each).
3. **Direct src calls** wherever they are not on the hot path:
   `level_structure`, `Register.chain`, protocol classes, `compile_exact`,
   decay metadata, `backend="exact_ode"` reference solves. The batched
   DOP853 kernel stays custom (src's exact_ode rebuilds H per RHS, ~10×).

## Hard constraints (regression-locked, non-negotiable)

The live new-era stores must remain byte-compatible. Literals captured
2026-07-25 from the running stores:

| store | physics_hash | model_hash | pulse_hash |
|---|---|---|---|
| ode a3.0 | `d66867f0f1f9404203933778250f859bb672e6c3081d266b2acaa734b0d06f3c` | `2b0a443017c9769519b0e493bd5372d379eee95df8b7da0e0cfb658d508817df` | `671a54574d9ad674f211086f11a20174d0734427c6dd2077d4ff4635752d4f3e` |
| 297 a3.0 | `a6653e742bd4592e499a56f7586c50f743049db8b97a38f5a297aa696e7897ca` | `17dfcb524e1ddcadaaacf833d019b93e8f4edfb57f90c32fcbd028516f925cc3` | `7e8bb1b09a93508ab3fe0f17c1659ca29804ba7bcdf48d54e88cc5ed336de4e9` |

- `ScanConfig` physics payloads, pulse formulas, and the model-build recipe
  (operator bytes) must reproduce these hashes exactly (fast tests for
  physics/pulse; slow ARC tests for model).
- Chunk/scatter NPZ schemas and key-field names are frozen per script
  (`delta_idx` in ode, `n_idx` in 297 — they are serialized in the stores).
- `status`/resume against both live stores must keep working; a
  compat-lock test file gates every refactor task.

## Architecture

`scripts/sweeplib/` package:

- `axes.py` — nested-grid math (`canon_coord`, `coord_value_mhz`,
  `axis_coords`, level tables) + `make_pointkey_type(panel_field, id_prefix,
  omega_anchors, dsweep_anchors, panel_len)` factory so each script keeps its
  serialized field name and id prefix.
- `solver.py` — quintic/envelope/envelope_integral, `BlockMaxDOP853` +
  `make_block_solver_class`, `integrate_batch` (segmented restarts, t_eval
  trajectory sampling, swap reconstruction, per-column shifts) taking an
  injected `rhs_factory(ops, cols, t_gate, ramp)`; the two-drive+Stark vs
  single-drive difference lives only in each script's factory.
- `store.py` — Store, atomic NPZ writes, three-hash provenance gates, flock,
  chunk/scatter series, parameterized by key type + provenance columns.
- `runner.py` — Runner, CostModel, worker context, batching; gammas uniformly
  keyed by panel row index (ode: same dict for all rows — format unchanged).
- `plotting.py` — log-linear interpolation, LOO veil, credibility floor,
   8×9 grid renderer with per-script row-variable labeller and metric list.
- `cli.py` — shared parser scaffold (`--output`, `--spacing-um`, workers,
  tolerances, `--panels`) and derived-output resolution.

Scripts keep: locked axes constants, `ScanConfig`, pulse formulas +
`pulse_hash`, model layer (direct src calls), RHS factory, scatter channel
tables, CLI wiring (metrics, store root, docstring).

## Single-pass mechanics

- `run` batches always integrate with `t_eval = n_eval_trajectory` samples;
  the worker computes `scattering_integrals` and returns both terminal states
  and scatter payloads; the writer appends the coherent chunk and the scatter
  records in the same completion step.
- The trajectory-equivalence gate moves from `cmd_scatter` to a store-level
  setup check: run once per store, recorded in `reports/scatter_gate.json`,
  skipped when present-and-ok.
- `scatter` subcommand remains as an incremental backfill (covers points
  computed by pre-merge code; a no-op otherwise). The old sequential driver
  therefore stays runnable, but the new driver relies on merged `run`.

## Deletions

- pilot `--bench-workers` throughput benchmark (always skipped in practice).
- `run`'s budget/reserve-hours P90-ETA ladder and `--target-level auto`;
  `--target-level` becomes explicit with default `13`. The locked driver
  invocation `run --spacing-um A --workers auto --batch-size auto
  --target-level 13` must keep parsing.
- `audit` stays (credibility-floor data source). Pilot keeps the packing
  acceptance gate and setup verification.

## Deployment (decided)

- The old sequential driver loop was stopped 2026-07-25 and the a4.0
  coherent run SIGINT-drained (checkpointed; its completed points are valid
  and resumable — physics hashes unchanged). a3.0 completed fully under the
  old code.
- After the refactor passes its final review: a new driver
  (`~/spacing_family_run2.sh`) resumes a4.0 and runs spacings 4/5/7/10 with
  merged single-pass stages (pilot → run 13×13 → scatter-backfill → export →
  plots), then runs the 297 a3.0 campaign (run 13×13 → backfill → export →
  five plots) sequentially on the same 40 workers.
- The 297 script's refactor task waits for the in-flight Task-5 smoke commit
  so the two edit streams never race.

## Testing

- `tests/test_sweep_compat_locks.py` (new, Task 1): hash literals above,
  real-chunk round-trips and `status` smokes against both live stores
  (skip-if-absent), slow ARC model-hash locks.
- Shared-machinery tests deduplicate from the two script test files into
  `tests/test_sweeplib.py`; script files keep physics-specific tests.
- Full fast repo suite green at every task boundary.

## Out of scope

Changing any physics, axes, tolerances, or store contents; moving sweep
machinery into `src/`; deleting `audit`; re-scanning a3.0 (either store).
