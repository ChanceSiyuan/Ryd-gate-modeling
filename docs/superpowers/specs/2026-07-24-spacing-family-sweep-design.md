# Spacing-family leakage sweep & store reorganization — design

Date: 2026-07-24
Status: approved in conversation; implementation plan to follow.

## Goal

Extend `scripts/max_leakage_ode_sweep.py` so the coherent-leakage + scattering
map family (the six `{metric}_8x9.pdf` maps) can be produced at atom spacings
3, 4, 5, 7, 10 µm, and reorganize `results/max_leakage_ode` into one
self-contained sub-store per spacing with the old-era data archived.

## Decisions (settled with the user)

1. **New physics era.** All five spacings are computed from current `main`
   (ARC-computed C6 ≈ 2π·862.7 GHz·µm⁶). The existing store was produced by
   pre-rewrite src (pinned C6 = 2π·874) and becomes a read-only archive;
   3 µm is re-scanned fresh so all five spacings are strictly comparable.
   The `~/mlods_f394c08` worktree recipe is no longer needed for new scans.
2. **Depth: 13×13 first.** Each spacing scans the nested inner grid
   (Ω420 × D_sweep) to the 13×13 level (72 panels × 169 = 12 168 keys);
   deepening to 25×25 stays available per spacing via the resumable ladder.
3. **Layout: parent directory + sub-stores.** No spacing axis in `PointKey`;
   `spacing_um` stays a manifest-level physics scalar. No cross-spacing
   comparison figures — the user compares per-spacing PDFs manually.
4. **Per-panel PNGs retired.** The `panel_{metric}_dX_tY.png` outputs are
   dropped from `cmd_plot`; the existing 72 files were deleted 2026-07-24.

## Target layout

```
results/max_leakage_ode/
├── legacy_c6-874/                 # old-era store, moved wholesale; read-only archive
│   └── manifest.json chunks/ scatter/ trajectories/ exports/ plots/ reports/ logs/
├── a3.0/                          # new-era sub-stores, identical internal structure
├── a4.0/
├── a5.0/
├── a7.0/
└── a10.0/
```

Each sub-store is a complete, independent store: its own manifest (with its
own physics/model/pulse hashes), chunk series, scatter series, exports and
plots. All existing subcommands work on a sub-store by pointing `--output`
at it.

## Script changes (`scripts/max_leakage_ode_sweep.py`)

1. **`--spacing-um` flag** (float, default 3.0) on the shared parser.
   Threaded into the single CLI-side `ScanConfig(...)` construction in
   `setup_run` (line ~2024); read/scatter paths reconstruct `ScanConfig`
   from the manifest payload and inherit spacing automatically.
   The `--output` default changes from the fixed `results/max_leakage_ode`
   to the derived `results/max_leakage_ode/a{spacing_um:.1f}` (argparse
   default `None`, resolved after parse); an explicit `--output` overrides.
   No new anti-mixing mechanism is needed: `spacing_um` is already part of
   the physics payload, so the existing physics_hash gate refuses
   cross-spacing appends.
2. **Remove per-panel PNG rendering** from `cmd_plot`. The six
   `{metric}_8x9.{png,pdf}` maps are unchanged.
3. **`.gitignore` globs** updated for the new nesting:
   `results/max_leakage_ode/reports/status.json` →
   `results/max_leakage_ode/*/reports/status.json`, and
   `results/max_leakage_ode/exports/` →
   `results/max_leakage_ode/*/exports/` (globs also cover `legacy_c6-874/`).

Nothing else changes: `PointKey`, the Store hash validation, the nested-grid
resume machinery, and the scatter pipeline are untouched.

## One-time migration

```
mkdir results/max_leakage_ode/legacy_c6-874
git mv  results/max_leakage_ode/{manifest.json,chunks,scatter,plots,trajectories,reports}  results/max_leakage_ode/legacy_c6-874/
mv      results/max_leakage_ode/{exports,logs}  results/max_leakage_ode/legacy_c6-874/     # git-ignored content
```

Verification: `python scripts/max_leakage_ode_sweep.py status --output
results/max_leakage_ode/legacy_c6-874` must read the archived store.

## Run plan (informational; execution is separate from this change)

Per spacing, from current main:
`pilot → run --level 13 → scatter --level 13 → export → plot`, each with
`--spacing-um {a}` (output auto-derived). Estimated ~10 h + ~7 h wall per
spacing at 40 workers; five spacings sequentially ≈ 4 days.

Physics expectation (ARC C6): V/2π ≈ 1.18 GHz (3 µm), 211 MHz (4 µm),
55 MHz (5 µm), 7.3 MHz (7 µm), 0.86 MHz (10 µm) — blockade is marginal at
7 µm and broken at 10 µm; the maps are expected to show gate breakdown
there, which is the point of the scan. `beam_area_um2 = 140 · spacing_um`
keeps the intensity/scattering model spacing-consistent automatically.

## Testing

- The existing fast tests in `tests/scripts/test_max_leakage_ode_sweep.py`
  (43) must keep passing.
- New tests: (a) default-output derivation from `--spacing-um`;
  (b) plot smoke asserts no `panel_*.png` is emitted;
  (c) `ScanConfig(spacing_um=4.0)` yields a different physics_hash than the
  default (anti-mixing guard exercised).
- The first `a3.0` pilot's built-in checks (H equivalence vs
  `backend="exact_ode"`, error-norm seam, swap symmetry) serve as the
  new-era validation.

## Downstream notes

- The error-budget table cell in
  `scripts/notebooks/04_quench_and_state_prep.ipynb` points at the store
  root; after migration it must point at `legacy_c6-874/` (old-era numbers)
  or `a3.0/` (new era, once scanned). Not modified by this work.
- Docs/memory referencing `results/max_leakage_ode` as a single store are
  updated after implementation.

## Out of scope

Running the scans themselves; 25×25 deepening policy; any cross-spacing
comparison figures.
