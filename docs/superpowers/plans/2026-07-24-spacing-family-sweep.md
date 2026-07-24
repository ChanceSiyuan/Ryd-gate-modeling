# Spacing-Family Sweep & Store Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `scripts/max_leakage_ode_sweep.py` scan atom spacings 3/4/5/7/10 µm into per-spacing sub-stores under `results/max_leakage_ode/`, and archive the old-era store as `legacy_c6-874/`.

**Architecture:** `spacing_um` is already a `ScanConfig` physics field covered by `physics_hash`; the change threads a CLI flag into the one CLI-side `ScanConfig` construction (`setup_run`) and derives the default `--output` as `results/max_leakage_ode/a{spacing:.1f}`. The store internals (PointKey, hash gates, resume, scatter) are untouched; each spacing is an independent sub-store. Per-panel PNG output is retired.

**Tech Stack:** Python/argparse/numpy/scipy script; pytest (`tests/test_max_leakage_ode_sweep.py` loads the script via importlib as `mls`); git for the one-time migration.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-24-spacing-family-sweep-design.md`.
- Sub-store naming: `results/max_leakage_ode/a{spacing_um:.1f}` (e.g. `a3.0`, `a10.0`); archive dir name: `legacy_c6-874`.
- Default spacing: `3.0` µm; explicit `--output` always overrides the derived default.
- No new anti-mixing mechanism: physics_hash already covers `spacing_um`.
- `cmd_plot` must no longer emit `panel_*.png`; the six `{metric}_8x9.{png,pdf}` outputs are unchanged.
- The repo checkout is an sshfs mount of DGX `chance@100.106.69.117:~/Ryd-gate-modeling`; run pytest and all git operations on the remote (`ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && …'`, append `< /dev/null`). Local file edits via normal tools are fine.
- Test command (remote): `export PATH=$HOME/.local/bin:$PATH && uv run --extra dev pytest -q tests/test_max_leakage_ode_sweep.py`.
- Commit after every task; end commit messages with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Do not push.

---

### Task 1: `--spacing-um` flag + derived `--output` default

**Files:**
- Modify: `scripts/max_leakage_ode_sweep.py` (`common()` inside `build_parser()` ~line 3043; `setup_run()` ~line 2024; `main()` ~line 3129; new helper `_default_output` just above `build_parser()`; usage lines in the module docstring ~lines 45–52)
- Test: `tests/test_max_leakage_ode_sweep.py`

**Interfaces:**
- Produces: `mls._default_output(spacing_um: float) -> str` returning `os.path.join("results", "max_leakage_ode", f"a{spacing_um:.1f}")`; every subcommand's args gain `spacing_um: float` (default 3.0) and `output` defaults to `None` at the parser level (resolved in `main()`).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_max_leakage_ode_sweep.py` next to `test_cli_parser_covers_subcommands_and_locked_invocation`:

```python
def test_spacing_flag_and_derived_output_default():
    parser = mls.build_parser()
    args = parser.parse_args(["run", "--dry-run"])
    assert args.spacing_um == 3.0 and args.output is None
    assert mls._default_output(args.spacing_um) == os.path.join(
        "results", "max_leakage_ode", "a3.0")
    args = parser.parse_args(["scatter", "--level", "13", "--spacing-um", "7"])
    assert args.spacing_um == 7.0
    assert mls._default_output(args.spacing_um) == os.path.join(
        "results", "max_leakage_ode", "a7.0")
    args = parser.parse_args(
        ["status", "--output", "results/max_leakage_ode/legacy_c6-874"])
    assert args.output == "results/max_leakage_ode/legacy_c6-874"


def test_spacing_um_changes_physics_hash():
    assert (mls.ScanConfig(spacing_um=4.0).physics_hash()
            != mls.ScanConfig().physics_hash())
```

- [ ] **Step 2: Run tests to verify the expected failure**

Run (remote): `uv run --extra dev pytest -q tests/test_max_leakage_ode_sweep.py -k spacing`
Expected: `test_spacing_flag_and_derived_output_default` FAILS (`--spacing-um` unrecognized / `_default_output` missing); `test_spacing_um_changes_physics_hash` already PASSES (spacing is in the payload today — it stays as a regression guard).

- [ ] **Step 3: Implement**

In `build_parser()`, replace the `--output` line inside `common()`:

```python
    def common(sp, compute: bool = False):
        sp.add_argument("--output", default=None,
                        help="scan store directory (default: "
                             "results/max_leakage_ode/a{spacing:.1f})")
        sp.add_argument("--spacing-um", type=float, default=3.0,
                        help="atom spacing in um (physics-hash relevant; also "
                             "selects the default store directory)")
```

Add above `build_parser()`:

```python
def _default_output(spacing_um: float) -> str:
    return os.path.join("results", "max_leakage_ode", f"a{spacing_um:.1f}")
```

In `main()`:

```python
def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.output is None:
        args.output = _default_output(args.spacing_um)
    args.func(args)
```

In `setup_run()`:

```python
    cfg = ScanConfig(
        spacing_um=args.spacing_um,
        rtol_production=args.rtol, atol_production=args.atol,
        rtol_audit=args.audit_rtol, atol_audit=args.audit_atol,
    )
```

In the module docstring usage block, change the `status`/`run` example lines to show the new convention, e.g. `python scripts/max_leakage_ode_sweep.py run --spacing-um 5 ...  # store: results/max_leakage_ode/a5.0`.

- [ ] **Step 4: Run the full fast file to verify pass**

Run (remote): `uv run --extra dev pytest -q tests/test_max_leakage_ode_sweep.py`
Expected: all pass (watch `test_cli_parser_covers_subcommands_and_locked_invocation` — it passes explicit `--output`, unaffected).

- [ ] **Step 5: Commit**

```bash
git add scripts/max_leakage_ode_sweep.py tests/test_max_leakage_ode_sweep.py
git commit -m "max_leakage_ode_sweep: --spacing-um flag with derived per-spacing store dir"
```

---

### Task 2: retire per-panel PNG output

**Files:**
- Modify: `scripts/max_leakage_ode_sweep.py` (`cmd_plot` block ~lines 3008–3024; `--individual/--no-individual` args ~lines 3122–3124)
- Test: `tests/test_max_leakage_ode_sweep.py` (`test_plot_and_status_smoke` ~line 806)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `cmd_plot` args no longer include `individual`; `_panel_plot_data`/`_draw_panel` remain (used by the main 8×9 grid).

- [ ] **Step 1: Update the smoke test to the new contract** — in `test_plot_and_status_smoke`, drop `individual=False` from the Namespace and assert no panel files:

```python
    args = Namespace(output=store.root, dpi=60, veil=True,
                     metric="max_leakage")
    mls.cmd_plot(args)
    assert (Path(store.plots_dir) / "max_leakage_8x9.png").exists()
    assert (Path(store.plots_dir) / "max_leakage_8x9.pdf").exists()
    assert not list(Path(store.plots_dir).glob("panel_*.png"))
```

- [ ] **Step 2: Run it to verify it fails**

Run (remote): `uv run --extra dev pytest -q tests/test_max_leakage_ode_sweep.py -k plot_and_status`
Expected: FAIL with `AttributeError: 'Namespace' object has no attribute 'individual'`.

- [ ] **Step 3: Implement** — delete the whole `if args.individual:` block at the end of `cmd_plot` (from `if args.individual:` through `print(f"individual panels -> {store.plots_dir}")`), and delete both parser lines:

```python
    sp.add_argument("--individual", action="store_true", default=True,
                    help="also write per-panel PNGs (default on)")
    sp.add_argument("--no-individual", dest="individual", action="store_false")
```

- [ ] **Step 4: Run the full fast file to verify pass**

Run (remote): `uv run --extra dev pytest -q tests/test_max_leakage_ode_sweep.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/max_leakage_ode_sweep.py tests/test_max_leakage_ode_sweep.py
git commit -m "max_leakage_ode_sweep: retire per-panel PNG plot output"
```

---

### Task 3: `.gitignore` sub-store globs

**Files:**
- Modify: `.gitignore` (lines 37 and 55)

**Interfaces:** none.

- [ ] **Step 1: Edit** — replace the two store-specific entries:

```
results/max_leakage_ode/reports/status.json  ->  results/max_leakage_ode/*/reports/status.json
results/max_leakage_ode/exports/             ->  results/max_leakage_ode/*/exports/
```

(`results/**/*.log` and `results/**/store.lock` already cover the nesting.)

- [ ] **Step 2: Verify with git check-ignore (remote)**

```bash
git check-ignore -v results/max_leakage_ode/legacy_c6-874/exports/x.npz \
                    results/max_leakage_ode/a3.0/exports/x.npz \
                    results/max_leakage_ode/a3.0/reports/status.json
```

Expected: all three match the new glob lines.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "gitignore: per-spacing sub-store globs for max_leakage_ode"
```

---

### Task 4: one-time migration to `legacy_c6-874/`

**Files:**
- Move (remote git): `results/max_leakage_ode/{manifest.json,chunks,scatter,plots,reports}` → `results/max_leakage_ode/legacy_c6-874/`
- Move (plain mv, git-ignored or possibly untracked): `exports`, `logs`, and `trajectories` if untracked

**Interfaces:** none (data-only).

- [ ] **Step 1: Check what is tracked** (remote): `git ls-files results/max_leakage_ode/trajectories | head -1` — decides `git mv` vs `mv` for `trajectories`.

- [ ] **Step 2: Migrate** (remote; use `git mv` for tracked paths, `mv` for the rest):

```bash
mkdir results/max_leakage_ode/legacy_c6-874
git mv results/max_leakage_ode/manifest.json results/max_leakage_ode/chunks \
       results/max_leakage_ode/scatter results/max_leakage_ode/plots \
       results/max_leakage_ode/reports results/max_leakage_ode/legacy_c6-874/
mv results/max_leakage_ode/exports results/max_leakage_ode/logs \
   results/max_leakage_ode/legacy_c6-874/          # plus trajectories if untracked
```

- [ ] **Step 3: Verify the archived store reads** (remote, main venv):

```bash
.venv/bin/python scripts/max_leakage_ode_sweep.py status \
    --output results/max_leakage_ode/legacy_c6-874
```

Expected: the usual records summary (45000 unique ok points; no model build, no hash gate).

- [ ] **Step 4: Report stale references** (no edits): `grep -rn "results/max_leakage_ode" docs scripts/notebooks --include='*.md' --include='*.ipynb' -l` — list hits in the task report; per spec, notebook 04's table cell is intentionally left for the user (point it at `legacy_c6-874/` for old numbers or `a3.0/` once scanned).

- [ ] **Step 5: Commit**

```bash
git add -A results/max_leakage_ode
git commit -m "Archive old-era (pinned C6=874) max_leakage_ode store as legacy_c6-874/"
```

---

### Task 5: full-suite gate

**Files:** none (verification only).

- [ ] **Step 1: Run the whole fast suite** (remote): `uv run --extra dev --extra tn --extra tn-2d pytest -q`
Expected: no failures (baseline was green at 481 tests post-PEPS-refactor; count has since grown).

- [ ] **Step 2: Commit any straggler** (should be none): `git status --porcelain` must be clean apart from ignored files.

---

## Run plan after implementation (execution is a separate decision, not part of this plan)

Per spacing `a ∈ {3,4,5,7,10}`, on the DGX from current main:
`pilot → run --level 13 → scatter --level 13 → export → plot`, each with `--spacing-um {a}` and no `--output` (derived `a{a:.1f}/`). ≈17 h wall per spacing at 40 workers; five sequential ≈ 4 days. The first `a3.0` pilot's built-in checks (H equivalence vs `backend="exact_ode"`, error-norm seam, swap) validate the new era.
