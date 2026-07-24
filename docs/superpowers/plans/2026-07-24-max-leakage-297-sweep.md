# 297 nm Single-Photon Leakage Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fork `scripts/max_leakage_ode_sweep.py` into `scripts/max_leakage_297_sweep.py` — same store/solver/plot machinery, 297 nm single-photon physics layer (n×T panels, Ω₂₉₇×D_sweep inner grids, p_ryd/p_r_garb scatter, five metric maps) — plus a test file `tests/test_max_leakage_297_sweep.py`.

**Architecture:** Copy the original 3135-line script, then swap only its physics layer; the Store/hash/resume/Runner/CostModel/export/plot scaffolding is kept byte-identical wherever possible so diffs against the original stay reviewable. The 297 model is the existing `rb87_297_clock_4` preset (clock encoding, 4 levels/atom, two-atom dim 16) driven by `Direct297CZProtocol` with the same quintic-envelope + cos-chirp waveform, no Stark-compensation term.

**Tech Stack:** Python/argparse/numpy/scipy DOP853, `ryd_gate` (`rb87_297_clock_4`, `Direct297CZProtocol`), ARC (per-n lifetimes/C6 via the preset), pytest with importlib script loading.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-24-max-leakage-297-sweep-design.md`.
- Panel axes: `RYD_N = (50, 53, 56, 60, 64, 68, 71, 73)` × `T_GATE_US = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)` (unchanged tuple).
- Inner anchors: `OMEGA297_ANCHORS_MHZ = (Fraction(9), Fraction(12), Fraction(15), Fraction(18))` (target-leg clock Ω₂₉₇/2π; power/optics never enter the model), `DSWEEP_ANCHORS_MHZ = (2, 10, 20, 30)` and `DSWEEP_HW_LIMIT_MHZ = 20.0` unchanged.
- `magnetic_field_G = 20.0`, `spacing_um = 3.0` default, `--spacing-um` flag + derived default output `results/max_leakage_297/a{spacing:.1f}` (same convention as the ode sweep).
- Scatter channels: `("p_ryd", "p_r_garb")` with level groups `{"p_ryd": (2,), "p_r_garb": (3,)}` (levels `0,1,r,r_garb`); Γ is n-dependent → per-panel gammas keyed by `n_idx`.
- Plot metrics: `max_leakage, p_ryd, p_r_garb, p_loss_total, total_error`; no `panel_*.png`.
- Key id prefix `n` (e.g. `n0_t1_om3-2_dw3-2`); PointKey field `n_idx` replaces `delta_idx`.
- No `omega_1013` concept anywhere (reference constant, manifest fields, prints all dropped).
- Repo is an sshfs mount; run pytest/git on the DGX (`ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && …' < /dev/null`); test command `export PATH=$HOME/.local/bin:$PATH && uv run --extra dev pytest -q tests/test_max_leakage_297_sweep.py`.
- ARC calls during tests/pilot may collide with the running five-spacing campaign's sqlite cache — if `sqlite3.IntegrityError: UNIQUE constraint failed: dipoleME…` appears, rerun with `HOME=/tmp/arc297home` (pre-synced copy of `~/.arc-data`).
- Commit after every task with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; do not push.

---

### Task 1: Fork file — locked config, keys, pulse layer

**Files:**
- Create: `scripts/max_leakage_297_sweep.py` (via `cp scripts/max_leakage_ode_sweep.py scripts/max_leakage_297_sweep.py`)
- Create: `tests/test_max_leakage_297_sweep.py`

**Interfaces:**
- Produces (consumed by later tasks): module constants `RYD_N`, `T_GATE_US`, `OMEGA297_ANCHORS_MHZ`, `DSWEEP_ANCHORS_MHZ`; `ScanConfig(spacing_um=3.0, magnetic_field_G=20.0, ryd_n=RYD_N, …)` with `physics_hash()`; `PointKey(n_idx, t_idx, om_num, om_den, dw_num, dw_den)` with `.id()` → `"n{n_idx}_t{t_idx}_om{num}-{den}_dw{num}-{den}"`, `.panel == (n_idx, t_idx)`, `.omega_mhz()`, `.dsweep_mhz()`; `make_key(n_idx, t_idx, om, dw)`; `panel_keys/all_panels/all_keys/pilot_keys`; `envelope/envelope_integral/quintic/quintic_antideriv` (unchanged); `chirp_rad_s(t, t_gate, d_sweep, ramp=0.15)` and `phase_rad(t, t_gate, d_sweep, ramp=0.15)` (the `dr_minus_d1` parameter and `stark_coefficients` are deleted); `pulse_hash()` (probes envelope/J/chirp/phase only).

- [ ] **Step 1: Copy the file and start the test file** with the loader block (mirrors the original test file's lines 10–29, module name `max_leakage_297_sweep`, alias `mls297`) plus the first tests:

```python
"""Tests for the 297 nm single-photon leakage sweep script (fork of the
two-photon max_leakage_ode_sweep tests; same importlib loading pattern)."""

import importlib.util
import json
import math
import os
import sys
from argparse import Namespace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "max_leakage_297_sweep", ROOT / "scripts" / "max_leakage_297_sweep.py")
mls297 = importlib.util.module_from_spec(_spec)
sys.modules["max_leakage_297_sweep"] = mls297
_spec.loader.exec_module(mls297)


def test_axes_are_the_locked_297_specification():
    assert mls297.RYD_N == (50, 53, 56, 60, 64, 68, 71, 73)
    assert mls297.T_GATE_US == (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)
    assert mls297.OMEGA297_ANCHORS_MHZ == (
        Fraction(9), Fraction(12), Fraction(15), Fraction(18))
    assert mls297.DSWEEP_ANCHORS_MHZ == (
        Fraction(2), Fraction(10), Fraction(20), Fraction(30))
    assert mls297.DSWEEP_HW_LIMIT_MHZ == 20.0
    assert len(mls297.all_panels()) == 72
    assert len(mls297.all_keys(1)) == 72 * 49
    assert len(mls297.all_keys(3)) == 72 * 625


def test_20mhz_is_a_dsweep_node_and_grids_nest():
    for level in range(4):
        vals = mls297.axis_values_mhz(mls297.DSWEEP_ANCHORS_MHZ, level)
        assert Fraction(20) in vals
    coarse = set(mls297.axis_coords(1))
    assert coarse <= set(mls297.axis_coords(2)) <= set(mls297.axis_coords(3))


def test_point_key_id_uses_n_prefix_and_canonicalizes():
    k = mls297.make_key(2, 4, (2, 4), (6, 8))
    assert k == mls297.make_key(2, 4, (1, 2), (3, 4))
    assert k.id() == "n2_t4_om1-2_dw3-4"
    assert k.panel == (2, 4)
    assert float(k.omega_mhz()) == 13.5          # 12 + (15-12)/2
    assert float(k.dsweep_mhz()) == 27.5         # 20 + (30-20)*3/4


def test_phase_is_exact_integral_of_chirp():
    t_gate, d_sweep = 1.7e-6, mls297.TAU * 13e6
    ts = np.linspace(0.0, t_gate, 4001)
    chirp = mls297.chirp_rad_s(ts, t_gate, d_sweep)
    phi_num = np.concatenate(
        [[0.0], np.cumsum((chirp[1:] + chirp[:-1]) * 0.5 * np.diff(ts))])
    phi = mls297.phase_rad(ts, t_gate, d_sweep)
    assert np.max(np.abs(phi - phi_num)) < 1e-6 * np.max(np.abs(phi))
    assert not hasattr(mls297, "stark_coefficients")


def test_physics_hash_covers_axes_and_spacing():
    base = mls297.ScanConfig().physics_hash()
    assert mls297.ScanConfig(spacing_um=4.0).physics_hash() != base
    assert mls297.ScanConfig(ryd_n=(50, 53)).physics_hash() != base
```

- [ ] **Step 2: Run to verify failure** — `uv run --extra dev pytest -q tests/test_max_leakage_297_sweep.py` fails (`RYD_N` undefined; fork still two-photon).

- [ ] **Step 3: Edit the fork's top layer.**
  1. Rewrite the module docstring: 297 single-photon scan, usage examples with `--spacing-um` and `results/max_leakage_297/a3.0`, note "no intermediate state → no p_mid".
  2. Constants block: delete `DELTA_E_GHZ` and `OMEGA_1013_REFERENCE_RAD_S`; add `RYD_N = (50, 53, 56, 60, 64, 68, 71, 73)`; `OMEGA_ANCHORS_MHZ` → `OMEGA297_ANCHORS_MHZ = (Fraction(9), Fraction(12), Fraction(15), Fraction(18))`; keep `T_GATE_US`, `DSWEEP_ANCHORS_MHZ`, level tables, `DSWEEP_HW_LIMIT_MHZ`, `LOGICAL_INPUTS`.
  3. `ScanConfig`: fields become `spacing_um=3.0, magnetic_field_G=20.0, ramp_frac=0.15, rtol/atol ×4, ryd_n=RYD_N, t_gate_us=T_GATE_US, omega297_anchors_mhz=tuple(str(a) for a in OMEGA297_ANCHORS_MHZ), dsweep_anchors_mhz=…, credibility_floor_min, interp_space, n_eval_trajectory` — delete `ryd_level`, `detuning_sign`, `p1013_nominal_w`, `optics_loss`, `beam_factor`, `beam_area_um2` property. `physics_payload()` unchanged in form.
  4. `PointKey`: `delta_idx` → `n_idx` (global rename in the file: `delta_idx` appears in keys, panels, plotting, exports); `.id()` prefix `d` → `n`; `omega_mhz()` uses `OMEGA297_ANCHORS_MHZ`. `all_panels()` ranges over `len(RYD_N)`. `pilot_keys()`: same center `((3,2),(3,2))` (13.5 MHz, 15 MHz) and extremes over `len(RYD_N)-1`.
  5. Pulse layer: delete `stark_coefficients`; `chirp_rad_s(t, t_gate, d_sweep, ramp=0.15)` returns `-d_sweep*cos(TAU*s)` only; `phase_rad(t, t_gate, d_sweep, ramp=0.15)` returns `(-d_sweep*t_gate/TAU)*sin(TAU*s)`; `pulse_hash()` probes envelope/J/chirp/phase only (keep `envelope_integral` — it still shapes the envelope and stays in the probe).
  Downstream call sites that still pass `drmd1`/use `delta` will break — that is Task 2/3's job; for Step 4 only the test file's imports must work, and it does not touch those layers yet.

- [ ] **Step 4: Run tests** — the five tests above pass (`pytest -q tests/test_max_leakage_297_sweep.py`; the module must import, so any top-level code touching deleted names gets fixed now, e.g. the `OMEGA_1013_REFERENCE_RAD_S` constant users).

- [ ] **Step 5: Commit** — `git add scripts/max_leakage_297_sweep.py tests/test_max_leakage_297_sweep.py && git commit -m "max_leakage_297_sweep: fork skeleton — 297 axes, keys, Stark-free pulse layer"`.

---

### Task 2: Model build layer (per-n panels)

**Files:**
- Modify: `scripts/max_leakage_297_sweep.py` (PanelOperators / build_system / aggregate_operators / hamiltonian_equivalence_error / model_decay_rates / warm_and_build / setup_run; delete `compute_omega_1013`)
- Test: `tests/test_max_leakage_297_sweep.py`

**Interfaces:**
- Produces: `PanelOperators(ryd_n, h_static_diag(16,), x297, y297, amplitude_scale, logical_indices, swap_perm, swap_symmetric)` with `hash_bytes()`; `build_system(cfg, ryd_n:int)`; `aggregate_operators(system, ryd_n) -> PanelOperators`; `hamiltonian_equivalence_error(system, ops, t_gate, omega_297, d_sweep, times, ramp)`; `model_decay_rates(system) -> {"p_ryd": float, "p_r_garb": float}`; `warm_and_build(cfg) -> (ops_by_n: dict[int, PanelOperators], model_hash: str, checks: dict)` where `checks["decay_rates_rad_s"]` is `{n_idx: {"p_ryd": …, "p_r_garb": …}}`.

- [ ] **Step 1: Write the (slow-marked) build test + fast structural tests**:

```python
def test_swap_permutation_local_dim_4():
    perm = mls297._swap_permutation(4)
    a, b = np.divmod(np.arange(16), 4)
    assert np.array_equal(perm, b * 4 + a)


@pytest.mark.slow
def test_warm_and_build_single_panel_matches_repo_hamiltonian():
    cfg = mls297.ScanConfig(ryd_n=(53,))
    ops_by_n, model_hash, checks = mls297.warm_and_build(cfg)
    ops = ops_by_n[0]
    assert ops.h_static_diag.shape == (16,)
    assert checks["hamiltonian_equivalence_rel_dev"] < 1e-12
    assert checks["swap_symmetric"]
    assert set(checks["decay_rates_rad_s"][0]) == {"p_ryd", "p_r_garb"}
    assert len(model_hash) == 64
```

- [ ] **Step 2: Run** — fast test fails only if `_swap_permutation` default changed; slow test deselected by default, run explicitly once in Step 4.

- [ ] **Step 3: Implement.** Key replacements (full code for the changed cores):

`build_system`:

```python
def build_system(cfg: ScanConfig, ryd_n: int):
    """Two-atom rb87_297_clock_4 system with a placeholder 297 protocol bound."""
    import ryd_gate as rg
    from ryd_gate.protocols import Direct297CZProtocol
    from ryd_gate.lattice import Register

    proto = Direct297CZProtocol(
        t_gate_s=1e-6, omega_297_max_rad_s=1.0,
        envelope_297=lambda t: 1.0, phase_297_rad=lambda t: 0.0,
    )
    return rg.RydbergSystem(
        level_structure=rg.level_structure(
            "rb87_297_clock_4", ryd_level=int(ryd_n),
            magnetic_field_G=cfg.magnetic_field_G),
        register=Register.chain(2, spacing_um=cfg.spacing_um),
        protocol=proto,
    )
```

`PanelOperators`: fields `ryd_n: int`, `h_static_diag`, `x297`, `y297`, `amplitude_scale`, `logical_indices`, `swap_perm`, `swap_symmetric`; `hash_bytes()` hashes `(h_static_diag, x297, y297, [float(ryd_n), amplitude_scale])`.

`aggregate_operators(system, ryd_n)`: same skeleton; diag-channel folding loop stays but any folded constant is just absorbed (no `delta_rad_s` retained); `ratios = {"297": {}}`; `b297 = sum(ratios["297"][ch] * ops[ch] …)`; `x297 = b297 + b297.conj().T`, `y297 = 1j*(b297 - b297.conj().T)`; swap check over `(x297, y297)`; `_swap_permutation(system._basis.local_dim)` (== 4).

`hamiltonian_equivalence_error(system, ops, t_gate, omega_297, d_sweep, times, ramp=0.15)`:

```python
    from ryd_gate.protocols import Direct297CZProtocol
    proto = Direct297CZProtocol(
        t_gate_s=t_gate, omega_297_max_rad_s=omega_297,
        envelope_297=lambda t: float(np.sqrt(envelope(t / t_gate, ramp))),
        phase_297_rad=lambda t: float(phase_rad(t, t_gate, d_sweep, ramp)),
    )
    # reference loop identical to the original; grouped evaluation:
    #   c297 = ops.amplitude_scale * omega_297 * amp * np.exp(-1j * phi)
    #   h_grp = diag + c297.real * ops.x297 + c297.imag * ops.y297
```

`model_decay_rates(system)` returns `{"p_ryd": rates["r"]["total"], "p_r_garb": rates["r_garb"]["total"]}`.

`warm_and_build(cfg)`: loop `for n_idx, n in enumerate(cfg.ryd_n)` building system+ops per n; `model_hash` = sha256 over concatenated `ops.hash_bytes()` in n order; checks: equivalence probe on the middle panel (n=60 row, T=1.2 µs, Ω=2π·13.5 MHz, D=2π·15 MHz, 41 times), `verify_scipy_error_norm`, swap flag AND over panels, `decay_rates_rad_s = {n_idx: model_decay_rates(sys_n)}`; no `omega_1013` in the return (signature `-> tuple[dict[int, PanelOperators], str, dict]`).

`setup_run`: adjust to the 3-tuple, print `[setup] panels(n) = 8 | H equivalence …` without Ω₁₀₁₃; manifest init drops the `omega_1013_*` fields (Store change lands in Task 4 — do the `init_or_validate_manifest` signature change together with it if the call breaks earlier, keep a `# Task 4` marker comment only if the module still imports).

- [ ] **Step 4: Run** — fast file passes; then run the slow test once on the DGX: `uv run --extra dev pytest -q tests/test_max_leakage_297_sweep.py -m slow -k warm_and_build` (minutes; ARC per-n; use `HOME=/tmp/arc297home` if the sqlite race appears). Expected: PASS with equivalence dev < 1e-12.

- [ ] **Step 5: Commit** — `"max_leakage_297_sweep: per-n model build layer (rb87_297_clock_4, Direct297CZ)"`.

---

### Task 3: Solver kernel + scatter layer

**Files:**
- Modify: `scripts/max_leakage_297_sweep.py` (`integrate_batch` RHS + column setup; `SCATTER_CHANNELS`/`_SCATTER_LEVEL_GROUPS`/`_scatter_weight_vectors`; per-panel gammas through `_set_worker_context`/`_WORKER_CTX`/scatter chunk writer; `_scatter_equivalence_gate` reference channels)
- Test: `tests/test_max_leakage_297_sweep.py`

**Interfaces:**
- Consumes: `PanelOperators.x297/y297`, `phase_rad(t, t_gate, d_sweep, ramp)`.
- Produces: `integrate_batch(ops, t_gate, omega_297, d_sweep, …)` (array `omega_297` rad/s per point); `SCATTER_CHANNELS = ("p_ryd", "p_r_garb")`; `_SCATTER_LEVEL_GROUPS = {"p_ryd": (2,), "p_r_garb": (3,)}`; `scattering_integrals(times, states, gammas)` unchanged in form; worker context `gammas: dict[int, dict[str, float]]` indexed by the batch panel's `n_idx`.

- [ ] **Step 1: Port + write tests** (adapt the originals; full code for the new ones):

```python
def _toy_ops(dim=16):
    rng = np.random.default_rng(7)
    b = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    x = b + b.conj().T
    y = 1j * (b - b.conj().T)
    diag = rng.normal(size=dim)
    perm = mls297._swap_permutation(4)
    return mls297.PanelOperators(
        ryd_n=53, h_static_diag=diag, x297=x, y297=y, amplitude_scale=1.0,
        logical_indices=np.array([0, 1, 4, 5]), swap_perm=perm,
        swap_symmetric=False)


def test_batched_kernel_matches_dense_reference():
    ops = _toy_ops()
    t_gate = 0.4e-6
    om = np.array([mls297.TAU * 12e6]); dw = np.array([mls297.TAU * 15e6])
    res = mls297.integrate_batch(ops, t_gate, om, dw, ("00", "01", "11"),
                                 rtol=1e-10, atol=1e-13)
    # dense reference: expm-free RK on H(t) = diag + Re/Im(c297)·(X, Y)
    from scipy.integrate import solve_ivp
    def href(t):
        s = t / t_gate
        amp = math.sqrt(float(mls297.envelope(s)))
        phi = float(mls297.phase_rad(t, t_gate, float(dw[0])))
        c = float(om[0]) * amp * np.exp(-1j * phi)
        return (np.diag(ops.h_static_diag).astype(complex)
                + c.real * ops.x297 + c.imag * ops.y297)
    for j, label in enumerate(("00", "01", "11")):
        psi0 = np.zeros(16, complex)
        psi0[ops.logical_indices[("00", "01", "10", "11").index(label)]] = 1.0
        sol = solve_ivp(lambda t, y: -1j * (href(t) @ y), (0, t_gate), psi0,
                        rtol=1e-11, atol=1e-13, dense_output=False)
        ref = sol.y[:, -1]
        got = res.states_final[0, ("00", "01", "11").index(label)]
        assert np.max(np.abs(got - ref)) < 1e-7


def test_scatter_weights_count_r_and_garb_atoms():
    w = mls297._scatter_weight_vectors(4)
    assert set(w) == {"p_ryd", "p_r_garb"}
    idx_rr = 2 * 4 + 2
    assert w["p_ryd"][idx_rr] == 2.0 and w["p_r_garb"][idx_rr] == 0.0
    idx_1g = 1 * 4 + 3
    assert w["p_r_garb"][idx_1g] == 1.0
```

Port from the original file, adapting names/dims: `test_error_norm_matches_installed_scipy`, `test_block_solver_matches_stock_dop853_per_block` (block 16), `test_swap_reconstruction_matches_direct_propagation`, `test_segmented_equals_unsegmented`, `test_batched_points_match_isolated_points`, `test_trajectory_sampling_and_time_dependent_restore`. (Adapt `integrate_batch(ops, t_gate, omega_297, d_sweep, …)` call signatures and any `_fake/_toy` helpers to dim 16 / `x297,y297` / no `omega_1013`, `drmd1`.) If a ported test's helper references the exact final-state attribute names, keep the original's names — the fork does not rename result containers.

- [ ] **Step 2: Run** — new tests fail (`integrate_batch` still expects `omega_420, omega_1013` etc.).

- [ ] **Step 3: Implement.** RHS core of `integrate_batch` (replacing the 745–778 block):

```python
    col_of_point = np.repeat(np.arange(n_points), n_states)
    om_cols = omega_297[col_of_point]
    dsw_cols = d_sweep[col_of_point]
    …
    x297_t = np.ascontiguousarray(ops.x297.T)
    y297_t = np.ascontiguousarray(ops.y297.T)
    ascale = ops.amplitude_scale
    sin_coef = -t_gate / TAU

    def rhs(t, y):
        s = t / t_gate
        amp = math.sqrt(float(envelope(s, ramp)))
        phi = (sin_coef * math.sin(TAU * s)) * dsw_cols
        c297 = (ascale * amp) * om_cols * np.exp(-1j * phi)
        ym = y.reshape(n_cols, dim)
        out = diag_row * ym
        out += c297.real[:, None] * (ym @ x297_t)
        out += c297.imag[:, None] * (ym @ y297_t)
        return (-1j * out).ravel()
```

Scatter layer: `SCATTER_CHANNELS = ("p_ryd", "p_r_garb")`; `_SCATTER_LEVEL_GROUPS = {"p_ryd": (2,), "p_r_garb": (3,)}`; `_scatter_weight_vectors(local_dim=4)`. Worker plumbing: `_set_worker_context(…, gammas=…)` stores the per-panel dict; the worker call site (`scattering_integrals(result.times, result.states, _WORKER_CTX["gammas"])`) becomes `…, _WORKER_CTX["gammas"][spec_n_idx])` where `spec_n_idx` is the batch's panel row (all keys in a batch share one panel — assert it). The scatter chunk writer's `gamma_mid` column is deleted; `gamma_ryd`/`gamma_r_garb` are filled from the batch's panel gammas. `_scatter_equivalence_gate`: reference `exact_ode` trajectory of the 297 system, `ref_levels = {"p_ryd": (2,), "p_r_garb": (3,)}`, `ref_rates` from `model_decay_rates`.

- [ ] **Step 4: Run** — full 297 test file green.
- [ ] **Step 5: Commit** — `"max_leakage_297_sweep: single-drive kernel + p_ryd/p_r_garb scatter with per-n gammas"`.

---

### Task 4: Store manifest, export, plot, CLI

**Files:**
- Modify: `scripts/max_leakage_297_sweep.py` (`Store.init_or_validate_manifest` axes/fields; export CSV columns; `cmd_plot` labels + metric list; `build_parser` defaults/choices; `_default_output`)
- Test: `tests/test_max_leakage_297_sweep.py`

**Interfaces:**
- Consumes: everything above.
- Produces: manifest `axes = {ryd_n, t_gate_us, omega297_anchors_mhz, dsweep_anchors_mhz, dsweep_hw_limit_mhz, level_sizes}` and no `omega_1013_*` keys; `_default_output(spacing_um)` → `results/max_leakage_297/a{spacing_um:.1f}`; plot x-label `Ω₂₉₇/2π (MHz)`, panel titles `n = {RYD_N[n_idx]}, T = {T} us`; `--metric` choices the five spec metrics.

- [ ] **Step 1: Port + write tests.** Port with mechanical adaptation: `test_manifest_guard_rejects_mismatched_provenance`, `test_chunk_guard_refuses_to_merge_foreign_provenance`, `test_resume_dedup_and_tier_preference`, `test_failed_points_recorded_and_not_counted_done`, `test_export_store_writes_merged_npz_and_csv`, `test_chunk_roundtrip_preserves_states_exactly`, `test_atomic_write_rejects_object_arrays_and_leaves_no_file`, `test_stale_tmp_files_are_ignored_by_the_loader`, `test_group_batches_stays_within_panel_and_orders_axes`, `test_plot_and_status_smoke` (five-metric loop + `assert not …glob("panel_*.png")`), `test_cli_parser_covers_subcommands_and_locked_invocation`. New:

```python
def test_manifest_has_297_axes_and_no_1013(tmp_path):
    _, manifest = _mini_store(tmp_path)
    assert manifest["axes"]["ryd_n"] == list(mls297.RYD_N)
    assert manifest["axes"]["omega297_anchors_mhz"] == ["9", "12", "15", "18"]
    assert not any("1013" in k for k in manifest)


def test_default_output_derivation():
    parser = mls297.build_parser()
    args = parser.parse_args(["run", "--dry-run"])
    assert args.spacing_um == 3.0 and args.output is None
    assert mls297._default_output(3.0) == os.path.join(
        "results", "max_leakage_297", "a3.0")


def test_plot_metric_choices_are_the_five_297_metrics():
    parser = mls297.build_parser()
    for m in ("max_leakage", "p_ryd", "p_r_garb", "p_loss_total", "total_error"):
        assert parser.parse_args(["plot", "--metric", m]).metric == m
    with pytest.raises(SystemExit):
        parser.parse_args(["plot", "--metric", "p_mid"])
```

(`_mini_store`/`_mini_cfg`/`_fake_result` helpers: port from the original test file, dim 16, no omega_1013 argument.)

- [ ] **Step 2: Run** — fails on manifest keys / choices.
- [ ] **Step 3: Implement**: manifest init (drop `omega_1013_rad_s/omega_1013_over_2pi_MHz/omega_1013_reference_rad_s`, axes rename `delta_e_ghz`→`ryd_n` with values `list(cfg.ryd_n)`, add `omega297_anchors_mhz`); export CSV header `omega297_mhz` column instead of `omega420_mhz` (keep other columns); `cmd_plot`: metric choices/`--metric` help, x-label `r"$\Omega_{297}/2\pi$ (MHz)"`, panel title `f"n = {RYD_N[ni]}, T = {tg[ti]:g} us"`, suptitle mentions "297 single-photon"; `total_error` join = coherent + `p_ryd` + `p_r_garb`; `_default_output` root `results/max_leakage_297`; docstring usage block final pass.
- [ ] **Step 4: Run** — full 297 file green.
- [ ] **Step 5: Commit** — `"max_leakage_297_sweep: 297 store schema, five-metric plots, CLI"`.

---

### Task 5: End-to-end smoke + suite gate

**Files:** none new (verification; possible small fixes).

- [ ] **Step 1: Restricted real pilot on the DGX** (one panel; ARC + solver end-to-end; throttled to leave the campaign's workers alone):

```bash
ssh chance@100.106.69.117 'cd ~/Ryd-gate-modeling && HOME=/tmp/arc297home OMP_NUM_THREADS=1 nice -n 15 .venv/bin/python scripts/max_leakage_297_sweep.py pilot --panels 1,1 --workers 4 --batch-size 8 --bench-workers ""' < /dev/null
```

Expected: `[setup]` line reports H equivalence ≲1e-12, error-norm dev ~1e-16, swap ok; pilot completes its restricted keys; `results/max_leakage_297/a3.0/` gains manifest + first chunk. Then `status --output results/max_leakage_297/a3.0` prints a coherent summary.

- [ ] **Step 2: Full fast suite** — `uv run --extra dev --extra tn --extra tn-2d pytest -q` (both sweep test files + whole repo green).
- [ ] **Step 3: Commit any fixes**, working tree clean, report per-point runtime measured by the pilot (informs the 13×13 campaign schedule).

---

## Run plan after implementation (separate decision)

`pilot → run --target-level 13 → scatter --level 13 → export → plot ×5` on `results/max_leakage_297/a3.0`, after (or throttled beside) the five-spacing campaign. The pilot's measured s/point decides whether it can run beside the campaign at reduced workers.
