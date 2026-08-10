# Zeeman-Resolved 53P Pair-Potential Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ineffective overlap color bar with a shared marker-size legend, remove scheme-2 summary plots, and generate scheme-1 pair-potential figures for all four `53P3/2` Zeeman doorway states.

**Architecture:** Keep `scripts/check_297_pair_channels.py` as the sole producer. Extend its declarative manifold table, give all 53P manifolds a shared plot-scale group, and use one marker-area function for gray full-spectrum points and colored tracked-branch points. Preserve the current JSON case schema and regenerate it under the stricter five-manifold configuration; remove only scheme-2 renderers and images, not `weak_shift_weight` or complete-spectrum arrays.

**Tech Stack:** Python 3, NumPy, SciPy/ARC 3.10.2, Matplotlib, pytest, Markdown, repository-local `results-report` validator.

## Global Constraints

- Preserve the existing `53P3_2` key and filenames for the `mj=-1.5` doorway.
- Add `mj=-0.5`, `+0.5`, and `+1.5`; keep the `70S1_2` benchmark.
- Keep `B={20,40,60}` G, seven `theta` values, `phi=0`, the 41-point `R=2.5--8.0 um` grid, 10 GHz energy window, `n_range=3`, `l_max=2`, dipole-dipole coupling, and current numerical gates.
- Retain the complete local spectrum in neutral gray and encode `p_k=|<rr|Psi_k>|^2` by marker area for both plot layers.
- The size legend must show `p_k={0.1,0.5,1.0}`; line color continues to mean anchor rank only.
- Four 53P manifolds share one y limit; 70S has a separate limit shared across its three fields.
- Remove scheme-2 code, prose, and three `pair_potential_summary_B*G.png` outputs, but retain JSON `weak_shift_weight` and complete-spectrum arrays.
- Do not alter the 30 GHz `pair_channels.json` audit, laser model, gate dynamics, or pulse optimization.
- Work directly in the current checkout because the required staged pair-potential implementation is not present at HEAD; preserve unrelated user changes.

---

### Task 1: Specify Zeeman manifolds and plotting behavior with failing tests

**Files:**
- Modify: `tests/test_check_297_pair_channels.py`
- Read: `scripts/check_297_pair_channels.py`

**Interfaces:**
- Consumes: existing `POTENTIAL_MANIFOLDS`, `_plot_curve_panel`, `_pair_potential_y_limits`, and `render_pair_potential_figures`.
- Produces: executable expectations for `_overlap_marker_area(overlap) -> np.ndarray`, five manifold definitions, shared-scale limits, a size-only overlap legend, fifteen output paths, and absence of scheme-2 renderers.

- [ ] **Step 1: Add the manifold and case-count test**

```python
def test_pair_potential_manifolds_cover_all_53p_zeeman_levels():
    expected_53p = {
        "53P3_2": -1.5,
        "53P3_2_mj_m1_2": -0.5,
        "53P3_2_mj_p1_2": 0.5,
        "53P3_2_mj_p3_2": 1.5,
    }
    actual_53p = {
        key: manifold["mj"]
        for key, manifold in pair.POTENTIAL_MANIFOLDS.items()
        if manifold["n"] == 53
    }
    assert actual_53p == expected_53p
    assert pair.POTENTIAL_MANIFOLDS["70S1_2"]["mj"] == -0.5
    assert (
        len(pair.POTENTIAL_MANIFOLDS)
        * len(pair.POTENTIAL_FIELDS_G)
        * len(pair.POTENTIAL_THETA_DEG)
        == 105
    )
```

- [ ] **Step 2: Add the shared marker-area and panel-collection test**

Create a minimal case with two distances, one full-spectrum state per distance,
and one tracked branch. Use a real Agg Matplotlib axis, call
`_plot_curve_panel`, and assert both scatter collections use
`_overlap_marker_area`:

```python
def test_curve_panel_uses_one_overlap_marker_area_for_both_layers():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlaps = np.array([0.1, 1.0])
    case = {
        "curves": {
            "distance_um": [2.5, 3.0],
            "spectrum_shift_mhz": [[-5.0], [6.0]],
            "spectrum_rr_overlap": [[0.1], [1.0]],
            "branches": [{"shift_mhz": [-5.0, 6.0], "rr_overlap": overlaps}],
        }
    }
    fig, ax = plt.subplots()
    pair._plot_curve_panel(ax, case, 100.0, show_spectrum=True)
    np.testing.assert_allclose(
        ax.collections[0].get_sizes(), pair._overlap_marker_area(overlaps)
    )
    np.testing.assert_allclose(
        ax.collections[1].get_sizes(), pair._overlap_marker_area(overlaps[[0]])
    )
    plt.close(fig)
```

- [ ] **Step 3: Add the shared-scale test**

Build a synthetic result containing the five expected state keys. Give the
four 53P records maximum absolute shifts `10, 20, 30, 40` and 70S a maximum
of `100`. Assert the returned state-key limits are `41.6` for every 53P key
and `104.0` for 70S:

```python
limits = pair._pair_potential_y_limits(result)
np.testing.assert_allclose([limits[key] for key in expected_53p], 41.6)
assert limits["70S1_2"] == pytest.approx(104.0)
```

Use the same minimal case shape for each synthetic state; only the spectrum
shift values need differ.

- [ ] **Step 4: Add the figure-legend test**

Construct seven angle cases for one synthetic state, temporarily prevent
`plt.close`, render with `_render_state_field_potential`, and inspect the live
figure:

```python
pair._render_state_field_potential(result, tmp_path, "53P3_2", 20.0, 100.0)
fig = plt.gcf()
assert len(fig.axes) == 8  # seven panels plus legend panel; no colorbar axis
legend_text = {
    text.get_text()
    for ax in fig.axes
    for legend in ax.findobj(matplotlib.legend.Legend)
    for text in legend.get_texts()
}
assert {r"$p_k=0.1$", r"$p_k=0.5$", r"$p_k=1.0$"} <= legend_text
```

- [ ] **Step 5: Add the renderer-output and scheme-2-removal test**

Monkeypatch configuration/completeness checks and the state renderer so no
real images or ARC calls occur. First assert `_anchor_spectral_density` and
`_render_field_summary` are absent. Then install a raising sentinel named
`_render_field_summary` with `raising=False`, call
`render_pair_potential_figures`, and assert:

```python
assert not hasattr(pair, "_anchor_spectral_density")
assert not hasattr(pair, "_render_field_summary")
monkeypatch.setattr(
    pair,
    "_render_field_summary",
    lambda *args, **kwargs: pytest.fail("scheme-2 renderer was called"),
    raising=False,
)
assert len(paths) == 15
assert all("pair_potential_summary_" not in path.name for path in paths)
```

- [ ] **Step 6: Run the focused tests and record RED**

Run:

```bash
pytest tests/test_check_297_pair_channels.py -m "not slow" -v
```

Expected: the new tests fail because three 53P manifolds and
`_overlap_marker_area` are absent, 53P scales are per-state, a colorbar axis
exists, and the summary renderer is still called.

---

### Task 2: Implement the manifold and scheme-1 renderer changes

**Files:**
- Modify: `scripts/check_297_pair_channels.py:60-85`
- Modify: `scripts/check_297_pair_channels.py:872-1185`
- Test: `tests/test_check_297_pair_channels.py`

**Interfaces:**
- Consumes: tests and existing sidecar schema from Task 1.
- Produces: `_overlap_marker_area(overlap)`, five `POTENTIAL_MANIFOLDS`, grouped y limits, gray complete-spectrum rendering, combined rank/size legends, and a scheme-1-only `render_pair_potential_figures`.

- [ ] **Step 1: Extend `POTENTIAL_MANIFOLDS` minimally**

Keep `53P3_2` first, add the three new 53P keys, and add `scale_group` to all
records:

```python
POTENTIAL_MANIFOLDS = {
    "53P3_2": {
        "n": 53, "l": 1, "j": 1.5, "mj": -1.5,
        "label": r"$53P_{3/2},\,m_j=-3/2$", "scale_group": "53P3_2",
    },
    "53P3_2_mj_m1_2": {
        "n": 53, "l": 1, "j": 1.5, "mj": -0.5,
        "label": r"$53P_{3/2},\,m_j=-1/2$", "scale_group": "53P3_2",
    },
    "53P3_2_mj_p1_2": {
        "n": 53, "l": 1, "j": 1.5, "mj": 0.5,
        "label": r"$53P_{3/2},\,m_j=+1/2$", "scale_group": "53P3_2",
    },
    "53P3_2_mj_p3_2": {
        "n": 53, "l": 1, "j": 1.5, "mj": 1.5,
        "label": r"$53P_{3/2},\,m_j=+3/2$", "scale_group": "53P3_2",
    },
    "70S1_2": {
        "n": 70, "l": 0, "j": 0.5, "mj": -0.5,
        "label": r"$70S_{1/2},\,m_j=-1/2$", "scale_group": "70S1_2",
    },
}
```

Do not add `label` or `scale_group` to the physical manifold fingerprint;
`n/l/j/mj` remain the calculation-defining keys.

- [ ] **Step 2: Implement the common marker-area function**

```python
def _overlap_marker_area(overlap):
    return 2.0 + 58.0 * np.asarray(overlap, dtype=float)
```

In `_plot_curve_panel`, render the full spectrum with `color="0.55"`, no
colormap/norm, the shared area function, and the existing subdued alpha. Use
the same area function for sampled tracked-branch markers.

- [ ] **Step 3: Group 53P y limits**

Compute each state's raw maximum as today, reduce those maxima by
`POTENTIAL_MANIFOLDS[state_key]["scale_group"]`, multiply each group maximum
by `1.04`, and return a mapping keyed by each state key. Do not include timing
or metadata values in scale calculation.

- [ ] **Step 4: Replace the colorbar with two legends in the eighth panel**

Retain the rank handles. Add three gray marker-only `Line2D` handles whose
`markersize` is the square root of `_overlap_marker_area(value)` for values
`0.1`, `0.5`, and `1.0`. Add the rank legend as an artist, then add the size
legend below it with title `marker area: $p_k$`. Remove `ScalarMappable`,
`Normalize`, the `fig.colorbar` call, and their imports.

- [ ] **Step 5: Remove scheme-2 rendering code**

Delete `_anchor_spectral_density` and `_render_field_summary`. Remove the
field-summary loop from `render_pair_potential_figures` and update its
docstring to `Render seven-angle pair spectra for every state and field.`

- [ ] **Step 6: Run focused tests and record GREEN**

Run:

```bash
pytest tests/test_check_297_pair_channels.py -m "not slow" -v
```

Expected: all focused non-slow tests pass.

- [ ] **Step 7: Review the surgical diff**

Run:

```bash
git diff -- tests/test_check_297_pair_channels.py scripts/check_297_pair_channels.py
git diff --check -- tests/test_check_297_pair_channels.py scripts/check_297_pair_channels.py
```

Expected: only the new tests, manifold table, marker/legend logic, grouped
limits, and scheme-2 deletions appear; no adjacent physics code changes.

- [ ] **Step 8: Commit the test and renderer change only if the dirty index can be preserved**

Use path-limited staging/commit only after confirming the pre-existing staged
versions of these files are intentionally part of the feature. Otherwise
leave the verified diff uncommitted and report that choice rather than
disturbing the user's index.

---

### Task 3: Regenerate the five-manifold sidecar and fifteen scheme-1 figures

**Files:**
- Regenerate: `results/297_to_calibration/pair_potential_curves.json`
- Regenerate (gitignored): `results/297_to_calibration/pair_potential_*.png`
- Delete (gitignored): `results/297_to_calibration/pair_potential_summary_B20G.png`
- Delete (gitignored): `results/297_to_calibration/pair_potential_summary_B40G.png`
- Delete (gitignored): `results/297_to_calibration/pair_potential_summary_B60G.png`

**Interfaces:**
- Consumes: five-manifold configuration and scheme-1 renderer from Task 2.
- Produces: schema-1 complete sidecar with 105 cases and fifteen scheme-1 PNGs.

- [ ] **Step 1: Invoke and read the mandatory `results-report` skill**

Read its full `SKILL.md` before any command writes the result sidecar. Follow
its provenance and validation requirements throughout Tasks 3 and 4.

- [ ] **Step 2: Remove exactly the three obsolete summary PNGs**

First list the exact files, then remove only:

```bash
rm results/297_to_calibration/pair_potential_summary_B20G.png \
   results/297_to_calibration/pair_potential_summary_B40G.png \
   results/297_to_calibration/pair_potential_summary_B60G.png
```

These are generated, gitignored outputs and are recoverable from the previous
renderer revision.

- [ ] **Step 3: Run the full calculation from a fresh configuration**

Do not pass `--resume`, because the stored two-manifold fingerprint must not be
accepted as the five-manifold artifact:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl297 \
  .venv/bin/python scripts/check_297_pair_channels.py --pair-potentials
```

Expected: 105 `calculate:` cases, `status=complete`,
`params.completed_cases=105`, `params.completed_cases_this_run=105`, and
fifteen printed PNG paths.

- [ ] **Step 4: Validate numerical gates from the generated JSON**

Run a read-only Python check that asserts:

```python
assert data["status"] == "complete"
assert data["params"]["completed_cases"] == 105
assert set(data["manifolds"]) == set(POTENTIAL_MANIFOLDS)
assert min(all_captured_rr_overlap) >= 1.0 - 1e-9
assert max(all_eigensystem_residual_mhz) <= POTENTIAL_EIGENSYSTEM_RESIDUAL_MHZ
assert min(all_branch_match_overlaps) >= POTENTIAL_MATCH_TARGET
```

Print the actual extrema and runtime for README provenance.

- [ ] **Step 5: Verify output inventory**

List `pair_potential_*.png`; assert there are exactly fifteen files and no
filename contains `summary`. Confirm each of the four 53P state keys has 20,
40, and 60 G images and 70S has three images.

- [ ] **Step 6: Inspect representative figures**

Open at least these four PNGs at original resolution:

- `pair_potential_53P3_2_B20G.png`;
- `pair_potential_53P3_2_mj_m1_2_B20G.png`;
- `pair_potential_53P3_2_mj_p1_2_B20G.png`;
- `pair_potential_53P3_2_mj_p3_2_B20G.png`.

Verify gray background spectra remain visible, no colorbar exists, both
legends are legible, titles identify `mj`, panels are unclipped, and all four
53P figures use the same y limits.

---

### Task 4: Rewrite and validate the result report from generated data

**Files:**
- Modify: `results/297_to_calibration/README.md`
- Read: `results/297_to_calibration/pair_potential_curves.json`
- Validate with: `.agents/skills/results-report/validate.py`

**Interfaces:**
- Consumes: validated numerical output and figure inventory from Task 3.
- Produces: a provenance-complete README describing the final five-manifold calculation without scheme 2.

- [ ] **Step 1: Extract the README values mechanically**

Print, from JSON rather than transcription from plots:

- `completed_cases`, `completed_cases_this_run`, `elapsed_s`, and
  `wall_s_this_run`;
- maximum removed ARC degeneracy offset;
- maximum eigensystem residual;
- minimum adjacent branch-match overlap;
- `W_weak` at `R=3 um` for every 53P `mj`, field, and
  `theta={0,45,90}`.

- [ ] **Step 2: Update the model/data/provenance sections**

Change the sidecar description from 53P/70S with 42 cases to four 53P Zeeman
doorways plus 70S with 105 cases. Update runtime and PNG count using generated
values. State explicitly that only `mj=-3/2` is the current sigma-minus gate's
target and that the other three are pair-spectrum comparisons, not gate
predictions.

- [ ] **Step 3: Replace the figure explanation and embeds**

Describe the full local spectrum as neutral gray and the common marker-area
encoding as `p_k`; explain that branch color is anchor rank. Embed all twelve
53P Zeeman-resolved figures grouped by `mj`, then the three retained 70S
benchmark figures. Remove every scheme-2 paragraph, equation specific to the
summary panel, and `pair_potential_summary_B*G.png` reference.

- [ ] **Step 4: Add the compact Zeeman comparison table**

Use one row per `(mj, B)` and columns for `theta=0,45,90` containing generated
`W_weak(R=3 um)` values. Explain that this is a diagnostic overlap weight in
the 10 GHz visualization basis, not a gate error and not a basis-convergence
claim.

- [ ] **Step 5: Run the result-report validator**

Use the exact command specified by the `results-report` skill for
`results/297_to_calibration/README.md`. Expected: exit code 0 with every
required section/provenance check passing.

- [ ] **Step 6: Search for stale scheme-2 claims**

Run:

```bash
rg -n "方案 2|方向谱热图|Gaussian 展宽|pair_potential_summary|九张 pair|42 例" \
  results/297_to_calibration/README.md scripts/check_297_pair_channels.py
```

Expected: no matches.

---

### Task 5: Reproduction and final verification

**Files:**
- Verify: `scripts/check_297_pair_channels.py`
- Verify: `tests/test_check_297_pair_channels.py`
- Verify: `results/297_to_calibration/pair_potential_curves.json`
- Verify: `results/297_to_calibration/README.md`

**Interfaces:**
- Consumes: completed implementation and result report.
- Produces: evidence that replay is read-only, tests pass, and the delivered diff is scoped.

- [ ] **Step 1: Verify `--plot-only` does not rewrite JSON**

Hash `pair_potential_curves.json`, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl297 \
  .venv/bin/python scripts/check_297_pair_channels.py --plot-only
```

Hash the JSON again and assert the two hashes match. Confirm exactly fifteen
paths are printed.

- [ ] **Step 2: Run focused and slow tests**

```bash
pytest tests/test_check_297_pair_channels.py -m "not slow" -v
pytest tests/test_check_297_pair_channels.py -m slow -v
```

Expected: both commands pass.

- [ ] **Step 3: Invoke `superpowers:verification-before-completion`**

Follow the skill before claiming the task is complete. Re-run its required
fresh checks rather than relying on earlier output.

- [ ] **Step 4: Audit the final diff and user-change boundaries**

```bash
git diff --check -- tests/test_check_297_pair_channels.py \
  scripts/check_297_pair_channels.py \
  results/297_to_calibration/README.md
git status --short -- tests/test_check_297_pair_channels.py \
  scripts/check_297_pair_channels.py \
  results/297_to_calibration/README.md \
  results/297_to_calibration/pair_potential_curves.json
```

Confirm every changed line traces to the approved spec. Do not stage, unstage,
or alter unrelated repository files.

- [ ] **Step 5: Deliver the result**

Report the fifteen generated figures, the four-Zeeman comparison, deleted
summary outputs, test/validator commands and outcomes, and the fact that the
other three `mj` spectra do not by themselves define realizable gate targets.
