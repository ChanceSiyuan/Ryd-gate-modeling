# `rr` Pair-Potential Curves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three reproducible pair-potential figures at 20, 40, and 60 G that continuously track the six explicit-pair eigenstates with the largest `rr` overlap at 3 micrometres.

**Architecture:** Keep the study in `scripts/check_297_pair_channels.py`. Separate the distance grid, eigenvector assignment, branch tracking, ARC eigensolver adapter, and rendering into small functions so the numerical identity logic and plotting can be tested without a 2276-state ARC run. Store curve data in schema-3 JSON, render PNGs from that artifact, and retain 160 G as a single-distance audit.

**Tech Stack:** Python 3.11, NumPy, SciPy sparse `eigsh`, SciPy Hungarian assignment, ARC 3.10.2, Matplotlib Agg, pytest, JSON, Markdown.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-10-rr-pair-potential-curves-design.md`.
- Potential plots are only for 20, 40, and 60 G; the full-pair single-distance audit remains at 20, 40, 60, and 160 G.
- The distance grid has exactly 81 points from 2.5 to 8.0 micrometres and contains 3.0 exactly.
- Seed six branches at 3.0 micrometres by descending `rr` overlap, then track adiabatic eigenvectors in both directions using Hungarian assignment on squared eigenvector overlaps.
- Candidate extraction starts at 64 eigenpairs and can expand to 256; require captured `rr` weight at least 0.995 and adjacent branch match at least 0.25.
- Plot frequency shifts relative to the uncoupled `rr` basis energy, not absolute ARC energies and not algebraically largest cutoff-edge eigenvalues.
- Preserve the user's uncommitted edits in `results/297_to_calibration/README.md`; add content surgically around them.
- PNGs are generated but remain untracked. Do not use `git add -f`.
- Any write under `results/` is incomplete until the `results-report` validator passes.
- Use explicit paths for every `git add` and `git commit --only`; do not include unrelated staged or untracked work.

---

### Task 1: Distance grid and eigenvector branch assignment

**Files:**
- Modify: `tests/test_check_297_pair_channels.py`
- Modify: `scripts/check_297_pair_channels.py`

**Interfaces:**
- Produces: `potential_distance_grid() -> np.ndarray`
- Produces: `match_eigenbranches(previous_vectors, candidate_vectors) -> tuple[np.ndarray, np.ndarray]`
- Consumes later: both functions are used by `track_rr_branches` in Task 2.

- [ ] **Step 1: Re-read the TDD test-quality rules**

Read `superpowers/test-driven-development/writing-good-tests.md` completely before editing the test file.

- [ ] **Step 2: Add failing grid and matching tests**

Append these tests:

```python
def test_potential_distance_grid_has_requested_extent_and_exact_anchor():
    distances = pair.potential_distance_grid()

    assert distances.shape == (81,)
    assert distances[0] == pytest.approx(2.5)
    assert distances[-1] == pytest.approx(8.0)
    assert np.count_nonzero(distances == 3.0) == 1
    assert np.all(np.diff(distances) > 0.0)


def test_match_eigenbranches_follows_vectors_instead_of_energy_order():
    previous = np.eye(3, dtype=complex)[:, :2]
    candidates = np.eye(3, dtype=complex)[:, [1, 0, 2]]

    indices, qualities = pair.match_eigenbranches(previous, candidates)

    np.testing.assert_array_equal(indices, [1, 0])
    np.testing.assert_allclose(qualities, [1.0, 1.0])
```

- [ ] **Step 3: Run the two tests and verify RED**

Run:

```bash
uv run --with pytest pytest tests/test_check_297_pair_channels.py \
  -k 'potential_distance_grid or match_eigenbranches' -v
```

Expected: both tests fail because the two functions do not exist.

- [ ] **Step 4: Implement the grid and Hungarian assignment**

Add `linear_sum_assignment` to the SciPy imports, define the field/curve constants, and implement:

```python
FULL_PAIR_FIELDS_G = (20.0, 40.0, 60.0, 160.0)
POTENTIAL_FIELDS_G = (20.0, 40.0, 60.0)
POTENTIAL_BRANCH_COUNT = 6
POTENTIAL_INITIAL_EIGENPAIRS = 64
POTENTIAL_MAX_EIGENPAIRS = 256
POTENTIAL_CAPTURE_TARGET = 0.995
POTENTIAL_MATCH_TARGET = 0.25


def potential_distance_grid() -> np.ndarray:
    left = np.linspace(2.5, 3.0, 9)
    right = np.linspace(3.0, 8.0, 73)
    return np.concatenate((left, right[1:]))


def match_eigenbranches(previous_vectors, candidate_vectors):
    overlaps = np.abs(previous_vectors.conj().T @ candidate_vectors) ** 2
    rows, columns = linear_sum_assignment(-overlaps)
    assignment = np.empty(previous_vectors.shape[1], dtype=int)
    qualities = np.empty(previous_vectors.shape[1], dtype=float)
    assignment[rows] = columns
    qualities[rows] = overlaps[rows, columns]
    return assignment, qualities
```

Keep the old `B_FIELDS_G` name only if all existing consumers are migrated in the same edit; do not leave two competing field constants.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run the Step 3 command. Expected: 2 passed.

- [ ] **Step 6: Run all fast pair-channel tests**

```bash
uv run --with pytest pytest tests/test_check_297_pair_channels.py -m 'not slow' -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 1 paths only**

```bash
git add scripts/check_297_pair_channels.py tests/test_check_297_pair_channels.py
git commit --only scripts/check_297_pair_channels.py tests/test_check_297_pair_channels.py \
  -m "test: define rr pair-curve tracking"
```

---

### Task 2: Adaptive curve eigensystems and schema-3 ARC output

**Files:**
- Modify: `tests/test_check_297_pair_channels.py`
- Modify: `scripts/check_297_pair_channels.py`

**Interfaces:**
- Produces: `extract_curve_eigenpairs(hamiltonian, reference_ghz, bare_index, *, previous_vectors=None, initial_k, max_k, capture_target, match_target) -> tuple[np.ndarray, np.ndarray, dict]`
- Produces: `track_rr_branches(distances_um, solve, *, bare_index, reference_ghz, basis_states, branch_count, anchor_um=3.0) -> dict`
- Produces: `calculate_rr_potential_curves(calc) -> dict`
- Extends: `calculate_full_pair_field(atom, b_gauss)` with `rr_potential_curves` at 20, 40, and 60 G.
- Extends: `build_output(atom)` to schema 3 and four full-pair fields.

- [ ] **Step 1: Add a failing adaptive-candidate test**

Add:

```python
def test_extract_curve_eigenpairs_expands_to_recover_previous_branch():
    values = np.array([-0.40, -0.20, -0.10, -0.02, 0.01,
                       0.05, 0.12, 0.30, 0.45, 0.60])
    hamiltonian = diags(values, format="csr")
    previous = np.eye(values.size)[:, [3, 7]]

    eigenvalues, eigenvectors, meta = pair.extract_curve_eigenpairs(
        hamiltonian,
        reference_ghz=0.0,
        bare_index=3,
        previous_vectors=previous,
        initial_k=4,
        max_k=8,
        capture_target=0.99,
        match_target=0.9,
    )

    assert eigenvalues.shape == (8,)
    assert eigenvectors.shape == (10, 8)
    assert meta["candidate_eigenpairs"] == 8
    assert meta["captured_rr_overlap"] == pytest.approx(1.0)
    assert min(meta["assigned_match_overlap"]) == pytest.approx(1.0)
```

- [ ] **Step 2: Run the test and verify RED**

```bash
uv run --with pytest pytest tests/test_check_297_pair_channels.py \
  -k extract_curve_eigenpairs_expands -v
```

Expected: failure because `extract_curve_eigenpairs` is absent.

- [ ] **Step 3: Implement deterministic adaptive curve candidates**

Reuse the deterministic ramp `v0` convention from `extract_local_eigenpairs`. On each attempt:

```python
eigenvalues, eigenvectors = eigsh(
    hamiltonian,
    k=k,
    sigma=reference_ghz,
    which="LM",
    tol=1e-9,
    v0=v0,
)
order = np.argsort(eigenvalues)
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]
captured = float(np.sum(np.abs(eigenvectors[bare_index, :]) ** 2))
```

If `previous_vectors` is present, call `match_eigenbranches` and require all assignment qualities to reach `match_target`. Double `k` until both capture and matching pass. At `max_k`, raise `RuntimeError` naming which threshold failed. Return metadata with exactly these keys:

```python
{
    "candidate_eigenpairs": int(k),
    "captured_rr_overlap": captured,
    "assigned_indices": assignment.tolist() if assignment is not None else None,
    "assigned_match_overlap": qualities.tolist() if qualities is not None else None,
}
```

- [ ] **Step 4: Verify the candidate test GREEN**

Run the Step 2 command. Expected: 1 passed.

- [ ] **Step 5: Add a failing synthetic branch-tracking test**

Use a two-branch rotation and deliberately permute candidate column order away from energy order:

```python
def test_track_rr_branches_seeds_at_anchor_and_tracks_both_directions():
    q0 = np.array([np.sqrt(0.8), np.sqrt(0.2), 0.0])
    q1 = np.array([-np.sqrt(0.2), np.sqrt(0.8), 0.0])
    q2 = np.array([0.0, 0.0, 1.0])
    anchor_vectors = np.column_stack((q0, q1, q2))
    records = {
        2.5: (np.array([-0.20, 0.10, 0.40]), anchor_vectors[:, [1, 0, 2]]),
        3.0: (np.array([-0.10, 0.20, 0.50]), anchor_vectors),
        4.0: (np.array([-0.05, 0.30, 0.60]), anchor_vectors[:, [1, 0, 2]]),
    }

    def solve(distance_um, previous_vectors):
        eigenvalues, eigenvectors = records[float(distance_um)]
        assigned = None
        quality = None
        if previous_vectors is not None:
            assigned, quality = pair.match_eigenbranches(
                previous_vectors, eigenvectors
            )
        return eigenvalues, eigenvectors, {
            "candidate_eigenpairs": 3,
            "captured_rr_overlap": 1.0,
            "assigned_indices": None if assigned is None else assigned.tolist(),
            "assigned_match_overlap": None if quality is None else quality.tolist(),
        }

    result = pair.track_rr_branches(
        np.array([2.5, 3.0, 4.0]),
        solve,
        bare_index=0,
        reference_ghz=0.0,
        basis_states=[
            [53, 1, 1.5, -1.5, 53, 1, 1.5, -1.5],
            [53, 0, 0.5, -0.5, 54, 0, 0.5, -0.5],
            [52, 2, 2.5, -0.5, 53, 0, 0.5, -0.5],
        ],
        branch_count=2,
        anchor_um=3.0,
    )

    assert result["distance_um"] == [2.5, 3.0, 4.0]
    assert len(result["branches"]) == 2
    assert result["branches"][0]["anchor_rr_overlap"] == pytest.approx(0.8)
    assert result["branches"][0]["shift_mhz"] == pytest.approx(
        [100.0, -100.0, 300.0]
    )
    assert result["branches"][0]["adjacent_match_overlap"] == pytest.approx(
        [1.0, 1.0, 1.0]
    )
```

The expected shift sequence follows vector `q0` despite the candidate permutations. If exact synthetic eigenvalue ordering needs adjustment during RED setup, change only `records`; do not weaken the identity assertions.

- [ ] **Step 6: Run the branch test and verify RED**

```bash
uv run --with pytest pytest tests/test_check_297_pair_channels.py \
  -k track_rr_branches_seeds -v
```

Expected: failure because `track_rr_branches` is absent.

- [ ] **Step 7: Implement branch tracking**

Implement `track_rr_branches` with this data contract:

```python
result = {
    "anchor_um": float(anchor_um),
    "branch_count": int(branch_count),
    "distance_um": np.asarray(distances_um, dtype=float).tolist(),
    "candidate_eigenpairs": candidate_counts.astype(int).tolist(),
    "captured_rr_overlap": captured_overlaps.astype(float).tolist(),
    "branches": branch_records,
}

branch_record = {
    "anchor_rank": int(seed_index + 1),
    "anchor_shift_mhz": float(shifts_mhz[anchor_index]),
    "anchor_rr_overlap": float(rr_overlaps[anchor_index]),
    "anchor_top_components": anchor_top_components,
    "shift_mhz": shifts_mhz.astype(float).tolist(),
    "rr_overlap": rr_overlaps.astype(float).tolist(),
    "adjacent_match_overlap": match_overlaps.astype(float).tolist(),
    "min_adjacent_match_overlap": float(np.min(match_overlaps)),
}
```

At the anchor, sort candidate columns by
`abs(eigenvectors[bare_index])**2`, select six, and set match overlap to 1.0.
Traverse `range(anchor_index + 1, len(distances_um))` and
`range(anchor_index - 1, -1, -1)`, each time using the previously selected
branch vectors. Preserve seed order in every output array. Use
`_pair_state_label` and the four largest absolute-square coefficients for
`anchor_top_components`.

For every non-anchor solve, select columns using
`meta["assigned_indices"]` in seed order and copy
`meta["assigned_match_overlap"]` into the corresponding grid position. Pass
those selected columns, not the entire candidate eigenspace, as
`previous_vectors` on the next step. This is what makes the synthetic
permutation test follow `q0` instead of energy ordering.

- [ ] **Step 8: Verify branch tracking GREEN**

Run the Step 6 command. Expected: 1 passed.

- [ ] **Step 9: Add the failing schema contract changes**

Update `test_build_output_separates_authoritative_and_comparison_models`:

```python
assert output["schema_version"] == 3
assert set(output["full_pair"]["fields"]) == {
    "20.0", "40.0", "60.0", "160.0"
}
assert output["params"]["potential_fields_gauss"] == [20.0, 40.0, 60.0]
assert output["params"]["potential_distance_points"] == 81
```

Change its fake `calculate_full_pair_field` to return an
`rr_potential_curves` sentinel only for values in `POTENTIAL_FIELDS_G`, then assert 160 G has no curve record.

- [ ] **Step 10: Run the schema test and verify RED**

```bash
uv run --with pytest pytest tests/test_check_297_pair_channels.py \
  -k build_output_separates_authoritative -v
```

Expected: failure because the output is still schema 2 and has only two fields.

- [ ] **Step 11: Connect ARC to the branch tracker**

Implement:

```python
def calculate_rr_potential_curves(calc) -> dict:
    distances = potential_distance_grid()
    bare_index = find_basis_state_index(
        calc.basisStates,
        (N, L, J, -1.5, N, L, J, -1.5),
    )
    reference_ghz = float(np.real(calc.matDiagonal[bare_index, bare_index]))

    def solve(distance_um, previous_vectors):
        return extract_curve_eigenpairs(
            assemble_pair_hamiltonian(calc, float(distance_um)),
            reference_ghz,
            bare_index,
            previous_vectors=previous_vectors,
            initial_k=POTENTIAL_INITIAL_EIGENPAIRS,
            max_k=POTENTIAL_MAX_EIGENPAIRS,
            capture_target=POTENTIAL_CAPTURE_TARGET,
            match_target=POTENTIAL_MATCH_TARGET,
        )

    return track_rr_branches(
        distances,
        solve,
        bare_index=bare_index,
        reference_ghz=reference_ghz,
        basis_states=calc.basisStates,
        branch_count=POTENTIAL_BRANCH_COUNT,
        anchor_um=3.0,
    )
```

Call it inside `calculate_full_pair_field` before discarding `calc`, only when `b_gauss in POTENTIAL_FIELDS_G`. Record its elapsed time separately as `potential_curve_s`. Migrate all field loops to `FULL_PAIR_FIELDS_G`, bump `schema_version` to 3, and add the curve constants to `params`.

- [ ] **Step 12: Verify schema and all fast tests GREEN**

```bash
uv run --with pytest pytest tests/test_check_297_pair_channels.py -m 'not slow' -q
```

Expected: all fast tests pass.

- [ ] **Step 13: Add and run a reduced real-ARC curve test**

Add this slow test. It builds the same reduced ARC basis used by the existing
Bz characterization test but exercises the real distance-dependent matrices:

```python
@pytest.mark.slow
def test_reduced_arc_tracks_rr_branches_across_distance():
    from arc import PairStateInteractions, Rubidium87

    calc = PairStateInteractions(
        Rubidium87(),
        53, 1, 1.5,
        53, 1, 1.5,
        -1.5, -1.5,
        interactionsUpTo=1,
    )
    calc.defineBasis(np.pi / 2, 0.0, 1, 2, 1e9, Bz=20e-4)
    bare_index = pair.find_basis_state_index(
        calc.basisStates,
        (53, 1, 1.5, -1.5, 53, 1, 1.5, -1.5),
    )
    reference_ghz = float(np.real(calc.matDiagonal[bare_index, bare_index]))
    distances = np.array([2.9, 3.0, 3.1])

    def solve(distance_um, previous_vectors):
        cap = min(32, len(calc.basisStates) - 2)
        return pair.extract_curve_eigenpairs(
            pair.assemble_pair_hamiltonian(calc, distance_um),
            reference_ghz,
            bare_index,
            previous_vectors=previous_vectors,
            initial_k=min(8, cap),
            max_k=cap,
            capture_target=0.90,
            match_target=0.10,
        )

    result = pair.track_rr_branches(
        distances,
        solve,
        bare_index=bare_index,
        reference_ghz=reference_ghz,
        basis_states=calc.basisStates,
        branch_count=2,
        anchor_um=3.0,
    )

    assert result["distance_um"] == [2.9, 3.0, 3.1]
    assert len(result["branches"]) == 2
    for branch in result["branches"]:
        assert len(branch["shift_mhz"]) == 3
        assert np.all(np.isfinite(branch["shift_mhz"]))
        assert np.all(np.asarray(branch["rr_overlap"]) >= 0.0)
        assert np.all(np.asarray(branch["rr_overlap"]) <= 1.0)
        assert branch["min_adjacent_match_overlap"] >= 0.10
```

Run:

```bash
HOME=/tmp/arc297home MPLCONFIGDIR=/tmp/mpl297 \
uv run --with pytest pytest tests/test_check_297_pair_channels.py -m slow -v
```

Expected: both ARC slow tests pass.

- [ ] **Step 14: Commit Task 2 paths only**

```bash
git add scripts/check_297_pair_channels.py tests/test_check_297_pair_channels.py
git commit --only scripts/check_297_pair_channels.py tests/test_check_297_pair_channels.py \
  -m "feat: track rr pair-potential branches"
```

---

### Task 3: Render the three field-resolved figures and support replay

**Files:**
- Modify: `tests/test_check_297_pair_channels.py`
- Modify: `scripts/check_297_pair_channels.py`

**Interfaces:**
- Produces: `rr_potential_figure_path(output_dir: Path, b_gauss: float) -> Path`
- Produces: `render_rr_potential_figures(result: dict, output_dir: Path) -> list[Path]`
- Changes: `main(argv: list[str] | None = None)` accepts `--plot-only`.

- [ ] **Step 1: Add a failing synthetic render test**

Add `json` to the test imports, then add this schema-3 fixture. It uses five
distances and six branches without invoking ARC:

```python
def synthetic_curve_result():
    distances = np.array([2.5, 3.0, 4.0, 6.0, 8.0])
    fields = {}
    for field in (20.0, 40.0, 60.0):
        branches = []
        for index in range(6):
            rank = index + 1
            anchor_shift = (-1.0) ** rank * (20.0 + 15.0 * index) + field
            shifts = anchor_shift + (distances - 3.0) * (4.0 + 2.0 * index)
            overlaps = np.clip(
                0.82 - 0.11 * index - 0.04 * np.abs(distances - 3.0),
                0.01,
                1.0,
            )
            branches.append({
                "anchor_rank": rank,
                "anchor_shift_mhz": float(anchor_shift),
                "anchor_rr_overlap": float(overlaps[1]),
                "anchor_top_components": [{
                    "state": f"synthetic component {rank}",
                    "weight": float(overlaps[1]),
                }],
                "shift_mhz": shifts.tolist(),
                "rr_overlap": overlaps.tolist(),
                "adjacent_match_overlap": [1.0, 0.99, 0.98, 0.97, 0.96],
                "min_adjacent_match_overlap": 0.96,
            })
        fields[str(field)] = {
            "rr_potential_curves": {
                "anchor_um": 3.0,
                "branch_count": 6,
                "distance_um": distances.tolist(),
                "candidate_eigenpairs": [64] * distances.size,
                "captured_rr_overlap": [0.999] * distances.size,
                "branches": branches,
            }
        }
    return {
        "schema_version": 3,
        "params": {
            "potential_fields_gauss": [20.0, 40.0, 60.0],
            "weak_shift_threshold_mhz": 83.07,
        },
        "full_pair": {"fields": fields},
    }
```

Add:

```python
def test_render_rr_potential_figures_writes_three_named_images(tmp_path):
    result = synthetic_curve_result()

    paths = pair.render_rr_potential_figures(result, tmp_path)

    assert [path.name for path in paths] == [
        "pair_rr_potential_B20G.png",
        "pair_rr_potential_B40G.png",
        "pair_rr_potential_B60G.png",
    ]
    assert all(path.stat().st_size > 10_000 for path in paths)
```

- [ ] **Step 2: Run the render test and verify RED**

```bash
MPLCONFIGDIR=/tmp/mpl297 uv run --with pytest pytest \
  tests/test_check_297_pair_channels.py -k render_rr_potential -v
```

Expected: failure because `render_rr_potential_figures` is absent.

- [ ] **Step 3: Implement deterministic Agg rendering**

Import Matplotlib lazily inside the renderer, set the `Agg` backend before
importing `pyplot`, and use:

```python
FIGURE_STEM = "pair_rr_potential_B{field:g}G.png"


def rr_potential_figure_path(output_dir: Path, b_gauss: float) -> Path:
    return output_dir / FIGURE_STEM.format(field=b_gauss)
```

In `render_rr_potential_figures`:

- validate that the three plot fields and six branches exist;
- compute one global maximum absolute finite shift from all 18 curves;
- use a `(9.6, 7.2)` two-row figure with shared x axis;
- call `ax_energy.set_yscale("symlog", linthresh=WEAK_SHIFT_THRESHOLD_MHZ)`;
- apply common symmetric y limits to all figures;
- shade the weak window with `axhspan`, mark zero and 3 micrometres;
- use `tab10` colours 0 through 5 in anchor-rank order;
- plot the full energy line and scatter every fourth point with size
  `12 + 70 * rr_overlap`;
- plot the lower-panel overlap on `[0, 1.02]`;
- construct each legend entry as
  `branch {rank}: Δ3={shift:+.1f} MHz, p3={overlap:.3f}, {largest_state}`;
- save with `dpi=200`, `bbox_inches="tight"`, then close the figure.

- [ ] **Step 4: Verify render GREEN and inspect test artifacts**

Run Step 2. Expected: 1 passed. Open one generated temporary PNG if pytest
retains it; otherwise call the renderer once on the fixture in a temporary
directory and inspect it with `view_image`.

- [ ] **Step 5: Add a failing plot-only behavior test**

Refactor `main` to accept an optional argument list, then add a test that
writes the synthetic JSON to a temporary path, monkeypatches `pair.OUT`, and
monkeypatches `pair.build_output` to raise if called:

```python
def test_plot_only_reads_json_without_running_arc(tmp_path, monkeypatch):
    output = tmp_path / "pair_channels.json"
    output.write_text(json.dumps(synthetic_curve_result()))
    monkeypatch.setattr(pair, "OUT", output)
    monkeypatch.setattr(
        pair,
        "build_output",
        lambda atom: (_ for _ in ()).throw(AssertionError("ARC was called")),
    )

    pair.main(["--plot-only"])

    assert (tmp_path / "pair_rr_potential_B20G.png").exists()
    assert (tmp_path / "pair_rr_potential_B40G.png").exists()
    assert (tmp_path / "pair_rr_potential_B60G.png").exists()
```

Add `json` to the test imports.

- [ ] **Step 6: Run plot-only test and verify RED**

```bash
MPLCONFIGDIR=/tmp/mpl297 uv run --with pytest pytest \
  tests/test_check_297_pair_channels.py -k plot_only -v
```

Expected: `main` rejects the argument or calls ARC.

- [ ] **Step 7: Implement CLI replay and default rendering**

Use `argparse` with one flag:

```python
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="render pair-potential PNGs from the existing JSON",
    )
    return parser.parse_args(argv)
```

In `main(argv=None)`, if `plot_only`, load `OUT`, render to `OUT.parent`, print
the paths, and return before importing or constructing `Rubidium87`. On the
default path, build output, write JSON, then call the same renderer. End with
`main()` under the script guard.

- [ ] **Step 8: Verify plot-only and all fast tests GREEN**

```bash
MPLCONFIGDIR=/tmp/mpl297 uv run --with pytest pytest \
  tests/test_check_297_pair_channels.py -m 'not slow' -q
```

Expected: all fast tests pass.

- [ ] **Step 9: Commit Task 3 paths only**

```bash
git add scripts/check_297_pair_channels.py tests/test_check_297_pair_channels.py
git commit --only scripts/check_297_pair_channels.py tests/test_check_297_pair_channels.py \
  -m "feat: plot rr pair-potential curves"
```

---

### Task 4: Production calculation, figure review, and results report

**Files:**
- Modify: `results/297_to_calibration/pair_channels.json`
- Modify: `results/297_to_calibration/README.md`
- Modify: `results/README.md`
- Generate, do not track: `results/297_to_calibration/pair_rr_potential_B20G.png`
- Generate, do not track: `results/297_to_calibration/pair_rr_potential_B40G.png`
- Generate, do not track: `results/297_to_calibration/pair_rr_potential_B60G.png`

**Interfaces:**
- Consumes: the schema-3 producer and renderer from Tasks 2–3.
- Produces: final JSON, three locally rendered figures, appendix interpretation, provenance, and copy-pasteable replay/full-run commands.

- [ ] **Step 1: Run all pair-channel tests before production**

```bash
MPLCONFIGDIR=/tmp/mpl297 uv run --with pytest pytest \
  tests/test_check_297_pair_channels.py -m 'not slow' -q
HOME=/tmp/arc297home MPLCONFIGDIR=/tmp/mpl297 \
uv run --with pytest pytest tests/test_check_297_pair_channels.py -m slow -q
```

Require both commands to pass.

- [ ] **Step 2: Run the full deterministic ARC calculation**

```bash
HOME=/tmp/arc297home MPLCONFIGDIR=/tmp/mpl297 \
.venv/bin/python scripts/check_297_pair_channels.py
```

This overwrites `pair_channels.json` and generates the three PNGs. Keep the
command in a pollable terminal session and report progress at least once per
minute while it runs.

- [ ] **Step 3: Validate the generated curve schema numerically**

Run:

```bash
jq -e '
  .schema_version == 3 and
  (.params.potential_fields_gauss == [20,40,60]) and
  ([.full_pair.fields["20.0"], .full_pair.fields["40.0"],
    .full_pair.fields["60.0"]]
   | all(.rr_potential_curves.branch_count == 6)) and
  ([.full_pair.fields["20.0"], .full_pair.fields["40.0"],
    .full_pair.fields["60.0"]]
   | all((.rr_potential_curves.distance_um | length) == 81)) and
  ([.full_pair.fields["20.0"], .full_pair.fields["40.0"],
    .full_pair.fields["60.0"]]
   | all([.rr_potential_curves.branches[].min_adjacent_match_overlap]
         | min >= 0.25))
' results/297_to_calibration/pair_channels.json
```

Also print, for each field and branch, anchor shift, anchor overlap, minimum
matching overlap, and min/max shift. Use those exact fields when writing the
README; do not transcribe values from terminal memory.

- [ ] **Step 4: Exercise the cheapest replay path**

```bash
HOME=/tmp/arc297home MPLCONFIGDIR=/tmp/mpl297 \
.venv/bin/python scripts/check_297_pair_channels.py --plot-only
```

Record that this command was actually verified and confirm it does not change
the JSON checksum.

- [ ] **Step 5: Inspect all three figures visually**

Use `view_image` on each PNG. Check:

- all six energy branches and six overlap curves are visible;
- the energy and distance axes match across all fields;
- the weak-window shading, zero line, and 3-micrometre guide are legible;
- legends do not cover data or clip long state labels;
- no branch contains a discontinuity inconsistent with its stored adjacent
  match diagnostic;
- marker sizes visibly encode `rr` overlap without hiding curves.

If a visual defect is found, add a failing renderer test when possible before
changing code, rerun the fast suite, regenerate with `--plot-only`, and inspect
again.

- [ ] **Step 6: Update the README appendix without overwriting user edits**

First inspect:

```bash
git diff -- results/297_to_calibration/README.md
```

Preserve that diff. Add a subsection after “显式基结果” that:

- defines $\Delta_k(B,R)/h$ and $p_k(B,R)$;
- explains seeding at 3 micrometres and bidirectional Hungarian matching;
- embeds the three images using their relative filenames;
- gives an anchor table with six rows per field from
  `full_pair.fields.<B>.rr_potential_curves.branches`;
- reports minimum adjacent-match diagnostics and any visible avoided crossing;
- states that colour identifies an adiabatic numerical branch while the lower
  panel measures its instantaneous `rr` relevance;
- warns that exact-degenerate eigenvectors are basis-dependent;
- does not call $p_k$ a gate leakage probability.

Update the data table to schema 3 and four single-distance fields. Add the
verified cheap replay command before the full ARC command:

```bash
HOME=/tmp/arc297home MPLCONFIGDIR=/tmp/mpl297 \
.venv/bin/python scripts/check_297_pair_channels.py --plot-only
```

Update `results/README.md` so the `297_to_calibration` row mentions the
20/40/60 G `rr` potential curves and date 2026-08-10.

- [ ] **Step 7: Run the mandatory results-report validator**

```bash
.venv/bin/python .claude/skills/results-report/validate.py \
  results/297_to_calibration/README.md results/README.md
```

Expected:

```text
ok       results/297_to_calibration/README.md
ok       results/README.md

2/2 reports pass
```

- [ ] **Step 8: Run final focused and regression verification**

```bash
MPLCONFIGDIR=/tmp/mpl297 uv run --with pytest pytest \
  tests/test_check_297_pair_channels.py -m 'not slow' -q
HOME=/tmp/arc297home MPLCONFIGDIR=/tmp/mpl297 \
uv run --with pytest pytest tests/test_check_297_pair_channels.py -m slow -q
uv run --with pytest pytest \
  tests/core/test_rb87_297_system.py \
  tests/protocols/test_direct_297_cz_protocol.py \
  tests/physics/test_direct_297_physics.py \
  tests/protocols/test_direct_297_protocol.py -q
```

Also run `python -m py_compile` on the script and test, `git diff --check` on
all task paths, and the JSON `jq -e` assertion from Step 3.

- [ ] **Step 9: Commit tracked result paths only**

Do not stage any PNG. Stage only:

```bash
git add \
  results/297_to_calibration/pair_channels.json \
  results/297_to_calibration/README.md \
  results/README.md
git commit --only \
  results/297_to_calibration/pair_channels.json \
  results/297_to_calibration/README.md \
  results/README.md \
  -m "docs: report rr pair-potential curves"
```

- [ ] **Step 10: Hand off with exact evidence**

Report the three figure paths, JSON schema version, field/grid/branch counts,
minimum continuity diagnostics, production elapsed time, `--plot-only`
verification, test totals, validator output, commits, and the unrelated dirty
files that were preserved. State explicitly that the PNGs are intentionally
untracked and regenerable.
