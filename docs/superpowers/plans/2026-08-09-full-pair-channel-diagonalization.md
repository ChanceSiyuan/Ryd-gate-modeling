# Full Pair-Channel Diagonalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the explicit ARC pair-basis Hamiltonian with field-dependent intermediate-state energies the authoritative 53P pair-interaction diagnosis, retain the old effective-C6 calculation only as a comparison, and regenerate the documented result.

**Architecture:** Keep this study-specific workflow in `scripts/check_297_pair_channels.py`. Add small, testable sparse-matrix and spectral-summary functions to that script; build one ARC pair Hamiltonian per magnetic field and reuse it for both bare channels. Store authoritative and comparison models in separate JSON sections, then rewrite the existing result README from the regenerated values.

**Tech Stack:** Python 3.11, NumPy, SciPy sparse matrices/ARPACK, ARC 3.10.2, pytest, JSON, Markdown.

## Global Constraints

- The authoritative model is ARC `defineBasis(..., Bz=...)` with `nRange=5`, `lrange=2`, `energyDelta=30e9`, `interactionsUpTo=1`, `theta=pi/2`, `phi=0`, and `R=3 um`.
- Magnetic fields are exactly 20 G and 160 G and are converted to tesla before passing to ARC.
- The README must call this a truncated explicit pair-basis calculation with ARC's linear Zeeman approximation, not an unqualified exact Hamiltonian.
- The old `getC6perturbatively + PP-manifold Zeeman` path remains only under `effective_c6_comparison`.
- `radial_defect_ranking` must state that it omits angular factors and denominator sign.
- Preserve all unrelated tracked, untracked, and staged user changes. Every commit command must use explicit task paths.
- A write to `results/297_to_calibration/` is incomplete until the mandatory `results-report` validation succeeds.

---

### Task 1: Sparse Hamiltonian and spectral diagnostics

**Files:**
- Create: `tests/test_check_297_pair_channels.py`
- Modify: `scripts/check_297_pair_channels.py`

**Interfaces:**
- Produces: `assemble_pair_hamiltonian(calc, spacing_um) -> scipy.sparse.csr_matrix`
- Produces: `find_basis_state_index(basis_states, quantum_numbers) -> int`
- Produces: `extract_local_eigenpairs(hamiltonian, reference_ghz, bare_index, weak_threshold_mhz, *, initial_k, max_k, capture_target) -> tuple[np.ndarray, np.ndarray, dict]`
- Produces: `summarize_eigenpairs(eigenvalues_ghz, eigenvectors, *, reference_ghz, bare_index, basis_states, target_manifold_indices, weak_threshold_mhz, report_overlap_cutoff) -> dict`

- [ ] **Step 1: Read the TDD test-quality rules**

Read `superpowers/test-driven-development/writing-good-tests.md` completely before writing the first test.

- [ ] **Step 2: Create the script-import fixture and failing assembly/reference tests**

Use the repository's existing script-test pattern:

```python
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_297_pair_channels", ROOT / "scripts" / "check_297_pair_channels.py"
)
pair = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pair
SPEC.loader.exec_module(pair)


def test_assemble_pair_hamiltonian_applies_arc_distance_powers():
    spacing_um = 2.0
    r_m = spacing_um * 1e-6
    calc = SimpleNamespace(
        matDiagonal=diags([0.1, 0.2], format="csr"),
        matR=[csr_matrix(np.array([[0.0, 0.05], [0.05, 0.0]]) * r_m**3)],
    )

    actual = pair.assemble_pair_hamiltonian(calc, spacing_um).toarray()

    np.testing.assert_allclose(actual, [[0.1, 0.05], [0.05, 0.2]])


def test_find_basis_state_index_requires_one_exact_pair_state():
    states = [[53, 1, 1.5, -1.5, 53, 1, 1.5, -1.5]]
    target = (53, 1, 1.5, -1.5, 53, 1, 1.5, -1.5)
    assert pair.find_basis_state_index(states, target) == 0
    with pytest.raises(ValueError, match="exactly one"):
        pair.find_basis_state_index(states, (*target[:-1], -0.5))
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
pytest tests/test_check_297_pair_channels.py -m "not slow" -v
```

Expected: failures because `assemble_pair_hamiltonian` and
`find_basis_state_index` do not exist.

- [ ] **Step 4: Implement only Hamiltonian assembly and exact state lookup**

Add:

```python
def assemble_pair_hamiltonian(calc, spacing_um: float):
    matrix = calc.matDiagonal.copy()
    distance_m = spacing_um * 1e-6
    for power, term in enumerate(calc.matR, start=3):
        matrix = matrix + term / distance_m**power
    return matrix.tocsr()


def find_basis_state_index(basis_states, quantum_numbers) -> int:
    matches = [
        i for i, state in enumerate(basis_states)
        if tuple(state[:8]) == tuple(quantum_numbers)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one basis state; found {len(matches)}")
    return matches[0]
```

- [ ] **Step 5: Verify GREEN**

Run the focused command from Step 3 and require both tests to pass.

- [ ] **Step 6: Add failing adaptive-spectrum and summary tests**

Add tests that use a ten-state synthetic sparse Hamiltonian:

```python
def test_extract_local_eigenpairs_expands_until_weak_window_is_bracketed():
    values = np.array([-0.20, -0.12, -0.06, -0.01, 0.02,
                       0.07, 0.11, 0.15, 0.25, 0.40])
    eigenvalues, eigenvectors, meta = pair.extract_local_eigenpairs(
        diags(values, format="csr"),
        reference_ghz=0.005,
        bare_index=3,
        weak_threshold_mhz=80.0,
        initial_k=4,
        max_k=8,
        capture_target=0.99,
    )
    shifts_mhz = (eigenvalues - 0.005) * 1e3
    assert shifts_mhz.min() < -80.0
    assert shifts_mhz.max() > 80.0
    assert meta["window_bracketed"] is True
    assert meta["eigenpairs"] == 8
    assert eigenvectors.shape == (10, 8)


def test_extract_local_eigenpairs_rejects_unbracketed_window():
    values = np.linspace(-0.2, 0.2, 10)
    with pytest.raises(RuntimeError, match="did not bracket"):
        pair.extract_local_eigenpairs(
            diags(values, format="csr"), 0.0, 4, 150.0,
            initial_k=2, max_k=4, capture_target=0.99,
        )


def test_summarize_eigenpairs_uses_channel_reference_and_overlap_weights():
    eigenvalues = np.array([-0.05, 0.02, 0.12])
    eigenvectors = np.array([
        [0.5, np.sqrt(0.75), 0.0],
        [np.sqrt(0.75), -0.5, 0.0],
        [0.0, 0.0, 1.0],
    ])
    summary = pair.summarize_eigenpairs(
        eigenvalues, eigenvectors,
        reference_ghz=0.0,
        bare_index=0,
        basis_states=[[53, 1, 1.5, -1.5, 53, 1, 1.5, -1.5],
                      [53, 1, 1.5, -0.5, 53, 1, 1.5, -0.5],
                      [53, 0, 0.5, -0.5, 54, 0, 0.5, 0.5]],
        target_manifold_indices=[0, 1],
        weak_threshold_mhz=80.0,
        report_overlap_cutoff=0.01,
    )
    assert summary["weak_shift_weight"] == pytest.approx(1.0)
    assert summary["captured_overlap"] == pytest.approx(1.0)
    assert summary["states"][0]["overlap"] == pytest.approx(0.75)
    assert summary["states"][0]["shift_mhz"] == pytest.approx(20.0)
```

- [ ] **Step 7: Verify the new tests fail for missing behavior**

Run the focused command and confirm failures name the two missing spectral
functions.

- [ ] **Step 8: Implement deterministic adaptive ARPACK extraction and summaries**

Use `scipy.sparse.linalg.eigsh(..., sigma=reference_ghz, which="LM",
tol=1e-9)` with a deterministic normalized ramp as `v0`. Double `k` up to
`min(max_k, dimension - 2)`. Stop when the weak window is bracketed and the
captured bare overlap reaches `capture_target`; if the cap is reached with a
bracketed window, return the spectrum and mark capture convergence false; if
the window is not bracketed, raise `RuntimeError`.

`summarize_eigenpairs` must use absolute squares, convert GHz shifts to MHz,
sum weak weight only inside the threshold, sort reported states by bare
overlap, calculate target-manifold weight, and list the four largest basis
components using a compact spectroscopic label.

- [ ] **Step 9: Verify all fast tests pass**

Run the focused command and require zero failures.

- [ ] **Step 10: Commit Task 1 paths only**

```bash
git add scripts/check_297_pair_channels.py tests/test_check_297_pair_channels.py
git commit --only scripts/check_297_pair_channels.py tests/test_check_297_pair_channels.py \
  -m "test: define full pair-spectrum diagnostics"
```

---

### Task 2: Authoritative ARC full-pair calculation and JSON schema

**Files:**
- Modify: `scripts/check_297_pair_channels.py`
- Modify: `tests/test_check_297_pair_channels.py`

**Interfaces:**
- Consumes: all Task 1 helpers.
- Produces: `calculate_full_pair_field(atom, b_gauss) -> dict`
- Produces: `calculate_effective_c6_comparison(atom) -> dict`
- Produces top-level JSON keys `schema_version`, `params`, `full_pair`, `effective_c6_comparison`, and `radial_defect_ranking`.

- [ ] **Step 1: Add a failing authoritative-output contract test**

```python
def test_build_output_separates_authoritative_and_comparison_models(monkeypatch):
    monkeypatch.setattr(
        pair, "calculate_full_pair_field",
        lambda atom, b: {"b_gauss": b}, raising=False,
    )
    monkeypatch.setattr(
        pair, "calculate_effective_c6_comparison",
        lambda atom: {"model": "effective"}, raising=False,
    )
    monkeypatch.setattr(
        pair, "radial_defect_ranking", lambda atom: [], raising=False,
    )

    output = pair.build_output(object())

    assert set(output) == {
        "schema_version", "params", "full_pair",
        "effective_c6_comparison", "radial_defect_ranking",
    }
    assert output["full_pair"]["authoritative"] is True
    assert set(output["full_pair"]["fields"]) == {"20.0", "160.0"}
    assert output["effective_c6_comparison"]["authoritative"] is False
```

- [ ] **Step 2: Run the fast test and verify RED**

```bash
pytest tests/test_check_297_pair_channels.py \
  -m "not slow" \
  -k build_output_separates_authoritative_and_comparison_models -v
```

Expected: failure because `build_output` does not exist.

- [ ] **Step 3: Implement the full-pair and comparison output paths**

Implement `calculate_full_pair_field`, `calculate_effective_c6_comparison`,
`radial_defect_ranking`, and `build_output` as specified in Steps 4–6 below,
then rerun the contract test until it passes.

- [ ] **Step 4: Add the reduced-basis ARC integration test**

```python
@pytest.mark.slow
def test_arc_bz_changes_intermediate_pair_state_references():
    from arc import PairStateInteractions, Rubidium87

    calculations = []
    for b_tesla in (0.0, 20e-4):
        calc = PairStateInteractions(
            Rubidium87(), 53, 1, 1.5, 53, 1, 1.5,
            -1.5, -1.5, interactionsUpTo=1,
        )
        calc.defineBasis(np.pi / 2, 0.0, 1, 2, 1e9, Bz=b_tesla)
        calculations.append(calc)

    zero, field = calculations
    assert zero.basisStates == field.basisStates
    rr = pair.find_basis_state_index(
        zero.basisStates, (53, 1, 1.5, -1.5, 53, 1, 1.5, -1.5)
    )
    intermediate = [
        i for i, state in enumerate(zero.basisStates)
        if not (state[0:3] == [53, 1, 1.5] and state[4:7] == [53, 1, 1.5])
    ]
    relative_change_mhz = (
        field.matDiagonal.diagonal() - zero.matDiagonal.diagonal()
        - (field.matDiagonal[rr, rr] - zero.matDiagonal[rr, rr])
    ) * 1e3
    assert intermediate
    assert np.max(np.abs(relative_change_mhz[intermediate])) > 1.0
```

- [ ] **Step 5: Run the reduced-basis ARC integration test**

Run:

```bash
MPLCONFIGDIR=/tmp/mpl297 pytest -m slow \
  tests/test_check_297_pair_channels.py::test_arc_bz_changes_intermediate_pair_state_references -v
```

This verifies the ARC dependency behavior on which the already-red/green
authoritative output path relies; it is not used as a substitute for the RED
contract test.

- [ ] **Step 6: Complete `calculate_full_pair_field`**

For each field:

```python
calc = PairStateInteractions(
    atom, N, L, J, N, L, J, -1.5, -1.5, interactionsUpTo=1
)
calc.defineBasis(
    THETA, PHI, N_RANGE, L_MAX, ENERGY_DELTA_HZ,
    Bz=b_gauss * 1e-4,
)
hamiltonian = assemble_pair_hamiltonian(calc, R_UM)
```

Find the two bare-channel indices and all 16 target-manifold indices from
`calc.basisStates`. For each channel, use its `matDiagonal` entry as the bare
reference, extract the local spectrum, and summarize it. Record basis
dimension, Hamiltonian nonzeros, field, build time, diagonalization time, and
the channel records.

- [ ] **Step 7: Move the old calculation behind an explicit comparison name**

Extract the current perturbative reconstruction into
`calculate_effective_c6_comparison(atom)`. Preserve its numerical behavior but
rename `channels.*.dressed` to `channels.*.pp_zeeman_dressed`, attach a model
description and limitations, and keep the exchange element only in this
section.

Rename `intermediate_channel_inventory` to `radial_defect_ranking`; add fields
to its JSON description that state angular factors are omitted and absolute
defects are used for ranking.

- [ ] **Step 8: Verify the top-level output contract**

Factor JSON assembly into `build_output(atom)`. The Step 1 test is limited to
schema wiring; numerical behavior remains covered by real sparse-matrix tests
and the reduced ARC test.

- [ ] **Step 9: Run fast and slow focused tests**

```bash
pytest tests/test_check_297_pair_channels.py -m "not slow" -v
MPLCONFIGDIR=/tmp/mpl297 pytest -m slow tests/test_check_297_pair_channels.py -v
```

Require zero failures.

- [ ] **Step 10: Commit Task 2 paths only**

```bash
git add scripts/check_297_pair_channels.py tests/test_check_297_pair_channels.py
git commit --only scripts/check_297_pair_channels.py tests/test_check_297_pair_channels.py \
  -m "feat: diagonalize explicit Rydberg pair basis"
```

---

### Task 3: Regenerate and inspect the authoritative result

**Files:**
- Modify: `results/297_to_calibration/pair_channels.json`

**Interfaces:**
- Consumes: `build_output(atom)` from Task 2.
- Produces: generated schema-versioned numerical evidence for README Task 4.

- [ ] **Step 1: Read the mandatory results-report skill completely**

Read `.claude/skills/results-report/SKILL.md` and every directly required
template/validation instruction before writing new result data.

- [ ] **Step 2: Run the production calculation**

```bash
HOME=/tmp/arc297home MPLCONFIGDIR=/tmp/mpl297 \
  uv run python scripts/check_297_pair_channels.py
```

Allow roughly four minutes. Record the actual elapsed time printed by the
script.

- [ ] **Step 3: Inspect physical and numerical convergence fields**

```bash
jq '{schema_version, params, full_pair, effective_c6_comparison: {
  model: .effective_c6_comparison.model,
  channels: .effective_c6_comparison.channels
}}' results/297_to_calibration/pair_channels.json
```

For both fields and both channels require `window_bracketed=true`. Report, but
do not silently hide, any `capture_converged=false`. Check that field-specific
basis dimensions and Hamiltonian nonzero counts are recorded.

- [ ] **Step 4: Re-run the script for determinism**

Run the production command again and compare the physics sections while
excluding elapsed-time provenance. Use `jq -S` to normalize both copies. Any
physics mismatch must be fixed before documentation.

---

### Task 4: Rewrite the appendix and validate the result report

**Files:**
- Modify: `results/297_to_calibration/README.md`
- Modify: `results/297_to_calibration/pair_channels.json`

**Interfaces:**
- Consumes: generated values and timings from Task 3.
- Produces: a self-contained five-section-compliant result report whose
  appendix identifies the authoritative full-pair model and its limitations.

- [ ] **Step 1: Update provenance and reproduction instructions**

Update the artefact table entry for `pair_channels.json`, the script command,
runtime, ARC version, basis dimension, truncations, and schema names. Do not
change unrelated calibration provenance.

- [ ] **Step 2: Rewrite the appendix around the explicit Hamiltonian**

The appendix must:

- define the retained pair basis and show `matDiagonal + matR[0]/R^3`;
- state that `Bz` shifts every retained pair state before diagonalization;
- report the generated 20 G and 160 G spectra and weak weights;
- compare them with `effective_c6_comparison` without treating it as final;
- label `radial_defect_ranking` as non-angular screening only;
- explain sparse local-spectrum extraction and captured overlap;
- list the remaining `n`, `l`, energy-window, dipole-only, linear-Zeeman,
  hyperfine, diamagnetic, and inter-`j` mixing limitations;
- revise any conclusion about whether the 20 G or 160 G gate model is
  self-consistent according to the newly generated evidence.

- [ ] **Step 3: Run the mandatory result validator**

```bash
python .claude/skills/results-report/validate.py \
  results/297_to_calibration
```

Require a successful exit and retain its output as completion evidence.

- [ ] **Step 4: Run final targeted and repository checks**

```bash
pytest tests/test_check_297_pair_channels.py -m "not slow" -v
MPLCONFIGDIR=/tmp/mpl297 pytest -m slow tests/test_check_297_pair_channels.py -v
pytest tests/test_documentation_structure.py tests/test_readme_examples.py -v
```

Require zero failures.

- [ ] **Step 5: Review the surgical diff**

```bash
git diff -- scripts/check_297_pair_channels.py \
  tests/test_check_297_pair_channels.py \
  results/297_to_calibration/pair_channels.json \
  results/297_to_calibration/README.md
```

Every changed line must trace to the approved model replacement or its
verification. Do not stage any unrelated pre-existing change.

- [ ] **Step 6: Commit only the implementation result paths**

```bash
git add scripts/check_297_pair_channels.py \
  tests/test_check_297_pair_channels.py \
  results/297_to_calibration/pair_channels.json \
  results/297_to_calibration/README.md
git commit --only scripts/check_297_pair_channels.py \
  tests/test_check_297_pair_channels.py \
  results/297_to_calibration/pair_channels.json \
  results/297_to_calibration/README.md \
  -m "docs: validate full 53P pair interaction model"
