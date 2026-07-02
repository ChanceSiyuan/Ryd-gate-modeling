# Backend Simplification Plan

## Purpose

The goal is to make this repository a focused Rydberg lattice simulator with a
small, readable backend surface:

- `exact`: reference and small-system production backend.
- `mps`: 1D and quasi-1D tensor-network backend.
- `peps`: 2D rectangular-grid PEPS backend, implemented through YASTN.

This plan is not TFIM-only. The core should continue to represent general
Rydberg lattice problems: `1r`, `01r`, `analog_3` (`g/e/r` physical ladder),
`rb87_7`, and hand-built `LevelStructureSpec` models where applicable. TFIM is
one preset/workflow on top of that core, not the organizing principle of the
backend architecture.

The simplification target is:

1. Keep physical problem construction general.
2. Keep backend capabilities explicit.
3. Stop maintaining multiple competing PEPS engines in the public path.
4. Move experimental or external bridge backends out of the main dispatcher.
5. Make the YASTN PEPS adapter small enough to read and reason about.

## Current Architecture Review

### Public entry points

The public entry point is `ryd_gate.simulate.simulate`. It routes exact backends
directly and routes tensor-network backends through `tn_common.simulate_tn`.

Current public backend names include:

- `exact_dense`
- `exact_sparse`
- `mps`
- `peps`
- `gputn`
- `pepskit`

For `backend="peps"`, there is an additional hidden backend selector:

```python
backend_options={"engine_package": "rydtn" | "yastn"}
```

This is the main source of API confusion. Users see one backend name, but the
actual algorithm changes through a second option. It also makes capability
reporting inaccurate because "peps" does not mean one concrete implementation.

### Core problem layer

The core layer is in good shape conceptually:

- `core/level_structures.py` defines the central local level presets.
- `core/system.py` builds `RydbergSystem` from geometry, level structure,
  interactions, and protocol.
- `ir.py` compiles a protocol-bound system into a backend-neutral
  `HamiltonianIR`.
- `protocols/` owns the time-dependent drive semantics.
- `lattice.py` owns atom positions and register geometry.

Important supported model classes:

- `1r`: two-level Rydberg lattice model.
- `01r`: three-level digital/analog lattice model.
- `analog_3`: physical `g/e/r` ladder with local blocks.
- `rb87_7`: seven-level Rb87 gate model.
- Custom `LevelStructureSpec`: symbolic/custom local models.

The core should stay general. The simplification should happen in backend
selection and backend lowering, not by reducing the problem model to TFIM.

### Tensor-network common layer

`backends/tn_common/` already provides useful shared pieces:

- `TNLatticeSpec`
- `TNCompiler`
- `TNProtocolContext`
- level-structure validation helpers
- `simulate_tn`

Current limitations:

- `TNLatticeSpec` assumes a rectangular 2D grid with `Lx`, `Ly`, and snake
  mappings.
- `tn_lattice_spec_from_system` rejects non-rectangular geometry.
- The capability system does not distinguish geometry constraints.
- The capability system also does not distinguish concrete PEPS engines.

This is acceptable for PEPS, but not as a universal TN problem spec. The
long-term model should distinguish a generic graph/lattice problem from a
rectangular-grid PEPS problem.

### PEPS layer

There are two PEPS implementations:

1. `src/ryd_gate/backends/peps2d.py`
   - YASTN-backed PEPS adapter.
   - Uses `yastn.tn.fpeps`.
   - Supports `1r` and `01r` today.
   - Explicitly rejects `analog_3` today.

2. `src/ryd_gate/backends/rydtn/`
   - Self-written dense PEPS engine.
   - Implements a dense NTU-NN update, not the full YASTN environment family.
   - Supports `analog_3` today.
   - Also contains its own PEPS tensors, gates, operators, measurements,
     boundary-MPS contraction, and backend selection.

This duplication is expensive:

- operator semantics are duplicated between YASTN PEPS and rydtn;
- Trotter schedule lowering is duplicated;
- measurements are duplicated;
- `rydtn` is the default PEPS backend even though YASTN is the professional
  PEPS implementation.

`peps2d.py` itself is too large and mixes too many responsibilities:

- lazy YASTN import and config;
- payload construction;
- initial state lowering;
- operator factory;
- real-time evolution;
- imaginary-time evolution;
- YASTN gate construction;
- update environment selection;
- CTM/BP measurement;
- CTM energy evaluation.

This file should be split into a small package.

### Other backend layer

Additional current backends:

- `gputn.py`: experimental cuTensorNet/CuPy backend.
- `pepskit.py`: Julia PEPSKit bridge.

Both are useful experiments, but they do not fit the desired core surface:
`exact`, `mps`, `peps`. They should be removed from the main dispatcher and
either deleted, moved to an experimental namespace, or maintained outside the
main package.

### Documentation and dependency inconsistencies

The README says the `tn-2d` extra includes YASTN, but `pyproject.toml` currently
puts YASTN in `tn-2d-validation`, not `tn-2d`.

Current capability reporting says `analog_3` supports `peps`, but the YASTN PEPS
adapter rejects `analog_3`. This is only true because the default PEPS engine is
currently `rydtn`. Once PEPS means YASTN, the capability matrix must be updated
or YASTN `analog_3` support must be implemented first.

## Feasibility Assessment

Reducing the public backend set to `exact`, `mps`, and `peps` is feasible.

The main blocker is not YASTN itself. YASTN can represent local Hilbert spaces of
dimension 2, 3, or larger when given the correct tensors and gates. The blocker
is the current adapter code:

- `01r` is already supported by the YASTN PEPS path.
- `analog_3` is currently rejected by policy/code, not by a fundamental YASTN
  limitation.
- `rb87_7` should remain exact-only for now because the full gate model is stiff,
  high-dimensional, and currently tied to exact local matrix blocks and decay/noise
  semantics.

Therefore the clean three-backend architecture is feasible if we do one of the
following:

1. Preferred: implement `analog_3` lowering in the YASTN PEPS adapter, then make
   YASTN the official `peps` backend.
2. Transitional: make YASTN the official `peps` backend for `1r` and `01r`, and
   make `analog_3 + backend="peps"` raise a clear error recommending `mps` or
   `rydtn_experimental`.

The preferred route is better because it preserves the current PEPS capability
claim for `analog_3`.

## Proposed Target Architecture

### Public backends

The public backend set should become:

```text
exact
mps
peps
```

Recommended aliases:

- `exact` should be the public exact backend family.
- Internally, keep dense/sparse solver selection through options:
  `solver="dense"` or `solver="sparse"`.
- If keeping the explicit names is important for backward compatibility, keep
  `exact_dense` and `exact_sparse` as accepted aliases, but document `exact` as
  the primary backend.

Remove from the public dispatcher:

- `gputn`
- `pepskit`
- `engine_package="rydtn"`
- `engine_package="yastn"`

Optional transitional aliases:

- `backend="rydtn_experimental"`
- `backend="gputn_experimental"`
- `backend="pepskit_experimental"`

These should not appear in the main README quickstart or capability matrix
unless the project explicitly wants an experimental section.

### PEPS package layout

Replace `backends/peps2d.py` with:

```text
src/ryd_gate/backends/peps/
  __init__.py
  backend.py
  loader.py
  capabilities.py
  lowering.py
  operators.py
  gates.py
  environments.py
  measurements.py
  energy.py
```

Responsibilities:

- `backend.py`
  - Owns `PEPSBackend`.
  - Runs real-time and imaginary-time orchestration.
  - Should be thin and mostly read like pseudocode.

- `loader.py`
  - Lazy-imports YASTN.
  - Builds YASTN config.
  - Handles CPU/GPU backend/device validation.

- `capabilities.py`
  - Declares supported level structures and geometry constraints.
  - Checks rectangular grid requirement.

- `lowering.py`
  - Converts `TNEvolutionIR` into a PEPS evolution plan.
  - Should not be named `build_yastn_peps_payload` after the refactor because
    the plan is physics/backend-adapter data, not YASTN itself.

- `operators.py`
  - Builds YASTN local vectors and operators.
  - Owns local operator factory for `1r`, `01r`, and `analog_3`.

- `gates.py`
  - Builds second-order Trotter gate layers.
  - Supports real-time and imaginary-time gates.
  - Caches distinct local and NN gates.

- `environments.py`
  - Constructs YASTN update environments: NTU, approximate, CTM.
  - Constructs measurement environments: BP, CTM.

- `measurements.py`
  - Implements `n_r`, `n_mean`, `n_0`, `n_1`, `n_g`, `n_e`, `sigma_z`,
    `m_s`, `czz`, and related observables.

- `energy.py`
  - Contains CTM energy evaluation for imaginary-time ground-state checks.

### Common lattice/problem specs

Introduce two explicit TN spec concepts:

```text
TNGraphSpec
RectGridSpec
```

`TNGraphSpec`:

- arbitrary site coordinates;
- arbitrary interaction pairs;
- level structure;
- protocol and schedule data;
- useful for exact, MPS, and future graph-based methods.

`RectGridSpec`:

- `Lx`, `Ly`;
- row-major and snake mappings;
- rectangular grid validation;
- required by PEPS.

Short-term implementation can keep `TNLatticeSpec`, but add an explicit
`geometry_kind` or helper:

```python
spec.is_rectangular_grid
spec.require_rectangular_grid("peps")
```

Long-term implementation should split the type.

### Capability system

Move from a backend-name-only truth table to concrete backend capabilities.

Suggested concrete backends:

```text
exact
mps
peps
```

Capabilities should include:

- level structures;
- geometry constraints;
- real-time support;
- imaginary-time/ground-state support;
- noise support;
- required optional dependencies;
- measurement environment options.

For example:

```text
exact:
  levels: 01, 1r, 01r, analog_3, rb87_7
  geometry: any finite register
  noise: supported where exact noise layer supports it

mps:
  levels: 1r, 01r, analog_3
  geometry: chain or ordered graph, depending on TeNPy lowering support

peps:
  levels: 1r, 01r, analog_3 after YASTN analog_3 support lands
  geometry: rectangular 2D grid only
  implementation: YASTN
```

Until YASTN `analog_3` support lands, mark:

```text
peps + analog_3: no
```

and make the error message explicit.

## YASTN analog_3 Support

The current YASTN adapter rejects `analog_3`. This should be changed by porting
the existing rydtn local-gate logic into the YASTN gate builder.

Current `analog_3` local Hamiltonian semantics:

```python
H(t) = static + coeff * drive_420 + conj(coeff) * drive_420.conj().T
```

The PEPS plan should carry:

- `local_blocks.static`
- `local_blocks.drive_420`
- `drive_coeffs["drive_420"]`

The YASTN operator factory should support generic matrix operators, not only
`X_01`, `X_1r`, `n_0`, `n_1`, `n_r`.

Implementation outline:

1. Add or reuse `YASTNPEPSOps.matrix(values)` support.
2. In PEPS gate builder, branch on `payload["lattice"]["local_blocks"]`.
3. If `local_blocks is not None`, build one spatially uniform local gate:

   ```python
   H = static + coeff * drive_420 + coeff.conjugate() * drive_420.conj().T
   gate = gates.gate_local_exp(step_coeff, ops.I, ops.matrix(H), site=coord)
   ```

4. Keep the existing `1r`/`01r` profile-based branch.
5. Add smoke and exact-comparison tests for a 2-site `analog_3` chain.

This makes `peps` a real replacement for the current rydtn default.

## Migration Plan

### Phase 0: Freeze behavior with tests

Before moving code, add characterization tests for current public behavior:

- exact small-system smoke for `1r`, `01r`, `analog_3`, `rb87_7`;
- MPS smoke for `1r`, `01r`, `analog_3`;
- YASTN PEPS smoke for `1r`, `01r`;
- PEPS rectangular geometry validation;
- PEPS non-rectangular geometry rejection;
- current `analog_3 + YASTN PEPS` rejection, if analog_3 support is not added
  in the same phase.

### Phase 1: Add explicit capabilities

Add `backends/capabilities.py` or similar.

Do not change runtime behavior yet. The first goal is to make the current truth
visible and testable.

Update docs generation to use the new capability objects instead of
`LevelStructureSpec.supports_backend` alone.

### Phase 2: Split YASTN PEPS adapter

Move `peps2d.py` into the new `backends/peps/` package.

Keep compatibility imports temporarily:

```python
# src/ryd_gate/backends/peps2d.py
from ryd_gate.backends.peps.backend import PEPSBackend as YASTNPEPSBackend
from ryd_gate.backends.peps.lowering import build_peps_evolution_plan as build_yastn_peps_payload
```

This avoids breaking tests and downstream code during the split.

### Phase 3: Implement YASTN analog_3

Add `analog_3` local matrix gates to the YASTN PEPS path.

Tests:

- `analog_3` PEPS smoke with `n_g`, `n_e`, `n_r`;
- exact vs PEPS comparison on a 2-site chain with a short pulse;
- conservation of per-site population for unitary `analog_3` evolution.

After this phase, `peps` can honestly support `analog_3`.

### Phase 4: Make YASTN the only public PEPS backend

Change `tn_common.simulate._run_peps`:

- remove `engine_package` from the public path;
- route `backend="peps"` directly to the YASTN PEPS backend;
- optionally accept `engine_package="yastn"` as a deprecated no-op for one
  release;
- reject `engine_package="rydtn"` with a migration message.

Suggested message:

```text
backend='peps' now uses YASTN. The old engine_package='rydtn' path moved to
backend='rydtn_experimental' and is not part of the stable API.
```

### Phase 5: Move or remove experimental backends

Move these out of the stable dispatcher:

- `rydtn`
- `gputn`
- `pepskit`

Options:

1. Delete them from the main package.
2. Move them to `src/ryd_gate/backends/experimental/`.
3. Move them to separate branches or separate packages.

Recommended:

- Move `rydtn` to `experimental/rydtn` for a short transition because it is still
  useful as a dense debug/reference path.
- Move or delete `gputn` and `pepskit` from the public dispatcher. If retained,
  mark them experimental and remove them from README quickstarts.

### Phase 6: Clean public API and docs

Update:

- README backend list;
- optional dependency extras;
- capability matrix;
- tests that reference old backend names;
- docstrings in `simulate.py` and `tn_common/simulate.py`;
- error messages.

Recommended dependency extras:

```toml
tn = ["physics-tenpy>=1.0"]
peps = ["yastn @ git+https://github.com/yastn/yastn.git"]
tn-2d = ["physics-tenpy>=1.0", "yastn @ git+https://github.com/yastn/yastn.git"]
```

If YASTN remains optional, `backend="peps"` should fail with a clear install
message.

## Proposed Final Public API

Examples:

```python
simulate(system, backend="exact", backend_options={"solver": "dense"})
simulate(system, backend="mps", backend_options={"chi_max": 128})
simulate(system, backend="peps", backend_options={"chi_max": 64, "update_environment": "ntu"})
```

If preserving explicit exact names:

```python
simulate(system, backend="exact_dense")
simulate(system, backend="exact_sparse")
```

but the documentation should prefer `backend="exact"` unless there is a strong
reason to keep solver choice in the backend name.

## Removal Checklist

Remove from stable docs and dispatcher:

- `backend="gputn"`
- `backend="pepskit"`
- `backend_options={"engine_package": "rydtn"}`
- `backend_options={"engine_package": "yastn"}`

Keep or migrate tests:

- Keep exact tests.
- Keep MPS tests.
- Keep YASTN PEPS tests.
- Convert current rydtn analog_3 tests to YASTN PEPS analog_3 tests after the
  implementation lands.
- Move rydtn internals tests to experimental tests or delete them.
- Delete public gputn/pepskit dispatch tests if those backends leave the stable
  API.

## Main Risks

1. YASTN `analog_3` support may expose missing generic local-matrix support in
   the current adapter.
   - Mitigation: implement generic matrix operators first and test on 2 sites.

2. Removing `rydtn` default may break users relying on `analog_3 + peps`.
   - Mitigation: implement YASTN `analog_3` before changing the default.

3. Changing exact backend names may create unnecessary churn.
   - Mitigation: keep `exact_dense` and `exact_sparse` aliases for now; simplify
     docs first, code later.

4. PEPS geometry restrictions may surprise users with triangular/custom lattices.
   - Mitigation: add capability checks and clear errors before backend execution.

5. Optional dependency behavior may become confusing if YASTN is not installed.
   - Mitigation: make the `peps` extra explicit and error messages actionable.

## Recommended Decision

Proceed with the three-backend architecture:

```text
exact
mps
peps
```

Use YASTN as the only stable PEPS implementation. Do not vendor YASTN source into
this repository. Treat the repository value as:

- problem construction;
- protocol lowering;
- model-specific schedules;
- observables and result objects;
- clean backend adapters;
- validation against exact/MPS on small systems.

Move the current self-written `rydtn` engine out of the stable public path after
YASTN `analog_3` support is in place.

## Script and Notebook Migration Requirement

The backend simplification is not complete until all runnable artifacts under
`scripts/` are updated to match the revised source layout and public API.

Required scope:

- Update every Python script under `scripts/**/*.py`.
- Update every notebook under `scripts/**/*.ipynb`.
- Preserve the original function of each script or notebook.
- Replace removed backend names and options with the new stable API:
  `exact`, `mps`, and `peps`.
- Remove or rewrite usages of `engine_package`, `rydtn`, `gputn`, and `pepskit`
  unless the artifact is explicitly moved to an experimental area.
- Update imports that reference moved modules such as `peps2d.py` or `rydtn`.
- Execute or smoke-check the migrated scripts/notebooks where practical.
- For notebooks that are expensive to run, at minimum validate imports,
  backend options, and code-cell syntax after migration.

This requirement should be part of the same migration, not left as a separate
cleanup task, because scripts and notebooks are user-facing examples of the
repository architecture.

## simplify.md Compliance Requirement

All implementation work for this backend simplification must follow `simplify.md`.
This plan is a simplification pass, not a feature-development pass.

Hard constraints for coding:

- Do not add new features, plugin systems, registries, compatibility layers, or abstractions for hypothetical future use.
- Prefer deletion, consolidation, renaming, and narrowing of interfaces over adding new modules or modes.
- Keep the public API minimal: top-level `ryd_gate` should expose only common stable workflows.
- Delete low-value wrappers, old aliases, shims, and forwarding functions unless README, examples, tests, or docs clearly require them.
- Narrow function signatures instead of preserving many unrelated input styles.
- Keep one concept in one place; do not duplicate model, schedule, observable, or backend semantics across modules.
- Keep backend internals internal; do not export backend helpers from top-level APIs.
- Do not change numerical or physical behavior covered by tests.
- Every code change must state how it reduces complexity: API surface, wrappers, file count, duplicated logic, or signature breadth.

Workflow constraints:

- Start with an audit-only phase before code edits.
- Produce the API simplification audit requested in `simplify.md`: current public API, usage, wrapper deletion candidates, over-flexible functions, consolidation candidates, proposed minimal API, and a patch plan.
- Limit implementation to at most 6 small patches after the audit is approved.
- Each patch should perform one kind of simplification only.
- Do not start by editing numerical backend internals unless the change is purely visibility, wrapper, import, or deletion related.
- After each patch, run or explicitly report the status of `uv run pytest -q`, `uv run ruff check src tests docs examples`, and `uv run mypy src/ryd_gate`.

If any part of this backend plan conflicts with `simplify.md`, `simplify.md` takes precedence.
