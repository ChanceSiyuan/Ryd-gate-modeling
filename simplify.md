# Prompt: Simplify `src/ryd_gate` and Its Public API

You are maintaining the repository `ChanceSiyuan/Ryd-gate-modeling`.

Treat this task as an **API and source-structure simplification pass**, not as a feature-development task.

The goal is to make `src/ryd_gate` smaller, clearer, easier to read, and easier to teach. Prefer deletion, consolidation, renaming, and narrowing of interfaces over adding new abstractions.

---

## Project Context

This repository models Rydberg neutral-atom many-body dynamics and gate protocols.

The main user-facing workflows should remain clear and stable.

### 1. Many-body / TFIM simulation workflow

```python
from ryd_gate import Register, RydbergSystem, TFIMQuenchProtocol, simulate

system = RydbergSystem.from_lattice(...)
result = simulate(system, ...)
```

### 2. CZ gate workflow

```python
from ryd_gate.gates import TOProtocol, cz_gate_report

report = cz_gate_report(...)
```

The current codebase has accumulated too much complexity in `src/ryd_gate`.

The problems I want you to address are:

- Redundant wrapper functions that only forward calls, rename functions, or preserve old compatibility layers.
- Functions with overly broad input signatures, too many optional arguments, too many modes, or too much implicit behavior.
- Too many scattered `.py` files containing small helper functions with unclear ownership.
- Too many exports from `ryd_gate.__init__`, making internal primitives look like stable public API.
- Too many adapters, aliases, shims, and compatibility paths for old scripts or notebooks.

---

## Main Goal

Simplify the `src/ryd_gate` package and its public API.

Focus on reducing:

- API surface area
- wrapper functions
- unnecessary compatibility code
- overly flexible function signatures
- scattered helper files
- unclear module boundaries

Do **not** expand the functionality of the project.

---

## Hard Rules

1. **Do not add new features.**  
   Do not add parameters, registries, factories, adapters, plugin systems, compatibility layers, or abstractions for hypothetical future use.

2. **Minimize the public API.**  
   Keep only the symbols that are actually used in README files, examples, tests, or clearly intended user-facing workflows.

3. **Delete low-value wrappers.**  
   If a function only forwards to another function, renames another function, or preserves an old calling style without adding real semantic value, delete it.

4. **Narrow function signatures.**  
   Do not let one function support many unrelated input styles. Prefer one clear signature over many optional modes.

5. **One concept should live in one place.**  
   Avoid having the same concept partially implemented across `core`, `ir`, `simulate`, `analysis`, `gates`, and `protocols`.

6. **Keep internal code internal.**  
   Backend helpers, dispatch helpers, numerical utilities, and private implementation details should not be exported from `ryd_gate.__init__`.

7. **Do not perform a large architecture rewrite.**  
   This is a simplification pass, not a framework redesign.

8. **Do not preserve unnecessary backward compatibility.**  
   Remove old aliases, shims, and compatibility wrappers unless README, examples, or tests clearly require them.

9. **Every change must reduce complexity.**  
   For every proposed change, explain whether it reduces API surface, removes a wrapper, consolidates files, or simplifies a function signature.

10. **Do not change numerical or physical behavior.**  
    Refactoring must not change simulation results, gate reports, fidelity calculations, noise behavior, or backend behavior covered by tests.

---

## Desired Package Shape

Aim for a structure where a reader can quickly understand the purpose of each module.

Suggested target structure:

```text
src/ryd_gate/
  __init__.py
  lattice.py
  ir.py
  simulate.py
  gates.py
  physics.py
  noise.py

  core/
    ...

  protocols/
    ...

  backends/
    ...

  analysis/
    ...

  schemas/
    ...
```

Use the following intended responsibilities:

### `core/`

Core data structures and physical system definitions only.

Examples:

- `RydbergSystem`
- level structures
- basis/model definitions
- core Hamiltonian-related data structures

Do not place high-level user convenience wrappers here.

---

### `protocols/`

User-selectable pulse and control protocols only.

Examples:

- `TFIMQuenchProtocol`
- `TFIMAnnealProtocol`
- `SweepProtocol`
- CZ-related pulse protocols if they are genuinely protocol definitions

Avoid excessive inheritance or abstract base classes unless they clearly remove repeated logic.

---

### `backends/`

Backend implementations only.

Examples:

- exact evolution
- MPS backend
- PEPS backend
- GPU backend
- backend-specific helpers

Backend internals should not be exported from top-level `ryd_gate`.

---

### `ir.py`

Cross-backend intermediate representation only.

Examples:

- Hamiltonian IR
- pulse IR
- `EvolutionResult`
- backend-independent result containers

Do not let this become a general utility file.

---

### `simulate.py`

The main simulation dispatcher only.

It should expose a clear `simulate(...)` function.

Avoid making `simulate(...)` responsible for too many old calling conventions, string modes, aliases, or hidden coercions.

---

### `lattice.py`

Geometry and register layout only.

Examples:

- `Register`
- `RegisterLayout`
- lattice constructors
- position geometry

---

### `gates.py`

User-facing CZ gate API only.

Examples:

- `TOProtocol`
- `ARProtocol`
- `DoubleARPProtocol`
- `cz_gate_report`
- `CZGateReport`

Avoid duplicating gate logic between `gates.py` and `protocols/gate_cz.py`.

---

### `analysis/`

User-level analysis APIs only.

Do not expose backend-internal metrics or implementation details as user-facing analysis functions.

---

### `physics.py`

Real physical helper functions only.

Do not let this become a miscellaneous dumping ground for unrelated helper functions.

---

## Proposed Minimal Public API

Public API should have two levels.

---

### 1. Top-level API: `ryd_gate`

The top-level import should stay small and beginner-friendly.

Suggested top-level exports:

```python
from ryd_gate import (
    Register,
    RydbergSystem,
    simulate,
    TFIMQuenchProtocol,
    TFIMAnnealProtocol,
    SweepProtocol,
    EvolutionResult,
)
```

Only include the most common and stable objects here.

Do not export backend internals, helper functions, private utilities, schemas, adapters, or low-level primitives from the top level.

---

### 2. Submodule APIs

Specialized APIs should live in submodules.

#### `ryd_gate.gates`

```python
from ryd_gate.gates import (
    TOProtocol,
    ARProtocol,
    DoubleARPProtocol,
    cz_gate_report,
    CZGateReport,
)
```

#### `ryd_gate.protocols`

Expose the complete protocol collection here.

#### `ryd_gate.analysis`

Expose user-facing analysis functions here.

#### `ryd_gate.backends`

Backend internals should generally remain internal and should not be treated as stable user-facing API.

---

## Specific Areas to Inspect

Pay special attention to these locations:

### `src/ryd_gate/__init__.py`

Check whether too many advanced primitives are exported.

Classify each export as:

- keep at top level
- move to submodule only
- make private
- delete

---

### `src/ryd_gate/simulate.py`

Check whether `simulate(...)` has accumulated too much compatibility logic.

Look for:

- too many optional arguments
- too many accepted input forms
- string modes that could be replaced by clearer objects
- hidden coercions
- wrapper functions around the actual dispatcher

---

### `src/ryd_gate/gates.py`

Check whether CZ gate APIs are duplicated with `protocols/gate_cz.py`.

Look for:

- duplicate protocol definitions
- wrapper functions
- report helpers that simply forward calls
- old aliases
- unclear distinction between user API and implementation detail

---

### `src/ryd_gate/protocols/`

Check whether protocol classes are over-abstracted.

Look for:

- unnecessary base classes
- inheritance that does not reduce real duplication
- too many constructor options
- protocol aliases or compatibility names

---

### `src/ryd_gate/backends/`

Check whether backend implementation details are exposed as public API.

Look for:

- backend dispatch helpers exported from top-level API
- backend-specific utilities imported by user-facing modules unnecessarily
- helper functions that should be private

---

### `src/ryd_gate/analysis/`

Check whether analysis APIs mix user-facing logic with backend-internal implementation details.

Look for:

- scattered metrics
- duplicated observables
- low-level numerical helpers exposed as analysis functions

---

### `src/ryd_gate/physics.py`

Check whether this file has become a miscellaneous helper module.

Classify functions into:

- real physics helpers to keep
- helpers that belong in a more specific module
- private implementation helpers
- unused functions to delete

---

## Workflow

Follow this workflow strictly.

---

## Phase 1: Audit Only

Do not modify files in this phase.

First inspect:

- `src/ryd_gate`
- `tests`
- `examples`
- `docs`
- `README.md`
- `pyproject.toml`
- any package-level `__init__.py` files

Then produce an audit report.

The audit report must include:

### Current Public API

List the current exported symbols from:

- `ryd_gate.__init__`
- `ryd_gate.gates`
- `ryd_gate.protocols`
- `ryd_gate.analysis`

For each symbol, mark one of:

- `keep`
- `move to submodule`
- `make private`
- `delete`

Also explain why.

---

### README / Examples / Tests Usage

List which symbols are actually used by:

- README
- examples
- tests
- docs

This should identify which APIs are truly user-facing and which are only accidentally public.

---

### Wrapper Deletion Candidates

Find functions or classes that are redundant wrappers.

Use this table format:

| Symbol | File | Current Role | Why It Is Redundant | Proposed Action | Risk | Tests / Docs To Update |
|---|---|---|---|---|---|---|

Possible actions:

- `delete`
- `inline`
- `merge`
- `rename`
- `make private`
- `keep`

Risk levels:

- `low`
- `medium`
- `high`

---

### Over-Flexible Function Candidates

Find functions with overly complex input signatures.

Use this table format:

| Function | File | Current Signature Problem | Proposed Simpler Signature | Migration Impact | Risk |
|---|---|---|---|---|---|

Look for:

- too many optional arguments
- parameters that accept many unrelated types
- string modes
- hidden defaults
- compatibility branches
- input coercion that belongs outside the function
- functions that behave like several functions combined

---

### Scattered Helper / File Consolidation Candidates

Find small modules or helpers that should be merged, moved, or made private.

Use this table format:

| Current File / Helper | Current Role | Proposed Destination | Reason | Risk |
|---|---|---|---|---|

---

## Phase 2: Propose the Target Structure

Do not modify files yet.

Propose a simpler target structure for `src/ryd_gate`.

Include:

- which files should remain
- which files should be merged
- which files should be deleted
- which helpers should become private
- which modules should own which concepts

Use this format:

```text
src/ryd_gate/
  __init__.py          # public top-level API only
  lattice.py           # register and geometry
  ir.py                # cross-backend IR and result containers
  simulate.py          # one simulation entry point
  gates.py             # user-facing CZ gate API
  physics.py           # real physical helper functions only
  noise.py             # noise models only
  core/
  protocols/
  backends/
  analysis/
  schemas/
```

For every proposed move or deletion, explain why it reduces complexity.

---

## Phase 3: Define the Minimal Public API

Do not modify files yet.

Provide a proposed new `__all__` for:

- `src/ryd_gate/__init__.py`
- `src/ryd_gate/gates.py`
- `src/ryd_gate/protocols/__init__.py`
- `src/ryd_gate/analysis/__init__.py`

Use this format:

```python
__all__ = [
    ...
]
```

Also provide migration notes for old exports:

| Old Export | New Location | Action | Reason |
|---|---|---|---|

Actions may include:

- `keep`
- `move`
- `make private`
- `delete`

---

## Phase 4: Patch Plan

Do not modify files yet.

Create a small-step patch plan with at most 6 patches.

Each patch should do only one kind of simplification.

Good patch types:

- delete one wrapper or alias group
- simplify one function signature
- reduce one `__init__.py` export list
- merge one group of helpers
- make one group of backend helpers private
- update tests/examples/docs for one API change

Bad patch types:

- rewrite the whole simulation system
- redesign all backends
- move many unrelated files at once
- add new abstractions
- preserve old APIs through new wrappers

Use this format:

## Patch Plan

### Patch 1: `<short title>`

- Goal:
- Files touched:
- What gets deleted:
- What gets moved:
- Public API impact:
- Tests / docs to update:
- Verification command:

### Patch 2: `<short title>`

- Goal:
- Files touched:
- What gets deleted:
- What gets moved:
- Public API impact:
- Tests / docs to update:
- Verification command:

Continue up to at most 6 patches.

---

## Verification Requirements

After each patch, run or explain the status of:

```bash
uv run pytest -q
uv run ruff check src tests docs examples
uv run mypy src/ryd_gate
```

If a command cannot be run, explain:

- which command was not run
- why it was not run
- what local or partial verification was used instead

Do not claim tests passed unless they were actually run.

---

## Refactoring Priority

Use this priority order:

1. Low-risk wrappers, aliases, and shims.
2. Over-exported symbols in `ryd_gate.__init__`.
3. Over-flexible function signatures.
4. Scattered helper files and module consolidation.
5. Backend internals only if they are clearly wrappers, imports, or visibility problems.

Do not start by modifying numerical backend code.

---

## Output Format

Your first response should be an audit and plan only.

Use exactly this structure:

# API Simplification Audit

## Current Public API

List current exports and classify each as:

- `keep`
- `move to submodule`
- `make private`
- `delete`

## README / Examples / Tests Usage

List which APIs are actually used.

## Wrapper Deletion Candidates

| Symbol | File | Current Role | Why It Is Redundant | Proposed Action | Risk | Tests / Docs To Update |
|---|---|---|---|---|---|---|

## Over-Flexible Function Candidates

| Function | File | Current Signature Problem | Proposed Simpler Signature | Migration Impact | Risk |
|---|---|---|---|---|---|

## File / Module Consolidation Plan

| Current File / Helper | Current Role | Proposed Destination | Reason | Risk |
|---|---|---|---|---|

## Proposed Minimal API

Show the intended user imports:

```python
from ryd_gate import Register, RydbergSystem, TFIMQuenchProtocol, simulate
from ryd_gate.gates import TOProtocol, cz_gate_report
```

Then provide proposed `__all__` blocks.

## Patch Plan

Provide at most 6 small patches.

Each patch must include:

- goal
- files touched
- what gets deleted
- what gets moved
- public API impact
- tests / docs to update
- verification command

---

## Important Constraint

Do not perform large-scale code edits before I approve the patch plan.

If you are in an editable environment, only inspect the repository and produce the audit report first.

Do not modify files until I explicitly approve a specific patch.