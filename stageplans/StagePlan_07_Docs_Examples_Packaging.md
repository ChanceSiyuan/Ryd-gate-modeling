# Stage 7 Plan: Docs, Examples, Packaging

## Purpose

Stage 7 makes the product API usable outside the development loop: a real docs
site, executed tutorials, a generated capability matrix, product-API notebook
migration, and packaging/CI polish.

This is also the first stage allowed to edit notebooks. The Stage 1-6 runtime
surface is now stable enough that examples can become public contracts instead
of exploratory scripts.

## Stage 7 Dependency

Stage 7 starts only after Stage 6 passes its acceptance, including schema and
Pulser interop tests.

## No-Wrapper / No-Fake Rules

1. Documentation examples must run against the real installed package. Do not
   mock imports or use snippets that cannot execute.
2. Capability docs are generated from code, not hand-maintained tables.
3. Notebook migration uses product APIs where they exist. Research-only
   protocol workflows may stay on `Protocol` + `simulate`, but removed names
   must not reappear.
4. Packaging metadata must describe actual optional dependency groups. Do not
   advertise backend extras that cannot import.
5. Type checking starts pragmatic and ratchets upward. Stage 7 may configure
   scoped mypy checks, but it must not hide real import/type failures with broad
   global ignores.
6. The legacy internal `("ger", param_set="analog_3")` inference branch is
   removed here, as promised in Decision Log D11. After Stage 7, bare `"ger"`
   is symbolic regardless of `param_set`.

## Allowed File Operations

Create:

```text
CHANGELOG.md
src/ryd_gate/py.typed
docs/getting_started.md
docs/fundamentals.md
docs/how_to_sequences.md
docs/how_to_noise.md
docs/how_to_gates.md
docs/how_to_interop.md
docs/capability_matrix.md
docs/_scripts/build_capability_matrix.py
docs/api.rst
tests/docs/__init__.py
tests/docs/test_capability_matrix.py
tests/docs/test_readme_examples.py
```

Modify:

```text
README.md
pyproject.toml
docs/index.rst
docs/conf.py
docs/stage1_api.md
examples/*
scripts/bench_quench_check.py
scripts/notebooks/*.ipynb
src/ryd_gate/core/factories.py          (remove D11 legacy inference branch)
tests/core/test_rydberg_system_model.py (update D11 expectation)
tests/core/test_init.py                 (import/package contract updates)
stageplans/README.md                    (status/table only, after implementation)
```

Do not modify:

```text
src/ryd_gate/backends/*                 (except no-op docstrings if needed)
src/ryd_gate/ir/*
src/ryd_gate/protocols/*
src/ryd_gate/sequence.py
src/ryd_gate/results.py
src/ryd_gate/noise.py
```

If notebook migration exposes a source bug, fix it in the owning stage module
only after updating this plan or opening a follow-up task. Stage 7 is primarily
documentation and packaging, not feature development.

## Documentation Plan

### Sphinx structure

`docs/index.rst` becomes the product entry point:

```text
Getting Started
Fundamentals: units, conventions, Hamiltonian lowering
How-To: build and run Sequence programs
How-To: NoiseModel
How-To: CZ gate reports and error budgets
How-To: Pulser interop
Capability Matrix
API Reference
```

`docs/api.rst` uses `sphinx.ext.autodoc` for stable public modules:

```text
ryd_gate
ryd_gate.lattice
ryd_gate.pulse
ryd_gate.devices
ryd_gate.sequence
ryd_gate.results
ryd_gate.noise
ryd_gate.gates
ryd_gate.interop.pulser
```

Do not autodoc internal backend modules as public API.

### Capability matrix

`docs/_scripts/build_capability_matrix.py` generates
`docs/capability_matrix.md` from code. The table must cover at least:

```text
level structure x backend support
result handle capabilities by backend
NoiseModel runtime support by backend
Sequence support by backend
Pulser interop supported subset
```

Generation rules:

1. Use `LevelStructureSpec.supports_backend` where available.
2. Use `QuantumStateHandle.capabilities` names from Stage 3.
3. Use `NoiseModel.validate_for(...)` for backend noise support.
4. Emit deterministic Markdown.
5. Tests fail if the generated file is stale.

## README and Examples Plan

### README

Rewrite the README quickstart around two examples:

1. Stage 2 sequence example:

   ```python
   seq = Sequence(Register.chain(1, 20.0), DeviceSpec.virtual_rb87(), "1r")
   seq.declare_channel("ryd", "rydberg_global")
   seq.add(Pulse.constant_detuning(Waveform.blackman(1000, area=np.pi), 0.0), "ryd")
   result = simulate_sequence(seq)
   ```

2. Stage 5 CZ gate report example:

   ```python
   report = cz_gate_report(system, ARProtocol(), x)
   print(report.fidelity)
   ```

The README must include the optional dependency matrix:

```text
base
dev
schema
interop
tn
tn-2d
gputn-cu12
docs
```

### Examples

`examples/` should contain small, executable scripts with bounded runtime:

```text
examples/demo_sequence_exact.py
examples/demo_noise_model.py
examples/demo_cz_gate_report.py
examples/demo_pulser_interop.py
examples/demo_local_addressing.py          (migrated imports only if still relevant)
examples/demo_local_addressing_tn.py       (kept only if it runs or is marked optional)
```

Every example must either run in the base environment or clearly check and skip
missing optional dependencies.

### Notebooks

Notebook migration happens here:

```text
scripts/notebooks/cz_gate_validation_and_errors.ipynb
scripts/notebooks/01r_saffman_double_arp_exact.ipynb
scripts/notebooks/02_ac_stark_local_addressing.ipynb
scripts/notebooks/03_lattice_dynamics_and_annealing.ipynb
scripts/notebooks/01r_lattice_dynamics.ipynb
scripts/notebooks/01r_plus_quench_benchmark.ipynb
scripts/notebooks/plus_state_preparation.ipynb
scripts/notebooks/01r_pepskit_quench.ipynb
scripts/notebooks/run_quench_benchmark.ipynb
scripts/notebooks/01r_yastn_peps_convergence.ipynb
scripts/notebooks/01r_tfim_critical_field.ipynb
```

Rules:

1. Use `Register`, `DeviceSpec`, `Sequence`, `NoiseModel`, and `cz_gate_report`
   where those product APIs directly match the notebook goal.
2. Keep continuous-time research protocols on `Protocol` + `simulate` when that
   is the real workflow.
3. Remove old `("ger", param_set="analog_3")` usage. Use
   `level_structure("analog_3")` for physical analog-3 construction and bare
   `"ger"` only for symbolic construction.
4. Keep notebook outputs light enough for review; do not commit large generated
   binary output unless the repository already tracks it intentionally.

## Packaging Plan

### `py.typed`

Add `src/ryd_gate/py.typed` and include it as package data. Public modules must
have enough annotations for `mypy` to type-check imports and the examples.

### Optional dependencies

Update `pyproject.toml` optional dependency groups:

```text
dev     -> pytest, pytest-cov, ruff, mypy, docs tooling
docs    -> sphinx, sphinx-rtd-theme, myst-parser
schema  -> jsonschema
interop -> jsonschema
tn      -> physics-tenpy
tn-2d   -> physics-tenpy, yastn, cotengra, autoray
gputn-cu12 -> existing CUDA stack
```

Do not move heavy backend dependencies into base `dependencies`.

### Versioning and changelog

Create `CHANGELOG.md` with an unreleased section covering:

```text
Stage 1 API foundation
Stage 2 Sequence exact simulation
Stage 3 backend-native result handles
Stage 4 NoiseModel
Stage 5 gate reports
Stage 6 serialization/interoperability
Stage 7 docs/packaging
```

Keep project version `0.1.0` unless a release task explicitly changes it.

### Type and lint gates

Add scoped mypy config in `pyproject.toml`. Initial target:

```text
src/ryd_gate/lattice
src/ryd_gate/pulse.py
src/ryd_gate/devices.py
src/ryd_gate/sequence.py
src/ryd_gate/results.py
src/ryd_gate/noise.py
src/ryd_gate/gates
src/ryd_gate/interop
```

Backend modules with heavy optional imports may be excluded initially, but
exclusions must be named, not blanket `src/ryd_gate/**`.

## D11 Cleanup Plan

Remove the legacy inference branch in `core/factories.py`:

```text
("ger", param_set="analog_3") -> analog_3 physical construction
```

After Stage 7:

```text
level_structure("ger")       -> symbolic ger construction
level_structure("analog_3")  -> physical analog-3 construction
```

Required tests:

1. Bare `"ger"` with any `param_set` no longer enters analog-3 physical local
   blocks.
2. `level_structure("analog_3")` keeps the existing physical behavior.
3. Notebook and example imports use the explicit name.

## Tests

### `tests/docs/test_capability_matrix.py`

Required tests:

1. Running the matrix generator produces exactly the checked-in
   `docs/capability_matrix.md`.
2. Matrix rows include all public level structures and documented backends.
3. Noise support entries come from `NoiseModel.validate_for`, not hard-coded
   strings.

### `tests/docs/test_readme_examples.py`

Required tests:

1. Extracted README quickstart snippets execute in a base environment.
2. Gate report snippet either executes fully or uses a small fixture with
   bounded runtime.
3. Optional-backend examples skip cleanly when dependencies are missing.

### Notebook execution

Executed notebooks are a Stage 7 acceptance item. *Revision (2026-06-11):*
Quarto is not installed in this environment; nbconvert 7.x is. The runner is
`jupyter nbconvert --to notebook --execute` driven by
`docs/_scripts/run_notebooks.py` with a per-notebook timeout. Notebooks whose
backends cannot import on the machine (yastn/PEPS 2D, cuQuantum GPU) are
skipped by an explicit, named skip list — all notebooks still get the import
migration, but only CPU notebooks with bounded runtime are execution-gated.
Do not require GPU notebooks on CPU-only CI.

## Acceptance

```bash
uv run pytest tests/docs -q
uv run pytest tests/core/test_rydberg_system_model.py -q
uv run pytest tests/core/test_init.py -q
uv run sphinx-build -b html docs docs/_build/html
uv run ruff check src tests docs examples
uv run mypy src/ryd_gate
OMP_NUM_THREADS=1 uv run pytest -m "not slow" -q
```

Notebook acceptance, using nbconvert (the runner installed in this repo):

```bash
uv run python docs/_scripts/run_notebooks.py
```

Stage 7 is complete only when the executed tutorial command is documented and
reproducible. The sphinx/mypy acceptance commands assume the `docs` extra and
`mypy` (added to `dev`) have been synced into the environment first
(`uv sync --extra dev --extra docs`); neither is installed before this stage.

## Non-Goals for Stage 7

No new physics features, no new backend algorithms, no cloud/QPU integration,
no full Pulser parity beyond Stage 6's subset, no broad type-strict rewrite of
optional backend internals, and no packaging release upload.
