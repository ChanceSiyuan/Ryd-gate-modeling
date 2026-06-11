# Stage 4 Plan: Declarative NoiseModel

## Purpose

Stage 4 adds a product-facing `NoiseModel` data object that configures the
noise machinery already present in the exact backend:

```python
noise = NoiseModel(
    runs=200,
    detuning_sigma_rad_per_us=0.02,
    amp_sigma=0.01,
    rydberg_decay=True,
)
```

The runner and Hamiltonian builders keep doing the physical work. `NoiseModel`
is the serializable, validated data layer that names what noise is requested,
converts product units to the existing backend units, and refuses unsupported
runtime fields precisely.

Stage 4 implements:

- `NoiseModel` in a new top-level domain module `noise.py`;
- deterministic `noise_types`, `summary()`, `to_dict()` / `from_dict()`;
- exact-runner configuration helpers for `MonteCarloRunner.setup_*`;
- decay physical-parameter helpers for the existing non-Hermitian local blocks;
- top-level exports: `from ryd_gate import NoiseModel, configure_monte_carlo_runner`.

## Why This Is Not Stage 1/2/3

Noise is a cross-cutting product concern: some effects are sampled by
`MonteCarloRunner`, while Rydberg/intermediate decay must be present when the
physical local blocks are built. Stage 1 introduced the data/validation pattern,
Stage 2 made sequences runnable, and Stage 3 made results capability-aware.
Stage 4 can now add a noise data contract without changing solver algorithms.

## Stage 4 Dependency

Stage 4 starts only after Stage 3 passes its acceptance, including the full fast
suite with TeNPy installed or cleanly skipped.

## No-Wrapper / No-Fake Rules

1. `NoiseModel` is data, not a new simulator. It configures
   `MonteCarloRunner` and physical block flags; it must not reimplement
   sampling, Hamiltonian perturbation construction, or decay rates.
2. `MonteCarloRunner` remains the exact Monte Carlo engine. The Stage 4 helper
   may call `setup_detuning_noise`, `setup_amplitude_noise`,
   `setup_local_rin_noise`, and `setup_position_noise`, but must not duplicate
   their logic.
3. Non-Hermitian decay support is applied by passing the existing
   `enable_rydberg_decay` / `enable_intermediate_decay` physical kwargs into
   `RydbergSystem.from_lattice` (or the sequence compile helper). Do not mutate
   an already-built system's local blocks.
4. No faked support: nonzero fields without current runtime machinery
   (`state_prep_error`, `p_false_pos`, `p_false_neg`, `temperature_uK`,
   `laser_waist_um`) are accepted as serializable data but raise typed
   validation errors when applied to a runtime path in Stage 4.
5. No TN noise support in Stage 4. `NoiseModel.validate_for(backend="mps")`
   and other TN backends report unsupported runtime codes instead of silently
   dropping noise.
6. No new mandatory dependencies. This stage uses only stdlib plus existing
   package dependencies.

## Allowed File Operations

Create:

```text
src/ryd_gate/noise.py
tests/noise/__init__.py
tests/noise/test_noise_model.py
tests/noise/test_noise_model_runner.py
```

Modify:

```text
src/ryd_gate/__init__.py                         (top-level exports)
src/ryd_gate/protocols/sequence_protocol.py      (optional: pass decay physical kwargs when compiling sequences)
src/ryd_gate/analysis/local_addressing.py        (optional: replace manual runner setup with NoiseModel helper)
src/ryd_gate/backends/exact/dense_ode.py         (bug fix only — see note below)
stageplans/README.md                             (status/table only, after implementation)
```

*Implementation note (2026-06-11):* the runner-identity test exposed a
pre-existing kernel bug: ``DenseODEBackend`` used ``np.asarray`` on IR
operators, which wraps scipy sparse matrices in a 0-d object array instead of
densifying them. Every ``MonteCarloRunner`` noise perturbation term is sparse,
so ``run_gate_fidelity`` with the default backend crashed; the notebook never
caught it because its MC cells default to cached data. Stage 4 adds the
one-function densification fix in ``dense_ode.py`` (no solver algorithm
change).

Do not modify:

```text
src/ryd_gate/backends/exact/monte_carlo_runner.py
src/ryd_gate/core/local_blocks.py
src/ryd_gate/core/factories.py
src/ryd_gate/ir/*
src/ryd_gate/backends/tenpy_mps/*
src/ryd_gate/backends/gputn/*
src/ryd_gate/backends/peps2d/*
scripts/notebooks/* and all *.ipynb
```

If implementation cannot meet the plan without changing a forbidden file, update
this plan first. In particular, do not add noise state to `DeviceSpec`,
`Register`, `Sequence`, `EvolutionResult`, or `SimulationResult`.

## Module Plan: `src/ryd_gate/noise.py` (new)

### Public objects

```python
@dataclass(frozen=True)
class NoiseModel:
    runs: int = 1
    detuning_sigma_rad_per_us: float = 0.0
    amp_sigma: float = 0.0
    local_rin_sigma: float = 0.0
    position_sigma_um: float | tuple[float, float, float] = 0.0
    rydberg_decay: bool = False
    intermediate_decay: bool = False
    state_prep_error: float = 0.0
    p_false_pos: float = 0.0
    p_false_neg: float = 0.0
    temperature_uK: float | None = None
    laser_waist_um: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_trajectories(self) -> int: ...
    @property
    def noise_types(self) -> tuple[str, ...]: ...
    def summary(self) -> str: ...
    def validate(self) -> list[ValidationIssue]: ...
    def validate_for(self, *, backend: str, level_structure=None, n_atoms: int | None = None) -> list[ValidationIssue]: ...
    def physical_kwargs(self) -> dict[str, bool]: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NoiseModel": ...

def configure_monte_carlo_runner(runner: MonteCarloRunner, noise: NoiseModel) -> MonteCarloRunner: ...
```

`runs` is the canonical Python field. `n_trajectories` is a read-only alias for
Pulser terminology. `from_dict()` accepts either `"runs"` or
`"n_trajectories"` but `to_dict()` writes `"runs"` only.

### Fixed noise type names

`noise_types` returns a tuple in this stable order, including only active
nonzero/true entries:

```text
state_prep
readout
detuning
amplitude
local_rin
position
rydberg_decay
intermediate_decay
temperature
laser_waist
```

`readout` is active if either `p_false_pos` or `p_false_neg` is nonzero.
`summary()` is deterministic and uses these same names; no table formatting
library.

### Validation rules

`validate()` is pure data validation and never raises:

- `runs` must be a positive integer, else `noise.runs`;
- probability fields (`state_prep_error`, `p_false_pos`, `p_false_neg`) must be
  in `[0, 1]`, else `noise.probability_range`;
- sigma fields (`detuning_sigma_rad_per_us`, `amp_sigma`, `local_rin_sigma`,
  `position_sigma_um`) must be finite and nonnegative, else
  `noise.nonnegative`;
- `position_sigma_um` must be scalar or length-3 tuple/list, else
  `noise.position_sigma_shape`;
- `temperature_uK` and `laser_waist_um`, when not `None`, must be finite and
  nonnegative, else `noise.nonnegative`;
- `metadata` must be JSON-compatible via `json_ready`, else
  `noise.metadata_json`.

`validate_for(...)` adds runtime capability checks:

- any active noise with `backend != "exact"` yields
  `noise.backend_unsupported`;
- `rydberg_decay` / `intermediate_decay` require level structures with physical
  local blocks (`"analog_3"` or `"rb87_7"`), else
  `noise.decay_level_structure_unsupported`;
- `position_sigma_um` requires `backend="exact"` and `n_atoms == 2`, because the
  existing `MonteCarloRunner.setup_position_noise` path builds a two-atom VdW
  perturbation, else `noise.position_two_atom_only`;
- active `state_prep`, `readout`, `temperature`, or `laser_waist` runtime fields
  yield `noise.runtime_not_stage4`;
- all issue codes are tested as API.

`configure_monte_carlo_runner(...)` calls `raise_for_errors` on both
`validate()` and `validate_for(backend="exact", level_structure=..., n_atoms=...)`
before mutating the runner.

### Serialization

`to_dict()` writes:

```python
{
    "schema": "ryd-gate/noise/v1",
    "runs": ...,
    "detuning_sigma_rad_per_us": ...,
    ...
    "metadata": json_ready(metadata),
}
```

`from_dict()` uses `check_schema(data, "noise")`, accepts older/interop payloads
with `"n_trajectories"` instead of `"runs"`, and ignores absent optional fields
by using dataclass defaults. Round-trip must preserve equality for JSON-ready
metadata.

### Unit conversions into `MonteCarloRunner`

`configure_monte_carlo_runner(runner, noise)` maps fields exactly:

```text
detuning_sigma_rad_per_us -> runner.setup_detuning_noise(
    sigma_detuning_hz = detuning_sigma_rad_per_us * 1e6 / (2*pi)
)
amp_sigma                -> runner.setup_amplitude_noise(amp_sigma)
local_rin_sigma          -> runner.setup_local_rin_noise(local_rin_sigma)
position_sigma_um        -> runner.setup_position_noise(tuple_um * 1e-6)
```

Zero-valued fields do not call setup methods. The helper returns the same runner
object for chaining and for identity tests.

### Decay physical kwargs

`NoiseModel.physical_kwargs()` returns only:

```python
{
    "enable_rydberg_decay": noise.rydberg_decay,
    "enable_intermediate_decay": noise.intermediate_decay,
}
```

These kwargs are passed at system construction time:

```python
system = RydbergSystem.from_lattice(
    register,
    level_structure("rb87_7"),
    protocol=protocol,
    **level_structure("rb87_7").physical_kwargs(),
    **noise.physical_kwargs(),
)
```

If sequence compilation is extended in this stage, add an optional
`noise_model: NoiseModel | None = None` parameter to `compile_sequence_to_system`
only to merge these physical kwargs before `RydbergSystem.from_lattice`. Do not
change `Sequence` storage or operation records.

## Integration Plan

### Top-level exports

`src/ryd_gate/__init__.py` exports:

```text
NoiseModel
configure_monte_carlo_runner
```

`from ryd_gate import NoiseModel, configure_monte_carlo_runner` must work.

### Exact Monte Carlo usage

The canonical Stage 4 exact usage is:

```python
noise = NoiseModel(detuning_sigma_rad_per_us=0.02, amp_sigma=0.01, runs=32)
runner = MonteCarloRunner(system.with_protocol(protocol), x)
configure_monte_carlo_runner(runner, noise)
result = runner.run_gate_fidelity(n_shots=noise.runs, seed=123)
```

This is deliberately runner-shaped. Stage 5 may wrap this in a gate-report API,
but Stage 4 does not introduce `simulate_noisy`, `NoisySimulationResult`, or a
second Monte Carlo result type.

### Local addressing helper

`analysis/local_addressing.py` may be migrated from manual `engine_kwargs`
setup to:

```python
noise = NoiseModel(
    detuning_sigma_rad_per_us=2*pi*sigma_detuning_hz / 1e6,
    local_rin_sigma=sigma_local_rin,
    amp_sigma=sigma_amplitude,
    runs=n_mc,
)
configure_monte_carlo_runner(engine, noise)
```

This migration is optional in Stage 4. If done, tests must show numerical
identity to the previous setup path.

## Tests

### `tests/noise/test_noise_model.py`

Required tests:

1. default model: `runs == 1`, `noise_types == ()`, `validate() == []`;
2. active type inference in the fixed order;
3. `summary()` contains each active type exactly once and is deterministic;
4. invalid `runs`, probabilities, negative sigmas, and bad position tuple shape
   return the specified `ValidationIssue.code`s;
5. `to_dict()` / `from_dict()` round-trip with schema tag
   `"ryd-gate/noise/v1"`;
6. `from_dict()` accepts `"n_trajectories"` as an alias for `"runs"`;
7. non-JSON metadata returns `noise.metadata_json`;
8. `physical_kwargs()` returns only the two existing decay flags.

### `tests/noise/test_noise_model_runner.py`

Required tests:

1. `configure_monte_carlo_runner` mutates and returns the same runner object;
2. detuning conversion is exact:
   `runner._sigma_detuning_rad == detuning_sigma_rad_per_us * 1e6`;
3. amplitude and local-RIN sigmas match the runner's existing private fields;
4. position sigma scalar expands to a length-3 tuple and is converted from um to
   meters before entering `setup_position_noise` (runner stores um internally);
5. `position_sigma_um` with `n_atoms != 2` raises through
   `noise.position_two_atom_only`;
6. nonzero `state_prep_error`, `p_false_pos`, `p_false_neg`,
   `temperature_uK`, or `laser_waist_um` raises through
   `noise.runtime_not_stage4` when configuring a runner;
7. decay physical kwargs reproduce the current manual flags by comparing
   metadata and the imaginary diagonal entries of `H_const` for `analog_3` or
   `rb87_7`;
8. a small `MonteCarloRunner.run_gate_fidelity(n_shots=2, seed=...)` using a
   `NoiseModel` configured runner gives identical samples/statistics to a runner
   configured manually with the same `setup_*` calls.

### Existing tests

All Stage 1-3 tests must pass unmodified except import-contract tests updated
to include the new top-level exports.

## Acceptance

```bash
uv run pytest tests/noise -q
uv run pytest tests/core/test_init.py -q
uv run pytest tests/analysis -q
OMP_NUM_THREADS=1 uv run pytest -m "not slow" -q
```

Targeted lint must pass for all touched Python files:

```bash
uv run ruff check src/ryd_gate/noise.py src/ryd_gate/__init__.py tests/noise
```

Repo-wide `ruff check` is not a Stage 4 acceptance blocker until the existing
notebook lint debt is handled; do not edit notebooks in this stage.

## Non-Goals for Stage 4

No noisy TN evolution, no state-preparation sampling, no readout corruption in
`SimulationResult.sample`, no Doppler/waist physics derived from
`temperature_uK` or `laser_waist_um`, no new noisy result type, no Pulser
import/export, no JSON Schema files, and no changes to exact solver algorithms.
