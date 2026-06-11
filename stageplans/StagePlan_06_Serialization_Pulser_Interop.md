# Stage 6 Plan: Serialization Freeze and Pulser Interop

## Purpose

Stage 6 freezes the `ryd-gate/<kind>/v1` serialization payloads and adds a
small Pulser abstract-representation bridge. The goal is switching-cost
reduction, not full Pulser parity:

```python
from ryd_gate.interop.pulser import from_pulser_abstract_repr, to_pulser_abstract_repr

seq = from_pulser_abstract_repr(payload)
payload2 = to_pulser_abstract_repr(seq)
```

The supported subset is intentionally narrow: registers, global Rydberg
channels, waveform/pulse/sequence objects that map onto the Stage 1-2 product
model, and `NoiseModel` fields with matching semantics. Unsupported Pulser
features raise typed errors naming the unsupported construct.

This stage also adds `EmulationConfig`-style observable schedules for tensor
network backends, built on the Stage 3 native observable measurement path.

## Why This Is Not Stage 1-5

Early stages needed schema-tagged `to_dict()` payloads, but they deliberately
did not freeze schemas before the object model settled. Stage 6 comes after:

- Stage 1-2 define the product data objects and sequence serialization.
- Stage 3 defines backend result capabilities and MPS-native observables.
- Stage 4 defines `NoiseModel`.
- Stage 5 defines gate-report serialization.

Now the `v1` payloads can become a compatibility promise.

## Stage 6 Dependency

Stage 6 starts only after Stage 5 passes its acceptance, including the full fast
suite and gate benchmark pins.

## No-Wrapper / No-Fake Rules

1. JSON Schemas describe existing `to_dict()` payloads; they do not create a
   parallel serialization framework or registry.
2. `jsonschema` is optional. Importing `ryd_gate` must not require it. Schema
   validation helpers raise a clear optional-dependency error if the extra is
   missing.
3. Pulser interop works on Pulser abstract-representation dictionaries or JSON
   strings. It must not require importing Pulser at runtime.
4. Unsupported Pulser constructs raise `PulserInteropError` with a stable code
   and path. Do not silently drop EOM, DMM, SLM, XY, local addressing, or
   measurement features that have no kernel counterpart.
5. The Pulser bridge lowers into native Stage 1-2 objects. It must not add
   Pulser-shaped wrapper classes under `src/ryd_gate`.
6. Observable schedules use Stage 3's backend-native measurement functions.
   They must not densify TN states and must respect result capabilities.

## Allowed File Operations

Create:

```text
src/ryd_gate/interop/__init__.py
src/ryd_gate/interop/pulser.py
src/ryd_gate/observables.py
src/ryd_gate/schemas/register.v1.schema.json
src/ryd_gate/schemas/register-layout.v1.schema.json
src/ryd_gate/schemas/level-structure.v1.schema.json
src/ryd_gate/schemas/channel.v1.schema.json
src/ryd_gate/schemas/device.v1.schema.json
src/ryd_gate/schemas/waveform.v1.schema.json
src/ryd_gate/schemas/pulse.v1.schema.json
src/ryd_gate/schemas/sequence.v1.schema.json
src/ryd_gate/schemas/noise.v1.schema.json
src/ryd_gate/schemas/cz-gate-report.v1.schema.json
src/ryd_gate/schemas/observable-config.v1.schema.json
tests/serialization/__init__.py
tests/serialization/test_json_schemas.py
tests/interop/__init__.py
tests/interop/test_pulser_subset.py
tests/sequence/test_observable_config.py
```

Modify:

```text
pyproject.toml                         (optional schema/interop extras and package data)
src/ryd_gate/core/serialization.py     (schema loading/validation helpers only)
src/ryd_gate/lattice/geometry.py       (RegisterLayout.define_register)
src/ryd_gate/simulate.py               (optional observables parameter for simulate_sequence)
src/ryd_gate/results.py                (observable-times result access if needed)
src/ryd_gate/backends/tenpy_mps/backends.py   (wire ObservableConfig through existing measurement path only)
src/ryd_gate/backends/tn_common/simulate.py   (metadata passthrough only)
src/ryd_gate/__init__.py
stageplans/README.md                   (status/table only, after implementation)
```

Do not modify:

```text
src/ryd_gate/core/local_blocks.py
src/ryd_gate/core/factories.py
src/ryd_gate/ir/hamiltonian.py
src/ryd_gate/protocols/gate_cz_*.py
src/ryd_gate/backends/exact/*
src/ryd_gate/backends/gputn/*
src/ryd_gate/backends/peps2d/*
scripts/notebooks/* and all *.ipynb
```

## JSON Schema Plan

### Schema files

Each schema file validates exactly one `to_dict()` payload kind and requires
the matching schema tag:

```text
ryd-gate/register/v1
ryd-gate/register-layout/v1
ryd-gate/level-structure/v1
ryd-gate/channel/v1
ryd-gate/device/v1
ryd-gate/waveform/v1
ryd-gate/pulse/v1
ryd-gate/sequence/v1
ryd-gate/noise/v1
ryd-gate/cz-gate-report/v1
ryd-gate/observable-config/v1
```

Schemas must be strict enough to catch missing required fields and unknown
type-shape errors, but not stricter than the Python validation for physics
semantics. Physics rules stay in object `validate()` methods.

### Schema helpers

Extend `src/ryd_gate/core/serialization.py` with:

```python
def schema_path(kind: str) -> Path: ...
def load_json_schema(kind: str) -> dict: ...
def validate_json_schema(data: Mapping[str, Any], kind: str) -> list[ValidationIssue]: ...
```

Rules:

1. `schema_path` and `load_json_schema` use `importlib.resources`, not
   hard-coded filesystem paths.
2. `validate_json_schema` imports `jsonschema` inside the function only.
3. If `jsonschema` is unavailable, return one error with code
   `serialization.jsonschema_missing`.
4. Schema validation returns `list[ValidationIssue]`; it does not raise.
5. Existing `check_schema`, `schema_tag`, and `json_ready` behavior stays
   unchanged.

`pyproject.toml` adds:

```toml
[project.optional-dependencies]
schema = ["jsonschema>=4"]
interop = ["jsonschema>=4"]
```

`setuptools` package data must include `src/ryd_gate/schemas/*.json`.

## Pulser Interop Plan

### Public objects

`src/ryd_gate/interop/pulser.py` exports:

```python
class PulserInteropError(ValueError):
    code: str
    path: tuple[str, ...]
    construct: str | None

def from_pulser_abstract_repr(data: Mapping[str, Any] | str) -> Sequence: ...
def to_pulser_abstract_repr(sequence: Sequence) -> dict: ...
def noise_from_pulser_abstract_repr(data: Mapping[str, Any] | str) -> NoiseModel: ...
def noise_to_pulser_abstract_repr(noise: NoiseModel) -> dict: ...
```

`src/ryd_gate/interop/__init__.py` re-exports those names.

### Supported Pulser subset

Import supports:

```text
Register coordinates with explicit qubit ids
RegisterLayout trap coordinates plus trap-to-qubit mapping
Rydberg.Global channel declarations
Constant waveform
Ramp waveform
Blackman waveform
Interpolated waveform
Custom waveform samples
Pulse amplitude, detuning, phase when phase is zero
Sequence delays and pulse additions on global channels
NoiseModel fields with matching semantics:
  state_prep_error
  p_false_pos
  p_false_neg
  amp_sigma
  detuning_sigma_rad_per_us
  runs / n_trajectories
```

Export supports only native objects that fall inside the same subset. If a
native object cannot be represented in the subset, raise `PulserInteropError`
instead of emitting a lossy payload.

Unsupported features and codes:

```text
pulser.eom_not_supported
pulser.dmm_not_supported
pulser.slm_not_supported
pulser.xy_not_supported
pulser.local_addressing_not_supported
pulser.phase_not_supported
pulser.measurement_not_supported
pulser.waveform_not_supported
pulser.channel_not_supported
pulser.noise_field_not_supported
```

Every error includes the path to the offending payload field when importing.

### RegisterLayout.define_register

Add to `RegisterLayout`:

```python
def define_register(
    self,
    trap_ids: Sequence[int],
    qubit_ids: Sequence[str] | None = None,
    *,
    center: bool = False,
) -> Register: ...
```

Rules:

1. `trap_ids` are integer indices into `trap_coords_um`.
2. ids must be unique and in range.
3. `qubit_ids=None` generates `q0..q{N-1}`.
4. `center=False` preserves Pulser/layout coordinates exactly by default.
5. The returned register attaches `self` as `layout`.
6. `spacing_um` is computed with the existing nearest-distance helper used by
   `Register.from_coordinates`.

This lands here because layout mapping is interop provenance, not a Stage 1
geometry default (Decision Log D10).

## Observable Schedule Plan

Create `src/ryd_gate/observables.py`:

```python
@dataclass(frozen=True)
class ObservableConfig:
    names: tuple[str, ...]
    times_ns: tuple[int, ...] | None = None
    every_ns: int | None = None

    def validate(self) -> list[ValidationIssue]: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservableConfig": ...
```

Rules:

1. Either `times_ns` or `every_ns` may be set, not both.
2. Times are nonnegative integer ns values.
3. Names are non-empty strings and are interpreted by the backend's existing
   observable resolver.
4. Serialization tag is `"ryd-gate/observable-config/v1"`.
5. `simulate_sequence(..., observables=config, backend="mps")` forwards the
   names/times into the existing TeNPy measurement path.
6. `backend="exact"` may ignore the streaming schedule and users can query the
   final state handle as before. It must not change exact backend behavior.
7. Unsupported backends raise `UnsupportedResultQuery` or `ValueError` with
   code `observables.backend_unsupported`.

## Tests

### `tests/serialization/test_json_schemas.py`

Required tests:

1. Every public `to_dict()` payload validates against its schema.
2. Removing each required field fails schema validation.
3. `validate_json_schema` returns `serialization.jsonschema_missing` when
   `jsonschema` is absent.
4. Schemas are loadable through `importlib.resources` from an installed package.

### `tests/interop/test_pulser_subset.py`

Required tests:

1. Import a minimal global Rydberg Pulser abstract-repr sequence into native
   `Sequence`.
2. Export the same native sequence back to the supported Pulser subset.
3. RegisterLayout trap mapping preserves trap order, selected coordinates, and
   qubit ids.
4. Constant, Ramp, Blackman, Interpolated, and Custom waveforms map to native
   `Waveform` kinds and round-trip where supported.
5. Unsupported EOM/DMM/local addressing payloads raise `PulserInteropError`
   with the specified codes and field paths.
6. Pulser noise fields map to `NoiseModel`; unsupported noise fields raise
   `pulser.noise_field_not_supported`.

### `tests/sequence/test_observable_config.py`

Required tests:

1. `ObservableConfig` validates name and time constraints.
2. Serialization round-trip validates against the schema.
3. MPS `simulate_sequence(..., observables=...)` records the requested
   observables through the Stage 3 native measurement path.
4. Unsupported backend handling is explicit and typed.

## Acceptance

```bash
uv run pytest tests/serialization -q
uv run pytest tests/interop -q
OMP_NUM_THREADS=1 uv run pytest tests/sequence/test_observable_config.py -q
OMP_NUM_THREADS=1 uv run pytest -m "not slow" -q
uv run ruff check src/ryd_gate/interop src/ryd_gate/observables.py src/ryd_gate/core/serialization.py tests/serialization tests/interop
```

If `jsonschema` is not installed, schema-dependent tests must skip or assert the
documented missing-extra issue. The package must still import without the extra.

## Non-Goals for Stage 6

No full Pulser parity, no Pulser package dependency, no QPU/cloud submission, no
EOM/DMM/SLM/XY support, no noisy TN simulation, no schema version migration
machinery beyond `v1`, and no notebook migration.
