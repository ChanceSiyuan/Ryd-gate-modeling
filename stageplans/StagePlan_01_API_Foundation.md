# Stage 1 Plan: Foundation for api

## Stage 1 User API Documentation

The detailed user-facing API contract for Stage 1 lives in [docs/stage1_api.md](../docs/stage1_api.md).

That document is normative for user-callable APIs: imports, one-line calls, inputs, outputs, units, and failure conditions. This stage plan only keeps engineering ownership, file operations, module implementation details, tests, and acceptance commands.

## Stage 1 File Responsibility Contract

This section fixes the responsibility of each file touched in Stage 1. Implementation must follow these ownership boundaries so the result is a product refactor of existing domain modules, not a new wrapper layer.

### New File: `src/ryd_gate/core/validation.py`

Primary responsibility:

- define the common validation issue primitive used by Stage 1 user-facing objects.

Functions and classes owned by this file:

- `ValidationSeverity = Literal["error", "warning"]`;
- `ValidationIssue`;
- `raise_for_errors`.

Exact behavior owned by this file:

- `ValidationIssue("error", code, message, path)` represents a blocking validation problem;
- `ValidationIssue("warning", code, message, path)` represents a non-blocking validation problem;
- `code` is the stable string that tests and downstream callers should branch on;
- `message` is human-readable text;
- `path` locates the invalid object field, for example `("register", "coords")` or `("pulse", "duration_ns")`;
- `raise_for_errors(issues)` turns accumulated validation errors into one `ValueError` only at an explicit boundary chosen by the user or caller.

What must not live here:

- no register geometry rules;
- no pulse sampling rules;
- no device constraints;
- no backend imports;
- no simulation imports.

How this contributes to the Stage 1 goal:

- it gives `Register`, `DeviceSpec`, `Pulse`, `ChannelSpec`, and later `Sequence` one shared validation vocabulary;
- it prevents every product API object from inventing its own error format.

### New File: `src/ryd_gate/devices.py`

Primary responsibility:

- define hardware and physical constraints for a neutral-atom device-like target.

Functions and classes owned by this file:

- `DeviceSpec`;
- `DeviceSpec.virtual_rb87`;
- `DeviceSpec.validate_register`;
- `DeviceSpec.validate_level_structure`;
- `DeviceSpec.validate_pulse`.

Exact behavior owned by this file:

- `DeviceSpec.virtual_rb87()` creates the default Stage 1 Rb87 virtual device;
- `validate_register(register)` checks geometry against dimension, atom count, minimum distance, and maximum radius constraints;
- `validate_level_structure(spec)` checks whether a level model is allowed on the device and whether species matches;
- `validate_pulse(pulse, channel_id)` checks channel existence, duration limits, clock-period divisibility, amplitude limits, and detuning limits;
- all validation methods return `list[ValidationIssue]` and do not raise.

What must not live here:

- no backend selection;
- no QPU job submission;
- no sequence compilation;
- no Hamiltonian construction;
- no waveform implementation;
- no atom coordinate storage.

How this contributes to the Stage 1 goal:

- it makes device constraints explicit and inspectable before any sequence or backend exists;
- it separates physical capability checks from simulation algorithms.

### New Test File: `tests/lattice/test_register.py`

Primary responsibility:

- lock down the user-facing register API and prevent old lattice names from remaining public API.

Required behavior tested here:

- `Register.chain` stable ids and coordinates;
- `Register.square` and `Register.rectangle` row-major ordering;
- `Register.from_coordinates` id generation, centering behavior, and coordinate normalization;
- duplicate ids raise;
- invalid coordinate dimensions raise;
- `coords_array` returns a copy;
- `coords_um` returns tuple-of-tuples;
- `index` and `id_at` preserve stable order;
- `distances_um` returns a symmetric matrix with zero diagonal;
- `distance_pairs` returns only upper-triangular pairs and respects cutoff;
- `blockade_edges` returns only upper-triangular edges within radius;
- `Register.validate(device)` delegates to `device.validate_register(register)`;
- `LatticeGeometry` is not importable from `ryd_gate.lattice`;
- old `make_*` geometry factory functions are not importable from `ryd_gate.lattice`.

What must not be tested here:

- no pulse behavior;
- no level-structure behavior;
- no device pulse validation;
- no simulation behavior.

How this contributes to the Stage 1 goal:

- it ensures the atom register is the stable public geometry object that all later state, bitstring, observable, and backend logic can rely on.

### New Test File: `tests/core/test_level_structures_product_api.py`

Primary responsibility:

- lock down `LevelStructureSpec` as the Stage 1 atom-model API.

Required behavior tested here:

- `level_structure("01")` returns a two-level computational model;
- `level_structure("1r")` returns a two-level Rydberg model with initial level `"1"`;
- `level_structure("01r")` returns a three-level model with initial level `"1"`;
- `level_structure("ger")` remains the explicit ladder model;
- `level_structure("analog_3")` is the official analog three-level preset;
- `level_structure("rb87_7")` returns the seven-level precision model;
- `initial_level_or_default` has deterministic output;
- `physical_kwargs` returns the exact physical factory kwargs required by current system construction;
- `supports_backend` returns the exact Stage 1 backend support matrix;
- `validate` catches malformed custom level specs.

What must not be tested here:

- no device geometry limits;
- no pulse waveform sampling;
- no backend compilation;
- no sequence construction.

How this contributes to the Stage 1 goal:

- it prevents creation of a duplicate `AtomModel` hierarchy and instead turns the existing level-structure concept into the product-level model API.

### New Test File: `tests/core/test_validation.py`

Primary responsibility:

- lock down the shared validation primitive.

Required behavior tested here:

- valid `"warning"` issues construct successfully;
- valid `"error"` issues construct successfully;
- invalid severity raises;
- empty code raises;
- non-tuple path raises;
- warning-only lists do not raise in `raise_for_errors`;
- one error raises `ValueError`;
- multiple errors appear on separate lines with both code and message.

What must not be tested here:

- no register, device, channel, or pulse domain rules except through simple synthetic issues.

How this contributes to the Stage 1 goal:

- it gives all Stage 1 public objects a deterministic validation result format that later APIs can compose.

### New Test File: `tests/core/test_devices.py`

Primary responsibility:

- lock down device/channel/register/level/pulse validation at the product boundary.

Required behavior tested here:

- `DeviceSpec.virtual_rb87()` exposes required channels;
- valid `Register` passes `validate_register`;
- too-close atoms return `register.min_distance`;
- wrong dimensions return `register.dimensions`;
- too many atoms return `register.max_atom_num` when a max is configured;
- too-large radial coordinate returns `register.max_radial_distance` when a max is configured;
- allowed level structures pass;
- unsupported level structures return `level_structure.unsupported`;
- incompatible species returns `level_structure.species`;
- unknown pulse channel returns `channel.unknown`;
- pulse duration below minimum returns `pulse.min_duration`;
- pulse duration above maximum returns `pulse.max_duration`;
- pulse duration not divisible by clock period returns `pulse.clock_period`;
- pulse amplitude above channel limit returns `pulse.amplitude_limit`;
- pulse detuning above channel limit returns `pulse.detuning_limit`.

What must not be tested here:

- no Hamiltonian construction;
- no protocol lowering;
- no backend execution;
- no sampling from a simulated state.

How this contributes to the Stage 1 goal:

- it makes hardware constraints enforceable before Stage 2 introduces `Sequence`.

### New Test File: `tests/core/test_pulse_api.py`

Primary responsibility:

- lock down `Waveform` and `Pulse` as the product pulse API.

Required behavior tested here:

- `Waveform.constant` samples constant values;
- `Waveform.ramp` starts and ends at the requested values;
- `Waveform.blackman` reproduces the intended current Blackman envelope through the new API;
- `Waveform.interpolated` performs piecewise-linear interpolation;
- `Waveform.custom` derives duration from sample count and `dt_ns`;
- `value_at_ns` clamps time and returns `rad/us`;
- `value_at_s` converts to `rad/s`;
- `sample` includes the final duration endpoint;
- `integral_rad` integrates `rad/us` over microseconds;
- invalid waveform durations raise;
- invalid interpolation grids raise;
- `Pulse.constant` creates matching amplitude and detuning waveforms;
- direct `Pulse(...)` rejects mismatched waveform durations;
- finite nonzero phase is stored;
- `Pulse.validate(channel)` returns validation issues without touching systems or backends;
- `blackman_pulse` is not importable from top-level `ryd_gate`.

What must not be tested here:

- no sequence timing;
- no protocol compilation;
- no backend execution;
- no atom geometry.

How this contributes to the Stage 1 goal:

- it moves pulse construction from loose helper functions to explicit user-level waveform and pulse objects while retaining the useful Blackman math as implementation detail.

### Modified File: `src/ryd_gate/lattice/geometry.py`

Primary responsibility after Stage 1:

- own the atom register product API and all geometry-only operations.

Functions and classes owned by this file after Stage 1:

- `RegisterLayout`;
- `Register`;
- `Register.from_coordinates`;
- `Register.chain`;
- `Register.square`;
- `Register.rectangle`;
- `Register.n_atoms`;
- `Register.dimensions`;
- `Register.coords_array`;
- `Register.coords_um`;
- `Register.index`;
- `Register.id_at`;
- `Register.distances_um`;
- `Register.distance_pairs`;
- `Register.blockade_edges`;
- `Register.validate`.

Exact refactor required in this file:

- replace the old `LatticeGeometry` class with `Register`;
- do not create a second geometry class;
- do not implement `Register` as a wrapper around `LatticeGeometry`;
- do not keep `LatticeGeometry` as a public alias;
- delete old module-level public factories `make_chain`, `make_square_lattice`, `make_triangular_lattice`, and `make_geometry_from_coords`;
- move official construction to `Register` classmethods.

What must not live here:

- no atom level model;
- no device constraints except delegation through `validate(device)`;
- no pulse or waveform logic;
- no backend-specific lattice spec logic;
- no simulation logic.

How this contributes to the Stage 1 goal:

- it turns the existing geometry kernel into the stable atom register API that determines atom order for basis states, bitstrings, observables, and future result handling.

### Modified File: `src/ryd_gate/lattice/__init__.py`

Primary responsibility after Stage 1:

- define the public lattice-domain exports.

Exact behavior owned by this file:

- export `Register`;
- export `RegisterLayout`;
- do not export `LatticeGeometry`;
- do not export old `make_*` geometry factories.

What must not live here:

- no implementation logic;
- no compatibility aliases;
- no backend imports.

How this contributes to the Stage 1 goal:

- it makes `from ryd_gate.lattice import Register` the single public lattice entry point.

### Modified File: `src/ryd_gate/core/level_structures.py`

Primary responsibility after Stage 1:

- own the level/model API for computational, Rydberg, ladder, analog, and precision local models.

Functions and classes owned by this file:

- `TransitionSpec`;
- `LevelStructureSpec`;
- `InteractionSpec`;
- `level_structure`;
- `DEFAULT_C6`.

Exact refactor required in this file:

- extend `LevelStructureSpec` instead of creating `AtomModel`;
- add `initial_level`, `species`, `interaction_kind`, and `params`;
- add `initial_level_or_default`;
- add `physical_kwargs`;
- add `supports_backend`;
- add `validate`;
- add `level_structure("01")`;
- add `level_structure("analog_3")` as the official analog three-level preset;
- keep `level_structure("ger")` as the explicit ladder preset;
- keep `level_structure("rb87_7")` as the precision model preset.

What must not live here:

- no register coordinates;
- no pulse waveforms;
- no device validation policy except data needed by `DeviceSpec.validate_level_structure`;
- no backend compiler implementation.

How this contributes to the Stage 1 goal:

- it turns existing level-structure machinery into the public atom-model API without duplicating it in a new module.

### Modified File: `src/ryd_gate/core/__init__.py`

Primary responsibility after Stage 1:

- expose stable core-domain primitives.

Exact behavior owned by this file:

- export `ValidationIssue`;
- export `raise_for_errors`;
- export `LevelStructureSpec`;
- export `level_structure`;
- export only the existing core-domain names that are deliberately kept as Stage 1 core API.

What must not live here:

- no new implementation logic;
- no compatibility aliases for removed lattice or pulse names.

How this contributes to the Stage 1 goal:

- it gives advanced users a core-domain import path without forcing them through top-level `ryd_gate`.

### Modified File: `src/ryd_gate/core/system.py`

Primary responsibility after Stage 1:

- accept `Register` as the geometry object stored on `RydbergSystem`.

Exact refactor required in this file:

- import `Register` instead of `LatticeGeometry`;
- annotate `geometry: Register | None`;
- annotate `RydbergSystem.from_lattice(geometry: Register, ...)`;
- store the provided `Register` on `system.geometry`;
- keep `RydbergSystem.from_lattice(...)` as the system-construction entry point.

What must not change in this file:

- no simulation entry point;
- no sequence support;
- no backend selection;
- no Hamiltonian materialization behavior;
- no protocol API redesign.

How this contributes to the Stage 1 goal:

- it makes `Register` the real object consumed by current system construction, not a superficial front-end class.

### Modified File: `src/ryd_gate/core/factories.py`

Primary responsibility after Stage 1:

- build `RydbergSystem` from `Register`, `LevelStructureSpec`, and `InteractionSpec` using the current physical construction path.

Exact refactor required in this file:

- import `Register` instead of `LatticeGeometry`;
- annotate `build_from_lattice(..., geometry: Register, ...)`;
- annotate internal interaction-pair helpers with `Register`;
- read `geometry.N`, `geometry.coords`, `geometry.sublattice`, and `geometry.spacing_um` from `Register`;
- preserve existing physical Hamiltonian construction behavior.

What must not change in this file:

- no new public sequence object;
- no backend result object;
- no MPS/PEPS/stabilizer contract changes;
- no pulse waveform object compilation.

How this contributes to the Stage 1 goal:

- it ensures the new register API reaches the existing system factory directly.

### Modified File: `src/ryd_gate/backends/tn_common/lattice_spec.py`

Primary responsibility after Stage 1:

- align tensor-network lattice-spec helpers with the product register construction path.

Exact refactor required in this file:

- replace internal use of `make_square_lattice(...)` with `Register.square(...)`;
- import `Register` from `ryd_gate.lattice.geometry`;
- do not introduce a public compatibility factory.

What must not change in this file:

- no tensor-network backend contract;
- no MPS state handle;
- no simulation result API;
- no sampling API.

How this contributes to the Stage 1 goal:

- it removes an internal dependency on deleted old geometry factories without changing backend behavior.

### Modified File: `src/ryd_gate/protocols/channels.py`

Primary responsibility after Stage 1:

- own typed channel specifications and canonical compiler channel ids.

Functions and classes owned by this file:

- `ChannelSpec`;
- existing compiler channel id constants only where current lowering code needs them.

Exact refactor required in this file:

- add `ChannelSpec`;
- validate channel id, kind, transition, addressing, duration limits, clock period, amplitude limits, and detuning limits;
- keep string constants only as internal compiler channel ids, not as the product channel API.

What must not live here:

- no pulse object;
- no waveform sampling;
- no device registry;
- no sequence compilation;
- no backend code.

How this contributes to the Stage 1 goal:

- it gives devices and pulses a typed channel contract while leaving protocol lowering untouched.

### Modified File: `src/ryd_gate/pulse.py`

Primary responsibility after Stage 1:

- own waveform and pulse construction for user-facing laser controls.

Functions and classes owned by this file:

- `Waveform`;
- `Pulse`;
- private helpers for Blackman math if useful.

Exact refactor required in this file:

- add `Waveform.constant`;
- add `Waveform.ramp`;
- add `Waveform.blackman`;
- add `Waveform.interpolated`;
- add `Waveform.custom`;
- add waveform evaluation, sampling, integration, and endpoint methods;
- add `Pulse.constant`;
- add `Pulse.duration_ns`;
- add `Pulse.validate`;
- stop treating `blackman_window`, `blackman_pulse`, and `blackman_pulse_sqrt` as top-level product API.

What must not live here:

- no sequence object;
- no backend compiler;
- no system construction;
- no device registry;
- no noise model.

How this contributes to the Stage 1 goal:

- it turns laser control data into explicit product objects that later `Sequence` can compose without forcing Stage 1 to implement sequence simulation.

### Modified File: `src/ryd_gate/__init__.py`

Primary responsibility after Stage 1:

- expose the intended Stage 1 user imports at package top level.

Exact behavior owned by this file:

- export `Register`;
- export `RegisterLayout`;
- export `DeviceSpec`;
- export `ChannelSpec`;
- export `ValidationIssue`;
- export `raise_for_errors`;
- export `Waveform`;
- export `Pulse`;
- export `LevelStructureSpec`;
- export `level_structure`;
- keep `RydbergSystem` available.

What must not be exported from this file after Stage 1:

- `LatticeGeometry`;
- `make_chain`;
- `make_square_lattice`;
- `make_triangular_lattice`;
- `make_geometry_from_coords`;
- `blackman_window`;
- `blackman_pulse`;
- `blackman_pulse_sqrt`.

How this contributes to the Stage 1 goal:

- it makes the clean product API visible from `ryd_gate` itself while avoiding a separate `ryd_gate.api` wrapper package.

## Stage 1 Goal

Stage 1 builds a clean user-facing foundation while preserving physical behavior, not old API shape:

- stable atom register API, implemented in `lattice.geometry`;
- model/level API, implemented by extending `core.level_structures`;
- channel/device validation API, implemented in `protocols.channels` and a new `devices.py`;
- waveform/pulse API, implemented in existing `pulse.py`;
- validation issue primitive, implemented in `core.validation`;
- top-level exports in `ryd_gate.__init__`.

Stage 1 does not run simulations from a sequence. Stage 1 does not modify backend contracts.

## Hard Scope Boundary

Do not implement any of the following in Stage 1:

- `Sequence`
- `SequenceProtocol`
- `simulate_sequence`
- `simulate(seq, ...)`
- `NoiseModel`
- `SimulationResult`
- `QuantumStateHandle`
- `ExactStateHandle`
- `MPSStateHandle`
- `PEPSStateHandle`
- backend-native result handling
- MPS sampling
- PEPS contraction
- dense statevector materialization policy
- Pulser import/export compatibility

These belong to later stages.

## Allowed File Operations

Create exactly these files:

```text
src/ryd_gate/core/validation.py
src/ryd_gate/devices.py
tests/lattice/test_register.py
tests/core/test_level_structures_product_api.py
tests/core/test_validation.py
tests/core/test_devices.py
tests/core/test_pulse_api.py
```

Modify exactly these existing files:

```text
src/ryd_gate/lattice/geometry.py
src/ryd_gate/lattice/__init__.py
src/ryd_gate/core/level_structures.py
src/ryd_gate/core/__init__.py
src/ryd_gate/core/system.py
src/ryd_gate/core/factories.py
src/ryd_gate/backends/tn_common/lattice_spec.py
src/ryd_gate/protocols/channels.py
src/ryd_gate/pulse.py
src/ryd_gate/__init__.py
```

Do not modify:

```text
src/ryd_gate/simulate.py
src/ryd_gate/ir/evolution.py
src/ryd_gate/ir/hamiltonian.py
src/ryd_gate/core/system_model.py
src/ryd_gate/core/rb87_params.py
src/ryd_gate/core/local_blocks.py
src/ryd_gate/protocols/base.py
src/ryd_gate/protocols/digital_analog.py
src/ryd_gate/protocols/sweep.py
src/ryd_gate/backends/* except src/ryd_gate/backends/tn_common/lattice_spec.py
scripts/notebooks/*
pyproject.toml
```

If implementation requires changing any forbidden file, stop and update this plan first.

## Public Import Contract

After Stage 1, these imports must work:

```python
from ryd_gate import (
    Register,
    RegisterLayout,
    DeviceSpec,
    ChannelSpec,
    ValidationIssue,
    Waveform,
    Pulse,
    LevelStructureSpec,
    level_structure,
)

from ryd_gate.lattice import Register, RegisterLayout
from ryd_gate.core import ValidationIssue, LevelStructureSpec, level_structure
from ryd_gate.devices import DeviceSpec
from ryd_gate.protocols.channels import ChannelSpec
from ryd_gate.pulse import Waveform, Pulse
```

## Current Repository Mapping


| Product concept | Current module                                                                  | Stage 1 action                                                                                                |
| --------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Register        | `lattice.geometry.LatticeGeometry` and geometry factories                       | Replace `LatticeGeometry` with `Register` as the single geometry/register class; update internal call sites |
| Atom model      | `core.level_structures.LevelStructureSpec`, `TransitionSpec`, `level_structure` | Extend `LevelStructureSpec`; do not create `AtomModel` class                                                  |
| Interaction     | `core.level_structures.InteractionSpec`, `core.interactions.vdw_couplings`      | No API expansion in Stage 1                                                                                   |
| Channel         | `protocols.channels` constants and `LevelStructureSpec` channels                | Replace constants-only channel surface with `ChannelSpec` plus canonical compiler channel ids                 |
| Device          | No dedicated module                                                             | Add `devices.py` with `DeviceSpec`                                                                            |
| Waveform        | `pulse.py` Blackman helpers; callable schedules in protocols                    | Refactor `pulse.py` around `Waveform`; Blackman math may remain as private implementation detail              |
| Pulse           | Implicit in protocol coefficient functions                                      | Add `Pulse` to `pulse.py`; do not modify protocols                                                            |
| Validation      | No reusable validation primitive                                                | Add `core.validation.ValidationIssue`                                                                         |


## Module Plan: `src/ryd_gate/core/validation.py`

Create this file.

Implement exactly:

```python
from dataclasses import dataclass
from typing import Literal

ValidationSeverity = Literal["error", "warning"]

@dataclass(frozen=True)
class ValidationIssue:
    severity: ValidationSeverity
    code: str
    message: str
    path: tuple[str, ...] = ()

def raise_for_errors(issues: list[ValidationIssue]) -> None: ...
```

Behavior:

- `severity` must be `"error"` or `"warning"`.
- `code` must be a stable machine-readable string.
- `path` must be a tuple of strings.
- `raise_for_errors` returns `None` if no error exists.
- `raise_for_errors` raises `ValueError` if any issue has `severity == "error"`.
- The raised message must include each error code and message on separate lines.

Export `ValidationIssue` and `raise_for_errors` from:

```text
src/ryd_gate/core/__init__.py
src/ryd_gate/__init__.py
```

## Module Plan: `src/ryd_gate/lattice/geometry.py`

Replace the existing `LatticeGeometry` class with `Register`.

Do not create a wrapper class. Do not keep a second `LatticeGeometry` class. Do not define `Register = LatticeGeometry`.

The final Stage 1 geometry object is:

```text
Register
```

The intended path after Stage 1 is:

```text
Register -> RydbergSystem.from_lattice(...)
```

Breaking the old `LatticeGeometry` API is allowed. The implementation must update the direct internal call sites in:

```text
src/ryd_gate/core/system.py
src/ryd_gate/core/factories.py
src/ryd_gate/lattice/__init__.py
src/ryd_gate/__init__.py
```

so that the repository uses `Register` as the geometry/register class.

### RegisterLayout

```python
@dataclass(frozen=True)
class RegisterLayout:
    name: str
    trap_coords_um: tuple[tuple[float, ...], ...]
    kind: Literal["chain", "square", "rectangle", "triangular", "custom"]
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

`RegisterLayout` is allowed because it is metadata about a trap pattern, not a wrapper around geometry/register state. It must not contain level structure, pulse, device, interaction, or backend information.

### Register Fields

Replace the current `LatticeGeometry` dataclass with exactly:

```python
@dataclass(frozen=True)
class Register:
    N: int
    coords: np.ndarray
    sublattice: np.ndarray
    spacing_um: float
    ids: tuple[str, ...]
    layout: RegisterLayout | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Rules:

- `coords` is the single stored coordinate field.
- `coords` units are micrometers.
- do not add a stored field named `coords_um`.
- `coords_um` may exist only as a computed property.
- `N`, `coords`, `sublattice`, and `spacing_um` are still real fields because lattice algorithms need these values, but they now belong to `Register`.

### Register Validation and Normalization

In `Register.__post_init__`, implement exactly:

1. `N` must be a positive integer.
2. `coords` must be convertible to a finite float numpy array of shape `(N, 2)` or `(N, 3)`.
3. `sublattice` must be convertible to a numpy array of shape `(N,)`.
4. `spacing_um` must be a finite nonnegative float.
5. `ids` must be non-empty.
6. `len(ids) == N`.
7. every id must be a non-empty string after `str(...)`.
8. ids must be unique.
9. normalized `ids` must be stored as `tuple[str, ...]`.
10. normalized `coords` must be stored as a numpy array with dtype `float`.
11. normalized `sublattice` must be stored as a numpy array.

Use `object.__setattr__` because the dataclass is frozen.

Do not mutate input arrays in place.

### Class Constructors

Add these classmethods to `Register`:

```python
@classmethod
def from_coordinates(
    cls,
    coords,
    ids: Sequence[str] | None = None,
    prefix: str = "q",
    center: bool = True,
) -> "Register": ...
```

Rules:

- `coords` may be list-like or numpy array.
- if `center=True`, subtract the mean coordinate vector.
- if `ids is None`, generate ids `f"{prefix}{i}"`.
- `prefix` must be non-empty.
- generated ids must preserve coordinate order.
- if no sublattice is supplied, use zeros.
- spacing is inferred using the existing `make_geometry_from_coords` behavior, or the same nearest-positive-x-spacing rule if implemented directly.

Add:

```python
@classmethod
def chain(cls, n_atoms: int, spacing_um: float, prefix: str = "q") -> "Register": ...

@classmethod
def square(cls, side: int, spacing_um: float, prefix: str = "q") -> "Register": ...

@classmethod
def rectangle(cls, rows: int, cols: int, spacing_um: float, prefix: str = "q") -> "Register": ...
```

Rules:

- `chain`: coordinates are `(i * spacing_um, 0.0)`.
- `square`: calls `rectangle(side, side, spacing_um, prefix)`.
- `rectangle`: row-major with index `i = row * cols + col`; coordinate is `(row * spacing_um, col * spacing_um)`.
- `chain`, `square`, and `rectangle` do not center coordinates.
- all integer sizes must be positive.
- `spacing_um` must be positive.
- these constructors return `Register` objects.

### Constructor API Requirement

The official Stage 1 geometry construction API is the `Register` classmethod set:

```python
Register.chain(...)
Register.square(...)
Register.rectangle(...)
Register.from_coordinates(...)
```

Delete the old module-level public factory functions from `lattice.geometry`:

```python
make_chain
make_square_lattice
make_triangular_lattice
make_geometry_from_coords
```

Do not export them from `ryd_gate.lattice` or `ryd_gate`.

If internal source files need these constructors, update them to call the `Register` classmethods. In Stage 1 this specifically includes:

```text
src/ryd_gate/backends/tn_common/lattice_spec.py
  make_square_lattice(...) -> Register.square(...)
```

Do not keep old factory functions as aliases. They are an old API shape, not the product API.

### Register Properties and Methods

Add:

```python
@property
def n_atoms(self) -> int: ...

@property
def dimensions(self) -> int: ...

@property
def coords_array(self) -> np.ndarray: ...

@property
def coords_um(self) -> tuple[tuple[float, ...], ...]: ...

def index(self, atom_id: str) -> int: ...

def id_at(self, index: int) -> str: ...

def distances_um(self) -> np.ndarray: ...

def distance_pairs(self, cutoff_um: float | None = None) -> tuple[tuple[int, int, float], ...]: ...

def blockade_edges(self, radius_um: float) -> tuple[tuple[int, int], ...]: ...

def validate(self, device: Any) -> list[ValidationIssue]: ...
```

Rules:

- `coords_array` returns a new numpy array.
- `coords_um` returns a tuple-of-tuples view of `coords` for serialization/user display.
- `index` raises `KeyError` for unknown id.
- `id_at` raises `IndexError` for invalid index.
- `distances_um` returns a symmetric `(N, N)` array with zero diagonal.
- `distance_pairs` returns upper-triangular `(i, j, distance_um)` pairs.
- `distance_pairs(cutoff_um=x)` omits pairs with distance greater than `x`.
- `blockade_edges(radius_um)` returns upper-triangular `(i, j)` pairs with distance `<= radius_um`.
- `validate(device)` returns `device.validate_register(self)`.

Do not implement `to_geometry()`. `Register` is already the geometry object consumed by `RydbergSystem.from_lattice(...)`.

### Required Internal Call-Site Updates

Update direct imports and annotations:

```text
src/ryd_gate/core/system.py
  from ryd_gate.lattice.geometry import Register
  RydbergSystem.__init__(..., geometry: Register | None = None, ...)
  RydbergSystem.from_lattice(geometry: Register, ...)

src/ryd_gate/core/factories.py
  from ryd_gate.lattice.geometry import Register
  build_from_lattice(..., geometry: Register, ...)
  _interaction_pairs(geometry: Register, ...)

src/ryd_gate/backends/tn_common/lattice_spec.py
  from ryd_gate.lattice.geometry import Register
  use Register.square(...) instead of make_square_lattice(...)
```

Do not rename `RydbergSystem.from_lattice` in Stage 1. The method name remains because it describes constructing from an atom array geometry; only the geometry class becomes `Register`.

Remove all direct imports of `LatticeGeometry` from source files touched in Stage 1.

### Exports

Export `Register` and `RegisterLayout` from:

```text
src/ryd_gate/lattice/__init__.py
src/ryd_gate/__init__.py
```

Do not export `LatticeGeometry` from these modules after Stage 1.

## Module Plan: `src/ryd_gate/core/level_structures.py`

Do not create `AtomModel`.

Extend `LevelStructureSpec` in place so it can serve as both:

- compiler-facing local level spec;
- user-facing atom model spec.

Change dataclass fields to exactly:

```python
@dataclass(frozen=True)
class LevelStructureSpec:
    name: str
    levels: tuple[str, ...]
    rydberg_levels: tuple[str, ...]
    transitions: tuple[TransitionSpec, ...] = ()
    detuning_levels: dict[str, str] = field(default_factory=dict)
    initial_level: str | None = None
    species: str = "Rb87"
    interaction_kind: Literal["none", "ising_c6", "xy_c3", "custom"] = "ising_c6"
    params: Mapping[str, Any] = field(default_factory=dict)
```

Migration rule:

- All new fields may have defaults to keep object construction concise, but Stage 1 is allowed to update internal call sites instead of preserving every old construction shape.

Add methods:

```python
def initial_level_or_default(self) -> str: ...

def physical_kwargs(self) -> dict[str, Any]: ...

def supports_backend(self, backend: str) -> bool: ...

def validate(self) -> list[ValidationIssue]: ...
```

Rules:

- `initial_level_or_default()` returns `initial_level` if not `None`, else `levels[0]`.
- `physical_kwargs()`:
  - for `name == "analog_3"`, return `{"param_set": "analog_3", **params_without_param_set}`;
  - for `name == "rb87_7"`, return `{"param_set": params.get("param_set", "our"), **params_without_param_set}`;
  - for all other names, return `{}`.
- `supports_backend("exact")` returns true for `01`, `1r`, `01r`, `ger`, `analog_3`, `rb87_7`.
- `supports_backend("mps")`, `"gputn"`, and `"peps"` return true only for `1r` and `01r`.
- `supports_backend("stabilizer")` returns true only for `01`.
- unknown backend returns false.

### Presets

Modify `level_structure(name: str)` so these names are valid:

```text
01
1r
01r
ger
analog_3
rb87_7
```

Preset details:

```text
01:
  levels=("0", "1")
  rydberg_levels=()
  transitions=()
  detuning_levels={}
  initial_level="0"
  interaction_kind="none"

1r:
  current compiler channel names unchanged: global_X, global_n
  initial_level="1"
  interaction_kind="ising_c6"

01r:
  current compiler channel names unchanged: drive_R, drive_hf, delta_R, delta_hf
  initial_level="1"
  interaction_kind="ising_c6"

ger:
  current compiler channel names unchanged: drive_420, H_1013, delta_e, delta_R
  initial_level="g"
  interaction_kind="ising_c6"

analog_3:
  same levels, transitions, detuning channels as ger
  name="analog_3"
  initial_level="g"
  params={"param_set": "analog_3"}

rb87_7:
  current levels unchanged
  initial_level="0"
  interaction_kind="ising_c6"
  params={"param_set": "our"}
```

Important migration rule:

- `level_structure("analog_3")` becomes the official Stage 1 way to request the analog three-level preset.
- The implementation may update internal examples/tests to use `level_structure="analog_3"` instead of `level_structure="ger", param_set="analog_3"`.
- Stage 1 does not need to preserve `ger + param_set="analog_3"` as a public compatibility path.

## Module Plan: `src/ryd_gate/protocols/channels.py`

Replace the constants-only channel module with a typed channel spec module.

Stage 1 may keep string constants only when they are canonical compiler channel ids used by existing lowering code. They are not retained as a compatibility public API; they are implementation constants.

Add:

```python
@dataclass(frozen=True)
class ChannelSpec:
    channel_id: str
    kind: Literal["rydberg", "raman", "microwave", "dmm", "custom"]
    transition: str
    addressing: Literal["global", "local"]
    amplitude_channels: Mapping[str, str] = field(default_factory=dict)
    detuning_channels: Mapping[str, str] = field(default_factory=dict)
    max_abs_amplitude_rad_per_us: float | None = None
    max_abs_detuning_rad_per_us: float | None = None
    min_duration_ns: int = 0
    max_duration_ns: int | None = None
    clock_period_ns: int = 1
    max_targets: int | None = None
    retarget_time_ns: int | None = None
```

Validation in `__post_init__`:

- `channel_id` non-empty.
- `kind` one of the declared literal values.
- `addressing` is `"global"` or `"local"`.
- `min_duration_ns >= 0`.
- `max_duration_ns` is `None` or `>= min_duration_ns`.
- `clock_period_ns > 0`.
- all max amplitude/detuning limits, if present, are positive.

Do not add pulse compilation logic here.

Export `ChannelSpec` from:

```text
src/ryd_gate/protocols/channels.py
src/ryd_gate/__init__.py
```

## Module Plan: `src/ryd_gate/devices.py`

Create this new top-level module. This mirrors Pulser's domain-module style more closely than `api/device.py`.

Implement:

```python
@dataclass(frozen=True)
class DeviceSpec:
    name: str
    dimensions: Literal[2, 3]
    atom_species: str
    allowed_level_structures: tuple[str, ...]
    default_level_structure: str
    min_atom_distance_um: float
    max_atom_num: int | None = None
    max_radial_distance_um: float | None = None
    interaction_coeffs: Mapping[str, float] = field(default_factory=dict)
    channels: Mapping[str, ChannelSpec] = field(default_factory=dict)
    supports_slm_mask: bool = False
    max_sequence_duration_ns: int | None = None
    max_runs: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Do not use field names `allowed_atom_models` or `default_atom_model` in Stage 1. The repo's existing concept is `LevelStructureSpec`, so the public device must use `allowed_level_structures`.

### Constructor

Implement:

```python
@classmethod
def virtual_rb87(cls) -> "DeviceSpec": ...
```

Fixed values:

```text
name="virtual_rb87"
dimensions=2
atom_species="Rb87"
allowed_level_structures=("01", "1r", "01r", "ger", "analog_3", "rb87_7")
default_level_structure="01r"
min_atom_distance_um=2.0
max_atom_num=None
max_radial_distance_um=None
interaction_coeffs={"C6_rad_s_um6": DEFAULT_C6}
supports_slm_mask=False
max_sequence_duration_ns=None
max_runs=None
```

Fixed channels:

```text
rydberg_global:
  kind="rydberg"
  transition="1_r"
  addressing="global"
  amplitude_channels={"1r": "global_X", "01r": "drive_R"}
  detuning_channels={"1r": "global_n", "01r": "delta_R"}

rydberg_local:
  kind="rydberg"
  transition="1_r"
  addressing="local"
  amplitude_channels={"01r": "drive_R"}
  detuning_channels={"01r": "delta_R"}

hyperfine_global:
  kind="microwave"
  transition="0_1"
  addressing="global"
  amplitude_channels={"01r": "drive_hf"}
  detuning_channels={"01r": "delta_hf"}
```

### Validation Methods

Implement:

```python
def validate_register(self, register: Register) -> list[ValidationIssue]: ...

def validate_level_structure(self, spec: LevelStructureSpec | str) -> list[ValidationIssue]: ...

def validate_pulse(self, pulse: "Pulse", channel_id: str) -> list[ValidationIssue]: ...
```

Rules:

- `validate_register` checks:
  - register dimensions equal device dimensions;
  - atom count <= `max_atom_num` if max is not `None`;
  - every pair distance >= `min_atom_distance_um`;
  - every radial distance <= `max_radial_distance_um` if max is not `None`.
- `validate_level_structure` checks:
  - spec name is in `allowed_level_structures`;
  - spec species equals `atom_species`.
- `validate_pulse` checks:
  - channel exists;
  - pulse duration respects min/max duration;
  - pulse duration is a multiple of `clock_period_ns`;
  - amplitude values respect `max_abs_amplitude_rad_per_us` if set;
  - detuning values respect `max_abs_detuning_rad_per_us` if set.
- Validation methods return `ValidationIssue` objects; they do not raise.

Export `DeviceSpec` from:

```text
src/ryd_gate/__init__.py
```

Do not create `src/ryd_gate/core/devices.py`.

## Module Plan: `src/ryd_gate/pulse.py`

Refactor this module around product-level `Waveform` and `Pulse`.

The old public helper functions are not a compatibility requirement:

```python
blackman_window
blackman_pulse
blackman_pulse_sqrt
```

The implementation may keep their numerical formulas as private helpers if useful, but Stage 1 public API is `Waveform` and `Pulse`.

### Waveform

Add:

```python
WaveformKind = Literal["constant", "ramp", "blackman", "interpolated", "custom"]
WaveformUnit = Literal["rad_per_us"]

@dataclass(frozen=True)
class Waveform:
    duration_ns: int
    kind: WaveformKind
    params: Mapping[str, Any] = field(default_factory=dict)
    samples: tuple[float, ...] | None = None
    unit: WaveformUnit = "rad_per_us"
```

Constructors:

```python
@classmethod
def constant(cls, duration_ns: int, value: float) -> "Waveform": ...

@classmethod
def ramp(cls, duration_ns: int, start: float, stop: float) -> "Waveform": ...

@classmethod
def blackman(cls, duration_ns: int, peak: float) -> "Waveform": ...

@classmethod
def interpolated(cls, duration_ns: int, times_ns, values) -> "Waveform": ...

@classmethod
def custom(cls, samples, dt_ns: int = 1) -> "Waveform": ...
```

Methods:

```python
def value_at_ns(self, t_ns: float) -> float: ...

def value_at_s(self, t_s: float) -> float: ...

def sample(self, dt_ns: int = 1) -> np.ndarray: ...

def integral_rad(self, dt_ns: int = 1) -> float: ...

def first_value(self) -> float: ...

def last_value(self) -> float: ...
```

Rules:

- `duration_ns` must be positive integer.
- values are public-facing `rad/us`.
- `value_at_ns` clamps into `[0, duration_ns]`.
- `value_at_s(t_s)` returns `rad/s`, equal to `value_at_ns(t_s * 1e9) * 1e6`.
- `sample(dt_ns)` returns samples at `0, dt_ns, ...` including `duration_ns`.
- `integral_rad` integrates `rad/us` over microseconds.
- no composite waveform in Stage 1.

### Pulse

Add:

```python
@dataclass(frozen=True)
class Pulse:
    amplitude: Waveform
    detuning: Waveform
    phase_rad: float = 0.0
    post_phase_shift_rad: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Constructor:

```python
@classmethod
def constant(
    cls,
    duration_ns: int,
    amplitude: float,
    detuning: float,
    phase_rad: float = 0.0,
    post_phase_shift_rad: float = 0.0,
) -> "Pulse": ...
```

Properties and methods:

```python
@property
def duration_ns(self) -> int: ...

def validate(self, channel: ChannelSpec) -> list[ValidationIssue]: ...
```

Rules:

- amplitude and detuning durations must match.
- phase values must be finite floats.
- nonzero phase is stored but not compiled in Stage 1.
- `Pulse.validate(channel)` checks only channel limits; no system or backend access.

Export `Waveform` and `Pulse` from:

```text
src/ryd_gate/__init__.py
```

## Export Plan

Modify `src/ryd_gate/lattice/__init__.py`:

```python
from .geometry import Register, RegisterLayout
```

Modify `src/ryd_gate/core/__init__.py`:

```python
from ryd_gate.core.validation import ValidationIssue, raise_for_errors
```

Modify `src/ryd_gate/__init__.py`:

Add imports:

```python
from .devices import DeviceSpec
from .lattice import Register, RegisterLayout
from .protocols.channels import ChannelSpec
from .pulse import Pulse, Waveform
from .core.validation import ValidationIssue, raise_for_errors
```

Update `__all__` accordingly.

Do not add lazy imports for these Stage 1 classes.

## Tests

### `tests/lattice/test_register.py`

Required tests:

1. `Register.chain(3, 4.0)` produces ids `("q0", "q1", "q2")` and coordinates `((0.0, 0.0), (4.0, 0.0), (8.0, 0.0))`.
2. `Register.rectangle(2, 3, 5.0)` uses row-major id order.
3. duplicate ids raise `ValueError`.
4. mixed coordinate dimensions raise `ValueError`.
5. `distances_um()` is symmetric with zero diagonal.
6. `distance_pairs(cutoff_um=...)` filters pairs.
7. `blockade_edges(radius_um=...)` returns only upper-triangular pairs.
8. `Register.square(...)` returns a `Register`.
9. `from ryd_gate.lattice import LatticeGeometry` is no longer a valid public import after Stage 1.
10. `from ryd_gate.lattice import make_square_lattice` is no longer a valid public import after Stage 1.

### `tests/core/test_level_structures_product_api.py`

Required tests:

1. `level_structure("01")` exists and has levels `("0", "1")`.
2. `level_structure("1r").initial_level_or_default() == "1"`.
3. `level_structure("01r").initial_level_or_default() == "1"`.
4. `level_structure("analog_3").name == "analog_3"`.
5. `level_structure("analog_3").physical_kwargs()["param_set"] == "analog_3"`.
6. `level_structure("rb87_7").physical_kwargs()["param_set"] == "our"`.
7. `supports_backend("mps")` is true only for `1r` and `01r`.
8. `supports_backend("stabilizer")` is true only for `01`.
9. `level_structure("ger")` remains a distinct explicit ladder preset with channels `drive_420`, `H_1013`, `delta_e`, and `delta_R`.

### `tests/core/test_validation.py`

Required tests:

1. warning-only issues do not raise.
2. one error issue raises `ValueError`.
3. multiple error issues appear on separate lines.

### `tests/core/test_devices.py`

Required tests:

1. `DeviceSpec.virtual_rb87()` contains `rydberg_global`.
2. register with distance below `2.0 um` returns error code `register.min_distance`.
3. unsupported level structure returns error code `level_structure.unsupported`.
4. pulse exceeding amplitude limit returns an error if a limit is set manually.
5. pulse duration not divisible by `clock_period_ns` returns error code `pulse.clock_period`.

### `tests/core/test_pulse_api.py`

Required tests:

1. `Waveform.blackman` reproduces the intended Blackman envelope shape through the new API.
2. `Waveform.constant` samples all equal value.
3. `Waveform.ramp` starts and ends at requested values.
4. `Waveform.value_at_s` converts `rad/us` to `rad/s`.
5. `Waveform.custom` respects `dt_ns`.
6. invalid waveform duration raises.
7. `Pulse.constant` creates matching amplitude/detuning durations.
8. mismatched waveform durations raise.
9. finite nonzero phase is stored.
10. `Pulse.validate(channel)` returns no errors for an unconstrained virtual channel.
11. `from ryd_gate import blackman_pulse` is no longer a valid public import after Stage 1.

## Acceptance Command

Run exactly:

```bash
pytest tests/lattice/test_register.py tests/core/test_level_structures_product_api.py tests/core/test_validation.py tests/core/test_devices.py tests/core/test_pulse_api.py
```

Stage 1 is complete only if:

1. the command passes;
2. no `src/ryd_gate/api/` directory exists;
3. no forbidden file changed;
4. new imports from `ryd_gate`, `ryd_gate.core`, `ryd_gate.lattice`, `ryd_gate.protocols`, and `ryd_gate.pulse` work;
5. `LatticeGeometry` is not exported as a public class from `ryd_gate` or `ryd_gate.lattice`;
6. `make_chain`, `make_square_lattice`, `make_triangular_lattice`, and `make_geometry_from_coords` are not exported as public functions from `ryd_gate` or `ryd_gate.lattice`;
7. `blackman_window`, `blackman_pulse`, and `blackman_pulse_sqrt` are not exported from top-level `ryd_gate`;
8. all internal call sites touched by Stage 1 use `Register`, not `LatticeGeometry`.

## Non-Goals for Stage 1

Stage 1 must not attempt to decide the final shape of `Sequence`, `SimulationResult`, or backend-native state handles. Those are later-stage integration problems. Stage 1 only ensures that the foundational public nouns live in the correct existing domain modules.
