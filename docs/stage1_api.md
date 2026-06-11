# Stage 1 User API

This page documents the Stage 1 public API target for `ryd_gate`. The API is planned as a product-facing foundation for neutral-atom simulation: registers, level models, devices, channels, waveforms, pulses, and validation. It is not a sequence API and it does not run simulations from a sequence.

Stage 1 keeps physical behavior stable while replacing old public API shapes with clearer product objects.

Shared unit rules:

- register coordinates are in micrometers;
- waveform duration inputs are in nanoseconds;
- waveform values are in `rad/us`;
- `Waveform.value_at_s(...)` returns `rad/s`;
- validation methods return `list[ValidationIssue]` and do not raise;
- constructors raise `ValueError` for invalid structural input.

Stage 1 does not create `Sequence`, does not support `simulate(seq, ...)`, and does not change backend contracts.

## Imports

Top-level import:

```python
from ryd_gate import (
    Register,
    RegisterLayout,
    DeviceSpec,
    ChannelSpec,
    ValidationIssue,
    raise_for_errors,
    Waveform,
    Pulse,
    LevelStructureSpec,
    level_structure,
    RydbergSystem,
)
```

Input:

- no runtime input.

Output:

- imported class and function objects;
- importing these names must not initialize a backend, allocate a Hamiltonian, compile a sequence, or run a simulation.

Domain imports:

```python
from ryd_gate.lattice import Register, RegisterLayout
from ryd_gate.core import ValidationIssue, raise_for_errors, LevelStructureSpec, level_structure
from ryd_gate.protocols.channels import ChannelSpec
from ryd_gate.devices import DeviceSpec
from ryd_gate.pulse import Waveform, Pulse
```

Removed public imports after Stage 1:

```python
from ryd_gate.lattice import LatticeGeometry
from ryd_gate.lattice import make_square_lattice
from ryd_gate import blackman_pulse
```

These imports must fail after Stage 1.

## Registers

`Register` is the stable atom register API. It owns atom ids, coordinates, stable order, lattice spacing metadata, sublattice metadata, and geometry-only helper methods.

### RegisterLayout

```python
layout = RegisterLayout(
    name="square_2x2",
    trap_coords_um=((0.0, 0.0), (0.0, 5.0), (5.0, 0.0), (5.0, 5.0)),
    kind="square",
)
```

Input:

- `name`: non-empty string;
- `trap_coords_um`: tuple of 2D or 3D coordinate tuples in micrometers;
- `kind`: one of `"chain"`, `"square"`, `"rectangle"`, `"triangular"`, `"custom"`;
- `metadata`: optional mapping.

Output:

- immutable `RegisterLayout`;
- no atom ids, no level structure, no device, no backend information.

### Direct Register Construction

```python
reg = Register(
    N=2,
    coords=[[0.0, 0.0], [4.0, 0.0]],
    sublattice=[0, 0],
    spacing_um=4.0,
    ids=("q0", "q1"),
)
```

Input:

- `N`: positive integer atom count;
- `coords`: finite float array-like of shape `(N, 2)` or `(N, 3)`;
- `sublattice`: array-like of shape `(N,)`;
- `spacing_um`: finite nonnegative float;
- `ids`: tuple or list of `N` unique non-empty strings;
- optional `layout` and `metadata`.

Output:

- immutable `Register`;
- normalized `coords` stored as `np.ndarray(dtype=float)`;
- normalized `sublattice` stored as `np.ndarray`;
- normalized `ids` stored as `tuple[str, ...]`;
- stable atom order exactly follows the order in `ids` and `coords`.

Failure:

- invalid `N`, invalid shape, non-finite coordinates, duplicate id, empty id, mismatched id count, or invalid spacing raises `ValueError`.

### Chain

```python
reg = Register.chain(3, 4.0)
```

Input:

- `n_atoms=3`: positive integer;
- `spacing_um=4.0`: positive float;
- optional `prefix="q"`.

Output:

- `Register` with `N == 3`;
- `ids == ("q0", "q1", "q2")`;
- `coords_um == ((0.0, 0.0), (4.0, 0.0), (8.0, 0.0))`;
- `sublattice == np.array([0, 0, 0])`;
- coordinates are not centered.

### Square

```python
reg = Register.square(2, 5.0, prefix="a")
```

Input:

- `side=2`: positive integer;
- `spacing_um=5.0`: positive float;
- `prefix="a"`.

Output:

- `Register` with `N == 4`;
- `ids == ("a0", "a1", "a2", "a3")`;
- row-major coordinates `((0.0, 0.0), (0.0, 5.0), (5.0, 0.0), (5.0, 5.0))`;
- coordinates are not centered.

### Rectangle

```python
reg = Register.rectangle(2, 3, 5.0)
```

Input:

- `rows=2`: positive integer;
- `cols=3`: positive integer;
- `spacing_um=5.0`: positive float;
- optional `prefix="q"`.

Output:

- `Register` with `N == 6`;
- row-major atom order with index `i = row * cols + col`;
- coordinates `((0.0, 0.0), (0.0, 5.0), (0.0, 10.0), (5.0, 0.0), (5.0, 5.0), (5.0, 10.0))`;
- ids `("q0", "q1", "q2", "q3", "q4", "q5")`;
- coordinates are not centered.

### From Coordinates

```python
reg = Register.from_coordinates(
    [(0.0, 0.0), (4.0, 0.0)],
    ids=("left", "right"),
    center=False,
)
```

Input:

- `coords`: finite array-like of 2D or 3D coordinates;
- optional `ids`: unique non-empty strings matching coordinate length;
- optional `prefix="q"` used only when `ids is None`;
- `center`: boolean.

Output:

- `Register` with stable atom order equal to coordinate order;
- if `center=False`, stored coordinates equal input coordinates converted to float;
- if `center=True`, mean coordinate vector is subtracted before storage;
- if `ids is None`, ids are generated as `f"{prefix}{i}"`;
- `sublattice` is all zeros;
- `spacing_um` is inferred from nearest positive x-coordinate separation.

Failure:

- empty coordinate list, mixed dimensions, duplicate ids, empty prefix when ids are generated, or failed positive spacing inference raises `ValueError`.

### Register Properties

```python
n = reg.n_atoms
dim = reg.dimensions
coords = reg.coords_array
coords_um = reg.coords_um
```

Output:

- `n_atoms`: integer equal to `reg.N`;
- `dimensions`: integer `2` or `3`, equal to `reg.coords.shape[1]`;
- `coords_array`: new `np.ndarray` copy of shape `(N, dimensions)`;
- `coords_um`: tuple-of-tuples coordinate representation suitable for serialization and display.

Mutating `coords_array` must not mutate the register.

### Register Indexing

```python
i = reg.index("q1")
atom_id = reg.id_at(1)
```

Input:

- `index(atom_id)`: atom id string;
- `id_at(index)`: integer atom index.

Output:

- `index(...)`: integer index of the id in stable register order;
- `id_at(...)`: id string at that stable index.

Failure:

- unknown id raises `KeyError`;
- negative or out-of-range index raises `IndexError`.

### Distances and Blockade Edges

```python
d = reg.distances_um()
pairs = reg.distance_pairs(cutoff_um=6.0)
edges = reg.blockade_edges(radius_um=6.0)
```

Input:

- optional `cutoff_um`: `None` or finite nonnegative float;
- `radius_um`: finite nonnegative float.

Output:

- `distances_um()`: symmetric `(N, N)` distance matrix in micrometers with zero diagonal;
- `distance_pairs(...)`: tuple of upper-triangular `(i, j, distance_um)` pairs with `i < j`;
- `blockade_edges(...)`: tuple of upper-triangular `(i, j)` pairs with distance `<= radius_um`.

### Register Validation

```python
issues = reg.validate(device)
```

Input:

- `device`: object implementing `validate_register(register)`.

Output:

- exactly `device.validate_register(reg)`;
- no exception unless the device object itself raises.

## Level Structures

`LevelStructureSpec` is the Stage 1 atom-model API. It describes local levels, Rydberg levels, transition channels, detuning channels, initial level, species, interaction kind, and physical parameters.

### Presets

```python
spec = level_structure("01r")
```

Input:

- `name`: one of `"01"`, `"1r"`, `"01r"`, `"ger"`, `"analog_3"`, `"rb87_7"`.

Output:

- `LevelStructureSpec`.

Failure:

- unknown name raises `ValueError`.

Required preset behavior:

```python
level_structure("01").levels == ("0", "1")
level_structure("1r").initial_level_or_default() == "1"
level_structure("01r").initial_level_or_default() == "1"
level_structure("ger").initial_level_or_default() == "g"
level_structure("analog_3").physical_kwargs()["param_set"] == "analog_3"
level_structure("rb87_7").physical_kwargs()["param_set"] == "our"
```

### Custom Level Structure

```python
custom = LevelStructureSpec(
    name="custom_gr",
    levels=("g", "r"),
    rydberg_levels=("r",),
    initial_level="g",
    interaction_kind="ising_c6",
)
```

Input:

- `name`: non-empty model name;
- `levels`: tuple of unique non-empty level labels;
- `rydberg_levels`: tuple whose entries must exist in `levels`;
- optional `transitions`;
- optional `detuning_levels`;
- optional `initial_level`;
- optional `species`;
- optional `interaction_kind`;
- optional `params`.

Output:

- immutable `LevelStructureSpec`;
- no backend object and no Hamiltonian object.

### Level-Structure Methods

```python
level = spec.initial_level_or_default()
kwargs = spec.physical_kwargs()
ok = spec.supports_backend("mps")
issues = spec.validate()
```

Output:

- `initial_level_or_default()`: `spec.initial_level` if set, otherwise `spec.levels[0]`;
- `physical_kwargs()`: physical factory kwargs needed by current system construction;
- `supports_backend(...)`: boolean support flag for a backend name;
- `validate()`: `list[ValidationIssue]`.

Backend support matrix:

- `"exact"` supports `"01"`, `"1r"`, `"01r"`, `"ger"`, `"analog_3"`, and `"rb87_7"`;
- `"mps"`, `"gputn"`, and `"peps"` support only `"1r"` and `"01r"`;
- `"stabilizer"` supports only `"01"`;
- unknown backend names return `False`.

Validation errors cover empty levels, duplicate levels, unknown initial level, Rydberg levels not present in levels, invalid detuning target level, and invalid interaction kind.

## Channels

`ChannelSpec` describes a physical control channel and its constraints. It does not compile pulses or sample waveforms.

```python
channel = ChannelSpec(
    channel_id="rydberg_global",
    kind="rydberg",
    transition="1_r",
    addressing="global",
)
```

Input:

- `channel_id`: non-empty string;
- `kind`: one of `"rydberg"`, `"raman"`, `"microwave"`, `"dmm"`, `"custom"`;
- `transition`: non-empty transition label;
- `addressing`: `"global"` or `"local"`;
- optional amplitude and detuning channel maps;
- optional amplitude, detuning, duration, clock, target, and retargeting limits.

Output:

- immutable `ChannelSpec`;
- no pulse compilation, no waveform sampling, no backend state.

Failure:

- empty `channel_id`, invalid `kind`, invalid `addressing`, negative duration limit, impossible min/max duration, nonpositive clock period, or nonpositive finite amplitude/detuning limits raises `ValueError`.

Device channel lookup:

```python
channel = DeviceSpec.virtual_rb87().channels["rydberg_global"]
```

Output:

- `ChannelSpec` with `kind == "rydberg"`, `transition == "1_r"`, and `addressing == "global"`.

## Devices

`DeviceSpec` describes hardware and physical constraints. It is not a backend and not a real QPU object.

### Virtual Rb87 Device

```python
device = DeviceSpec.virtual_rb87()
```

Input:

- no runtime input.

Output:

- immutable `DeviceSpec`;
- `name == "virtual_rb87"`;
- `dimensions == 2`;
- `atom_species == "Rb87"`;
- `allowed_level_structures == ("01", "1r", "01r", "ger", "analog_3", "rb87_7")`;
- `default_level_structure == "01r"`;
- `min_atom_distance_um == 2.0`;
- `channels` contains `"rydberg_global"`, `"rydberg_local"`, and `"hyperfine_global"`.

### Register Validation

```python
issues = device.validate_register(Register.chain(2, 4.0))
```

Input:

- `Register`.

Output:

- `list[ValidationIssue]`;
- empty list when register dimensions, atom count, pair distances, and radial extent satisfy the device.

Error codes:

- `register.dimensions`;
- `register.max_atom_num`;
- `register.min_distance`;
- `register.max_radial_distance`.

### Level-Structure Validation

```python
issues = device.validate_level_structure("01r")
```

Input:

- string level structure name or `LevelStructureSpec`.

Output:

- `list[ValidationIssue]`;
- empty list when name is allowed and species matches.

Error codes:

- `level_structure.unsupported`;
- `level_structure.species`.

### Pulse Validation

```python
issues = device.validate_pulse(
    Pulse.constant(1000, amplitude=1.0, detuning=0.0),
    "rydberg_global",
)
```

Input:

- `Pulse`;
- `channel_id` string.

Output:

- `list[ValidationIssue]`;
- empty list when channel exists and pulse satisfies channel limits.

Error codes:

- `channel.unknown`;
- `pulse.min_duration`;
- `pulse.max_duration`;
- `pulse.clock_period`;
- `pulse.amplitude_limit`;
- `pulse.detuning_limit`.

## Waveforms

`Waveform` describes a scalar time-dependent control value. Values are public-facing `rad/us`.

### Constant

```python
wf = Waveform.constant(1000, 2.5)
```

Input:

- `duration_ns=1000`: positive integer;
- `value=2.5`: finite float in `rad/us`.

Output:

- `Waveform` with `kind == "constant"`;
- `wf.value_at_ns(t) == 2.5` for any clamped `t`;
- `wf.sample(dt_ns=...)` returns all `2.5`.

### Ramp

```python
wf = Waveform.ramp(1000, start=0.0, stop=5.0)
```

Input:

- positive integer duration;
- finite `start` and `stop` values in `rad/us`.

Output:

- `Waveform` with `kind == "ramp"`;
- `wf.first_value() == 0.0`;
- `wf.last_value() == 5.0`;
- linear interpolation between start and stop.

### Blackman

```python
wf = Waveform.blackman(1000, peak=3.0)
```

Input:

- positive integer duration;
- finite peak value in `rad/us`.

Output:

- `Waveform` with `kind == "blackman"`;
- endpoint values follow the Blackman envelope used by the current pulse math;
- maximum sampled value is approximately `peak` within numerical sampling resolution.

### Interpolated

```python
wf = Waveform.interpolated(
    1000,
    times_ns=[0, 500, 1000],
    values=[0.0, 2.0, 0.0],
)
```

Input:

- positive integer duration;
- monotonic `times_ns` beginning at `0` and ending at `duration_ns`;
- finite values with the same length as `times_ns`.

Output:

- `Waveform` with `kind == "interpolated"`;
- piecewise-linear value interpolation.

Failure:

- unsorted times, mismatched lengths, missing endpoints, or non-finite values raises `ValueError`.

### Custom

```python
wf = Waveform.custom([0.0, 1.0, 0.0], dt_ns=10)
```

Input:

- finite sample values;
- positive integer `dt_ns`.

Output:

- `Waveform` with `kind == "custom"`;
- `samples == (0.0, 1.0, 0.0)`;
- `duration_ns == (len(samples) - 1) * dt_ns`.

Failure:

- fewer than two samples, non-finite sample, or nonpositive `dt_ns` raises `ValueError`.

### Evaluation and Sampling

```python
value_ns = wf.value_at_ns(250.0)
value_s = wf.value_at_s(250e-9)
samples = wf.sample(dt_ns=10)
area = wf.integral_rad(dt_ns=10)
first = wf.first_value()
last = wf.last_value()
```

Output:

- `value_at_ns(...)`: float value in `rad/us`, with time clamped into `[0, duration_ns]`;
- `value_at_s(...)`: float value in `rad/s`, equal to `value_at_ns(t_s * 1e9) * 1e6`;
- `sample(...)`: `np.ndarray` at `0, dt_ns, ...` including `duration_ns`;
- `integral_rad(...)`: pulse area in radians;
- `first_value()` and `last_value()`: endpoint values in `rad/us`.

## Pulses

`Pulse` combines amplitude, detuning, phase, and post phase shift. It does not compile itself into a sequence in Stage 1.

### Constant Pulse

```python
pulse = Pulse.constant(
    1000,
    amplitude=1.0,
    detuning=0.0,
    phase_rad=0.25,
)
```

Input:

- positive integer duration in nanoseconds;
- finite amplitude in `rad/us`;
- finite detuning in `rad/us`;
- finite phase in radians;
- finite optional post phase shift in radians.

Output:

- `Pulse`;
- `pulse.amplitude.kind == "constant"`;
- `pulse.detuning.kind == "constant"`;
- `pulse.duration_ns == 1000`;
- phase values are stored but not compiled into any backend in Stage 1.

### Pulse From Waveforms

```python
pulse = Pulse(
    amplitude=Waveform.constant(1000, 1.0),
    detuning=Waveform.constant(1000, 0.0),
)
```

Input:

- amplitude `Waveform`;
- detuning `Waveform`;
- optional finite phase values;
- optional metadata.

Output:

- immutable `Pulse`.

Failure:

- mismatched amplitude and detuning durations raises `ValueError`;
- non-finite phase raises `ValueError`.

### Pulse Properties and Validation

```python
duration = pulse.duration_ns
issues = pulse.validate(channel)
```

Output:

- `duration_ns`: integer duration equal to `pulse.amplitude.duration_ns`;
- `validate(channel)`: `list[ValidationIssue]`.

`Pulse.validate(channel)` checks only channel duration, clock, amplitude, and detuning limits. It does not access a system or backend.

## Validation

`ValidationIssue` is the common validation result object shared by registers, devices, pulses, channels, and level structures.

```python
issue = ValidationIssue(
    "error",
    "register.min_distance",
    "minimum atom distance is violated",
    ("register", "coords"),
)
```

Input:

- severity `"error"` or `"warning"`;
- stable machine-readable code string;
- human-readable message string;
- optional tuple path.

Output:

- immutable validation issue.

Failure:

- invalid severity, empty code, non-string message, or non-tuple path raises `ValueError`.

```python
raise_for_errors([issue])
```

Input:

- list of `ValidationIssue`.

Output:

- returns `None` when no issue has severity `"error"`;
- raises `ValueError` when at least one error issue exists;
- raised message includes each error code and message on separate lines.

## System Construction

Stage 1 connects the new register API to current system construction. It does not introduce sequence simulation.

```python
system = RydbergSystem.from_lattice(Register.chain(2, 4.0), "01r")
```

Input:

- `Register` as the first argument;
- level structure name or `LevelStructureSpec`;
- optional `InteractionSpec`;
- optional protocol;
- optional physical parameters accepted by the existing system factory.

Output:

- `RydbergSystem`;
- `system.geometry` is the same `Register`-type geometry object;
- no sequence is created;
- no simulation is run;
- no backend contract changes.

Failure:

- passing `LatticeGeometry` as a distinct public class is not supported after Stage 1 because that class no longer exists as the public geometry type.
