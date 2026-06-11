# Stage 1 User API

This page is the normative Stage 1 public API contract for `ryd_gate`. Stage 1 builds the product-facing data layer for neutral-atom simulation: registers, level structures (atom models), devices, channels, waveforms, pulses, validation, drawing, and serialization. The engineering companion (current-code refactor steps, file ownership, call-site migration, tests, acceptance) is [stageplans/StagePlan_01_API_Foundation.md](../stageplans/StagePlan_01_API_Foundation.md).

Object names and semantics are deliberately familiar to Pulser users (`Register`, `RegisterLayout`, `Pulse`, waveform kinds, device-as-validator), but every object *is* a refactored kernel object of this repo (`RydbergSystem` → `HamiltonianIR` → backends) — not a wrapper around it. Where Stage 1 documents a class, that class is the single implementation; there is no parallel internal class behind it.

Stage 1 keeps physical behavior stable while replacing old public API shapes with product objects. Stage 1 does not create `Sequence`, does not support `simulate(seq, ...)`, and does not change backend contracts — those arrive in Stages 2/3 (see `stageplans/README.md` for the roadmap).

Every callable below is specified with **Purpose** (what it is used for and why it must exist), **Input**, **Output**, and **Failure** where applicable.

## Shared Conventions

Units:

- register coordinates are in micrometers (µm);
- waveform and pulse durations are positive integers in nanoseconds (ns);
- waveform values (amplitude, detuning) are angular frequencies in `rad/us`;
- `Waveform.value_at_s(...)` returns `rad/s` (the unit the kernel solvers use);
- phases are in radians;
- interaction coefficients are keyed strings, e.g. `"C6_rad_s_um6"` is C6 in `rad/s · µm^6`.

Behavior:

- all Stage 1 objects are immutable (frozen dataclasses);
- validation methods return `list[ValidationIssue]` and do not raise (so callers can collect all problems at once);
- constructors raise `ValueError` for structurally invalid input (an object that exists is always well-formed);
- every Stage 1 object serializes via `to_dict()` / `from_dict(...)` with a schema tag (see [Serialization](#serialization));
- importing Stage 1 names must not initialize a backend, allocate a Hamiltonian, compile a sequence, or run a simulation;
- `draw(...)` methods import matplotlib lazily inside the method body;
- the public API is what this document (and later stage documents) specify; undocumented module internals (e.g. `ryd_gate.pulse.blackman_*`) are unsupported in user code and may change without notice.

## Quick Example

```python
import numpy as np
from ryd_gate import (
    DeviceSpec, Pulse, Register, RydbergSystem, Waveform,
    level_structure, raise_for_errors,
)

# 1. Atom geometry (coordinates in um)
reg = Register.square(2, spacing_um=5.0)
reg.draw(blockade_radius_um=8.0, show=False)      # matplotlib Figure

# 2. Hardware constraints
device = DeviceSpec.virtual_rb87()
raise_for_errors(device.validate_register(reg))
print(device.describe())
r_b = device.rydberg_blockade_radius_um(rabi_rad_per_us=2 * np.pi)

# 3. Atom model (level structure)
spec = level_structure("01r")

# 4. Laser control data — product pulses, for device validation and (Stage 2) sequences.
#    CZ-gate envelopes are NOT built this way; they use the Protocol API
#    (see "Choosing a Control Surface").
pulse = Pulse(
    amplitude=Waveform.blackman(1000, area=np.pi),   # pi-area window
    detuning=Waveform.constant(1000, 0.0),
)
raise_for_errors(device.validate_pulse(pulse, "rydberg_global"))

# 5. Existing kernel entry point (unchanged)
system = RydbergSystem.from_lattice(reg, spec)

# 6. Every object round-trips through plain JSON-compatible dicts
reg2 = Register.from_dict(reg.to_dict())
pulse2 = Pulse.from_dict(pulse.to_dict())

# Stage 2 preview (NOT part of Stage 1):
#   seq = Sequence(reg, device)
#   seq.declare_channel("ryd", "rydberg_global")
#   seq.add(pulse, "ryd")
#   result = simulate_sequence(seq, backend="exact")
```

## Choosing a Control Surface

`ryd_gate` has two pulse-control surfaces over one kernel (`RydbergSystem → HamiltonianIR → backends`). Pick by task:

| You want to | Use | Status |
|---|---|---|
| Build laser pulses by hand, validate them against a device, serialize/exchange them | `Waveform` / `Pulse` (this document); they enter `Sequence` in Stage 2 | Stage 1 |
| Optimize or simulate CZ gates (time-optimal, amplitude-robust, double-ARP) | `RydbergSystem.from_lattice(..., protocol=TOProtocol())` — the continuous-time `Protocol` API | existing kernel; productized in Stage 5 |
| Continuous flat-top envelopes (rise–hold–fall) | the Protocol API (gate protocols build these internally) — **not** `Waveform.blackman` | kernel-internal; `CompositeWaveform` reserved for a later stage |

Principle: protocols are the continuous-time surface for physics and optimization; waveforms are the discrete, hardware-shaped exchange format. Work that starts on the protocol side and must become a shareable schedule gets re-expressed as `Waveform`s and re-validated against the device (`device.validate_pulse`, and `Sequence.add` from Stage 2).

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
    InteractionSpec,
    RydbergSystem,
)
```

Purpose:

- the top-level namespace is the product surface (mirroring `pulser.Register`, `pulser.Pulse`, …); users should never need to know internal module paths for everyday work.

Input:

- no runtime input.

Output:

- imported class and function objects;
- importing these names must not initialize a backend, allocate a Hamiltonian, compile a sequence, or run a simulation.

Domain imports (same objects, explicit module paths for library code that wants narrow imports):

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
from ryd_gate.lattice import make_chain
from ryd_gate.lattice import make_square_lattice
from ryd_gate.lattice import make_triangular_lattice
from ryd_gate.lattice import make_geometry_from_coords
from ryd_gate import blackman_window
from ryd_gate import blackman_pulse
from ryd_gate import blackman_pulse_sqrt
```

These imports must fail after Stage 1. The geometry factories become `Register` classmethods.

The public pulse API is exactly `Waveform` and `Pulse` (top-level, or `from ryd_gate.pulse import Waveform, Pulse`). The continuous-time helpers `blackman_window`, `blackman_pulse`, and `blackman_pulse_sqrt` still exist inside `ryd_gate.pulse` for the kernel's own use, but they are excluded from `ryd_gate.pulse.__all__` and are **not part of the Stage 1 contract**: do not import them in user code — they are unsupported and may change without notice.

## Registers

`Register` is the stable atom register API. It owns atom ids, coordinates, stable order, lattice spacing metadata, sublattice metadata, and geometry-only helper methods. The stable order of `ids` defines the site order used by basis states, bitstrings, observables, and (in later stages) sampling output — this ordering promise is part of the public contract.

`Register` is the direct refactor of the internal `LatticeGeometry` dataclass (same `N`, `coords`, `sublattice`, `spacing_um` fields the kernel already consumes), extended with `ids`, validation, drawing, and serialization. It is **not** a wrapper: after Stage 1, `RydbergSystem.from_lattice` and the TN lattice spec consume `Register` itself.

Construction goes through the classmethods (`chain`, `square`, `rectangle`, `triangular`, `from_coordinates`) — they are the primary Stage 1 path. The `layout` field is optional provenance metadata (see RegisterLayout directly below); creating a `RegisterLayout` first is never a required step.

### RegisterLayout

```python
layout = RegisterLayout(
    name="square_2x2",
    trap_coords_um=((0.0, 0.0), (0.0, 5.0), (5.0, 0.0), (5.0, 5.0)),
    kind="square",
)
```

Purpose:

- records which trap pattern a register was built from, so registers stay self-describing through serialization and (later) Pulser-layout interop;
- necessary because hardware-faithful workflows distinguish the *trap layout* (what the tweezers can do) from the *register* (which traps are filled) — Pulser encodes the same distinction, and adding it later would break the serialization schema.

Input:

- `name`: non-empty string;
- `trap_coords_um`: tuple of 2D or 3D coordinate tuples in micrometers;
- `kind`: one of `"chain"`, `"square"`, `"rectangle"`, `"triangular"`, `"custom"`;
- `metadata`: optional mapping.

Output:

- immutable `RegisterLayout`;
- no atom ids, no level structure, no device, no backend information.

Failure:

- empty `name`, empty or dimension-mixed `trap_coords_um`, or invalid `kind` raises `ValueError`.

Relationship to `Register` construction (Stage 1 rules):

- the classmethod constructors are the primary path and always return registers with `layout is None` — they never synthesize a layout;
- a layout is attached only explicitly, through the direct constructor or `dataclasses.replace`:

```python
reg = Register.square(2, spacing_um=5.0)
assert reg.layout is None                                    # classmethods never attach layouts

import dataclasses
reg_with_layout = dataclasses.replace(reg, layout=layout)    # explicit provenance
```

- `from_dict` restores whatever layout the stored dict carries;
- Stage 1 performs **no consistency validation** between an attached layout and the register's coordinates — subset/tolerance matching is exactly the machinery the future mapping API will own; until then an attached layout is trusted metadata.

Future (not Stage 1): `layout.define_register(trap_ids, qubit_ids) -> Register` for sparsely filling a trap sheet (e.g. 4 atoms in a 10×10 layout), arriving with Pulser layout import in the Stage 6 roadmap. That is an *additional* path for trap-sheet workflows — classmethod-direct registers remain first-class permanently.

### Direct Register Construction

```python
reg = Register(
    N=2,
    coords=[[0.0, 0.0], [4.0, 0.0]],
    sublattice=[1, -1],
    spacing_um=4.0,
    ids=("q0", "q1"),
)
```

Purpose:

- the low-level constructor for fully explicit registers (e.g. loaded from files, produced by optimizers);
- necessary because every classmethod below must normalize through one validated entry point, and because the kernel (`build_from_lattice`, TN lattice spec) reads exactly these fields.

Input:

- `N`: positive integer atom count;
- `coords`: finite float array-like of shape `(N, 2)` or `(N, 3)`;
- `sublattice`: array-like of shape `(N,)` (±1 signs for staggered observables, or zeros when unused);
- `spacing_um`: finite nonnegative float;
- `ids`: tuple or list of `N` unique non-empty strings; optional — when omitted (`None`), ids are generated as `("q0", ..., f"q{N-1}")`;
- optional `layout` and `metadata`.

Output:

- immutable `Register`;
- normalized `coords` stored as `np.ndarray(dtype=float)` (input arrays are copied, never mutated);
- normalized `sublattice` stored as `np.ndarray`;
- normalized `ids` stored as `tuple[str, ...]`;
- stable atom order exactly follows the order in `ids` and `coords`.

Failure:

- invalid `N`, invalid shape, non-finite coordinates, duplicate id, empty id, mismatched id count, or invalid spacing raises `ValueError`.

The direct constructor is the low-level form; the classmethods below are the product construction path.

### Chain

```python
reg = Register.chain(3, spacing_um=4.0)
```

Purpose:

- builds a 1D chain along x — the default geometry for two-atom gate studies, 1D sweeps, and MPS benchmarks;
- necessary as the official replacement for the removed `make_chain` factory; it must reproduce `make_chain`'s coordinates and sublattice signs exactly so existing physics (staggered magnetization, basis site order) is unchanged.

Input:

- `n_atoms=3`: positive integer;
- `spacing_um`: positive float, default `4.0` (the old `make_chain` default);
- optional `prefix="q"`.

Output:

- `Register` with `N == 3`;
- `ids == ("q0", "q1", "q2")`;
- `coords_um == ((0.0, 0.0), (4.0, 0.0), (8.0, 0.0))`;
- `sublattice == np.array([1, -1, 1])` (alternating `(-1)**i`, preserving the existing staggered-observable convention);
- `layout is None`;
- coordinates are not centered.

Failure:

- nonpositive `n_atoms` or `spacing_um`, or empty `prefix`, raises `ValueError`.

### Square

```python
reg = Register.square(2, spacing_um=5.0, prefix="a")
```

Purpose:

- builds a side×side square lattice, the standard 2D geometry for blockade/AF-phase studies and PEPS benchmarks;
- necessary as the ergonomic special case of `rectangle` (mirrors `pulser.Register.square`); kept separate so user code reads as intent.

Input:

- `side=2`: positive integer;
- `spacing_um`: positive float, default `4.0`;
- `prefix="a"`.

Output:

- `Register` with `N == side**2`, identical to `Register.rectangle(side, side, spacing_um, prefix)`;
- `ids == ("a0", "a1", "a2", "a3")`;
- row-major coordinates `((0.0, 0.0), (0.0, 5.0), (5.0, 0.0), (5.0, 5.0))`;
- `sublattice == np.array([1, -1, -1, 1])` (checkerboard `(-1)**(row + col)`);
- `layout is None`;
- coordinates are not centered.

Failure:

- as `rectangle`.

### Rectangle

```python
reg = Register.rectangle(2, 3, spacing_um=5.0)
```

Purpose:

- builds a rows×cols rectangular lattice, the workhorse 2D geometry (TN quench benchmarks, domain dynamics);
- necessary as the official replacement for the removed `make_square_lattice(Lx, Ly, s)`; it must reproduce that function's atom order and coordinates exactly so basis-state site order is unchanged across the migration (`rows ≡ Lx`, `cols ≡ Ly`).

Input:

- `rows=2`: positive integer;
- `cols=3`: positive integer;
- `spacing_um`: positive float, default `4.0`;
- optional `prefix="q"`.

Output:

- `Register` with `N == 6`;
- row-major atom order with index `i = row * cols + col`;
- coordinates `(row * spacing_um, col * spacing_um)`, i.e. `((0.0, 0.0), (0.0, 5.0), (0.0, 10.0), (5.0, 0.0), (5.0, 5.0), (5.0, 10.0))`;
- ids `("q0", "q1", "q2", "q3", "q4", "q5")`;
- `sublattice` is checkerboard `(-1)**(row + col)`, i.e. `(1, -1, 1, -1, 1, -1)`;
- `layout is None`;
- coordinates are not centered.

Failure:

- nonpositive `rows`, `cols`, or `spacing_um`, or empty `prefix`, raises `ValueError`.

### Triangular

```python
reg = Register.triangular(2, 3, spacing_um=4.0)
```

Purpose:

- builds a row-staggered triangular lattice for frustrated-geometry studies;
- necessary because the repo already ships this geometry (`make_triangular_lattice`) and Pulser exposes `triangular_lattice`; dropping it in the migration would be a capability regression.

Input:

- `rows=2`: positive integer;
- `atoms_per_row=3`: positive integer;
- `spacing_um`: positive float, default `4.0`;
- optional `prefix="q"`.

Output:

- `Register` with `N == rows * atoms_per_row`;
- row-major atom order with index `i = row * atoms_per_row + col`;
- coordinates `(col * spacing_um + offset, row * spacing_um * sqrt(3) / 2)` with `offset = spacing_um / 2` on odd rows and `0.0` on even rows;
- `sublattice` is all zeros (matching the removed `make_triangular_lattice`);
- `layout is None`;
- coordinates are not centered.

`Register.triangular(rows, atoms_per_row, s)` reproduces the removed `make_triangular_lattice(Lx=atoms_per_row, Ly=rows, s)` exactly.

Failure:

- nonpositive sizes or spacing, or empty `prefix`, raises `ValueError`.

### From Coordinates

```python
reg = Register.from_coordinates(
    [(0.0, 0.0), (4.0, 0.0)],
    ids=("left", "right"),
    center=False,
)
```

Purpose:

- builds a register from arbitrary positions — irregular arrays, experimentally measured positions, optimizer output;
- necessary as the official replacement for the removed `make_geometry_from_coords`, with the same spacing-inference rule, plus id assignment and optional centering (matching `pulser.Register.from_coordinates` semantics).

Input:

- `coords`: finite array-like of 2D or 3D coordinates;
- optional `ids`: unique non-empty strings matching coordinate length;
- optional `prefix="q"` used only when `ids is None`;
- `center`: boolean, default `True`;
- optional `sublattice`: array-like of shape `(N,)`.

Output:

- `Register` with stable atom order equal to coordinate order;
- if `center=False`, stored coordinates equal input coordinates converted to float;
- if `center=True`, mean coordinate vector is subtracted before storage;
- if `ids is None`, ids are generated as `f"{prefix}{i}"`;
- if `sublattice is None`, `sublattice` is all zeros;
- `spacing_um` is inferred as the smallest positive difference between sorted x-coordinates (the existing `make_geometry_from_coords` rule), `0.0` for a single atom;
- `layout is None`.

Failure:

- empty coordinate list, mixed dimensions, duplicate ids, empty prefix when ids are generated, or mismatched sublattice length raises `ValueError`.

### Register Properties

```python
n = reg.n_atoms
dim = reg.dimensions
coords = reg.coords_array
coords_um = reg.coords_um
```

Purpose:

- `n_atoms`: the product-facing atom count (alias of the kernel field `N`), so user code does not depend on internal field names;
- `dimensions`: 2D/3D discrimination needed by device validation and drawing;
- `coords_array`: a safe numpy copy for numerical work — necessary so callers can compute without risk of mutating the frozen register;
- `coords_um`: a plain-tuple view — necessary for display, hashing-free comparisons, and the serialization payload.

Output:

- `n_atoms`: integer equal to `reg.N`;
- `dimensions`: integer `2` or `3`, equal to `reg.coords.shape[1]`;
- `coords_array`: new `np.ndarray` copy of shape `(N, dimensions)`;
- `coords_um`: tuple-of-tuples coordinate representation.

Mutating `coords_array` must not mutate the register.

### Register Indexing

```python
i = reg.index("q1")
atom_id = reg.id_at(1)
```

Purpose:

- translate between user-facing atom ids and the integer site indices the kernel uses (basis order, observable names like `n_r_3`);
- necessary because every later feature that names atoms (local targeting, per-site populations, bitstrings) must agree on one id↔index mapping, and that mapping must be queryable.

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

Purpose:

- `distances_um`: the pair-distance matrix needed for interaction strength estimates and device min-distance checks;
- `distance_pairs`: the upper-triangular `(i, j, d)` list in exactly the shape the interaction lowering (`vdw_couplings`) consumes — necessary so geometry questions are answered by `Register`, not re-derived ad hoc in analysis scripts;
- `blockade_edges`: the blockade graph at a given radius, used for connectivity reasoning and drawing.

Input:

- optional `cutoff_um`: `None` or finite nonnegative float;
- `radius_um`: finite nonnegative float.

Output:

- `distances_um()`: symmetric `(N, N)` distance matrix in micrometers with zero diagonal;
- `distance_pairs(...)`: tuple of upper-triangular `(i, j, distance_um)` pairs with `i < j`; pairs with distance greater than `cutoff_um` are omitted when a cutoff is given;
- `blockade_edges(...)`: tuple of upper-triangular `(i, j)` pairs with distance `<= radius_um`.

Failure:

- negative or non-finite `cutoff_um` / `radius_um` raises `ValueError`.

### Drawing

```python
fig = reg.draw(blockade_radius_um=None, show_ids=True, show=False)
```

Purpose:

- visual inspection of the array before running anything — atom placement, ids, and blockade connectivity at a chosen radius;
- necessary because "draw the register" is table-stakes UX in every competing package (Pulser, Bloqade), and because mislabeled geometry is the cheapest class of bug to catch visually.

Input:

- optional `blockade_radius_um`: `None` or positive float;
- `show_ids`: boolean, default `True`;
- `show`: boolean, default `True`.

Output:

- a `matplotlib.figure.Figure` with one marker per atom at its coordinates;
- atom ids annotated when `show_ids=True`;
- when `blockade_radius_um` is set: a line for every blockade edge (pairs within the radius) and a dashed circle of radius `blockade_radius_um / 2` around each atom;
- `show=True` additionally calls `matplotlib.pyplot.show()`;
- matplotlib is imported inside the method, never at module import time.

Failure:

- 3D registers raise `NotImplementedError` in Stage 1 (2D drawing only).

### Register Validation

```python
issues = reg.validate(device)
```

Purpose:

- convenience pivot so fluent code can ask "is this register valid on this device" from either object;
- necessary to keep the validation *rules* in exactly one place (`DeviceSpec`) — this method is pure delegation, never a second rule set.

Input:

- `device`: object implementing `validate_register(register)`.

Output:

- exactly `device.validate_register(reg)`;
- no exception unless the device object itself raises.

### Register Serialization

```python
data = reg.to_dict()
reg2 = Register.from_dict(data)
```

Purpose:

- persist and exchange registers (job specs, result provenance, future Pulser import/export);
- necessary because a product simulator must reproduce runs from stored configs; arrays and ids must round-trip exactly.

Output:

- `to_dict()`: JSON-compatible dict with `"schema": "ryd-gate/register/v1"`, coordinates as nested lists, sublattice as a list, plus `ids`, `spacing_um`, optional `layout` (nested `RegisterLayout` dict) and `metadata`;
- `from_dict(...)`: a `Register` whose `ids`, `coords_um`, `sublattice` entries, `spacing_um`, `layout`, and `metadata` equal the original's (field-level round-trip; `Register` objects are not compared with `==` because of array fields).

Failure:

- see the shared rules in [Serialization](#serialization).

`RegisterLayout` serializes the same way with `"schema": "ryd-gate/register-layout/v1"`.

## Level Structures

`LevelStructureSpec` is the Stage 1 atom-model API. It describes local levels, Rydberg levels, transition channels, detuning channels, initial level, species, interaction kind, and physical parameters. There is no separate `AtomModel` class: the existing compiler-facing level structure in `core/level_structures.py` is extended in place to be the user-facing model object. This avoids a duplicate hierarchy whose two halves would have to be kept in sync forever.

### Presets

```python
spec = level_structure("01r")
```

Purpose:

- one-call access to the validated, named atom models the kernel supports, with the right channels and parameter sets pre-wired;
- necessary because the level structure decides which Hamiltonian blocks exist and which backends apply — users should select models by name, not assemble them.

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

Preset semantics:

- `"01"`: two-level computational model (no Rydberg level), reserved for circuit/stabilizer-style backends;
- `"1r"`: two-level global Rydberg model (`|1> <-> |r>`), the analog many-body workhorse;
- `"01r"`: three-level model (`|0>` spectator, `|1> <-> |r>`, optional `|0> <-> |1>` hyperfine drive);
- `"ger"`: **symbolic** three-level ladder (`|g> <-> |e> <-> |r>`): only abstract transition blocks are registered, every drive coefficient (including the `e <-> r` coupling) is supplied by the bound protocol at run time, and no static `H_1013` term exists — full protocol control for research schedules;
- `"analog_3"`: the **physical** Rb87 ladder (previously spelled `level_structure="ger", param_set="analog_3"`): compiles through the analog block builder with preset detunings, a static `e <-> r` (`H_1013`) coupling, and the analog default interaction — same topology and channel names as `ger`, different Hamiltonian construction;
- `"rb87_7"`: full seven-level Rb87 precision model (hyperfine + intermediate + Rydberg + garbage levels).

Choosing between `ger`, `analog_3`, and `rb87_7`:

- you want full protocol control of every coefficient (custom sweeps, research schedules) → `"ger"`;
- you want a realistic Rb87 three-level analog simulation with physical numbers → `"analog_3"`;
- you want seven-level precision (intermediate-state scattering, garbage levels) → `"rb87_7"`, choosing the numerical set with `param_set="our"` or `"lukin"`.

Naming principle: a preset *name* changes when the Hamiltonian construction semantics change; `param_set` only switches numerical sets within identical semantics. That is why `our`/`lukin` are tags on `rb87_7` (same construction, different numbers) while `analog_3` is its own name rather than a tag on `ger` (different construction: symbolic vs physical) — and why forgetting a `param_set` can never silently switch you between a symbolic and a physical model.

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

Purpose:

- lets advanced users define atom models beyond the presets without forking the package;
- necessary because this repo's differentiator is multilevel-atom realism — the model layer must be open, not an enum.

Input:

- `name`: non-empty model name;
- `levels`: tuple of unique non-empty level labels;
- `rydberg_levels`: tuple whose entries must exist in `levels`;
- optional `transitions`;
- optional `detuning_levels`;
- optional `initial_level`;
- optional `species` (default `"Rb87"`);
- optional `interaction_kind`: one of `"none"`, `"ising_c6"`, `"xy_c3"`, `"custom"`;
- optional `params`.

Output:

- immutable `LevelStructureSpec`;
- no backend object and no Hamiltonian object.

Failure:

- structural violations raise `ValueError`; *semantic* problems are reported by `validate()` below (so a spec can be built, inspected, and fixed interactively).

### Level-Structure Methods

```python
level = spec.initial_level_or_default()
kwargs = spec.physical_kwargs()
ok = spec.supports_backend("mps")
issues = spec.validate()
```

Purpose and necessity, per method:

- `initial_level_or_default()` — the level each atom starts in when the user does not specify a state; necessary so Stage 2's default `psi0` and the GUI agree on one deterministic rule (`initial_level` if set, else `levels[0]`);
- `physical_kwargs()` — the exact keyword arguments the existing system factory (`build_from_lattice` → `local_blocks`) needs for this model (e.g. `param_set`); necessary so callers never hand-thread parameter-set strings that must match the preset;
- `supports_backend(name)` — capability flag per backend; necessary so `Sequence`/`simulate_sequence` (Stage 2/3) can reject unsupported model–backend pairs at the API boundary instead of deep inside a TN compiler;
- `validate()` — semantic self-check returning issues; necessary for custom specs built from user input or deserialized files.

Output:

- `initial_level_or_default()`: `spec.initial_level` if set, otherwise `spec.levels[0]`;
- `physical_kwargs()`: physical factory kwargs needed by current system construction (`{"param_set": "analog_3", ...}` for `analog_3`; `{"param_set": params.get("param_set", "our"), ...}` for `rb87_7`; `{}` otherwise);
- `supports_backend(...)`: boolean support flag;
- `validate()`: `list[ValidationIssue]`.

Backend support matrix (Stage 1 fixed truth table):

- `"exact"` supports `"01"`, `"1r"`, `"01r"`, `"ger"`, `"analog_3"`, and `"rb87_7"`;
- `"mps"`, `"gputn"`, and `"peps"` support only `"1r"` and `"01r"`;
- `"stabilizer"` supports only `"01"`;
- unknown backend names return `False`.

Validation errors cover empty levels, duplicate levels, unknown initial level, Rydberg levels not present in levels, invalid detuning target level, and invalid interaction kind.

### Level-Structure Serialization

```python
data = spec.to_dict()
spec2 = LevelStructureSpec.from_dict(data)
```

Purpose:

- persist custom atom models alongside registers and pulses so a job file is complete;
- necessary for reproducibility and the Stage 2 `Sequence` serialization, which embeds the level structure by value.

Output:

- `to_dict()`: JSON-compatible dict with `"schema": "ryd-gate/level-structure/v1"`; nested `TransitionSpec` entries serialize as dicts of their dataclass fields;
- `from_dict(...)`: an equal `LevelStructureSpec` (`spec2 == spec`).

## Channels

`ChannelSpec` describes a physical control channel and its constraints. It does not compile pulses or sample waveforms. Internally, the compiler keeps using the existing string channel ids (`global_X`, `drive_R`, `drive_420`, …); `ChannelSpec` carries the mapping from product channel to those compiler ids.

```python
channel = ChannelSpec(
    channel_id="rydberg_global",
    kind="rydberg",
    transition="1_r",
    addressing="global",
)
```

Purpose:

- types the previously implicit string-convention channels with hardware-style constraints (amplitude/detuning/duration/clock/targets), so pulses can be validated against a channel before any simulation;
- necessary because Stage 2's `Sequence.declare_channel`/`add` and `DeviceSpec.validate_pulse` need one constraint object per channel — without it, limits would be scattered through protocol code, which is exactly the research-script shape this refactor removes.

Input:

- `channel_id`: non-empty string;
- `kind`: one of `"rydberg"`, `"raman"`, `"microwave"`, `"dmm"`, `"custom"`;
- `transition`: non-empty transition label (e.g. `"1_r"`, `"0_1"`);
- `addressing`: `"global"` or `"local"`;
- optional `amplitude_channels` / `detuning_channels`: mappings from level-structure name to internal compiler channel id (e.g. `{"1r": "global_X", "01r": "drive_R"}`) — this is the lowering contract Stage 2 consumes;
- optional limits: `max_abs_amplitude_rad_per_us`, `max_abs_detuning_rad_per_us`, `min_duration_ns`, `max_duration_ns`, `clock_period_ns`, `max_targets`, `retarget_time_ns`.

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

Serialization: `to_dict()` with `"schema": "ryd-gate/channel/v1"`; `ChannelSpec.from_dict(channel.to_dict()) == channel`.

## Devices

`DeviceSpec` describes hardware and physical constraints. It is a frozen constraint dataclass — the device *is* the validator (the same pattern that makes Pulser devices trustworthy). It is not a backend, not a QPU job, and holds no state.

### Virtual Rb87 Device

```python
device = DeviceSpec.virtual_rb87()
```

Purpose:

- the default permissive device for this repo's Rb87-based models: real species, real C6 default, named channels, but no artificial atom-count or radius caps;
- necessary so every example and test has one canonical device to validate against, and so Stage 2 sequences always carry a device without users hand-building one.

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
- `max_atom_num is None`; `max_radial_distance_um is None`;
- `interaction_coeffs == {"C6_rad_s_um6": DEFAULT_C6}` (the existing `core.level_structures.DEFAULT_C6`);
- `channels` contains `"rydberg_global"`, `"rydberg_local"`, and `"hyperfine_global"` with the compiler-channel maps fixed in StagePlan_01.

### Register Validation

```python
issues = device.validate_register(Register.chain(2, 4.0))
```

Purpose:

- answers "can this device hold this geometry" before any Hamiltonian exists: dimensionality, atom count, pair distances, radial extent;
- necessary because geometry errors found at backend depth are expensive and cryptic; the device boundary is the right place to fail, and later stages (`Sequence.__init__`) call exactly this method.

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

Purpose:

- answers "does this device support this atom model" (allowed list + species match);
- necessary so model selection errors surface at sequence construction, not inside `local_blocks` matrix assembly.

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

Purpose:

- checks one pulse against one named channel's constraints (existence, duration window, clock divisibility, amplitude/detuning ceilings);
- necessary as the single enforcement point Stage 2's `Sequence.add` calls — hardware realism lives here, not in waveform code.

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

### Physics Helpers

```python
r_b = device.rydberg_blockade_radius_um(rabi_rad_per_us=2 * np.pi)
text = device.describe()
```

Purpose:

- `rydberg_blockade_radius_um` — converts the device's C6 and a drive amplitude into the blockade radius `(C6/Ω)^(1/6)`; necessary because choosing lattice spacing from the blockade radius is the first step of every analog workflow (Pulser exposes the same helper, and without it users hand-copy the formula with unit mistakes);
- `describe()` — one human-readable constraint sheet (name, species, limits, channels); necessary for notebooks/GUI and for support: "print your device.describe()" beats screenshots of dataclasses.

Input:

- `rabi_rad_per_us`: positive finite float.

Output:

- `rydberg_blockade_radius_um(...)`: `((interaction_coeffs["C6_rad_s_um6"] * 1e-6) / rabi_rad_per_us) ** (1 / 6)` in micrometers;
- `describe()`: multi-line string including at least the device name, dimensions, species, every constraint that is set, and one line per channel id.

Failure:

- `rydberg_blockade_radius_um` raises `ValueError` when `"C6_rad_s_um6"` is absent from `interaction_coeffs` or `rabi_rad_per_us` is not positive;
- `describe()` never raises.

### Device Serialization

```python
data = device.to_dict()
device2 = DeviceSpec.from_dict(data)
```

Purpose:

- ship device definitions as data (files, future cloud profiles) rather than code;
- necessary so third parties can define their hardware constraints without subclassing anything — the Pulser lesson worth copying.

Output:

- `to_dict()`: JSON-compatible dict with `"schema": "ryd-gate/device/v1"`, channels serialized as nested `ChannelSpec` dicts;
- `from_dict(...)`: an equal `DeviceSpec` (`device2 == device`).

## Waveforms

`Waveform` describes a scalar time-dependent control value. Values are public-facing `rad/us`. A waveform is a kind-tagged immutable dataclass (not a class hierarchy), which keeps serialization, equality, and validation uniform — one class, five constructors.

`Waveform` is the only supported way to express pulse shapes in user code: durations in integer ns, values in `rad/us`, serializable, and accepted by `Pulse`, `device.validate_pulse`, and (Stage 2) `Sequence`. The `blackman_*` kernel helpers in the same module are a different layer — continuous-time functions in seconds that build the gate protocols' flat-top envelopes internally; they are not serializable, not validated, and unsupported in user code (see [Choosing a Control Surface](#choosing-a-control-surface)). The two layers share window mathematics but neither wraps the other.

### Constant

```python
wf = Waveform.constant(1000, 2.5)
```

Purpose:

- a fixed drive value for a fixed duration — the most common building block (flat detuning, square amplitude segments);
- necessary as the base case every schedule uses; equivalent of `pulser.ConstantWaveform`.

Input:

- `duration_ns=1000`: positive integer;
- `value=2.5`: finite float in `rad/us`.

Output:

- `Waveform` with `kind == "constant"`;
- `wf.value_at_ns(t) == 2.5` for any clamped `t`;
- `wf.sample(dt_ns=...)` returns all `2.5`.

Failure:

- nonpositive/non-integer duration or non-finite value raises `ValueError`.

### Ramp

```python
wf = Waveform.ramp(1000, start=0.0, stop=5.0)
```

Purpose:

- linear sweep between two values — the standard detuning sweep in adiabatic state preparation;
- necessary because rise–sweep–fall adiabatic schedules are the canonical analog workflow (Pulser's `RampWaveform`); without it users fake ramps with interpolation points.

Input:

- positive integer duration;
- finite `start` and `stop` values in `rad/us`.

Output:

- `Waveform` with `kind == "ramp"`;
- `wf.first_value() == 0.0`;
- `wf.last_value() == 5.0`;
- linear interpolation between start and stop.

Failure:

- non-finite `start`/`stop` raises `ValueError`.

### Blackman

```python
wf = Waveform.blackman(1000, peak=3.0)
wf = Waveform.blackman(1000, area=np.pi)
```

Purpose:

- a smooth, spectrally clean envelope with zero endpoints — the repo's standard adiabatic turn-on/off shape, now as a product object;
- necessary because (a) it is the bridge from the existing `pulse.py` Blackman math to the product API, and (b) the `area` form is how users request π/2π pulses directly (mirroring `pulser.BlackmanWaveform(duration, area)`), eliminating manual peak-from-area algebra.

Input:

- positive integer duration;
- exactly one of:
  - `peak`: finite float in `rad/us`, or
  - `area`: finite nonzero float in radians (the target pulse area).

Output:

- `Waveform` with `kind == "blackman"`;
- the pure Blackman window
  `value_at_ns(t) = peak * (0.42 - 0.5 * cos(2*pi*t/T) + 0.08 * cos(4*pi*t/T))` with `T = duration_ns` (the same formula as the kernel `blackman_window` with `t_rise = T/2`);
- endpoint values are exactly `0.0`; the value at `t = T/2` is exactly `peak`;
- when constructed with `area`, the peak is `peak = area / (0.42 * duration_us)` (closed-form window integral), so `abs(wf.integral_rad(1) - area) <= 1e-3 * abs(area)`;
- a negative `area` produces a sign-flipped window (negative peak).

Failure:

- both `peak` and `area` given, or neither, raises `ValueError`.

When to use it (and not):

- **use** `Waveform.blackman` for product pulses: π/2π-area excitation inside a `Pulse`, device validation, serialization, and (Stage 2) `Sequence` scheduling;
- do **not** use it to reproduce CZ-gate envelopes: this is a single pure window (`0 → peak → 0`), while the TO/AR gate protocols use a *flat-top* envelope (Blackman rise, constant hold, Blackman fall) built internally from the kernel helper `blackman_pulse` — same window mathematics, different shape. Gate work goes through the Protocol API (see [Choosing a Control Surface](#choosing-a-control-surface));
- do **not** call the kernel helpers (`blackman_pulse` et al.) in user code: continuous-time seconds, no validation, no serialization, unsupported. Flat-top shapes may become expressible in the product layer later via `CompositeWaveform` (reserved vocabulary).

### Interpolated

```python
wf = Waveform.interpolated(
    1000,
    times_ns=[0, 500, 1000],
    values=[0.0, 2.0, 0.0],
)
```

Purpose:

- a waveform defined by control points — the natural parameterization for pulse *optimization* (optimizers move the points);
- necessary as the serializable replacement for ad-hoc Python callables in optimization workflows; equivalent of `pulser.InterpolatedWaveform`, which is Pulser's optimization workhorse.

Input:

- positive integer duration;
- monotonically increasing `times_ns` beginning at `0` and ending at `duration_ns`;
- finite values with the same length as `times_ns`.

Output:

- `Waveform` with `kind == "interpolated"`;
- piecewise-linear value interpolation.

Failure:

- unsorted times, mismatched lengths, missing endpoints, or non-finite values raises `ValueError`.

(Smooth interpolators such as PCHIP are a candidate later extension; Stage 1 is linear only so behavior is deterministic.)

### Custom

```python
wf = Waveform.custom([0.0, 1.0, 0.0], dt_ns=10)
```

Purpose:

- an explicit sample array for shapes the named kinds cannot express (measured waveforms, imported data, AWG exports);
- necessary as the escape hatch that keeps the closed `kind` set sufficient — anything else is expressible here.

Input:

- finite sample values (at least two);
- positive integer `dt_ns`.

Output:

- `Waveform` with `kind == "custom"`;
- `samples == (0.0, 1.0, 0.0)`;
- `duration_ns == (len(samples) - 1) * dt_ns`;
- `value_at_ns` interpolates linearly between stored samples.

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

Purpose and necessity, per method:

- `value_at_ns(t_ns)` — point evaluation in product units; the primitive everything else (plots, sampling, Stage 2 coefficient lowering) is built on;
- `value_at_s(t_s)` — the same value in kernel units (`rad/s`, seconds); necessary because the existing solvers integrate in SI time and the conversion must live in exactly one place;
- `sample(dt_ns)` — the discrete trace for plotting, hashing, and export; necessary because serializable schedules need deterministic sampling, not repeated callable evaluation;
- `integral_rad(dt_ns)` — pulse area in radians; necessary because pulse area is the physically meaningful knob (π pulses, adiabaticity), and validation/tests assert on it;
- `first_value()` / `last_value()` — endpoint values; necessary for continuity checks between adjacent schedule segments (Stage 2 uses them for amplitude-continuity warnings between adjacent pulses).

Output:

- `value_at_ns(...)`: float value in `rad/us`, with time clamped into `[0, duration_ns]`;
- `value_at_s(...)`: float value in `rad/s`, equal to `value_at_ns(t_s * 1e9) * 1e6`;
- `sample(dt_ns)`: `np.ndarray` of values at `0, dt_ns, ...` including `duration_ns`;
- `integral_rad(dt_ns)`: trapezoidal integral of `sample(dt_ns)` over time in microseconds;
- `first_value()` and `last_value()`: endpoint values in `rad/us`.

Failure:

- `sample(dt_ns)` and `integral_rad(dt_ns)` raise `ValueError` when `dt_ns` is not a positive integer dividing `duration_ns`.

### Waveform Serialization

`to_dict()` with `"schema": "ryd-gate/waveform/v1"` storing `kind`, `duration_ns`, and the kind's parameters (or samples). `Waveform.from_dict(wf.to_dict()) == wf`.

Purpose: waveforms are the leaves of every stored schedule; their round-trip exactness is what makes sequence files reproducible.

## Pulses

`Pulse` combines amplitude, detuning, phase, and post phase shift — the physical description of one laser drive segment. It does not know about targets (that is `Sequence.add`'s job in Stage 2) and does not compile itself.

### Constant Pulse

```python
pulse = Pulse.constant(
    1000,
    amplitude=1.0,
    detuning=0.0,
    phase_rad=0.25,
)
```

Purpose:

- the square pulse: fixed Ω, fixed δ for a duration — the first pulse every user writes;
- necessary as the one-line constructor for the most common case (equivalent of `pulser.Pulse.ConstantPulse`).

Input:

- positive integer duration in nanoseconds;
- finite amplitude in `rad/us`;
- finite detuning in `rad/us`;
- finite phase in radians (default `0.0`);
- finite optional post phase shift in radians (default `0.0`).

Output:

- `Pulse`;
- `pulse.amplitude.kind == "constant"`;
- `pulse.detuning.kind == "constant"`;
- `pulse.duration_ns == 1000`;
- phase values are stored but not compiled into any backend in Stage 1.

Failure:

- as the `Pulse` constructor below.

### Constant-Amplitude / Constant-Detuning Pulses

```python
pulse = Pulse.constant_detuning(Waveform.blackman(1000, area=np.pi), detuning=0.0)
pulse = Pulse.constant_amplitude(1.0, Waveform.ramp(1000, -5.0, 5.0))
```

Purpose:

- the two halves of the canonical adiabatic schedule: shaped amplitude at fixed detuning (rise/fall) and fixed amplitude under a detuning sweep;
- necessary because these two constructions appear in essentially every analog program; naming them removes a `Waveform.constant` boilerplate line each time (mirrors `pulser.Pulse.ConstantDetuning` / `ConstantAmplitude`).

Input:

- `constant_detuning(amplitude: Waveform, detuning: float, phase_rad=0.0, post_phase_shift_rad=0.0)`;
- `constant_amplitude(amplitude: float, detuning: Waveform, phase_rad=0.0, post_phase_shift_rad=0.0)`.

Output:

- a `Pulse` whose scalar argument becomes a `Waveform.constant` matching the other waveform's duration.

Failure:

- as the `Pulse` constructor below.

### Pulse From Waveforms

```python
pulse = Pulse(
    amplitude=Waveform.constant(1000, 1.0),
    detuning=Waveform.constant(1000, 0.0),
)
```

Purpose:

- the general form: any amplitude waveform with any detuning waveform plus phase bookkeeping;
- necessary as the single validated combination point — duration agreement between amplitude and detuning is enforced here and nowhere else.

Input:

- amplitude `Waveform`;
- detuning `Waveform`;
- optional finite `phase_rad` and `post_phase_shift_rad`;
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

Purpose:

- `duration_ns` — the single source of truth for a pulse's length (delegates to the amplitude waveform), used by scheduling and validation;
- `validate(channel)` — pulse-vs-channel constraint check from the pulse side; necessary so a pulse can be checked against a `ChannelSpec` without constructing a device (pure delegation to the same limit rules `DeviceSpec.validate_pulse` uses — one rule set, two entry points).

Output:

- `duration_ns`: integer duration equal to `pulse.amplitude.duration_ns`;
- `validate(channel)`: `list[ValidationIssue]`.

`Pulse.validate(channel)` checks only channel duration, clock, amplitude, and detuning limits. It does not access a system or backend.

### Pulse Serialization

`to_dict()` with `"schema": "ryd-gate/pulse/v1"`, amplitude and detuning as nested waveform dicts. `Pulse.from_dict(pulse.to_dict()) == pulse`.

Purpose: pulses are the unit Stage 2 sequences store per operation; their round-trip is a precondition for sequence files.

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

Purpose:

- one shared vocabulary for "what is wrong": severity, stable machine-readable code, human message, and a path locating the offending field;
- necessary because without it every object invents its own error format, and downstream tools (GUI, CI checks, Stage 2 `Sequence.validate`) cannot aggregate or branch on errors reliably. Codes are API: tests assert on them.

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

Purpose:

- the explicit boundary that converts accumulated error issues into one exception;
- necessary because validation itself never raises (so all problems can be collected); scripts still need a one-liner to fail fast.

Input:

- list of `ValidationIssue`.

Output:

- returns `None` when no issue has severity `"error"` (warnings never raise);
- raises `ValueError` when at least one error issue exists;
- raised message includes each error code and message on separate lines.

## Serialization

Every Stage 1 public object implements the same plain-dict contract:

```python
data = obj.to_dict()
obj2 = type(obj).from_dict(data)
```

Purpose:

- reproducibility (store and replay exact configurations), interoperability (the future Pulser import/export and JSON Schema freeze build on these dicts), and provenance (results can embed the inputs that produced them);
- necessary from Stage 1 — not later — because serialization added after the fact always misses fields; making it a Stage 1 acceptance criterion forces every object to stay data-shaped.

Rules:

- `to_dict()` returns a JSON-compatible dict (`str`/`int`/`float`/`bool`/`None`/`list`/`dict` values only; numpy arrays become nested lists, numpy scalars become Python scalars);
- the dict contains a `"schema"` key with value `"ryd-gate/<kind>/v1"`, where `<kind>` is one of `register-layout`, `register`, `level-structure`, `channel`, `device`, `waveform`, `pulse`;
- `json.dumps(obj.to_dict())` must succeed; `to_dict()` raises `ValueError` when `metadata` contains values that are not JSON-compatible;
- `from_dict(data)` raises `ValueError` on a missing or mismatched schema tag, an unsupported version, or an invalid payload (payload validation reuses the constructors — no second validation path);
- round-trip: objects with only scalar/tuple fields compare equal with `==`; `Register` round-trips field-wise (`ids`, `coords_um`, `sublattice` entries, `spacing_um`, `layout`, `metadata`).

Stage 1 ships plain dicts and schema tags only; frozen JSON Schema files and optional validation are a later stage (see stageplans/README roadmap). The tag is included from day one so files written now remain loadable after the schema freeze.

## System Construction

Stage 1 connects the new register API to current system construction. It does not introduce sequence simulation.

```python
system = RydbergSystem.from_lattice(Register.chain(2, 4.0), "01r")
```

Purpose:

- the unchanged kernel entry point, now consuming the product `Register` directly;
- necessary to prove the Stage 1 objects are the real kernel inputs, not a parallel facade — `Register` *is* the geometry object, so there is no `to_geometry()` conversion and no second geometry class.

Input:

- `Register` as the first argument;
- level structure name or `LevelStructureSpec`;
- optional `InteractionSpec`;
- optional protocol;
- optional physical parameters accepted by the existing system factory.

Output:

- `RydbergSystem`;
- `system.geometry` is the provided `Register`;
- basis site order follows `register.ids` order;
- no sequence is created;
- no simulation is run;
- no backend contract changes.

Failure:

- passing `LatticeGeometry` is not supported after Stage 1 because that class no longer exists as the public geometry type.

The same entry point is also the gate-optimization front door (the second control surface). This is existing kernel behavior, shown here for orientation; the gate library is productized in Stage 5:

```python
from ryd_gate import TOProtocol

system = RydbergSystem.from_lattice(
    Register.chain(2, spacing_um=3.0), "rb87_7", protocol=TOProtocol(),
)
# Continuous-time Protocol path: gate parameters x drive simulate(system, x, ...).
# Pulse shapes here are protocol-internal (flat-top Blackman), not Waveform objects.
```

## Out of Scope for Stage 1

The following are reserved vocabulary for later stages and must not be implemented in Stage 1 (see `stageplans/README.md` for the roadmap):

- `Sequence`, `simulate_sequence`, `SimulationResult`, `Sequence.draw()` — Stage 2;
- backend-native state handles (`QuantumStateHandle`, `MPSStateHandle`) — Stage 3;
- `NoiseModel` — Stage 4;
- gate-library productization (`ryd_gate.gates`, error budgets as documented API) — Stage 5;
- frozen JSON Schemas and Pulser subset import/export — Stage 6;
- trap-layout→register mapping (`layout.define_register(...)` for sparsely filled trap sheets) — Stage 6, with Pulser layout import;
- hardware features without a kernel counterpart yet (EOM mode, DMM/detuning maps, SLM masks, XY/microwave basis, mappable registers) — only if and when the physics layer needs them.
