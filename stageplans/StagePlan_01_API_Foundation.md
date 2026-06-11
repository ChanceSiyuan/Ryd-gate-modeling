# Stage 1 Plan: API Foundation

## Normative Split

The user-facing API contract for Stage 1 (every callable's purpose, input, output, failure) lives in [docs/stage1_api.md](../docs/stage1_api.md). That document is normative for behavior. This stage plan is normative for engineering: which existing code is refactored into which API, in what order, with what verification. If the two documents disagree, fix the disagreement before implementing.

## Stage 1 Goal

Refactor the existing data layer into the product API **in place** — no `src/ryd_gate/api/` package, no wrapper classes, no parallel hierarchies:

- atom register API: refactor `lattice/geometry.py::LatticeGeometry` into `Register` (the same kernel object, productized);
- atom-model API: extend `core/level_structures.py::LevelStructureSpec` (no new `AtomModel` class);
- channel/device API: type the existing string-channel conventions with `ChannelSpec` (in `protocols/channels.py`) and add a new `devices.py` with `DeviceSpec`;
- waveform/pulse API: add `Waveform`/`Pulse` to the existing `pulse.py`, keeping the kernel Blackman helpers;
- validation primitive: new `core/validation.py`;
- serialization helpers: new `core/serialization.py`; every Stage 1 object gets `to_dict()`/`from_dict()`;
- top-level exports in `ryd_gate/__init__.py`.

Stage 1 does not run simulations from a sequence and does not modify backend contracts.

## No-Wrapper Rules

These are the anti-patterns this refactor exists to avoid. Violating any of them fails review:

1. `Register` **replaces** `LatticeGeometry` (rename + extend). Do not keep `LatticeGeometry`, do not alias it, do not implement `Register` as a class that *holds* a geometry. After Stage 1 there is exactly one geometry class in the repo.
2. No `Register.to_geometry()`: `RydbergSystem.from_lattice(...)` consumes `Register` directly because `Register` keeps the kernel field names (`N`, `coords`, `sublattice`, `spacing_um`).
3. No `AtomModel`: `LevelStructureSpec` is both the compiler-facing and the user-facing model object. One class, extended in place.
4. `ChannelSpec` does not replace the internal compiler channel ids (`global_X`, `drive_R`, `drive_420`, …). It *carries the mapping* to them. Protocol lowering code is untouched.
5. `Waveform`/`Pulse` do not wrap the kernel Blackman helpers; both layers live in `pulse.py` and share formulas. `blackman_window`, `blackman_pulse`, `blackman_pulse_sqrt` stay as module functions because `protocols/gate_cz_to.py:57` and `protocols/gate_cz_ar.py:62` import them — only the *top-level* re-exports are removed. The helpers are **soft-closed**, not public: excluded from `ryd_gate.pulse.__all__`, docstring-marked as internal kernel API with no stability guarantee, and declared unsupported for user code in docs/stage1_api.md.
6. Validation rules live once: device-level rules in `DeviceSpec`, channel-limit rules in one helper that both `DeviceSpec.validate_pulse` and `Pulse.validate` call. `Register.validate(device)` is a one-line delegation.
7. No compatibility aliases for removed names (`make_*`, `LatticeGeometry`, top-level `blackman_*`). Internal call sites are migrated, not shimmed.

## Implementation Order

Execute as six commit-sized steps. Each step ends with its verify command green before the next starts.

```text
Step 1  core/validation.py + core/serialization.py (new)        -> uv run pytest tests/core/test_validation.py -q
Step 2  geometry.py refactor + all geometry call sites           -> uv run pytest -m "not slow" -q   (full fast suite)
Step 3  level_structures.py extension + factories inference      -> uv run pytest tests/core -q
Step 4  channels.py ChannelSpec + pulse.py Waveform/Pulse        -> uv run pytest tests/core/test_pulse_api.py tests/core/test_blackman.py -q
Step 5  devices.py DeviceSpec                                    -> uv run pytest tests/core/test_devices.py -q
Step 6  __init__.py exports + serialization round-trip tests     -> full acceptance (below)
```

Step 2 is the only step that touches many files; it must land as one unit (the rename breaks imports until all call sites move).

## Allowed File Operations

Create exactly these files:

```text
src/ryd_gate/core/validation.py
src/ryd_gate/core/serialization.py
src/ryd_gate/devices.py
tests/lattice/__init__.py
tests/lattice/test_register.py
tests/core/test_level_structures_product_api.py
tests/core/test_validation.py
tests/core/test_devices.py
tests/core/test_pulse_api.py
tests/core/test_serialization_roundtrip.py
```

Modify these source files (the API refactor itself):

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

Modify these files **only** to migrate call sites of removed names (see Call-Site Migration table — no behavior changes beyond the mechanical replacement):

```text
src/ryd_gate/protocols/digital_analog.py        (docstring example only)
tests/core/test_init.py
tests/core/test_simulate_entry.py
tests/core/test_rydberg_system_model.py
tests/protocols/test_sweep_protocol.py
tests/protocols/test_digital_analog_site_profiles.py
tests/protocols/test_lattice_dynamics_protocols.py
tests/backends/test_tn_dmrg.py
tests/backends/test_tn_lattice_spec.py
tests/backends/test_tn_tdvp.py
tests/backends/test_plus_init.py
app/pages/2_lattice_simulator.py
app/pages/4_local_addressing.py
scripts/bench_quench_check.py
examples/demo_local_addressing.py
examples/* (any other example importing a removed name)
```

Do not modify:

```text
src/ryd_gate/simulate.py
src/ryd_gate/ir/*
src/ryd_gate/core/system_model.py
src/ryd_gate/core/rb87_params.py
src/ryd_gate/core/local_blocks.py
src/ryd_gate/core/basis.py
src/ryd_gate/core/blocks.py
src/ryd_gate/core/observables.py
src/ryd_gate/protocols/base.py
src/ryd_gate/protocols/sweep.py
src/ryd_gate/protocols/gate_cz_*.py
src/ryd_gate/backends/*  except backends/tn_common/lattice_spec.py
scripts/notebooks/*  and all *.ipynb
pyproject.toml
```

If implementation requires changing any forbidden file, stop and update this plan first.

## Call-Site Migration

Mechanical replacement rules. The right-hand forms are behavior-identical to the left-hand forms (same coordinates, same atom order, same sublattice signs):

```text
make_chain(N)                      -> Register.chain(N)                       # default spacing 4.0 preserved
make_chain(N, s)                   -> Register.chain(N, s)
make_square_lattice(Lx, Ly, s)     -> Register.rectangle(Lx, Ly, s)           # rows=Lx, cols=Ly
make_square_lattice(L, L, s)       -> Register.square(L, s)
make_triangular_lattice(Lx, Ly, s) -> Register.triangular(Ly, Lx, s)          # rows=Ly, atoms_per_row=Lx — note the swap
make_geometry_from_coords(c)       -> Register.from_coordinates(c, center=False)
make_geometry_from_coords(c, sub)  -> Register.from_coordinates(c, sublattice=sub, center=False)
LatticeGeometry (annotation/import)-> Register
from ryd_gate import blackman_*    -> from ryd_gate.pulse import blackman_*   # kernel helpers keep working there
"ger" + param_set="analog_3"       -> level_structure("analog_3")             # app/pages/4_local_addressing.py:53; notebooks keep the old spelling until Stage 7
```

`center=False` is mandatory when migrating `make_geometry_from_coords` call sites: the old function never centered, while `from_coordinates` defaults to `center=True` (Pulser-style). Forgetting this silently shifts coordinates.

`tests/core/test_init.py` additionally changes meaning: it currently asserts top-level availability of `blackman_pulse` etc.; after Stage 1 it must assert the new top-level export list and assert that removed names raise `ImportError`.

## Module Plan: `src/ryd_gate/core/validation.py` (new)

Current state: no shared validation primitive exists; constraints raise ad hoc `ValueError`s near where they are checked.

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

Implementation steps:

1. `ValidationIssue.__post_init__` rejects invalid severity, empty `code`, non-string `message`, non-tuple `path` (`ValueError`).
2. `raise_for_errors` returns `None` when no issue has `severity == "error"`; otherwise raises `ValueError` whose message contains each error's `code` and `message` on its own line.

Must not contain: register/pulse/device rules, backend imports, simulation imports.

## Module Plan: `src/ryd_gate/core/serialization.py` (new)

Current state: no serialization support anywhere; configs are reproduced by re-running scripts.

Implement exactly (small helpers; each class still owns its own `to_dict`/`from_dict`):

```python
SCHEMA_PREFIX = "ryd-gate"

def schema_tag(kind: str) -> str:
    """Return 'ryd-gate/<kind>/v1'."""

def check_schema(data, kind: str) -> None:
    """Raise ValueError unless data is a mapping with data['schema'] == schema_tag(kind)."""

def json_ready(value, path: str = "metadata"):
    """Recursively convert numpy scalars/arrays to Python scalars/lists; raise
    ValueError naming `path` for values that are not JSON-compatible."""
```

Rationale: seven classes implement the dict contract; these ~30 lines keep the tag format and the numpy-conversion rule in one place. This is shared logic, not a framework — do not add registries, base classes, or decorators.

## Module Plan: `src/ryd_gate/lattice/geometry.py`

Current state (file is 181 lines):

- `LatticeGeometry` frozen dataclass, fields `N: int`, `coords: np.ndarray`, `sublattice: np.ndarray`, `spacing_um: float` (lines 27–49); no ids, no validation, no methods;
- `make_chain(N, spacing_um=4.0)` (line 51): coords `(i*s, 0)`, sublattice `(-1)**i`;
- `make_square_lattice(Lx, Ly, spacing_um=4.0)` (line 66): order `for ix in range(Lx) for iy in range(Ly)`, coords `(ix*s, iy*s)`, sublattice `(-1)**(ix+iy)`;
- `make_triangular_lattice(Lx, Ly, spacing_um=4.0)` (line 84): outer loop rows `iy`, inner columns `ix`, odd-row x-offset `s/2`, row pitch `s*sqrt(3)/2`, sublattice zeros;
- `make_geometry_from_coords(coords_um, sublattice=None)` (line 110): no centering, spacing = smallest positive difference of sorted x-coordinates, `0.0` for `N == 1`;
- module helpers `is_in_domain` (133), `nn_nnn_relative_pairs` (138), `cylinder_nn_nnn_pairs` (158) — pure functions used by TN/analysis code.

Implementation steps:

1. Add `RegisterLayout` frozen dataclass (`name`, `trap_coords_um`, `kind`, `metadata`) with `__post_init__` validation and `to_dict`/`from_dict` (`"ryd-gate/register-layout/v1"`).
2. **Rename** `LatticeGeometry` → `Register`, keeping the four kernel fields with the same names and order, and append:

   ```python
   ids: tuple[str, ...] | None = None
   layout: RegisterLayout | None = None
   metadata: Mapping[str, Any] = field(default_factory=dict)
   ```

   `ids=None` auto-generates `("q0", ..., f"q{N-1}")` in `__post_init__` — this keeps any internal positional construction valid and gives deserialized legacy data sane ids.
3. Implement `__post_init__` (use `object.__setattr__`; never mutate input arrays):
   - `N` positive int; `coords` → finite float ndarray of shape `(N, 2)` or `(N, 3)` (copy); `sublattice` → ndarray shape `(N,)` (copy); `spacing_um` finite nonnegative float;
   - ids: generate if `None`; else exactly `N` unique non-empty strings, stored as `tuple[str, ...]`.
4. Convert each factory **body** into the corresponding classmethod, preserving the loops verbatim and adding id generation:
   - `make_chain` body → `Register.chain(n_atoms, spacing_um=4.0, prefix="q")`;
   - `make_square_lattice` body → `Register.rectangle(rows, cols, spacing_um=4.0, prefix="q")` with `rows ≡ Lx`, `cols ≡ Ly`; `Register.square(side, spacing_um=4.0, prefix="q")` calls `rectangle(side, side, ...)`;
   - `make_triangular_lattice` body → `Register.triangular(rows, atoms_per_row, spacing_um=4.0, prefix="q")` with `rows ≡ Ly`, `atoms_per_row ≡ Lx`;
   - `make_geometry_from_coords` body → `Register.from_coordinates(coords, ids=None, prefix="q", center=True, sublattice=None)`; the spacing-inference block is copied unchanged; add the centering branch (subtract mean when `center=True`) and id handling;
   - every classmethod sets `layout=None`; do **not** synthesize a `RegisterLayout` — layout presence must remain meaningful provenance ("built from a trap pattern"), never decoration (Decision D10). Explicit attachment goes through the direct constructor or `dataclasses.replace(reg, layout=...)`.
5. Delete the four module-level factories. Do not leave aliases.
6. Add methods per docs/stage1_api.md: `n_atoms`, `dimensions`, `coords_array` (copy), `coords_um` (tuple-of-tuples), `index` (`KeyError` on unknown), `id_at` (`IndexError` on out-of-range), `distances_um`, `distance_pairs(cutoff_um=None)`, `blockade_edges(radius_um)`, `validate(device)` (one-line delegation), `draw(blockade_radius_um=None, show_ids=True, show=True)` (matplotlib imported inside the method; 2D only, `NotImplementedError` for 3D), `to_dict`/`from_dict` (`"ryd-gate/register/v1"`; coords/sublattice via `.tolist()`).
7. Leave `is_in_domain`, `nn_nnn_relative_pairs`, `cylinder_nn_nnn_pairs` untouched at module level (internal utilities, not product API, still imported elsewhere).
8. Update the module docstring: it currently advertises the `make_*` factories.

Must not live here: level structures, device constraints (beyond the `validate` delegation), pulses, backend lattice specs, simulation logic.

Verify: `uv run pytest tests/lattice/test_register.py -q`, then the full fast suite after call-site migration.

## Module Plan: `src/ryd_gate/lattice/__init__.py`

Current state: re-exports `LatticeGeometry` and the `make_*` factories; also hosts/forwards lattice plotting helpers.

Implementation steps:

1. Export `Register` and `RegisterLayout` from `.geometry`.
2. Remove `LatticeGeometry` and all `make_*` exports.
3. Keep existing plotting exports unchanged (`lattice/plotting.py` functions read `.coords`/`.sublattice` attributes, which `Register` preserves — no change needed there).

## Module Plan: `src/ryd_gate/core/level_structures.py`

Current state (file is 101 lines): `TransitionSpec` (line 19); `LevelStructureSpec` (line 33) with fields `name`, `levels`, `rydberg_levels`, `transitions`, `detuning_levels`; `InteractionSpec` (line 54) with `C6`/`max_range_um`/`mode` and `DEFAULT_C6`; `level_structure(name)` (line 62) with presets `1r` (66), `01r` (73), `ger` (83), `rb87_7` (93).

Do not create `AtomModel`. Extend `LevelStructureSpec` in place.

Implementation steps:

1. Append fields (all defaulted, so existing construction sites keep working):

   ```python
   initial_level: str | None = None
   species: str = "Rb87"
   interaction_kind: Literal["none", "ising_c6", "xy_c3", "custom"] = "ising_c6"
   params: Mapping[str, Any] = field(default_factory=dict)
   ```

2. Add methods exactly as specified in docs/stage1_api.md:
   - `initial_level_or_default()` — `initial_level` if set else `levels[0]`;
   - `physical_kwargs()` — `{"param_set": "analog_3", **params-without-param_set}` for `analog_3`; `{"param_set": params.get("param_set", "our"), **params-without-param_set}` for `rb87_7`; `{}` otherwise;
   - `supports_backend(name)` — the fixed Stage 1 truth table (exact: all six; mps/gputn/peps: `1r`, `01r`; stabilizer: `01`; unknown: `False`);
   - `validate()` — returns `ValidationIssue`s for empty/duplicate levels, unknown initial level, Rydberg levels not in `levels`, invalid detuning target, invalid interaction kind;
   - `to_dict()`/`from_dict()` (`"ryd-gate/level-structure/v1"`; `TransitionSpec` entries as nested field dicts).
3. Extend `level_structure(name)`:
   - add `"01"`: `levels=("0", "1")`, no Rydberg levels, `initial_level="0"`, `interaction_kind="none"`;
   - add `"analog_3"`: same levels/transitions/detuning channels as `ger`, `name="analog_3"`, `initial_level="g"`, `params={"param_set": "analog_3"}`. This is **not** a parameter alias of `ger` — the two names encode different Hamiltonian construction semantics (symbolic vs physical; Decision D11);
   - set `initial_level` on existing presets: `1r` → `"1"`, `01r` → `"1"`, `ger` → `"g"`, `rb87_7` → `"0"` (+ `params={"param_set": "our"}` on `rb87_7`);
   - keep all existing channel names unchanged (`global_X`/`global_n`; `drive_R`/`drive_hf`/`delta_R`/`delta_hf`; `drive_420`/`H_1013`/`delta_e`/`delta_R`).
4. Migration rule: `level_structure("analog_3")` becomes the official way to request the analog three-level model. The `ger + param_set="analog_3"` spelling is not preserved as a *public* path; non-notebook call sites that used it are migrated in this stage, while the internal `_physical_model_for` branch keeps working for frozen notebooks until Stage 7 (see the factories plan below).

Must not live here: register coordinates, waveforms, device policy beyond data, backend compiler code.

## Module Plan: `src/ryd_gate/core/factories.py`

Current state: `build_from_lattice(cls, geometry: LatticeGeometry, level_structure=..., interaction=..., ...)` reads `geometry.N`, `geometry.coords`, `geometry.sublattice`, `geometry.spacing_um`; infers the physical block builder from the level-structure name; generates `BasisSpec.site_labels` as `"0"..f"{N-1}"`.

Implementation steps:

1. Replace the `LatticeGeometry` import/annotations with `Register`. No attribute access changes (field names preserved).
2. Extend the physical-model inference in `_physical_model_for` (factories.py:169) with three pinned behaviors:
   - preset `analog_3` routes to the `_apply_analog_3_lattice_blocks` physical path (and the analog default interaction via `_default_interaction_for_physical_model`), sourcing `param_set` from `spec.physical_kwargs()`;
   - bare `ger` keeps the symbolic path: abstract transition blocks only, no static `H_1013` term — the existing kernel test `tests/core/test_rydberg_system_model.py::test_ger_transition_blocks_are_not_compiled_as_static_dense_terms` must remain green **unmodified**;
   - the legacy branch `("ger", param_set in {"analog", "analog_3"})` is **kept** as an internal, undocumented path: `scripts/notebooks/02_ac_stark_local_addressing.ipynb` still uses that spelling and notebooks are frozen until Stage 7, where the branch is removed (see README Stage 7 outline).

   `name == "01"` builds a bare two-level system (no drive blocks beyond projectors/observables — same generic path as other non-physical presets).
3. Keep `BasisSpec.site_labels` generation unchanged (`"0"..` strings). Register ids map to site indices through `Register.index`/`id_at`; unifying labels is out of scope for Stage 1.

Must not change: Hamiltonian block contents, observable registration, interaction lowering.

## Module Plan: `src/ryd_gate/core/system.py`

Current state: imports `LatticeGeometry` for the `geometry` field annotation and `from_lattice` signature.

Implementation steps: swap import and annotations to `Register` (`geometry: Register | None`, `from_lattice(geometry: Register, ...)`). Keep the method name `from_lattice` — it describes construction from an atom-array geometry; only the class is renamed. No behavioral change.

## Module Plan: `src/ryd_gate/backends/tn_common/lattice_spec.py`

Current state: imports `make_square_lattice` to build reference geometries.

Implementation steps: import `Register` from `ryd_gate.lattice.geometry`; replace `make_square_lattice(lx, ly, s)` with `Register.rectangle(lx, ly, s)` (identical coordinates/order). No TN contract changes; no public compatibility factory.

## Module Plan: `src/ryd_gate/protocols/channels.py`

Current state: string constants only (`DRIVE_420`, `DRIVE_420_DAG`, `LIGHTSHIFT_ZERO`, `GLOBAL_X`, `GLOBAL_N`) plus the docstring describing the channel conventions.

Implementation steps:

1. Keep all constants — they are the canonical compiler channel ids used by lowering code; they are implementation constants, not product API.
2. Add `ChannelSpec` frozen dataclass exactly as specified in docs/stage1_api.md:

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

   with `__post_init__` validation (non-empty id, valid kind/addressing, `min_duration_ns >= 0`, `max_duration_ns is None or >= min`, `clock_period_ns > 0`, limits `None` or positive) and `to_dict`/`from_dict` (`"ryd-gate/channel/v1"`).
3. No pulse compilation logic here. The `amplitude_channels`/`detuning_channels` maps are *data* consumed by Stage 2's sequence compiler.

## Module Plan: `src/ryd_gate/devices.py` (new)

Current state: no device concept; constraints are scattered (interaction defaults in `InteractionSpec`, physics presets in `rb87_params.py`, no geometry limits anywhere).

Implementation steps:

1. Implement `DeviceSpec` frozen dataclass:

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

   Field names use `*_level_structure*` (the repo's concept), not `*_atom_model*`.
2. `DeviceSpec.virtual_rb87()` with the fixed values from docs/stage1_api.md and these channels (the compiler-channel maps are the Stage 2 lowering contract — names verified against `core/channel_lowering.py`):

   ```text
   rydberg_global:   kind="rydberg",  transition="1_r", addressing="global",
                     amplitude_channels={"1r": "global_X", "01r": "drive_R"},
                     detuning_channels={"1r": "global_n", "01r": "delta_R"}
   rydberg_local:    kind="rydberg",  transition="1_r", addressing="local",
                     amplitude_channels={"01r": "drive_R"},
                     detuning_channels={"01r": "delta_R"}
   hyperfine_global: kind="microwave", transition="0_1", addressing="global",
                     amplitude_channels={"01r": "drive_hf"},
                     detuning_channels={"01r": "delta_hf"}
   ```

3. Validation methods returning `list[ValidationIssue]` with the exact error codes in docs/stage1_api.md:
   - `validate_register(register)` — dimensions, atom count, pair distances (via `register.distance_pairs()`), radial extent;
   - `validate_level_structure(spec_or_name)` — allowed-list membership, species match;
   - `validate_pulse(pulse, channel_id)` — channel existence, then delegate the limit checks to the shared channel-limit helper (the same one `Pulse.validate` uses; implement the helper once, in `pulse.py` or `channels.py`, not twice).
4. Helpers: `rydberg_blockade_radius_um(rabi_rad_per_us)` (formula and failure modes per docs/stage1_api.md) and `describe()` (name, dimensions, species, set constraints, one line per channel).
5. `to_dict`/`from_dict` (`"ryd-gate/device/v1"`, nested channel dicts).

Must not live here: backend selection, job submission, sequence compilation, Hamiltonian construction, waveform code, coordinate storage. Do not create `src/ryd_gate/core/devices.py`.

## Module Plan: `src/ryd_gate/pulse.py`

Current state (file is 75 lines): three continuous-time kernel helpers in seconds — `blackman_window(t, t_rise)`, `blackman_pulse(t, t_rise, t_gate)`, `blackman_pulse_sqrt(...)`. Imported by `protocols/gate_cz_to.py:57`, `protocols/gate_cz_ar.py:62`, legacy exact backends, and `tests/core/test_blackman.py`.

Implementation steps:

1. **Keep** the three kernel helpers' names, signatures, and numerical behavior unchanged, and add one line to each docstring: *"Internal kernel helper (gate protocols / legacy backends) — not public API; no stability guarantee."* They remain importable by name as `ryd_gate.pulse.blackman_*` (consumers: `protocols/gate_cz_to.py:57`, `protocols/gate_cz_ar.py:62`, legacy exact backends); only the top-level `ryd_gate` re-exports are removed. `tests/core/test_blackman.py` continues to pass untouched.
2. Add the product layer in the same module:

   ```python
   WaveformKind = Literal["constant", "ramp", "blackman", "interpolated", "custom"]

   @dataclass(frozen=True)
   class Waveform:
       duration_ns: int
       kind: WaveformKind
       params: Mapping[str, Any] = field(default_factory=dict)
       samples: tuple[float, ...] | None = None
       unit: Literal["rad_per_us"] = "rad_per_us"
   ```

   Constructors `constant`, `ramp`, `blackman(duration_ns, peak=None, area=None)` (exactly one; `peak = area / (0.42 * duration_us)` when area-form), `interpolated`, `custom(samples, dt_ns=1)`. Methods `value_at_ns` (clamped), `value_at_s` (`value_at_ns(t_s*1e9) * 1e6`), `sample(dt_ns)` / `integral_rad(dt_ns)` (both require `dt_ns` to divide `duration_ns`; trapezoid over µs), `first_value`, `last_value`, `to_dict`/`from_dict` (`"ryd-gate/waveform/v1"`). The blackman evaluation uses the same window expression as `blackman_window` with `t_rise = duration/2` — shared formula, not a call-wrapper (units differ: ns grid vs seconds).
3. Add `Pulse`:

   ```python
   @dataclass(frozen=True)
   class Pulse:
       amplitude: Waveform
       detuning: Waveform
       phase_rad: float = 0.0
       post_phase_shift_rad: float = 0.0
       metadata: Mapping[str, Any] = field(default_factory=dict)
   ```

   Constructors `constant(duration_ns, amplitude, detuning, phase_rad=0.0, post_phase_shift_rad=0.0)`, `constant_amplitude(amplitude: float, detuning: Waveform, ...)`, `constant_detuning(amplitude: Waveform, detuning: float, ...)`; property `duration_ns`; `validate(channel)` delegating to the shared channel-limit helper; `to_dict`/`from_dict` (`"ryd-gate/pulse/v1"`).
4. Implement the shared channel-limit helper here (module-level, private): given `(pulse, channel: ChannelSpec)` return the `pulse.*`-coded issues. Both `Pulse.validate` and `DeviceSpec.validate_pulse` call it.
5. Define `__all__ = ["Pulse", "Waveform"]` in `pulse.py` (soft closure): star-imports and IDE completion expose only the product API, while the `blackman_*` helpers remain importable by name for the kernel consumers listed in step 1.

Must not live here: sequence objects, backend compilers, system construction, device registry, noise.

## Module Plan: `src/ryd_gate/core/__init__.py` and `src/ryd_gate/__init__.py`

Current state: top-level `__init__.py` re-exports `blackman_pulse`, `blackman_pulse_sqrt`, `blackman_window` (line 69, `__all__` lines 125–127), the `make_*` factories, and shows `make_chain` in its docstring example (line 15).

Implementation steps:

1. `core/__init__.py`: add exports `ValidationIssue`, `raise_for_errors` (from `core.validation`), keep `LevelStructureSpec`, `level_structure`, `InteractionSpec` exported.
2. Top-level `__init__.py`:
   - add: `Register`, `RegisterLayout`, `DeviceSpec`, `ChannelSpec`, `ValidationIssue`, `raise_for_errors`, `Waveform`, `Pulse` (plus keep `LevelStructureSpec`, `level_structure`, `InteractionSpec`, `RydbergSystem`, protocols, `simulate`, …);
   - remove: `make_chain`, `make_square_lattice`, `make_triangular_lattice`, `make_geometry_from_coords`, `LatticeGeometry`, `blackman_window`, `blackman_pulse`, `blackman_pulse_sqrt`;
   - update `__all__` and the module docstring example (`make_chain(4)` → `Register.chain(4)`);
   - plain imports, no lazy-import machinery for Stage 1 names.

## Tests

### `tests/lattice/test_register.py`

1. `Register.chain(3, 4.0)`: ids `("q0","q1","q2")`, coords `((0,0),(4,0),(8,0))`, `sublattice == [1,-1,1]`.
2. `Register.rectangle(2, 3, 5.0)`: row-major order, coords per docs, checkerboard sublattice `[1,-1,1,-1,1,-1]`.
3. `Register.square(2, 5.0)` equals `Register.rectangle(2, 2, 5.0)` field-wise.
4. `Register.triangular(2, 3, 4.0)`: odd-row x-offset `2.0`, row pitch `4*sqrt(3)/2`, zero sublattice, row-major ids.
5. `Register.from_coordinates`: id generation, `center=True/False` behavior, `sublattice=` passthrough, spacing inference.
6. duplicate ids raise `ValueError`; mixed coordinate dimensions raise `ValueError`; omitted ids auto-generate `q0..`.
7. `coords_array` returns a copy; `coords_um` returns tuple-of-tuples.
8. `index`/`id_at` round-trip and raise `KeyError`/`IndexError`.
9. `distances_um` symmetric, zero diagonal; `distance_pairs(cutoff_um=...)` filters; `blockade_edges` upper-triangular within radius.
10. `draw(show=False)` returns a matplotlib `Figure` (Agg backend); 3D register draw raises `NotImplementedError`.
11. `validate(device)` returns exactly `device.validate_register(register)` (stub device).
12. `from ryd_gate.lattice import LatticeGeometry` raises `ImportError`; same for all four `make_*` names.
13. every classmethod constructor returns `layout is None`; `dataclasses.replace(reg, layout=...)` attaches a layout and re-runs `__post_init__` validation.

### `tests/core/test_level_structures_product_api.py`

1. `level_structure("01")` has levels `("0","1")`, `initial_level == "0"`, `interaction_kind == "none"`.
2. `initial_level_or_default()`: `"1"` for `1r`/`01r`, `"g"` for `ger`/`analog_3`, `"0"` for `rb87_7`.
3. `level_structure("analog_3").physical_kwargs()["param_set"] == "analog_3"`; `rb87_7` → `"our"`; `1r`/`01r`/`ger`/`01` → `{}`.
4. `supports_backend` truth table exactly as specified (including unknown backend → `False`).
5. `ger` keeps channels `drive_420`, `H_1013`, `delta_e`, `delta_R`.
6. `validate()` flags duplicate levels, unknown initial level, Rydberg level not in levels.
7. `RydbergSystem.from_lattice(Register.chain(2), level_structure("analog_3"))` builds and matches the old `ger + param_set="analog_3"` system (same basis, same block names).
8. semantic split pinned at product level: the `analog_3` system compiles with the physical analog blocks (static `H_1013` term present), while a bare-`ger` system with the same protocol does not (and the kernel test `test_ger_transition_blocks_are_not_compiled_as_static_dense_terms` stays green unmodified).

### `tests/core/test_validation.py`

1. valid warning/error issues construct; invalid severity / empty code / non-tuple path raise.
2. warning-only list does not raise in `raise_for_errors`; one error raises `ValueError`; multiple errors appear on separate lines with code and message.

### `tests/core/test_devices.py`

1. `virtual_rb87()` exposes `rydberg_global`, `rydberg_local`, `hyperfine_global` with the specified kinds/transitions/maps.
2. register below 2.0 µm spacing → `register.min_distance`; 3D register → `register.dimensions`; `max_atom_num`/`max_radial_distance_um` checks fire when configured on a custom `DeviceSpec`.
3. allowed level structures pass; unknown name → `level_structure.unsupported`; species mismatch → `level_structure.species`.
4. pulse validation codes: `channel.unknown`, `pulse.min_duration`, `pulse.max_duration`, `pulse.clock_period`, `pulse.amplitude_limit`, `pulse.detuning_limit` (limits set on a custom channel).
5. `rydberg_blockade_radius_um`: formula value for a known C6/Ω pair; `ValueError` on missing C6 key or nonpositive Ω.
6. `describe()` contains the device name and every channel id.
7. `Pulse.validate(channel)` and `DeviceSpec.validate_pulse(pulse, id)` return identical issue codes for the same violation (shared helper, not duplicated rules).

### `tests/core/test_pulse_api.py`

1. `Waveform.constant` samples all equal; `Waveform.ramp` endpoints correct.
2. `Waveform.blackman(T, peak=p)`: endpoints exactly 0, center exactly `p`, matches the window formula on a sample grid.
3. `Waveform.blackman(T, area=a)`: `abs(integral_rad(1) - a) <= 1e-3 * abs(a)`; negative area flips sign; both/neither of peak/area raise.
4. `Waveform.interpolated` piecewise-linear values; invalid grids raise.
5. `Waveform.custom` duration `(len-1)*dt_ns`; linear interpolation between samples.
6. `value_at_ns` clamps; `value_at_s` equals `value_at_ns(t*1e9)*1e6`.
7. `sample`/`integral_rad` raise when `dt_ns` does not divide `duration_ns`.
8. `Pulse.constant` builds matching constant waveforms; `constant_amplitude`/`constant_detuning` match the other waveform's duration; mismatched durations raise; non-finite phase raises.
9. kernel helpers still importable from `ryd_gate.pulse` (`blackman_window` et al.) and still pass their existing numeric checks (covered by untouched `tests/core/test_blackman.py`).
10. `from ryd_gate import blackman_pulse` raises `ImportError`.
11. `set(ryd_gate.pulse.__all__) == {"Pulse", "Waveform"}`.
12. `from ryd_gate.pulse import *` (executed in a fresh namespace) binds `Waveform` and `Pulse` but no `blackman_*` name.

### `tests/core/test_serialization_roundtrip.py`

1. For each of `RegisterLayout`, `Register` (one with `layout=None` and one with an explicitly attached `RegisterLayout`), `LevelStructureSpec` (one preset + one custom), `ChannelSpec`, `DeviceSpec.virtual_rb87()`, all five `Waveform` kinds, and a `Pulse`: `json.dumps(obj.to_dict())` succeeds and `from_dict(to_dict())` round-trips (`==` for scalar/tuple-field objects; field-wise for `Register`, including the restored nested layout).
2. every `to_dict()` carries the correct `"schema"` tag.
3. `from_dict` raises `ValueError` on a wrong tag and on an invalid payload (e.g. duplicate ids smuggled into a register dict).
4. non-JSON-compatible metadata raises `ValueError` in `to_dict()`.

## Acceptance

Run, in order, all green:

```bash
uv run pytest tests/lattice/test_register.py tests/core/test_level_structures_product_api.py \
  tests/core/test_validation.py tests/core/test_devices.py tests/core/test_pulse_api.py \
  tests/core/test_serialization_roundtrip.py -q
uv run pytest -m "not slow" -q
```

Stage 1 is complete only if additionally:

1. no `src/ryd_gate/api/` directory exists;
2. no forbidden file changed;
3. the import contract in docs/stage1_api.md holds (top-level and domain imports work; all eight removed imports fail);
4. `python -c "import ryd_gate"` triggers no backend/matplotlib import;
5. every internal call site uses `Register` / `Register.*` classmethods — `grep -rn "LatticeGeometry\|make_square_lattice\|make_chain\|make_triangular_lattice\|make_geometry_from_coords" src tests app examples scripts --include="*.py"` returns nothing;
6. `examples/demo_local_addressing.py --help` and `scripts/bench_quench_check.py --help` (or an equivalent smoke invocation) still run.

## Non-Goals for Stage 1

Stage 1 must not implement `Sequence`, `simulate_sequence`, results/state handles, `NoiseModel`, sequence drawing, Pulser interop, JSON Schema files, or any backend contract change. Stage 1 only ensures the foundational public nouns are the real kernel objects, validated, drawable, and serializable.
