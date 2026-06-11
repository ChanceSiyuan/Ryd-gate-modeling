# Stage 2 Plan: Sequence, Exact Simulation Bridge, Lazy Exact Result

## Purpose

Stage 2 makes the Stage 1 data layer runnable: a user composes a `Sequence` from `Register` + `DeviceSpec` + pulses, and runs it on the existing exact backend through the existing kernel path — `Sequence` compiles to a `Protocol` (the kernel's own scheduling abstraction), then `RydbergSystem.from_lattice` + `compile_hamiltonian_ir` + `simulate()` do what they already do today.

Stage 2 implements:

- `Sequence` (operations, validation, serialization, drawing) in a new top-level domain module `sequence.py`;
- `SequenceProtocol` — a `Protocol` subclass that lowers a sequence to drive coefficients — in `protocols/sequence_protocol.py`;
- `ExactStateHandle` + `SimulationResult` (lazy result layer) in a new `results.py`;
- `simulate_sequence(...)` added to the existing `simulate.py`.

Stage 2 must not change tensor-network backends and must not implement MPS/PEPS/GPUTN state handles (Stage 3).

## Consistency With Stage 1 (binding decisions)

Stage 2 is written against the Stage 1 design as implemented — earlier drafts of this file predated those decisions:

- there is **no `AtomModel`**: the sequence carries a `LevelStructureSpec` (`sequence.level_structure`), resolved via `level_structure(name)`;
- there is **no `register.to_geometry()`**: `Register` *is* the geometry object consumed by `RydbergSystem.from_lattice`;
- there is **no `src/ryd_gate/api/` package**: new code lives in domain modules (`sequence.py`, `results.py`, `protocols/sequence_protocol.py`);
- device fields are `default_level_structure` / `allowed_level_structures`;
- channel→compiler-channel maps come from `ChannelSpec.amplitude_channels[spec.name]` / `.detuning_channels[spec.name]`;
- top-level exports are the product surface: `from ryd_gate import Sequence, simulate_sequence, SimulationResult` must work after Stage 2.

## No-Wrapper Rules

1. `SequenceProtocol` **is** a `Protocol` (`protocols/base.py`). The backend compiler treats it like any other protocol; backends never see `Sequence`. Do not invent a second scheduling interface.
2. `Sequence.draw()` contains no plotting code: it builds `SequenceProtocol(self)` and calls the existing generic `Protocol.plot()` (protocols/base.py:104), feeding it physically labeled traces via `pulse_traces()` (protocols/base.py:94). One plotting implementation in the repo.
3. `EvolutionResult` (ir/evolution.py) stays a pure algorithm-agnostic container — no register/level-structure semantics are added to it. `SimulationResult` is the user-facing layer because bitstring sampling in register-id order, per-level populations, and caching are *new behavior*, not re-wrapped behavior; the raw `EvolutionResult` stays reachable at `result.raw`.
4. `SimulationResult` methods are one-line delegations to the state handle plus cache bookkeeping. Any logic beyond delegation/caching belongs in the handle (`ExactStateHandle` now, backend-native handles in Stage 3).
5. `simulate_sequence` calls the existing `ryd_gate.simulate.simulate(system, [], psi0, backend="exact", **kwargs)` — it must not re-implement dispatch, solver selection, or compilation.

## Stage 2 Dependency

Stage 2 starts only after Stage 1 passes its acceptance (including the full fast suite).

## Allowed File Operations

Create:

```text
src/ryd_gate/sequence.py
src/ryd_gate/protocols/sequence_protocol.py
src/ryd_gate/results.py
tests/sequence/__init__.py
tests/sequence/test_sequence.py
tests/sequence/test_compile_exact.py
tests/sequence/test_result_exact.py
```

Modify:

```text
src/ryd_gate/simulate.py          (ADD simulate_sequence only; the existing simulate() body is untouched)
src/ryd_gate/__init__.py          (exports)
src/ryd_gate/protocols/__init__.py (export SequenceProtocol if protocols are re-exported there)
```

Do not modify:

```text
src/ryd_gate/backends/*
src/ryd_gate/ir/*
src/ryd_gate/protocols/base.py
src/ryd_gate/core/*
src/ryd_gate/devices.py
src/ryd_gate/pulse.py
src/ryd_gate/lattice/*
scripts/notebooks/* and all *.ipynb
```

## Module Plan: `src/ryd_gate/sequence.py` (new)

### Operation records

```python
@dataclass(frozen=True)
class PulseOp:
    channel: str                  # declared channel name (user alias)
    pulse: Pulse
    t_start_ns: int
    targets: tuple[str, ...] | None = None   # always None in Stage 2

@dataclass(frozen=True)
class DelayOp:
    channel: str
    duration_ns: int
    t_start_ns: int

@dataclass(frozen=True)
class MeasureOp:
    basis: Literal["full-level", "rydberg", "computational"]
    t_start_ns: int
```

Purpose: an append-only, serializable record of what the user scheduled. No `TargetOp`, no `phase_shift` in Stage 2 (local addressing is a later stage).

### Sequence

```python
class Sequence:
    def __init__(
        self,
        register: Register,
        device: DeviceSpec,
        level_structure: LevelStructureSpec | str | None = None,
    ) -> None: ...

    def declare_channel(self, name: str, channel_id: str) -> None: ...
    def add(self, pulse: Pulse, channel: str) -> None: ...
    def delay(self, duration_ns: int, channel: str) -> None: ...
    def measure(self, basis: Literal["full-level", "rydberg", "computational"] = "rydberg") -> None: ...

    @property
    def duration_ns(self) -> int: ...
    @property
    def operations(self) -> tuple[PulseOp | DelayOp | MeasureOp, ...]: ...
    @property
    def declared_channels(self) -> Mapping[str, ChannelSpec]: ...

    def validate(self) -> list[ValidationIssue]: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Sequence": ...
    def draw(self, *, show: bool = True, **plot_kwargs): ...
```

`Sequence` is deliberately mutable (it is a builder); everything it stores is immutable records.

### Behavior rules (each is a required test)

1. **Constructor** — purpose: bind geometry + hardware + model and fail fast.
   - `level_structure=None` → `level_structure(device.default_level_structure)`; a string → `level_structure(name)`; a `LevelStructureSpec` → used as-is;
   - runs `raise_for_errors(device.validate_register(register))` and `raise_for_errors(device.validate_level_structure(spec))` — a constructed `Sequence` is always device-compatible.
2. **`declare_channel(name, channel_id)`** — purpose: bind a user alias to a device channel, mirroring Pulser's declaration step so retargeting devices later only changes the device argument.
   - `channel_id` must exist in `device.channels` (`ValueError`);
   - duplicate `name` raises `ValueError`;
   - Stage 2 scope gates, enforced at declaration (fail earliest): `addressing == "local"` → `NotImplementedError("sequence.local_not_stage2")`; `channel_id == "hyperfine_global"` (any channel whose maps don't include the bound level structure for *rydberg* driving) → `NotImplementedError("sequence.hyperfine_not_stage2")`;
   - missing model mapping (`spec.name not in channel.amplitude_channels`) → `ValueError("sequence.channel_model_mismatch")`;
   - two declared channels must not map to the same compiler channel for the bound level structure → `ValueError("sequence.compiler_channel_collision")` at declaration time (stricter than overlap-time detection, and sufficient while one global Rydberg channel is the Stage 2 scope).
3. **`add(pulse, channel)`** — purpose: schedule a pulse; the only way time advances besides `delay`.
   - `channel` must be declared (`ValueError`);
   - runs `raise_for_errors(pulse.validate(channel_spec))` — hardware limits enforced at add time;
   - appends `PulseOp` at the channel's current end time (append-only per channel ⇒ no same-channel overlap by construction; different channels may overlap in time);
   - raises `ValueError("sequence.measured")` after `measure()`.
4. **`delay(duration_ns, channel)`** — appends a channel-local `DelayOp` (positive integer duration; declared channel; same post-measure lock).
5. **`measure(basis)`** — appends one final `MeasureOp` at the current `duration_ns`; second call raises `ValueError`. The basis is metadata consumed by `SimulationResult.sample` defaults.
6. **`duration_ns`** — maximum end time over all declared channels and the measure op; `0` for an empty sequence.
7. **`validate()`** — re-runs the constructor checks plus per-op pulse-vs-channel checks and returns accumulated issues (needed for deserialized sequences; never raises).
8. **Serialization** — `to_dict()` with `"schema": "ryd-gate/sequence/v1"` embedding the register dict, device dict, level-structure dict, `{name: channel_id}` declarations, and the op list. `from_dict` rebuilds by **replaying** declarations and operations through the public methods (one construction path, validation re-runs for free). Round-trip preserves `operations`, `declared_channels`, `duration_ns`.
9. **`draw(...)`** — builds `proto = SequenceProtocol(self)`, `params = proto.unpack_params([], None)`, returns `proto.plot(params=params, time_scale=1e9, time_label="time (ns)", unit_label="rad/us", show=show, **plot_kwargs)`. Returns `(fig, ax)` like `Protocol.plot`. No new plotting code.

## Module Plan: `src/ryd_gate/protocols/sequence_protocol.py` (new)

### Grounding in the existing kernel

`Protocol` (protocols/base.py) requires exactly four members — `n_params` (property), `validate_params(x)`, `unpack_params(x, system) -> dict` containing `"t_gate"`, and `get_drive_coefficients(t, params) -> dict[str, complex]` — and provides overridable `required_channels` (base default is the two-photon set `{"drive_420", ...}`, so **`SequenceProtocol` must override it**), `drive_channels(system)`, `pulse_traces`, and `plot`. The coefficient conventions to reproduce are the ones `SweepProtocol` already uses for the same compiler channels: amplitude channel carries `Omega(t)/2`, detuning channel carries `-Delta(t)`, both in `rad/s`.

### Implementation

```python
class SequenceProtocol(Protocol):
    def __init__(self, sequence: Sequence) -> None: ...
```

Construction precomputes, per declared channel, the compiler-channel pair and the pulse intervals:

```text
amp_channel = channel_spec.amplitude_channels[spec.name]    # e.g. "global_X" (1r), "drive_R" (01r)
det_channel = channel_spec.detuning_channels[spec.name]     # e.g. "global_n" (1r), "delta_R" (01r)
intervals   = [(t_start_ns, t_end_ns, pulse), ...]          # from PulseOps, sorted
```

Rules (each is a required test):

1. `n_params == 0`; `validate_params(x)` accepts only an empty sequence/list.
2. `unpack_params(x, system)` returns exactly `{"t_gate": sequence.duration_ns * 1e-9, "n_sites": sequence.register.n_atoms}` (`system` is accepted but unused — the signature is fixed by the ABC).
3. `required_channels` (override) and `drive_channels(system)` return the frozenset of all mapped compiler channels.
4. `get_drive_coefficients(t, params)`: `t` is **seconds** (kernel convention); convert `t_ns = t * 1e9`; for every mapped compiler channel emit a coefficient — `0.0` outside pulse intervals; inside a pulse `p` starting at `t0_ns`:

   ```text
   omega_rad_s = p.amplitude.value_at_ns(t_ns - t0_ns) * 1e6
   delta_rad_s = p.detuning.value_at_ns(t_ns - t0_ns) * 1e6
   coefficients[amp_channel] = omega_rad_s / 2
   coefficients[det_channel] = -delta_rad_s
   ```

   The `1/2` factor and detuning sign live **only here** — users write physical Ω and δ; cross-check against `SweepProtocol`'s lowering before merging.
5. Nonzero `phase_rad` or `post_phase_shift_rad` on any pulse → `NotImplementedError("sequence.phase_not_stage2")` raised at compile time. Rationale: the `global_X`/`drive_R` lowering is a real symmetric drive block; phase modulation needs the complex conjugate-pair channel treatment the gate protocols use (`drive_420`/`drive_420_dag`). Phase-modulated pulses stay on the gate-protocol path until a later stage adds pair lowering to the sequence path.
6. `pulse_traces(t, params)` returns `{f"{name}.amp": Omega_rad_per_us, f"{name}.det": Delta_rad_per_us}` for each declared channel (real, physically labeled, product units) — this is what makes `Protocol.plot` render a clean sequence diagram.

```python
def compile_sequence_to_system(
    sequence: Sequence,
    interaction: InteractionSpec | None = None,
) -> RydbergSystem: ...
```

Steps: `raise_for_errors(sequence.validate())`; enforce rule 5; then

```python
return RydbergSystem.from_lattice(
    sequence.register,                  # Register IS the geometry — no conversion
    sequence.level_structure,
    interaction=interaction,
    protocol=SequenceProtocol(sequence),
    **sequence.level_structure.physical_kwargs(),
)
```

No `compile_hamiltonian_ir` call here — the backend entry (`simulate`) already does that. This function must stay ~15 lines; if it grows, logic is leaking out of the kernel.

## Module Plan: `src/ryd_gate/results.py` (new)

### ExactStateHandle

```python
@dataclass
class ExactStateHandle:
    psi: np.ndarray
    system: RydbergSystem
    register: Register
    level_structure: LevelStructureSpec
```

Methods, with purpose:

- `statevector(*, max_dim: int | None = None, copy: bool = True) -> np.ndarray` — the dense state; `max_dim` guard so code written today keeps working when Stage 3 routes TN handles through the same call (`ValueError` if `psi.size > max_dim`); returns a copy unless `copy=False`.
- `expectation(observable: str) -> float` — delegates to the existing `system.expectation(observable, psi)`; necessary so result objects answer physics questions without users digging out the system.
- `populations(level: str) -> np.ndarray` — per-site populations ordered by `register.ids`, computed from the existing per-site observables `f"n_{level}_{i}"`; necessary because "where is the Rydberg population" is the most common analysis question.
- `sample(n_shots, basis="rydberg", seed=None) -> dict[str, int]` — multinomial sampling from `abs(psi)**2`, computed on demand and never cached (different seeds give different draws):
  - `basis="full-level"`: keys concatenate local level labels in register order; if any label has more than one character, labels are joined with single spaces (e.g. `"0 r_garb"`);
  - `basis="rydberg"`: length-`N` bitstrings, `1` iff the local level is in `level_structure.rydberg_levels`;
  - `basis="computational"`: valid only when all probability (within `1e-10`) sits on levels `"0"`/`"1"`; otherwise `ValueError("result.noncomputational_population")`.

### SimulationResult

```python
@dataclass
class SimulationResult:
    raw: EvolutionResult
    state: ExactStateHandle
    backend: str                      # "exact" in Stage 2
    sequence: Sequence
    metadata: dict = field(default_factory=dict)
```

- `statevector/expectation/populations/sample` delegate to `state` (one line each); `expectation`/`populations` cache by `(method, args)` key with `cache=True` default; `sample` is never cached; `clear_cache()` empties the cache.
- `raw` keeps the kernel `EvolutionResult` (times/states/metadata) fully accessible — no information is hidden by the product layer.

## Module Plan: `src/ryd_gate/simulate.py` (modified — additive only)

Current state: `simulate(system, x, psi0="all_ground", *, backend="exact", **kwargs)` dispatching to `backends.exact.simulate` / `backends.tn_common.simulate_tn`. This function's body must not change.

Add:

```python
def simulate_sequence(
    sequence: Sequence,
    *,
    backend: str = "exact",
    psi0=None,
    interaction: InteractionSpec | None = None,
    **kwargs,
) -> SimulationResult: ...
```

Steps:

1. Stage 2 accepts only `backend="exact"`; anything else → `NotImplementedError("simulate_sequence.backend_not_stage2")`.
2. `system = compile_sequence_to_system(sequence, interaction=interaction)`.
3. `psi0 is None` → `system.product_state([sequence.level_structure.initial_level_or_default()] * register.n_atoms)`; strings and arrays are forwarded unchanged (existing `simulate` semantics).
4. `raw = simulate(system, [], psi0, backend="exact", **kwargs)` — the existing entry point, with `t_eval`/solver kwargs forwarded verbatim.
5. Return `SimulationResult(raw=raw, state=ExactStateHandle(raw.psi_final, system, sequence.register, sequence.level_structure), backend="exact", sequence=sequence)`.

Imports of `sequence`/`results` modules happen inside the function (keep module import light, matching the existing lazy-backend-import style of this file).

## Exports

`src/ryd_gate/__init__.py` adds: `Sequence`, `PulseOp`, `DelayOp`, `MeasureOp`, `simulate_sequence`, `SimulationResult`, `ExactStateHandle`, `SequenceProtocol` (the last may live under `ryd_gate.protocols` only — decide once, document in `__all__`).

## Tests

### `tests/sequence/test_sequence.py`

1. constructor defaults the level structure from `device.default_level_structure`; string and spec forms accepted.
2. constructor raises on a register violating the device (via `raise_for_errors`).
3. duplicate channel name raises; unknown `channel_id` raises.
4. local channel and `hyperfine_global` declarations raise the specified `NotImplementedError`s; missing model mapping raises `sequence.channel_model_mismatch`; double-mapping collision raises at declare time.
5. `add` on undeclared channel raises; `add` enforcing channel limits raises via pulse validation (custom limited device).
6. two pulses on one channel schedule back-to-back; `delay` extends only its channel; cross-channel overlap allowed.
7. `measure` appends at `duration_ns`; second `measure`, or `add`/`delay` after measure, raise `ValueError("sequence.measured")`.
8. `to_dict`/`from_dict` round-trip preserves operations, declarations, level structure, duration; schema tag `"ryd-gate/sequence/v1"`; `json.dumps` succeeds.
9. `draw(show=False)` returns `(fig, ax)` with a matplotlib `Figure` (Agg backend), and adds no top-level matplotlib import to `ryd_gate.sequence`.

### `tests/sequence/test_compile_exact.py`

1. `1r` global pulse compiles to a system with level structure `1r`; `drive_channels(system) == {"global_X", "global_n"}`.
2. `01r` compiles with `{"drive_R", "delta_R"}`.
3. coefficients: at a time inside a `Pulse.constant(1000, amplitude=1.0, detuning=2.0)`, `get_drive_coefficients` returns `amp = 0.5e6` and `det = -2.0e6` (rad/us → rad/s exact within float tolerance); zero outside intervals.
4. `unpack_params` returns exactly `t_gate = duration_ns * 1e-9` and `n_sites`.
5. nonzero `phase_rad` or `post_phase_shift_rad` → `NotImplementedError` at compile.
6. `compile_sequence_to_system` returns a `RydbergSystem` whose `geometry` **is** the sequence's register object (identity, not copy — no-wrapper check).
7. `n_params == 0`; `validate_params([0.1])` raises.

### `tests/sequence/test_result_exact.py`

1. `simulate_sequence` returns a `SimulationResult`; `result.raw` is the kernel `EvolutionResult`; `result.statevector()` equals `raw.psi_final`.
2. physics check (ties API to dynamics): one atom, `1r`, `Pulse.constant_detuning(Waveform.blackman(1000, area=np.pi), 0.0)` → `result.populations("r")[0] ≈ 1` within solver tolerance.
3. `result.populations("r")` matches `system.expectation("n_r_<i>")` per site, ordered by `register.ids`.
4. `result.sample(100, basis="rydberg", seed=1)` returns counts summing to 100 and is reproducible with the same seed.
5. `basis="computational"` raises `result.noncomputational_population` when Rydberg population is present.
6. `expectation` caches (underlying `system.expectation` called once for two identical queries — monkeypatch counter); `sample` is not cached; `clear_cache` works.
7. unsupported backend name raises `NotImplementedError("simulate_sequence.backend_not_stage2")`.

## Acceptance

```bash
uv run pytest tests/sequence -q
uv run pytest -m "not slow" -q
git diff --stat -- src/ryd_gate/backends src/ryd_gate/ir   # must be empty
```

Stage 2 is complete only if all three pass, `from ryd_gate import Sequence, simulate_sequence, SimulationResult` works, and the existing `simulate(system, x, ...)` call signature and behavior are bit-identical (covered by the untouched existing tests in the fast suite).

## Non-Goals for Stage 2

Local addressing/`target`, `phase_shift`, hyperfine driving through sequences, phase-modulated sequence pulses, TN backends, noise, mid-sequence measurement, and Pulser interop are all later stages. The gate protocols (`TOProtocol`, `ARProtocol`, …) remain the supported path for phase-modulated CZ work throughout Stage 2.
