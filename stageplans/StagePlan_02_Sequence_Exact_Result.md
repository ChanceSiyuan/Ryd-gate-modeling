# Stage 2 Plan: Sequence, Exact Simulation Bridge, Lazy Exact Result

## Purpose

Stage 2 connects the Stage 1 API to the existing exact simulation path. It implements:

- `Sequence`
- `SequenceProtocol`
- `simulate_sequence`
- exact-only lazy result wrapper

Stage 2 must not change tensor-network backends. It must not implement MPS/PEPS/GPUTN state handles.

## Stage 2 Dependency

Stage 2 starts only after Stage 1 passes its acceptance command.

## Allowed File Operations

Create:

```text
src/ryd_gate/api/sequence.py
src/ryd_gate/api/result.py
src/ryd_gate/api/simulate.py
src/ryd_gate/compiler/__init__.py
src/ryd_gate/compiler/sequence_compiler.py
tests/api/test_sequence.py
tests/api/test_sequence_compile_exact.py
tests/api/test_result_exact.py
```

Modify:

```text
src/ryd_gate/api/__init__.py
```

Do not modify:

```text
src/ryd_gate/simulate.py
src/ryd_gate/backends/*
src/ryd_gate/ir/evolution.py
src/ryd_gate/protocols/base.py
scripts/notebooks/*
```

## Import Contract

After Stage 2:

```python
from ryd_gate.api import Sequence, simulate_sequence
```

Top-level import is still not required:

```python
from ryd_gate import Sequence
```

Do not add top-level re-exports in Stage 2.

## Module: `api/sequence.py`

Implement operation dataclasses exactly:

```python
@dataclass(frozen=True)
class PulseOp:
    channel: str
    pulse: Pulse
    t_start_ns: int
    targets: tuple[str, ...] | None = None

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

Do not implement `TargetOp` or `phase_shift` in Stage 2.

Implement:

```python
class Sequence:
    def __init__(
        self,
        register: Register,
        device: DeviceSpec,
        atom_model: AtomModel | str | None = None,
    ) -> None: ...

    def declare_channel(self, name: str, channel_id: str) -> None: ...

    def add(self, pulse: Pulse, channel: str, targets=None) -> None: ...

    def delay(self, duration_ns: int, channel: str) -> None: ...

    def measure(self, basis: Literal["full-level", "rydberg", "computational"] = "rydberg") -> None: ...

    @property
    def duration_ns(self) -> int: ...

    @property
    def operations(self) -> tuple[PulseOp | DelayOp | MeasureOp, ...]: ...

    def validate(self) -> list[ValidationIssue]: ...

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Sequence": ...

    def compile_system(self, interaction: InteractionSpec | None = None) -> "RydbergSystem": ...
```

### Sequence rules

1. If `atom_model is None`, use `AtomModel.preset(device.default_atom_model)`.
2. If `atom_model` is a string, convert with `AtomModel.preset(atom_model)`.
3. `declare_channel(name, channel_id)`:
   - `name` is user alias;
   - `channel_id` must exist in `device.channels`;
   - duplicate `name` raises `ValueError`.
4. Stage 2 supports only global channels:
   - if the device channel has `addressing != "global"`, `add(...)` raises `NotImplementedError`.
5. `add(pulse, channel, targets=None)`:
   - `channel` must be declared;
   - `targets` must be `None`;
   - pulse is appended at current end time of that declared channel;
   - overlapping operations on the same declared channel are impossible because append-only scheduling is used;
   - overlapping operations on different declared channels are allowed.
6. `delay(duration_ns, channel)` appends a channel-local delay.
7. `duration_ns` is the maximum end time across all declared channels and measure ops.
8. `measure(...)` appends one final `MeasureOp` at current sequence duration.
9. Nonzero `Pulse.phase_rad` or `post_phase_shift_rad` is stored but `compile_system()` raises `NotImplementedError` if any nonzero phase is present.

## Module: `compiler/sequence_compiler.py`

Implement:

```python
class SequenceProtocol(Protocol):
    def __init__(self, sequence: Sequence) -> None: ...

    @property
    def n_params(self) -> int: ...

    def validate_params(self, x) -> None: ...

    def unpack_params(self, x, system) -> dict: ...

    @property
    def required_channels(self) -> frozenset[str]: ...

    def drive_channels(self, system) -> frozenset[str]: ...

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]: ...
```

### SequenceProtocol rules

1. `n_params == 0`.
2. `validate_params(x)` accepts only empty sequence.
3. `unpack_params` returns exactly:

```python
{
    "t_gate": sequence.duration_ns * 1e-9,
    "n_sites": sequence.register.n_atoms,
}
```

4. `drive_channels(system)` returns all compiler channel names required by active pulse operations.
5. Compiler channel mapping uses `ChannelSpec.amplitude_channels[atom_model.name]` and `ChannelSpec.detuning_channels[atom_model.name]`.
6. If a required mapping is missing, raise `ValueError` with text `"sequence.channel_model_mismatch"`.
7. For model `1r`:
   - amplitude channel is `global_X`;
   - coefficient is `Omega_rad_s / 2`;
   - detuning channel is `global_n`;
   - coefficient is `-Delta_rad_s`.
8. For model `01r`:
   - amplitude channel is `drive_R`;
   - coefficient is `Omega_rad_s / 2`;
   - detuning channel is `delta_R`;
   - coefficient is `-Delta_rad_s`.
9. Stage 2 does not compile `hyperfine_global`; if used, raise `NotImplementedError` with text `"sequence.hyperfine_not_stage2"`.
10. `get_drive_coefficients(t, params)` returns zeros for inactive pulse intervals.
11. Time `t` is seconds; convert to `t_ns = t * 1e9`.
12. If two declared channels map to the same compiler channel at the same time, raise `ValueError` with text `"sequence.compiler_channel_collision"`.

Implement:

```python
def compile_sequence_to_system(
    sequence: Sequence,
    interaction: InteractionSpec | None = None,
) -> RydbergSystem: ...
```

Rules:

- call `sequence.validate()` and raise for errors;
- reject nonzero phase;
- call `register.to_geometry()`;
- call `atom_model.to_level_structure()`;
- call `RydbergSystem.from_lattice(...)`;
- pass physical kwargs from `atom_model.physical_kwargs()`;
- bind `SequenceProtocol(sequence)` as protocol.

## Module: `api/result.py`

Implement exact-only lazy result wrapper.

```python
@dataclass
class ExactStateHandle:
    psi: np.ndarray
    system: RydbergSystem
    register: Register
    atom_model: AtomModel
```

Methods:

```python
def statevector(self, *, max_dim: int | None = None, copy: bool = True) -> np.ndarray: ...

def expectation(self, observable: str) -> float: ...

def populations(self, level: str) -> np.ndarray: ...

def sample(
    self,
    n_shots: int,
    basis: Literal["full-level", "rydberg", "computational"] = "rydberg",
    seed: int | None = None,
) -> dict[str, int]: ...
```

Rules:

- `statevector(max_dim=None)` returns dense vector for exact backend.
- If `max_dim` is not `None` and `psi.size > max_dim`, raise `ValueError`.
- `expectation(observable)` calls existing `system.expectation(observable, psi)`.
- `populations(level)` returns per-site population array ordered by `register.ids`.
- `sample` computes probabilities from `abs(psi)**2` on demand.
- `sample` must not cache random counts.
- `basis="full-level"` returns strings made by concatenating local level labels in register order. For multi-character labels like `r_garb`, join labels with spaces, e.g. `"0 r_garb"`.
- `basis="rydberg"` returns bitstrings of length `N`, where `1` means local level is in `atom_model.rydberg_levels`.
- `basis="computational"` is valid only when all sampled levels are `"0"` or `"1"` with total probability 1 within tolerance `1e-10`; otherwise raise `ValueError` with text `"result.noncomputational_population"`.

Implement:

```python
@dataclass
class SimulationResult:
    raw: EvolutionResult
    state: ExactStateHandle
    backend: Literal["exact"]
    sequence: Sequence
    metadata: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict)

    def statevector(self, *, max_dim: int | None = None, copy: bool = True) -> np.ndarray: ...
    def expectation(self, observable: str, *, cache: bool = True) -> float: ...
    def populations(self, level: str, *, cache: bool = True) -> np.ndarray: ...
    def sample(...): ...
    def clear_cache(self) -> None: ...
```

Rules:

- `expectation` and `populations` are lazy and cached by deterministic cache keys.
- `sample` is lazy but not cached.
- `raw` stores the existing `EvolutionResult`.

## Module: `api/simulate.py`

Implement:

```python
def simulate_sequence(
    sequence: Sequence,
    *,
    backend: Literal["exact"] = "exact",
    psi0=None,
    interaction: InteractionSpec | None = None,
    **kwargs,
) -> SimulationResult: ...
```

Rules:

- Stage 2 accepts only `backend="exact"`.
- Any other backend raises `NotImplementedError` with text `"simulate_sequence.backend_not_stage2"`.
- If `psi0 is None`, use product state with `sequence.atom_model.initial_level` on every atom.
- Compile system through `compile_sequence_to_system`.
- Call existing `ryd_gate.simulate.simulate(system, [], psi0, backend="exact", **kwargs)`.
- Wrap returned `EvolutionResult` in `SimulationResult`.

## Update `api/__init__.py`

Add exports:

```python
from .sequence import DelayOp, MeasureOp, PulseOp, Sequence
from .simulate import simulate_sequence
from .result import ExactStateHandle, SimulationResult
```

## Tests

### `tests/api/test_sequence.py`

Required tests:

1. sequence defaults atom model from device.
2. declare duplicate channel raises.
3. undeclared channel add raises.
4. local channel add raises `NotImplementedError`.
5. two pulses on same channel append sequentially.
6. delays extend channel-local duration.
7. measure appends at sequence duration.
8. serialization roundtrip preserves operations.

### `tests/api/test_sequence_compile_exact.py`

Required tests:

1. `1r` global pulse compiles to system with level structure `1r`.
2. `01r` global pulse compiles to system with level structure `01r`.
3. compiled protocol emits `global_X` and `global_n` for `1r`.
4. compiled protocol emits `drive_R` and `delta_R` for `01r`.
5. coefficient conversion from `rad/us` to `rad/s` is exact within floating tolerance.
6. nonzero phase raises `NotImplementedError`.
7. hyperfine channel raises `NotImplementedError`.

### `tests/api/test_result_exact.py`

Required tests:

1. `simulate_sequence` returns `SimulationResult`.
2. `result.statevector()` equals raw final state.
3. `result.populations("r")` matches existing `system.expectation("n_r_i")`.
4. `result.sample(100, basis="rydberg", seed=1)` returns exactly 100 shots.
5. `basis="computational"` raises when Rydberg population is nonzero.
6. expectation caching stores deterministic values.
7. sample counts are not cached.

## Acceptance Command

Run exactly:

```bash
pytest tests/api/test_register.py tests/api/test_atom_model.py tests/api/test_device.py tests/api/test_waveform.py tests/api/test_pulse.py tests/api/test_sequence.py tests/api/test_sequence_compile_exact.py tests/api/test_result_exact.py
```

Stage 2 is complete only if this command passes and no tensor-network backend file has changed.

