# Stage 3 Plan: Backend-Native Lazy Result Handles

## Purpose

Stage 3 introduces backend-native result handles so users can call:

```python
result.expectation(...)
result.sample(...)
result.statevector(...)
```

without requiring every backend to materialize a dense statevector.

This is the first stage where MPS/TN result behavior may be touched.

## Why This Is Not Stage 1

`MPSStateHandle` and related objects are not obvious API data structures. They require:

- deciding whether a backend returns final native state;
- deciding how long backend-native states remain valid;
- defining capabilities per backend;
- implementing MPS contraction or sampling;
- defining failure behavior for unsupported queries.

These are algorithm-interface decisions, not Stage 1 product API construction.

## Stage 3 Dependency

Stage 3 starts only after Stage 2 passes its acceptance command.

## Allowed File Operations

Create:

```text
src/ryd_gate/api/state_handle.py
tests/api/test_state_handle_capabilities.py
tests/backends/test_mps_result_handle.py
```

Modify:

```text
src/ryd_gate/api/result.py
src/ryd_gate/api/simulate.py
src/ryd_gate/backends/tenpy_mps/backends.py
src/ryd_gate/backends/tn_common/simulate.py
src/ryd_gate/backends/gputn/backend.py
src/ryd_gate/backends/peps2d/yastn_backend.py
```

Do not modify:

```text
src/ryd_gate/core/system.py
src/ryd_gate/core/factories.py
src/ryd_gate/core/level_structures.py
src/ryd_gate/ir/hamiltonian.py
scripts/notebooks/*
```

## Module: `api/state_handle.py`

Implement:

```python
class UnsupportedResultQuery(RuntimeError):
    pass

class StateMaterializationError(RuntimeError):
    pass
```

Implement protocol:

```python
class QuantumStateHandle(Protocol):
    kind: str
    n_atoms: int
    local_levels: tuple[str, ...]
    atom_ids: tuple[str, ...]

    @property
    def capabilities(self) -> frozenset[str]: ...

    def expectation(self, observable, *, cache: bool = True): ...

    def sample(self, n_shots: int, basis: str = "rydberg", seed: int | None = None): ...

    def statevector(self, *, max_dim: int | None = None, copy: bool = True): ...
```

Capability strings are fixed:

```text
expectation
sampling
statevector
statevector_materialization
```

Rules:

- `statevector` means dense statevector is already native and cheap to return.
- `statevector_materialization` means dense statevector can be produced only on explicit request and size guard.
- If a method is unsupported, raise `UnsupportedResultQuery`.
- If dense statevector would be too large, raise `StateMaterializationError`.

## Exact Handle Migration

Move or adapt Stage 2 `ExactStateHandle` to satisfy `QuantumStateHandle`.

Exact capabilities:

```python
frozenset({"expectation", "sampling", "statevector"})
```

No exact behavior may regress from Stage 2.

## MPS Handle

Implement:

```python
@dataclass
class MPSStateHandle:
    mps: Any
    spec: TNLatticeSpec
    register_ids: tuple[str, ...]
    metadata: dict = field(default_factory=dict)
```

Properties:

```python
kind = "mps"
n_atoms = spec.N
local_levels = spec.level_spec.levels
atom_ids = register_ids
capabilities = frozenset({"expectation", "sampling", "statevector_materialization"})
```

### MPS expectation support

Stage 3 supports exactly these observables:

```text
sum_n_<level>
n_<level>_<site_index>
n_<level>_<atom_id>
sum_nr
```

Rules:

- `sum_nr` sums all levels in `spec.level_spec.rydberg_levels`.
- `n_<level>_<atom_id>` resolves `atom_id` through `register_ids`.
- Unsupported observable raises `UnsupportedResultQuery`.
- Use backend-native MPS local expectation APIs. Do not convert to dense vector to compute expectation.

### MPS sampling support

Implement sequential MPS sampling only if the current TeNPy MPS object supports the required local projection/copy operations without mutating the stored final state.

If safe non-mutating sequential sampling is not implemented, `sample(...)` must raise:

```text
UnsupportedResultQuery("mps.sampling_not_implemented")
```

Do not implement approximate or mutating sampling silently.

### MPS statevector materialization

Implement:

```python
def statevector(self, *, max_dim: int | None = None, copy: bool = True):
```

Rules:

- compute `dim = local_dim ** N`;
- if `max_dim is None`, raise `StateMaterializationError("mps.statevector_requires_max_dim")`;
- if `dim > max_dim`, raise `StateMaterializationError("mps.statevector_too_large")`;
- only then contract/materialize dense vector.

## PEPS/GPUTN Handles

Stage 3 must not fake support.

If backend-native state is not available or query methods are not implemented, return a handle with limited capabilities:

```python
capabilities = frozenset()
```

Every query method raises `UnsupportedResultQuery` with backend-specific code:

```text
peps.expectation_not_implemented
peps.sampling_not_implemented
gputn.state_handle_not_implemented
```

Partial support can be added only with tests in the same stage.

## Backend Return Contract

Modify backend result metadata to include native final state only when available.

Required metadata keys:

```python
{
    "backend": "...",
    "state_handle_kind": "statevector" | "mps" | "unsupported",
}
```

For MPS backend, `EvolutionResult.psi_final` may remain backend-native object or current return value, but `simulate_sequence(..., backend="mps")` must wrap it into `MPSStateHandle`.

Do not require all backends to use the same native object type.

## `simulate_sequence` Backend Expansion

Stage 3 expands:

```python
simulate_sequence(sequence, backend="mps")
```

Rules:

- Check `sequence.atom_model.supports_backend(backend)`.
- If unsupported, raise `ValueError` with text `"atom_model.backend_unsupported"`.
- For `backend="mps"`, call existing `ryd_gate.simulate.simulate(system, [], psi0, backend="mps", **kwargs)`.
- Wrap final backend-native result in `MPSStateHandle`.
- If backend does not return enough native state to create a useful handle, raise `UnsupportedResultQuery("mps.native_state_missing")`.

## SimulationResult Changes

Modify `SimulationResult.state` type from exact-only to `QuantumStateHandle`.

Rules:

- `result.expectation(...)` delegates to `state.expectation`.
- `result.sample(...)` delegates to `state.sample`.
- `result.statevector(...)` delegates to `state.statevector`.
- `result.capabilities` returns `state.capabilities`.
- Existing exact tests from Stage 2 must still pass.

## Tests

### `tests/api/test_state_handle_capabilities.py`

Required tests:

1. exact result capabilities include `statevector`.
2. exact result sample still works.
3. unsupported backend handle raises `UnsupportedResultQuery`.
4. MPS `statevector(max_dim=None)` raises `StateMaterializationError`.
5. MPS `statevector(max_dim=small)` raises when dimension too large.

### `tests/backends/test_mps_result_handle.py`

Required tests:

1. Small `1r` MPS run returns `SimulationResult`.
2. result state kind is `"mps"`.
3. capabilities include `"expectation"`.
4. `result.expectation("sum_nr")` returns finite numeric value.
5. `result.statevector(max_dim=...)` works only for tiny systems where materialization is allowed.
6. unsupported observable raises `UnsupportedResultQuery`.

Tests must skip cleanly if TeNPy dependency is unavailable.

## Acceptance Command

Run exactly:

```bash
pytest tests/api/test_register.py tests/api/test_atom_model.py tests/api/test_device.py tests/api/test_waveform.py tests/api/test_pulse.py tests/api/test_sequence.py tests/api/test_sequence_compile_exact.py tests/api/test_result_exact.py tests/api/test_state_handle_capabilities.py tests/backends/test_mps_result_handle.py
```

Stage 3 is complete only if this command passes in an environment with TeNPy available, and all TeNPy-dependent tests skip cleanly when TeNPy is unavailable.

