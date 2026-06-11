# Stage 3 Plan: Backend-Native Lazy Result Handles

## Purpose

Stage 3 lets `simulate_sequence(...)` run on tensor-network backends and makes the result layer capability-aware, so users can call

```python
result.expectation("sum_nr")
result.sample(1000)
result.statevector(max_dim=4096)
```

and get either an answer computed natively (MPS expectation values without densification) or a precise, typed refusal — never a silent dense materialization of a 2^N vector.

This is the first stage allowed to touch TN backend files, and the changes are confined to *result plumbing* (what comes back), never to evolution algorithms.

## Why This Is Not Stage 1/2

State handles require algorithm-interface decisions: whether a backend retains its final native state, how expectations are computed without densifying, what sampling is safe without mutating the stored state, and how failures are typed. Those are backend contract changes, deliberately isolated here.

## Stage 3 Dependency

Stage 3 starts only after Stage 2 passes its acceptance (including the full fast suite).

## No-Wrapper / No-Fake Rules

1. The handle interface is a `typing.Protocol` (structural), so `ExactStateHandle` keeps working unchanged and backend handles implement it without a common base class. **Naming caution:** the repo already has `protocols.base.Protocol` (the physics scheduling ABC); in `results.py` import as `from typing import Protocol as TypingProtocol` and never re-export it.
2. MPS expectations must reuse the measurement code the TeNPy backend already has for its evolution-time `observables=[...]` feature — extract it into one shared function; do not write a second observable-measurement path.
3. No faked support: a backend without native-state retention returns a handle with empty capabilities whose every query raises `UnsupportedResultQuery` with a backend-specific code. Partial support may only be added together with tests in the same stage.
4. No silent densification: dense statevectors from TN states require an explicit `max_dim` and raise typed errors otherwise.

## Allowed File Operations

Create:

```text
tests/sequence/test_state_handle_capabilities.py
tests/backends/test_mps_result_handle.py
```

Modify:

```text
src/ryd_gate/results.py                       (handle protocol, errors, MPSStateHandle, SimulationResult.state typing)
src/ryd_gate/simulate.py                      (simulate_sequence backend expansion only)
src/ryd_gate/backends/tenpy_mps/backends.py   (native-state retention + shared observable function)
src/ryd_gate/backends/tn_common/simulate.py   (state_handle_kind metadata passthrough)
src/ryd_gate/backends/gputn/backend.py        (metadata: state_handle_kind="unsupported")
src/ryd_gate/backends/peps2d/yastn_backend.py (metadata: state_handle_kind="unsupported")
```

Do not modify:

```text
src/ryd_gate/core/*
src/ryd_gate/ir/hamiltonian.py
src/ryd_gate/protocols/*
src/ryd_gate/sequence.py
src/ryd_gate/devices.py
src/ryd_gate/pulse.py
scripts/notebooks/* and all *.ipynb
```

`ir/evolution.py` is not modified: `EvolutionResult.psi_final` is already typed `Any` and `metadata` is an open dict — the Stage 3 contract fits the existing container.

## Module Plan: `src/ryd_gate/results.py` (extended)

### Errors and interface

```python
class UnsupportedResultQuery(RuntimeError): ...
class StateMaterializationError(RuntimeError): ...

class QuantumStateHandle(TypingProtocol):
    kind: str                         # "statevector" | "mps" | "unsupported"
    n_atoms: int
    local_levels: tuple[str, ...]
    atom_ids: tuple[str, ...]

    @property
    def capabilities(self) -> frozenset[str]: ...
    def expectation(self, observable: str): ...
    def sample(self, n_shots: int, basis: str = "rydberg", seed: int | None = None): ...
    def statevector(self, *, max_dim: int | None = None, copy: bool = True): ...
```

Capability strings are fixed:

```text
expectation                  # native expectation values
sampling                     # bitstring sampling
statevector                  # dense state is native and cheap
statevector_materialization  # dense state producible on explicit, size-guarded request
```

Purpose: callers branch on `capabilities` instead of try/except; codes in raised errors make refusals scriptable.

### ExactStateHandle migration

Add `kind = "statevector"`, `n_atoms`/`local_levels`/`atom_ids` properties (from system/register), and `capabilities == frozenset({"expectation", "sampling", "statevector"})`. No Stage 2 behavior may regress (existing tests stay green unmodified).

### MPSStateHandle

```python
@dataclass
class MPSStateHandle:
    mps: Any                          # TeNPy MPS, opaque here
    spec: TNLatticeSpec
    register_ids: tuple[str, ...]
    metadata: dict = field(default_factory=dict)
```

- `kind = "mps"`; `n_atoms = spec.N`; `local_levels = spec.level_spec.levels`; `atom_ids = register_ids`;
- `capabilities = frozenset({"expectation", "statevector_materialization"})`, plus `"sampling"` only if the sampling rule below is actually implemented.

**Expectation** — supported observable names (resolved against `spec`):

```text
sum_n_<level> | n_<level>_<site_index> | n_<level>_<atom_id> | sum_nr
```

`sum_nr` sums the levels in `spec.level_spec.rydberg_levels`; atom ids resolve through `register_ids`. Anything else raises `UnsupportedResultQuery("mps.observable_unsupported: <name>")`. Implementation must call the shared `measure_mps_observable` function (see backend plan below) — never densify.

**Sampling** — implement sequential MPS sampling only if it can be done without mutating the stored state (operate on a copy / use TeNPy's non-destructive sampling if available). Otherwise `sample(...)` raises `UnsupportedResultQuery("mps.sampling_not_implemented")`. No approximate or mutating sampling, silently or otherwise.

**Statevector materialization**:

```text
max_dim is None                  -> StateMaterializationError("mps.statevector_requires_max_dim")
local_dim ** N > max_dim         -> StateMaterializationError("mps.statevector_too_large")
otherwise                        -> contract to a dense vector in register site order
```

### Unsupported handle

```python
@dataclass
class UnsupportedStateHandle:
    kind = "unsupported"
    backend: str
    reason_code: str                  # e.g. "peps.state_handle_not_implemented"
```

`capabilities = frozenset()`; every query raises `UnsupportedResultQuery(reason_code)`.

### SimulationResult changes

- `state: QuantumStateHandle` (was exact-only);
- add `capabilities` property returning `state.capabilities`;
- delegation rules from Stage 2 unchanged (one-line delegation + caching); caching keys unchanged.

## Backend Return Contract

Every backend's `EvolutionResult.metadata` gains one key (additive, no existing key changes):

```python
metadata["state_handle_kind"] = "statevector" | "mps" | "unsupported"
```

Implementation steps per backend:

1. **tenpy_mps/backends.py** — the two `EvolutionResult(...)` construction sites (lines ~113 and ~319):
   - determine what `psi_final` currently holds; if it is already the final TeNPy MPS, just set `state_handle_kind="mps"`; if it is something else (e.g. measured values), attach the final MPS as `metadata["native_state"]` behind a `keep_state: bool = True` backend option and set the kind accordingly (`"unsupported"` when `keep_state=False`);
   - extract the per-observable measurement logic used by the existing `observables=[...]` evolution feature into a module-level `measure_mps_observable(psi_mps, spec, name) -> float`, and re-route the existing loop through it (behavior-identical refactor; existing TN tests must not change numerically).
2. **tn_common/simulate.py** — pass `state_handle_kind`/`native_state` through unchanged when wrapping engine results (no recomputation).
3. **gputn/backend.py**, **peps2d/yastn_backend.py** — set `state_handle_kind="unsupported"` only. Reason codes used by the handle layer: `gputn.state_handle_not_implemented`, `peps.state_handle_not_implemented`.

## `simulate_sequence` Backend Expansion (`simulate.py`)

```python
simulate_sequence(sequence, backend="mps", **kwargs)
```

Steps:

1. capability gate first: `sequence.level_structure.supports_backend(backend)` must be `True`, else `ValueError("level_structure.backend_unsupported")` — the API-boundary refusal, instead of a TN-compiler stack trace;
2. compile via `compile_sequence_to_system` exactly as in Stage 2;
3. run the existing `simulate(system, [], psi0, backend=backend, **kwargs)` (TN kwargs like `backend_options={"chi_max": ...}` forward verbatim);
4. wrap by `metadata["state_handle_kind"]`:
   - `"statevector"` → `ExactStateHandle`;
   - `"mps"` → `MPSStateHandle(mps=<psi_final or metadata["native_state"]>, spec=<from metadata>, register_ids=sequence.register.ids)`;
   - `"mps"` requested but no native state present → `UnsupportedResultQuery("mps.native_state_missing")`;
   - anything else → `UnsupportedStateHandle` with the backend's reason code;
5. `backend="exact"` path is untouched.

Backends other than `{"exact", "mps"}` remain `NotImplementedError("simulate_sequence.backend_not_stage3")` for sequence entry (direct `simulate(system, ...)` use of gputn/peps is unaffected).

## Tests

### `tests/sequence/test_state_handle_capabilities.py`

1. exact result: `capabilities ⊇ {"expectation", "sampling", "statevector"}`; Stage 2 sampling/expectation behavior unchanged.
2. `UnsupportedStateHandle`: empty capabilities; every query raises `UnsupportedResultQuery` carrying the reason code.
3. MPS handle (constructed with a stub mps object): `statevector(max_dim=None)` → `StateMaterializationError("mps.statevector_requires_max_dim")`; too-small `max_dim` → `"mps.statevector_too_large"`.
4. unsupported observable name on the MPS handle raises `UnsupportedResultQuery`.
5. `SimulationResult.capabilities` mirrors `state.capabilities`.

### `tests/backends/test_mps_result_handle.py`

All tests `pytest.importorskip("tenpy")` and run with single-thread BLAS (see acceptance).

1. small `1r` chain sequence (e.g. 4 atoms, constant pulse) via `simulate_sequence(..., backend="mps", backend_options={"chi_max": 16, "dt": ...})` returns a `SimulationResult` with `state.kind == "mps"`.
2. `capabilities` include `"expectation"`.
3. `result.expectation("sum_nr")` is finite and agrees with the exact backend on the same 4-atom sequence within solver tolerance (cross-backend consistency — the real point of the contract).
4. `n_<level>_<atom_id>` resolves through register ids; bogus observable raises `UnsupportedResultQuery`.
5. `result.statevector(max_dim=4096)` materializes for the tiny system and matches the exact statevector up to global phase and truncation tolerance; `max_dim=8` raises.
6. `ValueError("level_structure.backend_unsupported")` for `rb87_7` + `backend="mps"`.
7. the refactored `measure_mps_observable` keeps the existing evolution-time `observables=[...]` results identical (regression pin on one existing TN test value).

## Acceptance

```bash
uv run pytest tests/sequence -q
OMP_NUM_THREADS=1 uv run pytest tests/backends/test_mps_result_handle.py -q
OMP_NUM_THREADS=1 uv run pytest -m "not slow" -q
```

(`OMP_NUM_THREADS=1` is the repo convention for TN runs on this machine — unpinned threads slow TeNPy by 10–40×.)

Stage 3 is complete only if all pass with TeNPy installed, the TeNPy-dependent tests skip cleanly without it, and all Stage 2 exact-result tests pass unmodified.

## Non-Goals for Stage 3

PEPS/GPUTN native handles (explicitly `unsupported` this stage), MPS sampling beyond the non-mutating rule, observable streaming configs (`EmulationConfig`-style observables-at-times — roadmap Stage 6), noise, and any evolution-algorithm change.
