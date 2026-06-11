# Stage 5 Plan: Gate Library and Error Budgets

## Purpose

Stage 5 productizes the repository's strongest differentiator: gate-level
Rydberg modeling. Pulser-style sequence simulation is now in place, but Pulser
does not have a two-qubit gate library, gate fidelity reports, or microscopic
error budgets. This stage turns the existing CZ protocols and analysis
functions into a documented product surface:

```python
from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.gates import ARProtocol, CZGateReport, cz_gate_report

protocol = ARProtocol()
system = RydbergSystem.from_lattice(
    Register.chain(2, 5.4),
    level_structure("rb87_7"),
    protocol=protocol,
)
report = cz_gate_report(system, protocol, x)
assert isinstance(report, CZGateReport)
```

The simulation and analysis work remains in the existing kernel:
`TOProtocol`, `ARProtocol`, `DoubleARPProtocol`,
`analysis.gate_metrics.average_gate_infidelity`, and
`analysis.gate_metrics.error_budget`. Stage 5 adds stable imports, stable
report data, serialization, and tests that pin the notebook benchmark values.

## Why This Is Not Stage 1-4

Gate reports need all prior layers:

- Stage 1 provides schema-tagged serialization and top-level product exports.
- Stage 2 provides runnable product sequences without changing gate protocols.
- Stage 3 provides result capability rules, keeping report data explicit.
- Stage 4 provides the declarative noise object that later gate reports can
  reference, while this stage keeps deterministic exact reports first.

The gate library is deliberately after the general product API because it is a
domain-specific flagship, not a replacement for the lower-level kernel.

## Stage 5 Dependency

Stage 5 starts only after Stage 4 passes its acceptance, including the full
fast suite.

## No-Wrapper / No-Fake Rules

1. Do not duplicate gate protocols. `ryd_gate.gates` re-exports the existing
   `TOProtocol`, `ARProtocol`, and `DoubleARPProtocol` classes; it does not
   define `TimeOptimalCZ`, `AmplitudeRobustCZ`, or any forwarding subclasses.
2. Do not duplicate fidelity math. Any report helper must call or refactor
   `analysis.gate_metrics.average_gate_infidelity` and `error_budget`; if phase
   diagnostics need shared overlap data, extract a private helper in
   `gate_metrics.py` and route the existing public functions through it.
3. Do not add optimizers to `src/`. Optimization workflows stay in
   `scripts/` and notebooks. Reusable protocols and metrics live in `src/`;
   search loops and fitting scripts do not.
4. `CZGateReport` is a data object, not a simulator. It stores metrics already
   computed by existing functions and serializes them.
5. No compatibility shims for old legacy classes. The exact legacy backend has
   been removed; Stage 5 works only through `RydbergSystem` + `Protocol`.
6. Existing top-level lazy exports for `average_gate_infidelity` and
   `error_budget` remain valid. Stage 5 may add eager exports for report
   objects, but must not break `from ryd_gate import average_gate_infidelity`.

## Allowed File Operations

Create:

```text
src/ryd_gate/gates/__init__.py
src/ryd_gate/gates/cz.py
tests/gates/__init__.py
tests/gates/test_gate_imports.py
tests/gates/test_cz_gate_report.py
tests/gates/test_cz_benchmark_pins.py
```

Modify:

```text
src/ryd_gate/__init__.py
src/ryd_gate/analysis/gate_metrics.py          (shared CZ overlap helper, docstrings, stable error messages)
src/ryd_gate/protocols/gate_cz_to.py           (docstrings only)
src/ryd_gate/protocols/gate_cz_ar.py           (docstrings only)
src/ryd_gate/protocols/gate_cz_double_arp.py   (docstrings only)
docs/stage1_api.md                             (gate surface note only, if needed)
stageplans/README.md                           (status/table only, after implementation)
```

Do not modify:

```text
src/ryd_gate/backends/*
src/ryd_gate/ir/*
src/ryd_gate/core/local_blocks.py
src/ryd_gate/core/factories.py
src/ryd_gate/simulate.py
src/ryd_gate/sequence.py
src/ryd_gate/results.py
scripts/notebooks/* and all *.ipynb
```

Notebook benchmark values may be copied into tests as constants, but notebooks
themselves remain untouched until Stage 7.

## Public API Plan

### `ryd_gate.gates`

`src/ryd_gate/gates/__init__.py` exports exactly:

```text
TOProtocol
ARProtocol
DoubleARPProtocol
average_gate_infidelity
error_budget
CZGateReport
cz_gate_report
```

The protocol names are imports from `ryd_gate.protocols.*`. The metric names
are imports from `ryd_gate.analysis.gate_metrics`. `ryd_gate.gates` is a
domain namespace, not a second implementation tree.

Top-level `ryd_gate` exports add:

```text
CZGateReport
cz_gate_report
```

`average_gate_infidelity` and `error_budget` keep their existing lazy top-level
behavior.

### `CZGateReport`

Implement in `src/ryd_gate/gates/cz.py`:

```python
@dataclass(frozen=True)
class CZGateReport:
    protocol: str
    parameters: tuple[float, ...]
    infidelity: float
    phase_error_rad: float
    theta_rad: float
    residuals: Mapping[str, float] = field(default_factory=dict)
    error_budget: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def fidelity(self) -> float: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CZGateReport": ...
```

Serialization uses schema tag `"ryd-gate/cz-gate-report/v1"`. `fidelity` is
computed as `1.0 - infidelity` and is not duplicated in the stored payload.

`parameters` are the exact `x` passed to the protocol, converted to a tuple of
floats. `protocol` is the class name, e.g. `"ARProtocol"`.

### Phase diagnostic

Add a private shared helper in `analysis/gate_metrics.py` if needed:

```python
def _cz_overlaps(system, protocol, x, *, amplitude_scale=1.0) -> dict[str, complex]:
    ...
```

It must use the same three input states as `average_gate_infidelity`. The phase
error reported by `CZGateReport` is:

```text
wrap_to_pi(arg(a11) - 2 * arg(a01) + arg(a00) - pi)
```

where `a00`, `a01`, and `a11` are the phase-corrected overlaps already used by
the Nielsen fidelity calculation. If `average_gate_infidelity` is refactored
onto `_cz_overlaps`, existing numerical outputs must be bit-identical within
the current test tolerances.

### `cz_gate_report`

```python
def cz_gate_report(
    system,
    protocol: Protocol,
    x: Sequence[float],
    *,
    include_error_budget: bool = True,
    include_residuals: bool = True,
    initial_states: list[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CZGateReport: ...
```

Rules:

1. `protocol.validate_params(list(x))` is called first.
2. The protocol must expose `unpack_params(... )["theta"]`, otherwise raise
   `ValueError("cz_report.protocol_not_cz")`.
3. `average_gate_infidelity(..., return_residuals=include_residuals)` computes
   infidelity and residuals.
4. `error_budget(...)` is called only when `include_error_budget=True`.
5. The helper never calls a backend directly except through the existing metric
   functions.
6. The returned report must round-trip through `to_dict()` / `from_dict()`.

## Benchmark Pins

Stage 5 must add a small test fixture that copies the CZ benchmark parameter
sets from `scripts/notebooks/cz_gate_validation_and_errors.ipynb`
(`X_TO_DARK`, `X_AR`, and the notebook's system kwargs:
`blackmanflag=True, detuning_sign=1` for TO dark, plain `our` system for AR)
into a Python test module. This is allowed because it does not edit the
notebook.

*Revision (2026-06-11, validated against the .ipynb):* the notebook renders
its result tables as Markdown display objects and stores **no numeric
infidelity outputs**, so reference values cannot be copied out of the file.
The pinned values are instead computed once from the current deterministic
exact solver at implementation time and recorded as constants next to the
parameter sets — they guard regressions from this point on. The Double-ARP
benchmark is also **not** in that notebook; its pin uses a deterministic
`DoubleARPProtocol` configuration taken from the existing protocol test suite
(`tests/protocols/test_double_arp_protocol.py`).

Required pins:

```text
TOProtocol dark-detuning benchmark infidelity (X_TO_DARK)
ARProtocol benchmark infidelity (X_AR)
DoubleARPProtocol deterministic-configuration infidelity
one representative error_budget total
```

The error-budget pin uses the default `["01", "11"]` initial states in the
fast suite; a cross-check against the cached 12-SSS-state values in
`data/det_dark.json` may be added as a `slow`-marked test if its runtime
exceeds the fast-suite budget. All pinned paths are deterministic (the exact
solver has no stochastic component here); tolerances reflect solver
tolerances (relative ~1e-6), not loosened to hide regressions.

## Tests

### `tests/gates/test_gate_imports.py`

Required tests:

1. `from ryd_gate.gates import TOProtocol, ARProtocol, DoubleARPProtocol` returns
   the same class objects as `ryd_gate.protocols.*`.
2. `average_gate_infidelity` and `error_budget` imported from `ryd_gate.gates`
   are the existing `analysis.gate_metrics` functions.
3. `from ryd_gate import CZGateReport, cz_gate_report` works.

### `tests/gates/test_cz_gate_report.py`

Required tests:

1. `CZGateReport.fidelity == 1.0 - infidelity`.
2. `to_dict()` writes `"ryd-gate/cz-gate-report/v1"` and round-trips exactly for
   JSON-ready metadata.
3. `cz_gate_report(..., include_error_budget=False)` does not call
   `error_budget`.
4. `cz_gate_report(..., include_residuals=False)` stores an empty residual map.
5. `phase_error_rad` is finite and wrapped into `[-pi, pi]`.
6. A protocol without `theta` raises `ValueError("cz_report.protocol_not_cz")`.

### `tests/gates/test_cz_benchmark_pins.py`

Required tests:

1. `cz_gate_report` reproduces the TO benchmark infidelity.
2. `cz_gate_report` reproduces the AR benchmark infidelity.
3. `cz_gate_report` reproduces the Double-ARP benchmark infidelity.
4. The report error budget matches direct `error_budget(...)` output and the
   pinned benchmark value.

## Acceptance

```bash
uv run pytest tests/gates -q
uv run pytest tests/analysis tests/protocols -q
OMP_NUM_THREADS=1 uv run pytest -m "not slow" -q
uv run ruff check src/ryd_gate/gates src/ryd_gate/analysis/gate_metrics.py tests/gates
```

Stage 5 is complete only if all existing analysis/protocol tests pass
unmodified except for import-contract additions.

## Non-Goals for Stage 5

No optimizer APIs, no pulse-parameter search loops, no new gate protocols, no
noisy Monte Carlo report aggregation, no TN gate fidelity path, no notebook
migration, and no changes to backend solvers.
