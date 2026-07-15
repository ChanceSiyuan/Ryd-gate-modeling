# Simulation

This page defines how calculations are requested and how their raw physical
results are read. Constructing the Hamiltonian is covered in [Model](model.md);
gate-specific post-processing is covered in [Gates](gates.md).

## Calculation entry points

| goal | entry point | result |
|---|---|---|
| time evolution | `simulate()` | `EvolutionResult` or a tuple of them |
| quasi-static noisy evolution | `simulate_ensemble()` | `EnsembleResult` |
| ground state of a frozen `1r` Hamiltonian | `system.ground_state()` | `GroundStateResult` |

All three consume a fully constructed, protocol-bound `RydbergSystem`. The
protocol fixes `system.t_gate`; a solver call cannot replace that duration.

## Time evolution

```python
import numpy as np

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.protocols import SweepProtocol

system = RydbergSystem(
    level_structure=level_structure("01r", ryd_level=70),
    register=Register.chain(2, spacing_um=9.0),
    protocol=SweepProtocol(
        t_gate_s=0.5e-6,
        omega_half_rad_s=lambda t: 2 * np.pi * 1e6,
        detuning_rad_s=lambda t: 0.0,
    ),
)

obs = system.observables
n_r = sum(obs.n("r", i) for i in range(system.N))
result = simulate(system, observables={"n_r": n_r})
```

The default backend is `"exact_ode"`.

## Initial states and return shape

Initial states use physical labels in Register site order:

| input | meaning | return |
|---|---|---|
| `None` | `|1...1>` | one result |
| `"plus"` | `(|0>+|1>)/sqrt(2)` at every site | one result |
| `['0', '1', ...]` | one product state | one result |
| `[['0','0'], ['0','1']]` | a batch of product states | tuple of results |

`"plus"` requires both `"0"` and `"1"` in the selected level structure. A
batch shares the same physical system and should be used for logical-basis gate
calculations:

```python
results = simulate(
    system,
    [["0", "0"], ["0", "1"], ["1", "0"], ["1", "1"]],
)
```

Dense vectors, backend-native states, previous results, and empty batches are
not public initial-state forms. A deterministic multi-stage experiment should
instead use one continuous piecewise protocol.

## Observables

The system supplies exactly two observable primitives:

```python
obs.E(ket, bra, site)  # |ket><bra|
obs.n(level, site)     # |level><level|
```

They form immutable scalar expressions with `+`, `-`, unary `-`, scalar `*`,
operator product `@`, `.dagger()`, and Python `sum()`:

```python
n_r = sum(obs.n("r", i) for i in range(system.N))
rr = obs.n("r", 0) @ obs.n("r", 1)
a = obs.E("r", "1", 0)
x_0 = a + a.dagger()
```

Complex intermediate expressions are allowed, but every expression passed to
a solver must be Hermitian. Expectations are real values. Backend capability is
checked before evolution begins.

## Measurement times

With `t_eval=None`, requested expectations are recorded only at the endpoint:

```python
result.times                 # array([system.t_gate])
result.expectation("n_r")    # real array with shape (1,)
```

An explicit measurement grid must be one-dimensional, finite, non-empty,
strictly increasing, and contained in `[0, system.t_gate]`:

```python
times = np.linspace(0.0, system.t_gate, 101)
result = simulate(
    system,
    t_eval=times,
    observables={"n_r": n_r, "rr": rr},
)
```

Explicit `t_eval` requires at least one observable. The exact requested grid is
returned; the public `times` array does not receive an implicit endpoint.
Intermediate backend states are discarded after the expectations are measured.
`amplitude()` and `sample()` nevertheless always read the true final state at
`system.t_gate`.

For PEPS, each measurement time requires an environment calculation, so cost
typically grows roughly with `len(t_eval)`. Multiple observables at one time
share the same requested environment.

## `EvolutionResult`

| interface | when evaluated | return |
|---|---|---|
| `result.times` | during evolution | 1-D array exposed with its write flag disabled |
| `result.expectation(name)` | eagerly at requested times | real array aligned with `times` |
| `result.amplitude(labels)` | lazily at the final state | complex scalar |
| `result.sample(shots=, seed=)` | lazily at the final state | `Counter[tuple[str, ...]]` |
| `result.peps_evidence` | snapshot only | PEPS evidence or `None` |

Only expectations requested in the original call exist:

```python
n_r_values = result.expectation("n_r")
amp_1r = result.amplitude(["1", "r"])
counts = result.sample(shots=1000, seed=123)
```

`shots` must be positive and `seed` a non-negative integer. Sampling does no
work unless it is called.

There is no public final backend state, complete state trajectory, complete
basis-probability vector, lazy new expectation, or generic solver metadata.
Result classes are imported only when type annotations are needed:

```python
from ryd_gate.results import EvolutionResult, GroundStateResult, EnsembleResult
```

## Backends

The following table is the single user-facing capability summary:

| backend | presets | geometry and interaction | observable terms |
|---|---|---|---|
| `exact_ode` | all presets | any valid Register and resolved pair set | any finite structured term |
| `mps` | `1r`, `01r` | any valid Register and resolved pair set | any finite structured term |
| `peps` | `1r`, `01r` | factory-built Cartesian grid, open boundary, graph-edge pairs only | BP: one-site terms; CTM: up to two-site terms |

The existence of a backend does not imply that every preset, geometry, or
observable is supported by it.

### Exact ODE

`exact_ode` uses adaptive DOP853 with stable defaults `rtol=1e-8` and
`atol=1e-12`. Options are optional:

```python
result = simulate(
    system,
    backend="exact_ode",
    backend_options={
        "hamiltonian_format": "auto",  # "auto", "dense", or "sparse"
        "rtol": 1e-8,
        "atol": 1e-12,
    },
)
```

Dense and sparse are storage/matrix-vector choices, not different physical
backends. There is no public `max_step`; the solver adapts its steps from the
error tolerances.

### MPS time evolution

MPS requires exactly three explicit options:

```python
mps_options = {
    "time_step_s": 1e-9,
    "bond_dimension": 128,
    "discarded_weight_tolerance": 1e-8,
}

result = simulate(system, backend="mps", backend_options=mps_options)
```

Requested measurement times and `system.t_gate` are exact step boundaries.
MPS is fail-closed for its declared cumulative discarded-weight tolerance: an
unmet tolerance raises instead of returning a successful-looking result.

## PEPS

PEPS is a numerical-estimate backend. It validates mathematical results and
reports numerical evidence, but it does not declare a finite calculation
converged on the user's behalf.

### Geometry and interactions

PEPS accepts only registers made by:

```python
Register.chain(...)
Register.rectangle(...)
Register.square(...)
```

It interprets them as finite open-boundary Cartesian tensor graphs. Direct
`Register(coords)` and `Register.triangular(...)` remain valid for exact/MPS but
are not accepted by this PEPS adapter.

Every nonzero physical pair term must be a Cartesian graph edge. The backend
never drops a long-range term silently. Select physical pairs with the system's
existing `interaction_cutoff_um`, for example:

```python
spacing_um = 5.0
register = Register.square(4, spacing_um=spacing_um)
peps_protocol = SweepProtocol(
    t_gate_s=0.5e-6,
    omega_half_rad_s=lambda t: 2 * np.pi * 1e6,
    detuning_rad_s=lambda t: 0.0,
)
peps_system = RydbergSystem(
    level_structure=level_structure("1r"),
    register=register,
    protocol=peps_protocol,
    interaction_cutoff_um=spacing_um,
)
peps_obs = peps_system.observables
peps_n_r = sum(peps_obs.n("r", i) for i in range(peps_system.N))
```

### Real-time options

All ten keys are mandatory:

```python
peps_options = {
    "time_step_s": 1e-9,
    "bond_dimension": 8,
    "svd_tolerance": 1e-12,
    "ntu_max_iterations": 20,
    "ntu_iteration_tolerance": 1e-10,
    "measurement_method": "ctm",  # or "belief_propagation"
    "environment_bond_dimension": 32,
    "environment_tolerance": 1e-8,
    "environment_max_iterations": 50,
    "device": "cuda",             # or "cpu"
}

result = simulate(
    peps_system,
    backend="peps",
    backend_options=peps_options,
    observables={"n_r": peps_n_r},
)
```

The time step controls physical-time resolution; `bond_dimension` and
`svd_tolerance` control the PEPS state; NTU keys control its local update; the
environment keys control expectation, amplitude/norm, and sampling
contractions. Device selection is explicit and never silently falls back from
CUDA to CPU.

`measurement_method="belief_propagation"` accepts sums of one-site terms.
`"ctm"` accepts terms acting on at most two sites. Unsupported BP expressions
raise before evolution instead of falling back to CTM.

CTM sampling requires a genuine two-dimensional grid: both grid dimensions
must be at least two. A real-time chain can therefore be sampled only with
`measurement_method="belief_propagation"`. PEPS ground-state readout always
uses CTM, so lazy ground-state sampling is unavailable on one-wide grids.

### Estimates and evidence

Finite truncation errors and residuals from eager expectation environments (or
the final ground-state CTM environment) are reported, not used as automatic
convergence gates. The configured NTU cap is retained in the option snapshot,
but actual NTU iteration counts or cap exhaustion are not recorded. Unsupported
capabilities, invalid inputs, failed tensor operations, non-finite values,
non-positive norms, or invalid sampling probabilities still raise.

PEPS results expose an immutable snapshot:

```python
evidence_before = result.peps_evidence
amp = result.amplitude(["1"] * peps_system.N)
evidence_after = result.peps_evidence
payload = evidence_after.to_dict()
```

Reading the property performs no contraction. The first contraction for each
distinct amplitude appends one record; a cached repeat does not. Every
successful `sample()` call appends one record. Later snapshots include those
records while an earlier snapshot never changes. Evidence contains the exact
PEPS options and already-produced truncation, environment, norm,
amplitude-contraction, and sampling summaries. A sampling record contains only
`shots` and `seed`; residuals or iteration counts from its private conditional
environment are not added. Evidence contains no physical result copies, full
traces, or `converged` boolean.

Exact, MPS, and DMRG results return `None` from `peps_evidence`. The three
evidence types are available from `ryd_gate.results` for type annotations.

Convergence studies belong in the caller: rerun with different time steps,
state/environment bond dimensions, and tolerances, then compare the physical
expectations, amplitudes, samples, or energies.

## Ground-state search

Ground-state search is not a `simulate()` backend. It freezes a complete `1r`
system at a mandatory time `at` and starts from an explicit label state.

```python
ground_system = RydbergSystem(
    level_structure=level_structure("1r", ryd_level=70),
    register=Register.square(2, spacing_um=7.0),
    protocol=SweepProtocol(
        t_gate_s=1e-6,
        omega_half_rad_s=lambda t: 2 * np.pi * 0.5e6,
        detuning_rad_s=lambda t: 2 * np.pi * 2e6,
    ),
    interaction_cutoff_um=7.0,
)
ground_obs = ground_system.observables
rr_ground = ground_obs.n("r", 0) @ ground_obs.n("r", 1)
ground_seed = ["1", "r", "1", "r"]
```

### DMRG

```python
ground = ground_system.ground_state(
    at=0.0,
    method="dmrg",
    initial_state=ground_seed,
    observables={"rr": rr_ground},
    method_options={
        "bond_dimension": 128,
        "discarded_weight_tolerance": 1e-10,
        "relative_energy_tolerance": 1e-8,
        "entropy_tolerance": 1e-6,
        "max_sweeps": 20,
    },
)
```

DMRG returns only after its energy, entropy, and discarded-weight criteria are
met.

### PEPS imaginary time

PEPS ground-state search uses the same PEPS geometry/pair restrictions and nine
mandatory options:

```python
ground = ground_system.ground_state(
    at=0.0,
    method="peps_imaginary_time",
    initial_state=ground_seed,
    observables={"rr": rr_ground},
    method_options={
        "bond_dimension": 8,
        "svd_tolerance": 1e-12,
        "ntu_max_iterations": 20,
        "ntu_iteration_tolerance": 1e-10,
        "environment_bond_dimension": 32,
        "environment_tolerance": 1e-8,
        "environment_max_iterations": 50,
        "imaginary_time_schedule": ((0.10, 30), (0.03, 30), (0.01, 40)),
        "device": "cuda",
    },
)
```

The Hamiltonian is normalized by a derived local energy scale, the complete
dimensionless schedule runs in decreasing step sizes, and one final CTM
measurement produces energy and requested observables. The returned estimate
and free numerical summaries follow the same PEPS evidence contract.

### `GroundStateResult`

`expectation("energy")` is always present and returns a real scalar. Other
explicitly requested expectations are also scalars. Because an eigenvector's
global phase is arbitrary, a complex amplitude needs a physical reference:

```python
energy = ground.expectation("energy")
amp = ground.amplitude(
    ["1"] * ground_system.N,
    phase_reference=ground_seed,
)
counts = ground.sample(shots=1000, seed=0)
```

The reference coefficient is rotated to non-negative real. If it is
numerically zero, choose another reference. Sampling needs no phase reference.

## Quasi-static noise

`NoiseModel` describes zero-mean Gaussian parameters that are constant within a
shot and independent between shots:

```python
from ryd_gate import NoiseModel, simulate_ensemble

noise = NoiseModel(
    position_sigma_um=(0.07, 0.07, 0.15),
)

ensemble = simulate_ensemble(
    system,
    noise=noise,
    shots=200,
    seed=123,
    observables={"n_r": n_r},
)
```

Physical laser names must match active laser protocols: `"420"`/`"1013"` for
the seven-level gate family and `"297"` for Direct-297. `SweepProtocol` and
`DigitalAnalogProtocol` emit effective Hamiltonian channels, so named physical
laser noise is rejected for them; position noise remains applicable.

Position noise samples three-dimensional offsets for the nominal two-dimensional
register and recomputes interaction weights on the already-selected pair
topology. It does not mutate `system.register`.

`EnsembleResult` is a raw shot-major container:

```python
ensemble.results[k]       # complete result for shot k
ensemble.realizations[k]  # random values actually applied to shot k
ensemble.seed
```

For a batched initial state, all states within one shot share the same
realization and `ensemble.results[k]` is the corresponding result tuple. The
container deliberately does not compute mean, standard deviation, fidelity, or
an error budget. Those formulas remain visible in the calling script. Large
ensembles should be explicitly split into batches when retaining every child
backend state would consume too much memory.

Time-dependent, correlated, or non-Gaussian noise is represented by explicitly
constructing the desired protocol/system realizations in the research script;
it is not an extra `NoiseModel` mode.

## Intentional non-features

The simulation layer does not provide:

- non-Hermitian no-jump evolution;
- raw or continued backend states;
- complete state trajectories or basis probabilities;
- gate reports or ensemble aggregation;
- derived correlations, FFTs, structure factors, or error budgets;
- optimizers, plotting, persistence, or report generation.

Radiative and blackbody decay are physical data on the level structure. Their
post-processing from explicitly requested population trajectories is described
in the [first-order gate error budget](gates.md#first-order-decay-budget). If
that approximation is insufficient, a future open-system model must be
implemented explicitly rather than inferred from hidden state-norm loss.
