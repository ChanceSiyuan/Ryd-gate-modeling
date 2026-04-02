# TeNPy Integration Plan for Large-Scale Lattice Simulation

## Goal

Add a tensor-network simulation path based on `TeNPy` so the lattice workflow can scale beyond exact sparse evolution and support physically meaningful simulations for systems up to roughly `16x16`, with the first target being the workflow currently demonstrated in `docs/lattice_demo.py`.

The plan should preserve the current dense/sparse exact path for small systems while introducing a new TN-friendly representation and backend for larger lattice simulations.

## Why TeNPy

- The project is already Python-based, so `TeNPy` has the lowest integration friction.
- The paper summarized in `docs/coarsen.tex` explicitly reports MPS/DMRG/TDVP simulations using `TeNPy`.
- `TeNPy` is a good fit for:
  - DMRG ground/low-energy states
  - finite-size MPS on 2D lattices mapped to 1D
  - 2-site TDVP real-time evolution
- It is more realistic for the current codebase than starting with PEPS/iPEPS.

## Non-Goals for the First Implementation

- Do not try to replace the current exact sparse path.
- Do not try to support arbitrary existing systems on day one; focus on the 2-level lattice Rydberg model.
- Do not target periodic-boundary-condition `16x16` production runs in v1.
- Do not try to fully solve long-time critical dynamics near the QCP in the first pass.
- Do not force all observables through dense matrices; TN observables need their own path.

## Current Constraints in the Repository

The current lattice stack is exact-Hilbert-space oriented:

- `src/ryd_gate/core/atomic_system.py:create_lattice_system()` constructs a `LatticeSystem` with prebuilt operators living in dimension `2**N`.
- `src/ryd_gate/lattice/operators.py:build_operators()` explicitly embeds global sparse matrices in the full Hilbert space.
- `src/ryd_gate/compilers/sparse_lattice.py` compiles those matrices into a sparse `HamiltonianIR`.
- `src/ryd_gate/solvers/sparse_expm.py` evolves a dense state vector with `expm_multiply`.

This architecture is fundamentally incompatible with `N=256`, so the TN path must avoid creating:

- full state vectors
- full sparse operators in dimension `2**N`
- observables defined only as full matrices

## Design Principles

1. Keep protocols as control logic.
   `SweepProtocol` should continue to describe time-dependent coefficients, local addressing, and ramp schedules.

2. Separate physical model description from numerical representation.
   A TN backend should consume geometry, site terms, interaction terms, and measurement recipes, not full-space matrices.

3. Preserve the high-level user entry point when practical.
   Long term, `simulate(...)` should be able to dispatch to exact or TN backends. Short term, a separate TN demo script is acceptable during bring-up.

4. Make TN support additive, not disruptive.
   Small-system exact tests should remain valid.

5. Prefer observable streaming over full trajectory storage.
   Large TN runs should record selected observables at each time, not save every MPS by default.

## High-Level Architecture

Introduce a second lattice representation alongside `LatticeSystem`:

- `LatticeSystem`:
  exact sparse, suitable for small `N`

- new TN-friendly lattice spec/model:
  stores only geometry and Hamiltonian coefficients needed to build a `TeNPy` model

Proposed new layers:

- `src/ryd_gate/tn/`
  TeNPy integration package

- `src/ryd_gate/tn/lattice_spec.py`
  lightweight TN-friendly system dataclass

- `src/ryd_gate/tn/model.py`
  TeNPy model builder for the 2-level Rydberg lattice

- `src/ryd_gate/tn/state.py`
  product-state and pinned-state MPS initializers

- `src/ryd_gate/tn/observables.py`
  MPS measurement utilities for occupations, staggered magnetization, and domain proxies

- `src/ryd_gate/tn/backends.py`
  DMRG and TDVP backend wrappers

- `src/ryd_gate/tn/results.py`
  result containers for TN runs, or reuse/extend `EvolutionResult`

## Dependency Plan

Add `TeNPy` as an optional dependency, not a hard dependency for the whole package.

Preferred approach:

- add an optional extra such as `tn`
- import `TeNPy` lazily inside TN modules
- raise a clear error if a TN backend is requested without `TeNPy` installed

The implementation should not import `tenpy` from top-level package import paths such as `src/ryd_gate/__init__.py`.

## Data Model Plan

### 1. Add a TN-friendly lattice specification

Create a new dataclass, e.g. `TensorNetworkLatticeSpec`, containing:

- `Lx`, `Ly`, `N`
- `coords`
- `sublattice`
- `vdw_pairs`
- `V_nn`
- `Omega`
- optional boundary-condition metadata
- snake-order mapping tables

This object must not contain any `2**N` matrices.

### 2. Reuse geometry logic

Reuse the existing square-lattice geometry and site indexing conventions so:

- addressing indices remain consistent with current `SweepProtocol`
- `sublattice = (-1) ** (x + y)` remains unchanged
- AF1/AF2 definitions remain consistent across exact and TN paths

### 3. Add a model wrapper if needed

If the existing `SystemModel` abstraction is kept, add a TN-aware lattice model variant that exposes symbolic blocks/observables without assuming they are matrices.

Possible options:

- minimal option:
  TN path bypasses `SystemModel` at first

- better option:
  extend `BlockRegistry`/observable plumbing to support symbolic TN terms

For v1, the minimal option is acceptable if it reduces risk.

## Hamiltonian Representation Plan

The TN implementation should represent the lattice Hamiltonian at the local-term level:

- onsite drive:
  `Omega(t)/2 * sum_i X_i`

- onsite detuning:
  `-sum_i (Delta(t) + delta_i) n_i`

- interaction:
  `sum_(i<j) V_ij n_i n_j`

where in the current lattice code `vdw_pairs` only includes NN and NNN.

### Mapping to TeNPy

Represent the 2D square lattice as a 1D snake-ordered MPS chain.

Store:

- `site_order_1d_to_2d`
- `site_order_2d_to_1d`

This mapping is needed for:

- building couplings in TeNPy
- applying local addressing
- plotting/measuring observables back in 2D

### Time Dependence

Use `SweepProtocol` as the source of:

- `Delta(t)`
- `Omega(t)`
- `delta_i`

The TDVP path should use piecewise-constant time slices, matching the current exact solver spirit:

- at each step, evaluate coefficients at the midpoint
- rebuild or update the time-local MPO
- evolve the MPS for one step

This keeps the protocol semantics aligned with the current sparse solver.

## Initial State Plan

Support the following initial-state constructors for TN runs:

1. all-ground product state
   needed for `docs/lattice_demo.py`

2. AF1 and AF2 product states
   needed for validation and local-domain construction

3. pinned checkerboard/domain product states
   needed for local-addressing/domain-wall workflows

4. DMRG-prepared low-energy state
   needed for Higgs-mode-like workflows and low-energy quenches

Implementation note:

- do not require users to pass a dense `psi0`
- allow `psi0` to be:
  - a product-state descriptor
  - a TeNPy MPS
  - a helper object returned by a TN state builder

## Observable Plan

Dense observable matrices are not appropriate for TN. Add dedicated MPS measurement helpers for:

- `n_i`
- mean Rydberg fraction
- staggered magnetization
- local staggered magnetization map
- domain proxy used in `demo_local_addressing.py`

Longer term, add:

- structure factor
- correlation function `G(r)`
- correlation length fit helpers

Important:

- the current `ObservableRegistry.measure()` assumes `obs.operator @ psi`
- that API is not suitable for MPS objects
- either add TN-specific observable execution or introduce a backend-dependent measurement interface

For v1, a separate TN measurement module is the lowest-risk path.

## Solver Plan

### Phase A: DMRG backend

Purpose:

- find ground states and low-energy pinned states
- validate the Hamiltonian against small exact results

Create a wrapper such as `TenpyDMRGBackend` that:

- takes a TN lattice spec and Hamiltonian builder
- constructs a TeNPy model/MPO
- runs finite DMRG
- returns an `EvolutionResult`-compatible result or a TN-specific result object

Outputs:

- final MPS
- energies
- convergence metadata
- optional measured observables

### Phase B: TDVP backend

Purpose:

- evolve the MPS under the sweep/hold protocol
- support short- to medium-time dynamics on larger lattices

Create `TenpyTDVPBackend` with:

- two-site TDVP
- configurable `chi_max`, truncation cutoff, and time step
- piecewise-constant Hamiltonian updates from `SweepProtocol`

Outputs:

- final MPS
- sampled observables vs time
- optional snapshots of MPS at selected times
- truncation/convergence metadata

### Why TDVP over TEBD first

- interactions are not purely nearest-neighbor in 1D after snake mapping
- TDVP handles long-range MPO evolution more naturally
- it is closer to the literature workflow already described in `docs/coarsen.tex`

## Integration Strategy with Existing `simulate(...)`

There are two realistic stages.

### Stage 1: Separate TN entry point

Add a new function, e.g.:

- `simulate_tn(...)`

This is the safest initial path because it avoids destabilizing the current dispatcher.

Suggested signature:

```python
simulate_tn(
    system_or_spec,
    protocol,
    x,
    initial_state,
    method="tdvp",
    t_eval=None,
    observables=None,
    backend_options=None,
)
```

Use this in a new demo script first.

### Stage 2: Fold TN into `simulate(...)`

Once stable, extend `simulate(...)` dispatch so that:

- exact `LatticeSystem` + ndarray state -> existing sparse path
- TN lattice spec/model + MPS/product-state descriptor -> TN path

This stage should happen only after validation and tests are in place.

## Proposed File-Level Implementation

### New modules

- `src/ryd_gate/tn/__init__.py`
- `src/ryd_gate/tn/lattice_spec.py`
- `src/ryd_gate/tn/model.py`
- `src/ryd_gate/tn/state.py`
- `src/ryd_gate/tn/observables.py`
- `src/ryd_gate/tn/backends.py`
- `src/ryd_gate/tn/simulate.py`

### Likely modified modules

- `src/ryd_gate/__init__.py`
  export TN entry points carefully and lazily if possible

- `src/ryd_gate/solvers/base.py`
  ensure result containers are MPS-compatible

- `src/ryd_gate/solvers/dispatch.py`
  only in the second integration stage

- `src/ryd_gate/protocols/sweep.py`
  possibly no major changes, but it may need tiny helpers to expose time-sliced coefficients cleanly

- `docs/lattice_demo.py`
  either add a TN version or replace the current exact example with a small exact / large-TN toggle

## Recommended User-Facing API

### Short-term

Keep exact demo intact and add a TN demo:

- `docs/lattice_demo.py`
  remains exact small-system example

- `docs/lattice_demo_tn.py`
  large-lattice TeNPy example

### Medium-term

Support a backend selector:

```python
result = simulate_tn(
    system,
    protocol,
    x,
    initial_state="all_ground",
    method="tdvp",
    t_eval=np.linspace(0, x[2], 101),
    observables=["m_s", "n_mean"],
    backend_options={"chi_max": 256, "dt": 0.1},
)
```

Later, if dispatch is unified:

```python
result = simulate(
    system,
    protocol,
    x,
    psi0,
    backend=TenpyTDVPBackend(...),
)
```

## Validation Plan

### Level 1: Exact cross-checks on small systems

For `2x2`, `3x3`, maybe `4x4` where feasible:

- compare energies from exact diagonalization vs DMRG
- compare short-time TDVP evolution vs exact sparse evolution
- compare:
  - `n_i(t)`
  - `m_s(t)`
  - final-state energy
  - norm preservation

Acceptance target:

- observables match within agreed tolerance for short times
- qualitative agreement survives moderate `chi`

### Level 2: Convergence checks

For TN-only regimes, always run:

- `chi_max` sweep
- time-step sweep
- truncation cutoff inspection
- optional energy drift tracking for TDVP

Acceptance target:

- chosen observables stable under modest parameter tightening

### Level 3: Reproduce known physics proxies

Reproduce the qualitative behavior already used in the paper and demo:

- AF1 / AF2 staggered magnetization signs
- pinned domain shrinking trend
- Higgs-mode-like oscillation frequency trends on accessible sizes

## Performance Plan

### Memory and runtime policy

For large TN runs:

- do not save every MPS unless explicitly requested
- default to storing only:
  - sampled times
  - selected observables
  - convergence metadata

### Recommended default run classes

1. smoke test
   `4x4`, small `chi`, short time

2. validation run
   `6x6` to `8x8`, compare against exact or near-exact

3. production TN run
   `10x10` to `16x16`, OBC, observables-only output

## Risks and Mitigations

### Risk 1: 2D to 1D entanglement growth limits reachable times

Mitigation:

- start with short-time dynamics
- stream observables only
- use two-site TDVP
- provide clear convergence controls

### Risk 2: Existing abstractions are too matrix-centric

Mitigation:

- build TN path as parallel modules first
- refactor common abstractions only after the TN workflow is proven

### Risk 3: Observable API mismatch

Mitigation:

- introduce TN-specific measurement utilities in v1
- do not force MPS into current `ObservableRegistry.measure()` path immediately

### Risk 4: User confusion around exact vs TN systems

Mitigation:

- keep naming explicit:
  - exact lattice system
  - TN lattice spec
- provide separate demo scripts initially

### Risk 5: Over-promising `16x16`

Mitigation:

- document realistic scope:
  - DMRG and short-time TDVP are the primary targets
  - long-time critical dynamics remain hard

## Phased Implementation Roadmap

### Phase 0: Preparation

- add optional `TeNPy` dependency strategy
- create TN module namespace
- add lightweight lattice spec and snake mapping helpers

Deliverable:

- no user-visible simulation yet, but TN scaffolding imports cleanly

### Phase 1: Static TN path

- implement TeNPy lattice model builder
- implement product-state MPS initializers
- implement DMRG backend
- validate energies and simple observables against exact small systems

Deliverable:

- can prepare all-ground, AF1, AF2, and pinned product states
- can run DMRG and measure `n_i`, `m_s`

### Phase 2: Time evolution

- implement TDVP backend
- implement sweep time slicing from `SweepProtocol`
- implement observable streaming
- validate short-time dynamics against exact sparse evolution

Deliverable:

- can run TN sweep dynamics on medium lattices

### Phase 3: Demo integration

- add `docs/lattice_demo_tn.py`
- support `10x10+` examples
- expose useful backend controls

Deliverable:

- documented large-lattice demo path based on TeNPy

### Phase 4: Feature parity with local-addressing workflows

- implement domain-related observables and local staggered maps
- support seeded-domain and checkerboard pinning workflows
- compare trends against `scripts/demo_local_addressing.py`

Deliverable:

- TN workflow can reproduce the key observables used in the local-addressing demo

### Phase 5: Optional dispatcher unification

- teach `simulate(...)` how to route TN-compatible inputs
- harmonize result objects and measurement API where sensible

Deliverable:

- one high-level simulation entry point for exact and TN paths

## Testing Plan

Add new tests in a dedicated TN test group, gated so they can be skipped when `TeNPy` is unavailable.

Suggested files:

- `tests/test_tn_lattice_spec.py`
- `tests/test_tn_dmrg.py`
- `tests/test_tn_tdvp_small.py`
- `tests/test_tn_observables.py`

Test coverage:

- indexing and snake-order mapping
- AF1/AF2 definitions preserved
- local addressing index mapping preserved
- DMRG energy regression on tiny systems
- TDVP short-time agreement with exact evolution
- observable measurement sanity checks

## Documentation Plan

- keep `docs/lattice_demo.py` as exact small-system reference until TN path is stable
- add `docs/lattice_demo_tn.py` for large-scale simulation
- document:
  - when to use exact vs TN
  - required TeNPy installation
  - realistic size/time expectations
  - convergence checklist

## Concrete Acceptance Criteria

The implementation should be considered successful when all of the following are true:

1. A user can run a TeNPy-based lattice simulation without constructing `2**N` matrices.
2. Small-system TN observables agree with exact sparse evolution over short times within documented tolerances.
3. A large-lattice example such as `10x10` or larger runs through the TN path and produces `m_s(t)` and site occupations.
4. The architecture cleanly separates exact sparse and TN representations.
5. The codebase has a documented path toward `16x16` OBC simulations using DMRG + TDVP.

## Recommended First Implementation Slice

If implementation begins immediately, the first milestone should be:

- add TN lattice spec
- add TeNPy product-state builder
- add DMRG backend
- add `m_s` and `n_i` measurements for MPS
- validate on `2x2` and `3x3`

This is the smallest slice that proves the architecture and de-risks the later TDVP integration.
