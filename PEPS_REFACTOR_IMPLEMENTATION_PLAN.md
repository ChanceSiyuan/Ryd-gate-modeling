# YASTN PEPS refactor implementation plan

Status: implementation-ready; no product or numerical decisions remain open.

Authority: implement the final semantics in `REFACTOR_DECISIONS.md`, especially
PEPS01–PEPS36. When an older E/ER/PEPS statement conflicts with a later PEPS
statement, the later PEPS statement wins. This plan makes those final decisions
executable; Claude must not reopen them or add compatibility behavior.

Baseline: the shared worktree is uncommitted on top of `f394c08`. Continue from
the current worktree. Do not reset to the baseline, stage files, or commit.

## 1. Required outcome

Replace the current monolithic, fail-closed `ryd_gate.backends.peps2d` with a
small private YASTN adapter that provides the following production capability:

- real-time PEPS evolution for `1r` and `01r` systems;
- imaginary-time PEPS ground-state search for `1r` systems;
- eager, explicitly requested expectation values;
- lazy normalized complex basis amplitudes;
- lazy native conditional sampling;
- immutable PEPS numerical evidence that reports what was calculated without
  deciding whether a physical estimate is converged;
- CPU and CUDA execution using the same algorithm and `complex128` tensors.

The current PEPS geometry is deliberately narrow:

- accept only registers created by `Register.chain(...)`,
  `Register.rectangle(...)`, or `Register.square(...)`;
- interpret all three as finite open-boundary Cartesian PEPS lattices;
- accept only physical pair interactions on PEPS graph nearest-neighbour edges;
- reject direct `Register(coords)` and `Register.triangular(...)`, even when the
  direct coordinates happen to form a rectangular grid.

An arbitrary two-dimensional unit-disk-graph tensor network is a future,
separate backend/algorithm. Do not create its backend name, dispatcher entry,
placeholder, public graph API, or PEPS mode in this refactor.

## 2. Guardrails

Before editing, record:

```bash
git status --short
git rev-parse HEAD
git diff --check
sha256sum simplify.md
```

The expected base commit is `f394c08`; the current `simplify.md` hash is
`5f99d252629ead5c03ea2c2797688b7d164a9f4e88f2f46a2497e8a099f9fdd2`.
`algsummary.tex` is absent from both the base tree and current tree; leave it
absent. Never edit `simplify.md`. Preserve every unrelated existing worktree
change.

Do not:

- run `git reset`, `git checkout --`, `git clean`, or an equivalent destructive
  command;
- stage or commit;
- modify research data or `results/`;
- edit generated `docs/_build` files by hand;
- treat the existing `425 passed` result as proof of this plan: several current
  tests explicitly reward behavior that this plan removes.

At the end of every implementation slice, run focused tests plus
`git diff --check`. Stop on an unexplained physics or site-ordering change; do
not hide it by widening a tolerance.

## 3. Final public capability

### 3.1 Geometry and topology

| Register construction | Exact | MPS | PEPS real time | PEPS ground state |
|---|---:|---:|---:|---:|
| `Register.chain(...)` | yes | yes | yes | yes for `1r` |
| `Register.rectangle(...)` | yes | yes | yes | yes for `1r` |
| `Register.square(...)` | yes | yes | yes | yes for `1r` |
| direct `Register(coords)` | yes | yes | no | no |
| `Register.triangular(...)` | yes | yes | no | no |

The PEPS interpretation is fixed:

```text
chain(N)              shape = (N, 1)
rectangle(rows, cols) shape = (rows, cols)
square(side)          shape = (side, side)
boundary              open (OBC)
site i                divmod(i, cols)
```

For a `(rows, cols)` grid, allowed physical interaction edges are exactly:

```text
(r, c) -- (r + 1, c)
(r, c) -- (r, c + 1)
```

when both endpoints exist. A subset of these edges and a system with no pair
interactions are valid. Diagonal, long-range, and wraparound nonzero pairs are
capability errors. The PEPS backend never inserts, drops, rounds, or truncates a
pair. `RydbergSystem(..., interaction_cutoff_um=...)` remains the only public
physical pair selector.

### 3.2 Real-time call

The exact mandatory schema is:

```python
result = simulate(
    system,
    initial_state,
    backend="peps",
    t_eval=t_eval,
    observables={"n_r_0": system.observables.n("r", site=0)},
    backend_options={
        "time_step_s": 1e-9,
        "bond_dimension": 8,
        "svd_tolerance": 1e-12,
        "ntu_max_iterations": 20,
        "ntu_iteration_tolerance": 1e-10,
        "measurement_method": "ctm",
        "environment_bond_dimension": 32,
        "environment_tolerance": 1e-8,
        "environment_max_iterations": 50,
        "device": "cuda",
    },
)
```

`measurement_method` is exactly `"belief_propagation"` or `"ctm"`; `device`
is exactly `"cpu"` or `"cuda"`. All ten keys are mandatory. There are no
defaults, aliases, deprecated keys, or arbitrary YASTN keyword passthroughs.

`time_step_s` is a maximum physical-seconds Trotter step. The existing anchor
planner must continue to shorten local steps so every requested `t_eval` and
the true `system.t_gate` are exact step boundaries.

### 3.3 Ground-state call

The exact mandatory schema is:

```python
ground = system.ground_state(
    at=at,
    method="peps_imaginary_time",
    initial_state=["1"] * system.N,
    observables={"n_r_0": system.observables.n("r", site=0)},
    method_options={
        "bond_dimension": 8,
        "svd_tolerance": 1e-12,
        "ntu_max_iterations": 20,
        "ntu_iteration_tolerance": 1e-10,
        "environment_bond_dimension": 32,
        "environment_tolerance": 1e-8,
        "environment_max_iterations": 50,
        "imaginary_time_schedule": (
            (0.10, 30),
            (0.03, 30),
            (0.01, 40),
        ),
        "device": "cuda",
    },
)
```

All nine keys are mandatory. The schedule is a nonempty tuple of two-element
tuples. `dtau` values are finite, positive, and strictly decreasing; step counts
are positive integers. The schedule is dimensionless and every declared step is
executed. There is no stage energy calculation, early stopping, or convergence
gate.

### 3.4 Result behavior

The three readout paths remain intentionally different in when they compute:

- `expectation(name)` reads values eagerly computed by `simulate()` or
  `ground_state()` because the caller explicitly requested the observables;
- `amplitude(labels)` performs a lazy terminal-state boundary contraction;
- `sample(shots=..., seed=...)` performs lazy terminal-state native sampling.

PEPS results additionally expose:

```python
evidence = result.peps_evidence  # PEPSEvidence
payload = evidence.to_dict()
```

Exact, MPS, and DMRG results return `None` from the same property. Accessing the
property never triggers a contraction. Lazy amplitude/sample calls append
successful operation records; every property access returns a new immutable
snapshot, so an older snapshot never changes.

## 4. Private module architecture

Delete `src/ryd_gate/backends/peps2d.py`; do not leave a compatibility shim.
Replace it with this private package:

```text
src/ryd_gate/backends/peps/
    __init__.py
    _options.py
    _layout.py
    _numerics.py
    _environment.py
    _boundary.py
    _sampling.py
    _readout.py
    _engine.py
```

Responsibilities are fixed:

- `__init__.py`: dependency-free option/layout re-exports plus lazy wrappers for
  `evolve_peps` and `solve_peps_ground_state`; importing this package must not
  import YASTN or `_engine`.
- `_options.py`: exact schema parsing into private frozen option records; no
  YASTN import.
- `_layout.py`: Register provenance to `_PEPSLatticeSpec`, site mapping, OBC
  edge construction, and compiled-pair validation; no YASTN import.
- `_numerics.py`: private `PEPSError`, scalar device-to-host conversion,
  mathematical validity helpers, NTU aggregation, and small private summary
  records.
- `_environment.py`: BP/CTM construction, environment summaries, observable
  lowering, expectation measurement, and physical ground energy.
- `_boundary.py`: deterministic product-bra and double-layer norm boundary-MPS
  contractions.
- `_sampling.py`: validated BP and CTM sequential conditional sampling.
- `_readout.py`: thread-safe lazy norm/coefficient caches, evidence ledger, and
  the evolution/ground-state reader objects.
- `_engine.py`: YASTN/device loading, local operators, product-state PEPS,
  Trotter gates, real-time orchestration, and normalized imaginary-time
  orchestration.

No class or helper from this package is added to `ryd_gate.__all__` or otherwise
made public. `PEPSError` remains an internal `RuntimeError`; do not add a public
exception hierarchy.

`_engine.py` must also have no module-scope YASTN import. The dispatcher first
imports `_options`/`_layout`, completes every dependency-free preflight, and
only then calls a lazy package wrapper that imports `_engine`; `_load_yastn()`
performs the actual third-party import. This import boundary is part of the
tested capability contract, not merely an optimization.

Dependency direction must be acyclic:

```text
options/layout/numerics
        ↓
environment/boundary/sampling
        ↓
readout
        ↓
engine
        ↓
private package __init__ / dispatchers
```

## 5. Register provenance and PEPS preflight

### 5.1 Private provenance

In `src/ryd_gate/lattice.py`, add a private frozen record equivalent to:

```python
@dataclass(frozen=True, slots=True)
class _RegisterOrigin:
    factory: Literal["custom", "chain", "rectangle", "square", "triangular"]
    grid_shape: tuple[int, int] | None
```

Change `Register.__slots__` to `("_coords", "_origin")` while preserving its
public constructor signature `Register(coords)` exactly.

- direct construction records `("custom", None)`;
- `chain(N)` records `("chain", (N, 1))`;
- `rectangle(rows, cols)` records `("rectangle", (rows, cols))`;
- `square(side)` records `("square", (side, side))`;
- `triangular(...)` records `("triangular", None)`.

Use a private construction helper so factories can set provenance without
adding a public constructor keyword. `square()` must build its own result rather
than delegating in a way that loses square provenance.

The helper must call the same coordinate validation/freezing path as direct
construction rather than bypassing it with `__new__`. Store shape components as
built-in `int` values.

Do not store spacing, edges, adjacency, topology, boundary, or interaction data
in the provenance. Do not add any public property. `register.coords`,
`register.N`, the direct constructor, and the four existing factories remain the
entire public Register surface; `__all__` remains `['Register']`. Do not expose
provenance through `repr`, and do not add or promise a Register serialization
API.

### 5.2 Private lattice specification

`_layout.py` derives:

```python
@dataclass(frozen=True, slots=True)
class _PEPSLatticeSpec:
    shape: tuple[int, int]
    site_to_coord: tuple[tuple[int, int], ...]
    allowed_edges: frozenset[tuple[int, int]]
```

For accepted factories, require `rows * cols == register.N` and set
`site_to_coord[i] = divmod(i, cols)`. `allowed_edges` uses sorted Register-site
index pairs, not coordinates. Do not inspect unique x/y values or use any float
tolerance to infer shape.

After all dependency-free checks pass, construct the only YASTN geometry as:

```python
geometry = fpeps.SquareLattice(dims=spec.shape, boundary="obc")
```

First require `terms.n_sites == register.N`. Validate every compiled
`TNTerms.pairs` entry before importing YASTN:

- indices must satisfy `0 <= i < j < terms.n_sites`; self-pairs are invalid;
- the coefficient must be finite;
- exact zero coefficients are ignored;
- every nonzero pair must be in `allowed_edges`;
- preserve its physical coefficient and map its endpoints through
  `site_to_coord` without changing it.

This validation is PEPS-only. Do not filter pairs in `compile_tn_terms()` and do
not restrict exact, MPS, or DMRG.

Position-noise realizations reuse the nominal Register and its private factory
provenance. Keep the same selected pair topology while allowing the canonical
lowering to change physical distances/directions/strengths; never infer a new
PEPS grid or edge set from noisy coordinates.

### 5.3 Error order

For real-time PEPS, the order is:

1. the top-level `simulate()` raw initial-state normalization plus common
   `t_eval`/observable request validation;
2. exact ten-key option validation;
3. Register provenance/layout validation;
4. `1r`/`01r` canonical TN compilation;
5. call `initial_local_amplitudes(terms, initial_state)` to validate labels or
   `"plus"` and produce the `(N, d)` amplitudes that the engine will consume;
6. observable capability validation;
7. compiled pair/topology validation;
8. lazy YASTN import and device probe;
9. tensor allocation and evolution.

For PEPS ground state, first validate the existing method/`1r`/`at`/raw initial
container contract, then run: nine-key options, Register layout, canonical TN
compilation, `validate_labels(terms, initial_state)`, fixed-CTM observable
capability, pair topology, lazy YASTN/device loading, and tensor allocation—in
that exact order.

An unsupported direct/triangular register must therefore fail even on a machine
without YASTN. Error messages must name the accepted factories and suggest
exact/MPS for unsupported geometry. Never fall back automatically.

## 6. Exact option validation

`_options.py` returns private `frozen=True, slots=True` records rather than
untyped dictionaries. Evidence later receives a canonical built-in-value copy.

Validation is fixed:

- option containers must implement `collections.abc.Mapping`; copy them to a
  canonical dict and require that key set to exactly equal the relevant schema;
- all integer fields accept Python/NumPy integers, reject booleans, and are
  at least one;
- real numeric fields accept Python/NumPy real numbers, reject booleans,
  strings, and complex values, and must be finite;
- `time_step_s > 0`;
- `0 < svd_tolerance < 1`;
- `ntu_iteration_tolerance > 0`, with no upper bound;
- `0 < environment_tolerance < 1`;
- `measurement_method` is exactly `"belief_propagation"` or `"ctm"`;
- `device` is exactly `"cpu"` or `"cuda"`;
- the ground schedule outer object and every entry must be tuples, not lists;
- each schedule entry has length two; `dtau` is finite and positive and is
  strictly smaller than the previous entry; steps are positive non-boolean
  integers.

Delete all PEPS uses of:

```text
time_step
discarded_weight_tolerance
truncation_error_tolerance
relative_energy_tolerance
_NTU_MAX_ITER
_NTU_TOL_ITER
min(user_tolerance, 1e-12)
```

No hidden clamp or default may rewrite a valid user input.

PEPS08 also already fixed the MPS real-time name. In
`backends/tenpy_mps/backends.py`, change only the MPS schema key
`time_step -> time_step_s`; also rename the private `plan_segments` formal
parameter/local variable to `time_step_s`, update usage/messages, and migrate all
MPS callers/tests. Keep MPS `bond_dimension`,
`discarded_weight_tolerance`, its fail-closed semantics, and DMRG unchanged.

## 7. YASTN runtime contract

Pin the optional dependency in `pyproject.toml` to the already locked and
audited commit:

```toml
"yastn @ git+https://github.com/yastn/yastn.git@30b1d8bb4dc691a25bf6394b061c564128ede8e0"
```

Regenerate `uv.lock` without upgrading unrelated packages. Add a focused
characterization test for the YASTN primitives this adapter relies on. Do not
add runtime version introspection or network access.

The device configuration is fixed:

- CPU: YASTN NumPy backend, `sym="none"`, `complex128`;
- CUDA: YASTN Torch backend, `sym="none"`, default device `"cuda"`,
  `complex128`;
- CUDA requests validate PyTorch import and `torch.cuda.is_available()` before
  allocating PEPS tensors;
- no silent device selection or CPU fallback.

The engine receives the already validated `(N, d)` initial local amplitudes
from preflight and only converts them to device tensors/product PEPS. It must not
defer physical-label or `"plus"` validation until after YASTN loading.

The fixed private NTU algorithm is:

```python
method = "mpo"
initialization = "EAT_SVD"
fix_metric = 0
pinv_cutoffs = (
    1e-12, 1e-11, 1e-10, 1e-9, 1e-8,
    1e-7, 1e-6, 1e-5, 1e-4,
)
opts_post_truncation = None
```

Every `evolution_step_` receives user values verbatim:

```python
opts_svd = {
    "D_total": bond_dimension,
    "tol": svd_tolerance,
}
max_iter = ntu_max_iterations
tol_iter = ntu_iteration_tolerance
```

Do not expose the fixed NTU choices as public options or accept `**kwargs`.

## 8. Real-time evolution semantics

Keep the existing second-order Strang decomposition and physical units:

```text
local exp(-i H_local(t_mid) dt / 2)
all Rydberg NN pair gates exp(-i V_ij n_r n_r dt)
local exp(-i H_local(t_mid) dt / 2)
```

Use Register site order for local gates and lexicographic `(i, j)` order for
pair gates. The local Hamiltonian remains the complete `1r`/`01r` matrix from
`TNTerms.local_hamiltonians(t_mid)`, including all DigitalAnalogProtocol `3x3`
couplings. Do not add TFIM-specific lowering or a PEPS Hamiltonian shortcut.
At every midpoint, require every local-Hamiltonian matrix element finite before
constructing a gate; do not let a later aggregate or matrix exponential mask a
NaN/Inf input.

`plan_segments` continues to split every interval into equal substeps no larger
than `time_step_s`, evaluated at each substep midpoint. Do not normalize the
real-time Hamiltonian.

After every YASTN update, validate every returned `Evolution_out.truncation_error`
as finite and nonnegative. Define the two evidence values exactly:

```text
max_ntu_truncation_error
    = maximum individual Evolution_out.truncation_error over the full run

one_step_error
    = max over bonds of sum(errors for repeated gates on that bond in this step)

cumulative_ntu_truncation_error
    = sum(one_step_error over all real-time steps)
```

An empty report contributes zero. Neither value is compared with a tolerance.
Large finite error or an exhausted NTU iteration cap still returns the final
estimate. Negative, NaN, or infinite error is a numerical validity failure.

Perform an O(N) final local tensor scan: every site tensor Frobenius norm must be
finite and strictly positive. Return the validated positive scales to the lazy
reader so amplitude does not repeat the scan. This is not a global norm
contraction and does not enter evidence.

## 9. Expectation environments

### 9.1 Fixed construction

For belief propagation:

```python
env = fpeps.EnvBP(psi)
out = env.iterate_(
    max_sweeps=environment_max_iterations,
    diff_tol=environment_tolerance,
)
```

For CTM:

```python
env = fpeps.EnvCTM(psi, init="eye")
out = env.iterate_(
    {
        "D_total": environment_bond_dimension,
        "tol": environment_tolerance,
    },
    moves="hv",
    method="2x2 corner",
    max_sweeps=environment_max_iterations,
    corner_tol=environment_tolerance,
)
```

Do not branch on `out.converged`. A finite high residual and cap exhaustion are
reportable estimates, not failures.

### 9.2 Summary semantics

Use a private summary `(residual: float | None, iterations: int)`:

- `iterations` is the returned positive `sweeps`, and may not exceed the user
  cap;
- BP residual is `max_diff`, which must be finite and nonnegative;
- CTM residual is `max_dsv`;
- CTM with exactly one returned sweep may have `max_dsv=NaN`; this one
  structural case becomes `None`;
- CTM after two or more sweeps with a nonfinite or negative residual fails;
- CTM `max_D` must be a positive integer no larger than
  `environment_bond_dimension`.

`environment_bond_dimension` does not control BP message dimensions. It remains
mandatory in a BP run because the same result may later need amplitude/norm
boundary contraction.

### 9.3 Measurement behavior

- Build no expectation environment when no observables were requested.
- At each requested measurement time, build one environment and share it across
  all requested expressions.
- BP expressions may contain any sum of terms acting on at most one distinct
  site each.
- CTM expressions may contain terms acting on at most two distinct sites each.
- Reject a BP two-site term before evolution; never fall back to CTM.
- Convert every complete Hermitian expectation to a real scalar only after all
  of its complex terms have been accumulated.

For real-time evidence, `environment_iterations` is the maximum over requested
measurement times. `environment_residual` is the maximum only when every
measurement produced an available residual; if any CTM one-sweep residual is
unavailable, it is `None`. With no observables, both fields are `None`.

The shared MPS `1e-6` real converter must not post-process PEPS results. PEPS
performs its own validity conversion before returning to `simulate_tn`; retain
the existing MPS behavior separately.

## 10. Mathematical validity rules

Define one private constant:

```python
_VALIDITY_RTOL = np.sqrt(np.finfo(np.float64).eps)
```

For a theoretically real complex scalar `z`, require:

```python
isfinite(z.real) and isfinite(z.imag)
abs(z.imag) <= _VALIDITY_RTOL * max(1.0, abs(z))
```

Then return `float(z.real)`. Apply this to:

- complete Hermitian expectations;
- complete physical ground energy;
- double-layer norm;
- raw conditional sampling weights.

This threshold is independent of `environment_tolerance`. Violating it is a
mathematical validity failure, not a convergence failure.

All state tensors, projected networks, boundary MPS objects, and environments
stay on the requested device. Never convert a tensor/network/environment to a
dense host object. Scalar `.item()` transfers are permitted only for:

- mathematical validity reductions such as site norms and conditional weights;
- conditional sampling RNG branching;
- final public amplitude/expectation/energy values;
- bounded evidence summaries.

Catch and contextualize the pinned YASTN tensor exception type. Do not use a
broad exception handler, retry with a different algorithm, or return a partial
result.

## 11. Normalized imaginary-time ground state

The method remains restricted to a fully specified `1r` system and explicit
initial physical labels.

Freeze `h_local = terms.local_hamiltonians(at)` and pair coefficients once.
First require every matrix element and pair coefficient finite. Compute each
local spectral norm separately, require every norm finite and nonnegative, then
compute the physical angular-frequency scale:

```python
Lambda = max(
    max(np.linalg.norm(h_local[i], ord=2) for i in range(N)),
    max((abs(V_ij) for _, _, V_ij in pairs), default=0.0),
)
```

`Lambda` must be finite and strictly positive. A zero Hamiltonian has no selected
ground-state representative and is a validity error.

Build imaginary-time gates from `h_local / Lambda` and `V_ij / Lambda`. For
every `(dtau, steps)` schedule entry, execute exactly `steps` second-order NTU
steps:

```text
local exp(-dtau h_tilde / 2)
all NN pair gates exp(-dtau V_tilde n_r n_r)
local exp(-dtau h_tilde / 2)
```

Do not:

- stop early;
- compare successive energies;
- construct a stage BP/CTM environment;
- calculate or store stage energies;
- reject a finite state because NTU error is large.

After the complete schedule:

1. run the local tensor validity scan;
2. construct exactly one final CTM using the ground options;
3. accumulate the full expectation of the original unscaled `h_local` and
   `V_ij` as a complex scalar;
4. apply the real validity check once to the complete energy;
5. use the same final CTM for every requested expectation;
6. return energy in `rad/s` as reserved `expectation("energy")`.

Retain this already computed final CTM privately in the ground-state reader for
lazy sampling. The sampling adapter must use temporary operator/window state and
must not mutate the retained CTM. Do not construct a second CTM merely to make
ground-state evidence.

Ground evidence stores `hamiltonian_scale_rad_s=Lambda`, the maximum individual
NTU error across the entire schedule, `cumulative_ntu_truncation_error=None`, and
the final CTM summary.

## 12. Boundary contraction and lazy amplitude

### 12.1 Remove the old implementation

Delete the hand-written `_single_layer_amplitude`, `_project_site`, and
`_compress_boundary` logic. It returns an unnormalized coefficient, uses the
state bond dimension for contraction, always sweeps one direction, and gates on
discarded weight; none of those semantics survive.

Never use `Peps.to_tensor()`, a full-state vector, a dense fallback, a reverse
sweep, direction averaging, or adaptive bond dimension.

### 12.2 Site rescaling and projected networks

Reuse the positive per-site Frobenius scales produced by the mandatory final
local validity scan. Keep only these small scalar scales in the reader; do not
rescan tensors and do not multiply the scales together.

Construct temporary networks without modifying the stored final PEPS:

- scaled double-layer source: each rank-5 tensor is `A_i / scale_i`;
- product-bra numerator: project the physical leg of `A_i / scale_i` onto the
  requested physical label, producing a rank-4 PEPS tensor.

Build temporary networks with `fpeps.Peps(psi.geometry, tensors=...)`. Use the
same positive scales for numerator and norm, so the unknown product cancels in
the normalized coefficient while the original complex global phase remains.

Use the pinned YASTN fused physical-leg convention. Construct the bra exactly
equivalent to:

```python
bra = (
    ops.vector(label)
    .add_leg(s=-1)
    .fuse_legs(axes=((0, 1),))
    .conj()
)
projected_site = yastn.tensordot(site_tensor, bra, axes=(4, 0))
```

The auxiliary leg signature is part of the adapter contract; contracting an
unfused `ops.vector(label).conj()` is invalid. Labels are placed through
`_PEPSLatticeSpec.site_to_coord` in Register order.

### 12.3 Deterministic orientation

For grid shape `(Nx, Ny)`:

- if `Nx <= Ny`, use `transfer_mpo(n=0..Ny-1, dirn="v")`;
- if `Nx > Ny`, use `transfer_mpo(n=0..Nx-1, dirn="h")`;
- a square therefore uses vertical transfers;
- all layers run from low coordinate to high coordinate.

Numerator and norm use the identical orientation and order.

### 12.4 Exact boundary algorithm

Locally implement the small identity transfer-boundary constructor used by the
pinned YASTN revision, using public tensor/MPS operations (`mps.Mps`,
`yastn.eye`, `yastn.ones`) rather than importing YASTN's private
`identity_tm_boundary`.

For the ordered transfer list:

1. create the low-side identity MPS from the first transfer;
2. create the high-side identity MPS from the transposed final transfer;
3. for every transfer except the final one, absorb it with:

   ```python
   next_boundary, discarded = mps.zipper(
       transfer,
       boundary,
       opts_svd={
           "D_total": environment_bond_dimension,
           "tol": environment_tolerance,
       },
       normalize=False,
       return_discarded=True,
   )

   compression_out = mps.compression_(
       next_boundary,
       (transfer, boundary),
       method="1site",
       overlap_tol=environment_tolerance,
       max_sweeps=environment_max_iterations,
       opts_svd={
           "D_total": environment_bond_dimension,
           "tol": environment_tolerance,
       },
       normalize=False,
   )
   ```

4. finish with `mps.vdot(high_boundary.conj(), final_transfer, boundary)`.

Both zipper and compression must use `normalize=False`; otherwise the complex
contraction factor is lost. Use exactly `environment_bond_dimension`, not its
square and not state `bond_dimension`. `svd_tolerance` never enters readout
contraction.

Define one contraction's evidence error as:

```text
max(
    every finite nonnegative zipper discarded ratio,
    abs(final compression_out.doverlap) for every absorbed layer,
    default=0.0,
)
```

Do not include `max_dSchmidt`, `max_discarded_weight`, a sum, or any hidden
instrumentation. The value is a heuristic summary, not a physical error bound.
Cap exhaustion does not fail; a nonfinite/negative summary or nonfinite final
scalar does.

### 12.5 Norm, coefficient, cache, and gauge

The first successful amplitude request performs one double-layer norm
contraction and caches `(positive_norm, norm_contraction_error)`. The norm must
be finite, real under the validity rule, and strictly positive.

For each distinct physical label tuple, perform at most one product-bra
contraction and cache:

```python
(normalized_coefficient, contraction_error)
```

where:

```python
normalized_coefficient = raw_product_bra / np.sqrt(norm)
```

Require only a finite complex coefficient. If an approximate contraction
returns `abs(coefficient) > 1`, return it unchanged; do not clamp or gate it.

Evolution amplitude returns the cached coefficient directly. Ground-state
amplitude resolves coefficients in this order:

1. `phase_reference`;
2. requested target, unless it is the same label tuple.

Require `abs(reference) > _VALIDITY_RTOL`, then return:

```python
target / (reference / abs(reference))
```

The reference is then nonnegative real in that gauge. A different reference
reuses any previously cached label coefficient.

Use one private `threading.RLock` around cache/ledger transactions. A public
amplitude call stages all new norm/coefficient values and evidence records, then
commits them atomically only after norm, reference, target, and phase checks all
succeed. A failed call publishes no norm error, coefficient cache entry, or
amplitude evidence record. Validated site scale caching may remain because it is
not a result or evidence value.

## 13. Validated lazy sampling

Do not call `EnvBP.sample()` or `EnvCTM.sample()` as the result implementation.
The pinned YASTN sampler uses backend RNG and the CTM path takes `.real` before
the caller can validate all candidate weights.

In `_sampling.py`, implement two private sequential conditional adapters by
porting only the necessary single-sample loops from YASTN commit
`30b1d8bb4dc691a25bf6394b061c564128ede8e0`. Keep an Apache attribution comment
beside the port. The CTM adapter uses `fpeps.EnvWindow` and public MPS
environment operations; the BP adapter uses copied BP local messages and never
constructs `EnvWindow` or CTM. Both own candidate-weight validation and RNG. Do
not import or call YASTN's private `_sample` function.

The shared candidate normalization is fixed. For candidate complex weights
`z_i` in `terms.levels` order:

1. require every real and imaginary part finite;
2. for each `z_i`, require
   `abs(imag_i) <= _VALIDITY_RTOL * max(1, abs(z_i))`;
3. set `w_i = real_i`;
4. for each candidate define its own
   `negative_slack_i = _VALIDITY_RTOL * max(1, abs(z_i))`;
5. clamp `-negative_slack_i <= w_i < 0` to zero;
6. reject any `w_i < -negative_slack_i`;
7. require `total = sum(w_i)` finite and strictly positive;
8. set `p = w / total`;
9. require `p_sum = sum(p)` finite and strictly positive, then set
   `p = p / p_sum` unconditionally;
10. require `cdf = np.cumsum(p)` finite and nondecreasing, then set
    `cdf[-1] = 1.0` exactly.

Given `u = rng.random()`, select
`index = min(np.searchsorted(cdf, u, side="right"), d - 1)` and require the
selected `p[index]` to be strictly positive before conditioning by it. This
prevents an exact `u == 0` from selecting a leading zero-probability outcome.
Use a heterogeneous-scale regression such as a very large positive candidate
beside a material negative candidate to prove one candidate cannot enlarge
another candidate's roundoff slack.

Use a local `np.random.default_rng(seed)` and draw one scalar at each conditional
step in the fixed traversal order. Do not retain or allocate a `shots * N`
diagnostic trace. Never call `np.random.seed`, `torch.manual_seed`, or a backend
global RNG.

Sampling order is deterministic:

- BP and CTM both use the same short-boundary orientation as amplitude: columns
  first when `Nx <= Ny`, rows first otherwise, always low-to-high within each
  layer;
- returned outcome tuples are always reordered into Register site order;
- candidate labels always follow `terms.levels` order.

The selected real-time `measurement_method` determines the sampler; never
fallback from BP to CTM. CTM boundary updates use exactly:

```python
opts_svd = {
    "D_total": environment_bond_dimension,
    "tol": environment_tolerance,
}
opts_var = {
    "method": "1site",
    "overlap_tol": environment_tolerance,
    "max_sweeps": environment_max_iterations,
    "normalize": True,
}
```

The CTM sampling zipper also uses `normalize=True`. Sampling needs conditional
probability ratios, not the complex global contraction factor; normalized
conditioned boundaries avoid scale overflow. The `normalize=False` rule in the
amplitude section applies only to amplitude/norm contractions.

After selecting a candidate, both adapters insert the matched physical
projector divided by its selected probability, `P_selected / p_selected`, and
then update only downstream conditional messages/boundaries. Each shot starts
from independent local copies and restores all temporary operators.

For BP, locally implement the pinned fused-ancilla projector matching using
public `eye`/`tensordot`/`fuse_legs` operations; a rank-2 physical projector must
be tensored with identity on the trivial auxiliary leg before matching the
fused PEPS physical leg. Also use a local private record containing copied
`tR/lR/bR/rR` messages rather than importing YASTN's unexported `EnvBP_local`.
Every downstream QR `R.norm()` must be finite and strictly positive before
normalizing the message.

For CTM, use `fpeps.EnvWindow` plus public MPS environment operations. Set and
restore the temporary transfer-MPO operator at every conditioned site; after a
layer, update the downstream boundary with the fixed normalized zipper and
compression controls above.

An exhausted cap remains report-only as long as every conditional distribution
is valid. Ground-state sampling uses the retained final CTM. Real-time sampling
constructs the selected final-state environment lazily; its residual is not
added to the eager environment summary because PEPS evidence intentionally
stores only `(shots, seed)` for sampling.

Return `Counter[tuple[str, ...]]`. Each successful public sample call appends
one evidence record, including repeated calls with identical arguments. A call
that fails at any site returns nothing and appends nothing. Serialize sampling
and evidence mutation with the reader's `RLock`.

## 14. Evidence contract

Add these public records to `src/ryd_gate/results.py`:

```python
@dataclass(frozen=True, slots=True)
class PEPSAmplitudeEvidence:
    labels: tuple[str, ...]
    contraction_error: float


@dataclass(frozen=True, slots=True)
class PEPSSampleEvidence:
    shots: int
    seed: int


@dataclass(frozen=True, slots=True)
class PEPSEvidence:
    parameters: Mapping[str, object]
    hamiltonian_scale_rad_s: float | None
    max_ntu_truncation_error: float
    cumulative_ntu_truncation_error: float | None
    environment_residual: float | None
    environment_iterations: int | None
    norm_contraction_error: float | None
    amplitudes: tuple[PEPSAmplitudeEvidence, ...]
    samples: tuple[PEPSSampleEvidence, ...]

    def to_dict(self) -> dict[str, object]: ...
```

Add all three names to `ryd_gate.results.__all__`; do not export them from the
top-level `ryd_gate` package.

Deep immutability must also hold when a caller constructs these public records
directly, not only for ledger-produced snapshots. Each dataclass therefore uses
`__post_init__` (with `object.__setattr__`) or an equivalent defensive public
constructor to:

- copy/canonicalize `labels`, `amplitudes`, and `samples` to tuples;
- recursively copy/freeze `parameters` into a new read-only mapping whose nested
  containers are immutable;
- canonicalize NumPy scalar inputs to built-in scalars;
- validate field types, finite/nonnegative error summaries, positive shots, and
  non-negative seeds.

Mutating a source dict/list after record construction must not affect the
record. Do not expose a separate unsafe dataclass constructor plus a safe private
factory.

The private ledger owns canonical eager fields and append-only lazy records.
Its `snapshot()` creates a fresh `PEPSEvidence` every time. Deep immutability
means:

- copy and wrap `parameters` in a new read-only mapping;
- preserve the schedule and all record collections as tuples;
- use only built-in immutable scalar values;
- never expose a list/dict owned by the ledger.

`to_dict()` is the only serializer. It returns a fresh JSON-compatible deep
copy, recursively converting mappings/tuples/records to plain dict/list values.
It performs no I/O and there is no `from_dict`, `json`, `save`, convergence
boolean, or generic metadata bag.

The exact field semantics are:

| Field | Real time | Ground state |
|---|---|---|
| `parameters` | exact canonical ten-key mapping | exact canonical nine-key mapping |
| `hamiltonian_scale_rad_s` | `None` | finite positive `Lambda` |
| `max_ntu_truncation_error` | maximum individual report | maximum individual report |
| `cumulative_ntu_truncation_error` | defined step aggregation | `None` |
| `environment_residual` | worst eager measurement residual, or `None` | final CTM residual, or `None` |
| `environment_iterations` | maximum eager measurement sweeps, or `None` | final CTM sweeps |
| `norm_contraction_error` | `None` until first successful amplitude | same |
| `amplitudes` | distinct successful coefficient contractions | same, including phase references |
| `samples` | one record per successful call | one record per successful call |

Amplitude evidence is a ledger of actual distinct coefficient contractions,
not public call pairs. A ground phase reference therefore receives its own
ordinary labels record. Cached labels do not create duplicate records.

Add `peps_evidence` to both result types. Keep the engine return seam a triple:

```python
(out_times, expectations, reader)
```

The PEPS reader owns the private ledger and implements a private
`_peps_evidence_snapshot()` method. The result property calls that method when
present and otherwise returns `None`. This avoids a backend-specific public
constructor argument and keeps lazy cache/ledger transactions co-located.
Calling the snapshot method must not build an environment, norm, amplitude, or
sample.

`EnsembleResult` does not aggregate evidence; each child PEPS result owns its
own ledger.

At the common result boundary, `shots` is a positive integer and `seed` is a
non-negative integer for exact, MPS, PEPS, DMRG, and PEPS ground-state sampling.
Reject booleans, negative seeds, and other types before dispatch. Each backend
then uses that validated seed with a local RNG.

## 15. Dispatcher and integration changes

### `src/ryd_gate/backends/tn_common/simulate.py`

- import the new private PEPS package instead of `peps2d`;
- run PEPS option/layout/observable/topology preflight before engine loading;
- let PEPS return already validated real expectation arrays;
- retain the current shared real conversion only for MPS;
- construct `EvolutionResult` with the existing reader seam.

### `src/ryd_gate/backends/ground_state.py`

- delete the unused duplicate `_PEPS_KEYS`;
- keep DMRG option validation and convergence behavior unchanged;
- import the new PEPS package only in the PEPS branch;
- use the identical Register/topology preflight as real time;
- do not add a PEPS measurement-method key; ground state is fixed to CTM.

### `src/ryd_gate/results.py`

- add the three evidence records and result properties;
- make the common sample validator reject negative seeds before any backend
  reader is called;
- keep result physical readout methods unchanged;
- keep top-level package exports unchanged.

### `src/ryd_gate/backends/tenpy_mps/backends.py`

- perform only the confirmed `time_step_s` rename;
- preserve all MPS algorithms, error thresholds, and result behavior.

## 16. Implementation slices

Implement in this order. Do not combine later numerical work into an earlier
slice merely to make old tests green.

### Slice 0 — characterize and protect

- record worktree/base/protected-file state;
- capture the current public top-level names and Register public surface;
- identify and mark old PEPS tests that assert convergence failures for
  replacement, not preservation;
- pin the exact YASTN commit in `pyproject.toml`/`uv.lock`.

Done when dependency resolution changes only YASTN metadata and guardrails are
clean.

### Slice 1 — contracts without YASTN

- add Register private provenance;
- add `_PEPSLatticeSpec` and topology validation;
- implement exact real-time/ground option records and validators;
- rename MPS `time_step_s` and migrate its focused tests;
- add evidence dataclasses, immutable snapshots, and `to_dict()`;
- add dependency-free tests.

Done when all schema, geometry, topology, immutability, and MPS rename tests pass
without importing YASTN.

### Slice 2 — module extraction and runtime seam

- create the private `backends/peps/` package;
- move the working operator/product-state/gate code without changing its
  physics;
- change both dispatchers to the new package;
- delete `peps2d.py` after all imports move;
- explicitly pin device/dtype and NTU fixed choices.

Done when a minimal supported chain evolution imports/runs and no source/caller
references `ryd_gate.backends.peps2d`.

### Slice 3 — report-only real-time NTU and environments

- implement user NTU controls and exact error aggregation;
- remove all truncation/convergence gates and hidden clamps;
- implement BP/CTM summaries and PEPS real validity conversion;
- attach the eager ledger to the PEPS reader;
- test high finite residual/high NTU error returning with evidence.

Done when old fail-closed tests are replaced by report-only assertions and no
PEPS convergence threshold raises remain.

### Slice 4 — normalized ground state

- compute `Lambda`;
- execute the complete normalized schedule;
- remove all stage CTM/energy logic;
- build one final same-device CTM;
- compute physical energy/expectations and ground evidence;
- retain final CTM for sampling.

Done when a small `1r` PEPS energy and requested observables agree with exact
diagonalization within explicitly justified integration-test tolerances, and
instrumentation proves every schedule step and exactly one CTM.

### Slice 5 — normalized lazy amplitude

- implement positive site rescaling;
- implement deterministic YASTN boundary contraction;
- implement lazy norm/coefficient caches and ground phase gauge;
- implement atomic evidence transactions;
- delete the old manual contraction.

Done when complex magnitude/phase, rescaling invariance, caching, error summary,
and no-dense tests pass on chain and 2D rectangle/square cases.

### Slice 6 — validated lazy sampling

- implement local BP and CTM conditional adapters;
- use local NumPy RNG and fixed ordering;
- validate every candidate distribution;
- integrate transactional sample evidence;
- remove all direct `env.sample()` and global RNG seeding.

Done when BP/CTM sampling is reproducible, preserves global RNG state, uses
Register label order, rejects invalid conditional weights, and never densifies.

### Slice 7 — callers and documentation

- migrate examples, scripts, notebooks, README, docs, and capability matrix;
- replace guarded/under-rewrite text with runnable workflows;
- demonstrate evidence collection and caller-owned convergence studies;
- remove stale notebook outputs that claim the backend is unavailable.

Done when all retained PEPS-capable callers use supported Register factories,
explicit nearest-neighbour physical cutoffs, and final schemas; exact/MPS-only
callers retain their intended physical interaction range.

### Slice 8 — full verification

- run all fast tests, lint, type checks, docs tests, and stale-name audits;
- run real YASTN CPU parity tests;
- run CUDA and retained large-workflow checks on the DGX;
- render docs/notebooks as applicable;
- confirm guardrails and final diff.

Do not stage or commit after verification.

## 17. Test plan

### 17.1 Dependency-free contract tests

Put these outside any module-level `pytest.importorskip("yastn")`:

- exact ten/nine key sets; missing/unknown/old keys rejected;
- booleans, strings, complex tolerances, zero, NaN, and infinity rejected;
- `svd_tolerance >= 1` and `environment_tolerance >= 1` rejected;
- `ntu_iteration_tolerance > 1` accepted;
- tuple-only, strictly decreasing ground schedule;
- chain/rectangle/square provenance, shape, mapping, and edges;
- direct coordinates identical to a factory rectangle still rejected;
- triangular, rotated, hole, duplicate, diagonal, long-range, and wraparound
  cases rejected;
- subset/zero interactions accepted;
- invalid geometry/topology fails before a sentinel YASTN loader is touched;
- with YASTN import deliberately blocked, custom/triangular/invalid-topology
  requests still raise their PEPS capability errors rather than `ImportError`;
- invalid explicit labels or an invalid initial-state shorthand fail before the
  YASTN loader/tensor allocator;
- exact/MPS remain capable of the same custom/triangular Register where
  otherwise supported;
- provenance survives a noise realization without being recomputed;
- no new public Register attribute;
- evidence deep immutability, old-snapshot stability, and JSON `to_dict()`;
- public evidence constructors defensively freezing caller-owned nested
  dict/list inputs;
- exact/MPS/DMRG `peps_evidence is None`;
- negative sampling seeds rejected consistently by every result/backend;
- NTU max/cumulative aggregation with empty and repeated-bond fake reports;
- CTM one-sweep residual `None`, later nonfinite residual failure;
- Hermitian validity boundary;
- conditional-weight normalization and negative-roundoff clamping.

Rotated/hole/duplicate direct-coordinate cases must all fail at the same
`factory="custom"` provenance gate. Their tests must not inspect or expect a
coordinate-specific diagnosis, because that would reintroduce shape inference.

### 17.2 Real YASTN CPU tests

Move `pytest.importorskip("yastn")` into an integration fixture or individual
tests. Cover:

- `Register.chain`, nonsquare `Register.rectangle`, and `Register.square`;
- real-time `1r` and full `01r` local `3x3` Hamiltonian paths;
- second-order gate ordering and physical `time_step_s` anchor behavior;
- BP one-site expectation and CTM one-/two-site expectation;
- multiple observables sharing one environment at one time;
- no observables causing no environment construction;
- high finite NTU/environment residual returning with evidence;
- final local tensor validity scan;
- normalized product-state and Bell-like complex amplitudes;
- a regression where raw PEPS scale is not one but normalized amplitude is;
- exact complex phase parity, not merely probability parity;
- `1x1`, `1xN`, `Nx1`, square, and both rectangular orientation branches;
- monkeypatched `Peps.to_tensor()` failure while amplitude still succeeds;
- numerator and norm using identical boundary controls and `normalize=False`;
- contraction error taking the exact specified maximum;
- norm once, labels once, ground reference/target cache semantics;
- failed amplitude leaving ledger/cache transaction unchanged;
- `abs(amplitude)>1` finite fake estimate returned unchanged;
- BP and CTM sampling, seed reproducibility, global RNG preservation, outcome
  ordering, and failure transaction;
- heterogeneous-scale conditional weights, leading zero-probability candidates,
  fused-ancilla projector matching, and finite-positive BP QR normalization;
- normalized `H/Lambda` ground state vs `scipy.sparse.linalg.eigsh` for a small
  system: energy, one-/two-site expectation, phase-referenced amplitude, sample;
- all schedule steps and exactly one final CTM;
- CPU device retained through lazy readouts.

Do not assert monotonic convergence across two approximate parameter settings.
Instead, run both, return both physical estimates/evidence, and prove the
caller-owned comparison workflow works.

### 17.3 YASTN compatibility characterization

Against the pinned commit, characterize:

- `Peps.transfer_mpo`;
- `mps.zipper(..., normalize=False, return_discarded=True)`;
- `mps.compression_(method="1site", overlap_tol=...,
  max_sweeps=..., normalize=False)` and `doverlap`;
- `mps.vdot`;
- BP `sweeps/max_diff/converged`;
- CTM `sweeps/max_dsv/max_D/converged`;
- the known first-sweep CTM `max_dsv=NaN` behavior;
- `fpeps.EnvWindow` boundary indexing and temporary-operator restoration;
- `DoublePepsTensor.set_operator_`/operator removal behavior used by the CTM
  adapter;
- `mps.Env.measure` and `mps.Env.update_env_` used during sequential CTM
  conditioning;
- BP local `tR/lR/bR/rR` message copying, fused-ancilla projector matching,
  and downstream QR updates;
- real BP and CTM sampling smoke tests in both rectangular aspect-ratio
  traversal branches;
- tensor/device preservation for the NumPy and Torch backends.

This test protects the intentionally narrow adapter. Do not generalize it into
support for arbitrary YASTN revisions.

### 17.4 DGX verification

Run at least:

- `4x4` CUDA real-time `1r` expectation, amplitude, and sampling;
- `4x4` CUDA real-time `01r` full local Hamiltonian smoke;
- `4x4` CUDA imaginary-time ground state with energy, requested expectation,
  amplitude, and sampling;
- the notebook-04 `10x10` finite-PEPS workflow as the single maintained large
  DGX validation case, with parameters and evidence written under a temporary
  validation directory outside the repository;
- CPU/CUDA comparison on a small deterministic case;
- all slow PEPS tests and the existing slow physics audit in a writable ARC
  environment.

Confirm with instrumentation that lazy tensor contractions stay on CUDA and
only permitted scalar reductions reach the host.

The maintained large-case entry is fixed. Implement
`scripts/run_peps_10x10.py` as an unconditional live run with no saved-result
fallback and no broad exception handler. Its default physical case is the
existing notebook-04 case:

```text
Register.rectangle(10, 10, spacing_um=5.0)
level_structure("1r", ryd_level=70)
interaction_cutoff_um=5.0
Omega = 2*pi*380 MHz
detuning = smooth round trip -2*pi*10 MHz -> +2*pi*10 MHz -> -2*pi*10 MHz
t_gate = 0.15 us
t_eval = 7 equally spaced points including both endpoints
initial state = |1>^100
```

Its default PEPS controls are:

```python
{
    "time_step_s": 0.15e-6 / 250,
    "bond_dimension": 8,
    "svd_tolerance": 1e-8,
    "ntu_max_iterations": 20,
    "ntu_iteration_tolerance": 1e-10,
    "measurement_method": "belief_propagation",
    "environment_bond_dimension": 32,
    "environment_tolerance": 1e-8,
    "environment_max_iterations": 50,
    "device": "cuda",
}
```

The script accepts `--device` and `--output-dir`; numerical values remain clear
named constants in the script rather than a generic kwargs parser. The DGX
acceptance command is:

```bash
.venv/bin/python scripts/run_peps_10x10.py \
  --device cuda \
  --output-dir /tmp/ryd_gate_peps_validation/10x10
```

Success means exit zero, exactly seven finite real mean/per-site Rydberg
occupation records, and JSON-compatible PEPS evidence whose parameters equal
the controls above. Write arrays and evidence only beneath the requested
temporary output directory; never write this validation run to repository
`results/`. Notebook 04 should present/import this maintained workflow without
executing it as part of ordinary low-resource docs rendering; the explicit
script command is the production resource opt-in, not an “under rewrite” guard.

## 18. Caller and documentation migration

Audit and migrate at least:

- `examples/demo_local_addressing_tn.py`;
- `scripts/bench_quench_check.py`;
- notebooks `03_lattice_dynamics_annealing.ipynb`,
  `04_quench_and_state_prep.ipynb`, and `05_tn_and_error_budget.ipynb`;
- README;
- `docs/getting_started.qmd`;
- `docs/fundamentals.qmd`;
- `docs/hamiltonians.qmd`;
- `docs/capability_matrix.qmd` and its generator/tests.

Requirements:

- build MPS and PEPS option dictionaries in separate branches;
- use `time_step_s` for both real-time TN backends;
- use only chain/rectangle/square factories for PEPS;
- only in workflows intended to run on PEPS, explicitly select the intended
  nearest-neighbour physical Hamiltonian through `interaction_cutoff_um` and
  explain that this changes the system, not just the backend approximation;
  do not add an NN cutoff to unrelated exact/MPS research workflows that retain
  long-range physics;
- remove old `chi_max`, `dt`, `svd_min`, `measurement_environment`,
  `discarded_weight_tolerance` PEPS keys, and all other engine-shaped inputs;
- remove broad `except Exception`, `NotImplementedError` guards, PEPS xfails,
  and “backend under rewrite” text;
- create the already-documented `scripts/run_peps_10x10.py` as the real
  headless DGX entry described below; remove saved-result fallback and false
  “under rewrite” behavior from notebook 04;
- show `result.peps_evidence.to_dict()` as provenance, not a convergence
  certificate;
- show convergence studies as multiple complete simulations with changed
  physical/numerical controls and caller-side comparison of energies,
  expectations, or amplitudes;
- keep analysis and plotting of those comparisons in scripts/notebooks, not
  `src`.

The capability matrix must qualify PEPS support as:

```text
1r / 01r real time: chain, rectangle, or square OBC Register with graph-local NN pairs
1r ground state:    same geometry/topology restriction
```

Do not state or imply that `Register.triangular` is globally invalid; only PEPS
rejects it.

## 19. Verification commands

Use the repository environment and run, in increasing cost order:

```bash
pytest -q tests/lattice/test_register.py
pytest -q tests/core/test_evolution_result.py
pytest -q tests/backends/test_peps_preflight.py
pytest -q tests/backends/test_peps_contracts.py
pytest -q tests/backends/test_tn_tdvp.py
pytest -q tests/backends/test_tn_yastn_peps_backend.py
pytest -q
ruff check src tests examples
mypy src/ryd_gate
pytest -q tests/docs
git diff --check
```

Use the actual filenames if focused tests are grouped slightly differently, but
keep dependency-free contracts separate from YASTN integration tests.

On the DGX, explicitly override the repository's default `not slow` filter when
running slow tests, then render the maintained notebooks/docs. Record exact
commands and results in Claude's handoff; do not put logs or convergence curves
inside `src`.

Final stale-name audit:

```bash
rg -n "peps2d|time_step\b|discarded_weight_tolerance|relative_energy_tolerance|_NTU_MAX_ITER|_NTU_TOL_ITER|_require_env_converged|env\.sample|under rewrite" src tests examples scripts README.md docs
```

Interpret matches by subsystem: MPS/DMRG legitimately retain
`discarded_weight_tolerance` and DMRG retains `relative_energy_tolerance`.
Generated docs output is not an implementation source.

## 20. Acceptance checklist

The refactor is complete only when every item is true:

- [ ] PEPS accepts only factory-provenance chain/rectangle/square registers.
- [ ] Direct coordinates and triangular registers fail before YASTN loading.
- [ ] PEPS uses OBC and only compiled nonzero Cartesian NN pair terms.
- [ ] No topology is inferred from floating coordinates.
- [ ] Real-time PEPS has exactly ten mandatory options.
- [ ] Ground-state PEPS has exactly nine mandatory options.
- [ ] MPS real time consistently uses `time_step_s`; MPS/DMRG numerics are
      otherwise unchanged.
- [ ] User SVD/NTU/environment controls reach YASTN without hidden clamps.
- [ ] High finite NTU/BP/CTM/boundary residuals do not gate a result.
- [ ] NaN/Inf, invalid realness, nonpositive norm, and invalid probabilities do
      fail clearly.
- [ ] Ground state uses `H/Lambda`, executes the whole schedule, and builds one
      final CTM.
- [ ] Energy is the unscaled physical `rad/s` expectation.
- [ ] Amplitude is lazy, normalized, complex, cached, deterministic in
      contraction convention, and never dense.
- [ ] Ground phase reference is label-based and transactionally cached.
- [ ] Sampling is lazy, BP/CTM-specific, mathematically validated, locally
      seeded, Register-ordered, and never dense.
- [ ] Every result rejects a negative sampling seed at the common boundary.
- [ ] PEPS evidence is bounded, immutable, append-only by successful lazy
      operation, and JSON-convertible.
- [ ] Exact/MPS/DMRG evidence is `None`; EnsembleResult does not aggregate it.
- [ ] No PEPS `NotImplementedError`, xfail, broad caller guard, or “under
      rewrite” path remains.
- [ ] Chain, rectangle, square, `1r`, `01r`, CPU, and CUDA paths have real tests.
- [ ] A maintained larger 2D workflow runs on the DGX.
- [ ] Top-level `ryd_gate` exports and public Register surface did not expand.
- [ ] No arbitrary unit-disk-graph backend placeholder or public graph API was
      added.
- [ ] `simplify.md` is byte-identical; `algsummary.tex` remains absent.
- [ ] Nothing is staged or committed.

## 21. Explicit non-goals

Do not implement or pre-design through public API:

- arbitrary unit-disk-graph tensor networks;
- a graph-PEPS mode;
- triangular-to-square embeddings;
- SWAP networks, PEPO long-range terms, graph edge coloring, or generic graph
  Trotter scheduling;
- periodic/cylindrical geometry;
- `Register.edges`, adjacency, topology, boundary, stored spacing, or a second
  geometry object;
- automatic interaction cutoff selection;
- automatic backend fallback;
- adaptive PEPS bond dimensions or internal convergence studies;
- a convergence certificate or `converged` result field;
- dense amplitude/sampling fallback;
- generic YASTN kwargs;
- evidence expansion for exact, MPS, or DMRG;
- result analysis, convergence plots, file I/O, or report generation in `src`.

The future unit-disk-graph TN should eventually consume the same canonical
compiled Hamiltonian/pair IR and result contracts, but it must be designed as a
separate algorithm after this PEPS adapter is complete. Nothing in this refactor
should constrain or pretend to implement it.
