# Rydberg Simulation

This context describes the physical and numerical language used by `ryd_gate` to distinguish quantum-simulation results from evidence about their tensor-network approximation.

## 01r control

**Computational-return pulse**:
A two-atom `01r` control pulse intended to leave the computational subspace during the evolution and return it there at the endpoint while accumulating logical phases. Its optimization is judged by endpoint return, leakage, and phase objectives; adiabatic evolution may motivate the initial pulse but is not a constraint or a claimed property.
_Avoid_: Treating an adiabatic-motivated search as an adiabaticity certificate, certified adiabatic path

**Computational endpoint**:
A pulse endpoint at which the `1-r` coupling vanishes, so the bare computational subspace is invariant under the endpoint Hamiltonian. The drive phase at that zero-amplitude endpoint is a gauge choice, not a physical boundary condition.
_Avoid_: Equal endpoint drive phase

**Minimum-entangling phase gate**:
A computational-return two-qubit phase gate whose conditional phase is not fixed to one target angle but must satisfy a configurable minimum entangling strength, `C_phi = |sin(Phi_ZZ / 2)|`. The default acceptance threshold is `C_phi >= 0.5`; above it, optimization is free to choose the conditional phase.
_Avoid_: Any nonzero phase, fixed-angle phase gate

**Return-weighted entangling score**:
The branch-free pulse-search score `S_phi = |q| C_phi^2`, where `q` is the gauge-invariant product of the four computational return amplitudes. It suppresses apparent phase from candidates with poor computational return; final gate acceptance still reports `C_phi` and endpoint leakage separately.
_Avoid_: Raw angle loss, unweighted phase of a leaky candidate

**Thresholded entangling penalty**:
The normalized squared-hinge penalty on the return-weighted entangling score. It equals one for a trivial phase, decreases to zero at the minimum accepted entangling strength, and remains exactly zero above that threshold so the search does not prefer a particular nontrivial angle.
_Avoid_: Fixed-angle phase loss, above-threshold phase reward

**Mean return loss**:
The average, over all four computational inputs, of one minus the probability to return to the same computational basis state. It is the smooth endpoint-return objective used during pulse search.
_Avoid_: Worst-case optimization loss

**Worst-case endpoint leakage**:
The largest same-input return loss among the four computational inputs. It is reported per candidate and used for final acceptance, not as the differentiable search objective. The default acceptance threshold is `1e-4`.
_Avoid_: Mean-only gate acceptance

**Endpoint-only pulse search**:
A pulse search whose evolution-dependent objective uses only final computational return amplitudes and entangling phase. Intermediate Rydberg occupation is allowed and is not penalized during the search.
_Avoid_: Rydberg-exposure optimization

**Rydberg exposure diagnostic**:
The time integral of total Rydberg occupation, evaluated for every logical input only when a continuation stage's selected pulse undergoes exact-backend validation. Per-input, mean, and worst-case exposures may be combined with a decay rate by the caller, but they are not part of the endpoint-only search objective or its iteration history.
_Avoid_: Endpoint leakage, built-in decay error

**Fixed-duration pulse search**:
A control-pulse search performed at one prescribed gate duration. Gate duration is varied only between separate searches, so a candidate cannot improve endpoint return merely by extending its own evolution time.
_Avoid_: Free-duration pulse search

**Pulse-duration continuation**:
A sequence of fixed-duration pulse searches that first finds an accepted computational-return pulse at a deliberately long duration, then progressively shortens the duration and warm-starts each search from the preceding pulse. Each shortened candidate is re-optimized and independently accepted or rejected; the sequence seeks the shortest accepted duration and does not itself certify that the path is adiabatic.
_Avoid_: Independent duration sweep, free-duration optimization, certified adiabatic compression

**Explicit long-running continuation**:
A duration-continuation search that is present in the notebook but disabled during its default complete execution. The default workflow validates the initial-duration branches; a clearly named switch must be enabled before the longer compression search runs.
_Avoid_: Automatic long search on Run All, omitted continuation workflow

**Continuation branch**:
A lineage of accepted computational-return pulses connected by pulse-duration continuation. Multiple branches may be retained from distinct long-duration seeds so that failure to compress one local solution is not mistaken for a physical minimum gate duration.
_Avoid_: Optimizer restart, proof of a speed limit

**Analytic seed branch pool**:
A bounded collection of public-backend-validated, physically distinct gapped seeds at the initial duration. The pool retains diversity in control scales and conditional phases rather than selecting seeds by leakage alone; each retained seed initializes a separate continuation branch. Random whole-waveform starts are not required to populate the pool.
_Avoid_: Random seed pool, duplicate local perturbations

**Polished continuation seed**:
An accepted computational-return pulse that continues to be optimized at its fixed duration before it initializes a shorter-duration search. Gate acceptance marks feasibility, not the end of polishing; the additional optimization supplies compression margin without changing the acceptance criteria.
_Avoid_: First-feasible seed, stricter hidden acceptance threshold

**Gapped adiabatic seed**:
A control path whose initial state belongs to an isolated instantaneous eigenbranch and whose relevant gap remains open in the explicitly chosen frame, so slower traversal can suppress transitions away from that branch. A long resonant pulse with a zero-coupling endpoint degeneracy is not a gapped adiabatic seed merely because its duration is large.
_Avoid_: Slow pulse, long-duration seed, guaranteed adiabatic random spline

**Endpoint-gapped control**:
An amplitude-chirp pulse whose coupling vanishes at both endpoints while its endpoint chirp remains nonzero and equal, leaving the bare computational and Rydberg states spectrally separated there. This boundary guarantee does not assert that the optimized path remains gapped or adiabatic in its interior.
_Avoid_: Globally gapped path, adiabaticity certificate

**Optimized endpoint chirp**:
The common nonzero chirp at both computational endpoints, varied as one bounded control coordinate independently of the interior chirp correction. Its allowed interval preserves a minimum endpoint gap while permitting duration continuation to retune the path's detuning scale.
_Avoid_: Fixed seed detuning, independent endpoint detunings

**Amplitude-chirp control**:
A `1-r` drive parameterized by a nonnegative Rabi amplitude and a real instantaneous chirp. The complex-drive phase is the time integral of the chirp, rather than an independent optimization coordinate; a separate freely optimized Rydberg detuning is not added on top of the same phase freedom.
_Avoid_: Independent Cartesian quadratures, simultaneous free phase and detuning

**Gapped-seed spline search**:
A pulse-search family whose first candidate is a gapped amplitude-and-chirp path and whose smooth spline coordinates then vary the admissible controls. Random perturbations may create distinct search branches around that physical seed, but a wholly random resonant waveform does not replace the seed.
_Avoid_: Random resonant start, guaranteed-optimal analytic pulse

**Seed-relative spline correction**:
A smooth control variation expressed relative to a validated analytic seed, with zero correction reproducing that seed exactly. Its parameterization preserves the pulse's endpoint conditions and physical amplitude bound while allowing the interior amplitude and chirp to change.
_Avoid_: Free spline refit of the seed, unconstrained additive waveform

**Symmetry-free pulse correction**:
A seed-relative correction whose coefficients on the first and second halves of the pulse are independent even when the analytic seed is time symmetric. Time symmetry may be measured afterward but is neither an optimization constraint nor a penalty.
_Avoid_: Mirrored correction, time-symmetry regularization

**Ansatz-limited smoothness**:
Pulse regularity supplied solely by the fixed smooth spline family and its finite resolution. The search objective contains no additional derivative, fluence, or bandwidth regularizer; those quantities may be inspected as diagnostics.
_Avoid_: Smoothness-regularized pulse search

**Spline-capacity check**:
A repeat of a stalled continuation stage with a richer spline basis, used to distinguish failure of the chosen control ansatz from evidence of a minimum reachable duration. It is a targeted diagnostic after all retained branches stall, not automatic basis growth during every stage.
_Avoid_: Adaptive spline growth, proof of a quantum speed limit

**Spline-GRAPE search**:
A pulse search that differentiates a time-discretized, finite-interaction Hamiltonian analytically with respect to amplitude and chirp and then maps those derivatives through the spline coordinates. It supplies efficient optimization gradients but does not replace independent continuous-time validation.
_Avoid_: Finite-difference GRAPE, hard-blockade optimization

**Study-owned pulse search**:
A research workflow whose gate objective, stage selection, continuation, and validation orchestration remain in a study script, while replay notebooks only load artifacts and plot them. Slice propagation, the discrete-adjoint gradient, and numerical optimization are consumed from `qoc` over the exported bilinear control model. A small scripts-local pulse-basis module may be shared after a second study needs it; the simulation package still gains no optimizer or study orchestration.
_Avoid_: Optimizer inside the simulation package, duplicated study orchestration

**GRAPE propagation time grid**:
The numerical time partition used to approximate one candidate's time-ordered evolution and analytic gradient during spline-GRAPE search. The optimizer still searches continuous spline coordinates; it does not independently optimize or enumerate a control value at every grid point.
_Avoid_: Search grid, control-parameter grid, GRAPE pixels

**Spline control coordinates**:
The low-dimensional coefficients that change overlapping amplitude and chirp basis functions together with the shared endpoint-chirp coordinate. They define continuous controls and are distinct from the many numerical time samples used to propagate one candidate.
_Avoid_: Time-grid values, pulse pixels

**Exact-ODE candidate validation**:
An independent public-backend evolution of a candidate produced by a discretized pulse search. Final gate acceptance uses this validation result rather than the search propagator's own estimate.
_Avoid_: Self-validation by the search grid

**Two-point stage selection**:
Selection at one continuation duration from only the inherited seed and the optimizer's terminal candidate. A feasible candidate is preferred over an infeasible one; when both are feasible, the one with lower worst-case endpoint leakage is independently validated. Optimization histories are retained for diagnostics but are not ranked as candidate archives.
_Avoid_: Candidate archive, blindly accepting the optimizer endpoint

**Fixed branch-optimization budget**:
A per-branch limit on optimizer iterations and objective evaluations that is declared before each fixed-duration search. Reaching the limit triggers ordinary two-point stage selection; the notebook does not silently extend the budget or introduce an adaptive retry policy.
_Avoid_: Unlimited optimization, adaptive hidden budget

**Optimizer trajectory**:
A diagnostic history containing the control coordinates and search-propagator metrics after every accepted optimizer iteration. It supports plots across optimization and duration continuation but does not participate in two-point stage selection. Internal line-search evaluations are not treated as optimizer iterations.
_Avoid_: Candidate-selection archive, line-search evaluation log

**Unwrapped optimization phase**:
A plotting-only continuation of the canonical conditional phase across accepted optimizer iterations within one branch. Stored and accepted results retain the canonical phase modulo `2*pi`; separate seed branches are never unwrapped into one artificial trajectory, and `C_phi` remains the periodic phase-strength diagnostic.
_Avoid_: New physical phase observable, cross-branch phase unwrapping

**Initial control guess**:
The starting spline coefficients that define the first candidate amplitude and chirp in a pulse search. It is control-parameter initialization, not a quantum state supplied to an evolution.
_Avoid_: Initial state

**Amplitude-bounded pulse**:
A control pulse constrained by a configurable hard upper bound on its Rabi amplitude. The bound applies to the complete continuous waveform and is enforced by construction rather than by sampled penalties or post-hoc clipping. The derived complex drive has the same magnitude, so chirp and accumulated phase cannot evade the bound.
_Avoid_: Unbounded Rabi search, phase-dependent amplitude bound

**Chirp-bounded pulse**:
A seed-relative amplitude-chirp pulse whose instantaneous chirp is constrained over the complete continuous waveform by a configurable symmetric physical limit. Zero correction exactly restores the analytic seed and its endpoint chirp; bounding only the seed's sweep amplitude is insufficient because spline corrections could otherwise leave the allowed range.
_Avoid_: Seed-only chirp bound, phase bound

**Phase-origin convention**:
The physically irrelevant constant phase of an amplitude-chirp pulse is fixed by setting its accumulated phase to zero at the initial time. Only phase differences generated by the chirp affect the control path.
_Avoid_: Free constant drive phase

**Control phase waveform**:
The physical-time function `phi(t)` obtained by integrating the instantaneous chirp and used in the complex `1-r` drive. Pulse-evolution plots show this function against real time; it is distinct from the logical conditional phase `Phi_ZZ` extracted from four endpoint amplitudes.
_Avoid_: Conditional gate phase, normalized-time phase trace

**Effective `01r` rotating frame**:
The bare three-level effective model in which `0`, `1`, and `r` have no preset static splitting. The protocol supplies every retained one-body energy and drive; laboratory hyperfine and optical carrier frequencies are absent from the simulated Hamiltonian.
_Avoid_: Laboratory-frame `01r` model

**Geometry-resolved blockade**:
The `rr` interaction obtained from the level structure's Rydberg-pair coefficient and the register distance as `C6 / R^6`. It is not an independently selected system parameter.
_Avoid_: Fixed preset blockade, user-specified interaction strength

**Bilinear control model export**:
The narrow public window returning a protocol-bound system's already-compiled drift matrix, quadrature-split Hermitian control operators, and requested initial-state vectors as plain arrays for search-side use. It shares the compiler with `simulate`, uses the bound protocol only for channel structure, refuses noise realizations, and does not replace exact-ODE candidate validation. Its basis ordering, angular-frequency units, and quadrature conventions are pinned by a constant-control parity test against `simulate`, not by exported metadata.
_Avoid_: Public compiler access, gradient mode of `simulate`, differentiable `simulate`

## Tensor-network numerics

**PEPS estimate**:
A finite numerical value produced by a PEPS calculation under explicitly chosen state, contraction, and resource controls. It is not, by itself, a claim that the PEPS sequence has converged.
_Avoid_: Converged PEPS result, certified result

**Convergence study**:
A comparison performed by the caller across PEPS estimates obtained with changed bond dimensions, step schedules, contraction controls, or initial states.
_Avoid_: Automatic convergence guarantee

**PEPS numerical evidence**:
The numerical provenance and bounded, already-computed error summaries retained only with a PEPS estimate so that a caller can compare runs. It never causes an extra contraction, does not contain full solver traces, and is not an acceptance gate applied by `src`.
_Avoid_: Generic metadata, convergence certificate

**Numerical provenance**:
The exact PEPS controls under which an estimate was produced, together with derived normalization information needed to interpret those controls.
_Avoid_: Convergence trace, physical observable

**PEPS evidence snapshot**:
An immutable view of all successful PEPS numerical evidence produced up to a particular point. A later lazy readout can produce a newer snapshot without changing an earlier one.
_Avoid_: Mutable diagnostics object, live backend log

**PEPS validity failure**:
A condition in which no mathematically meaningful estimate can be returned, such as a non-finite value, non-positive norm, invalid probability distribution, unsupported observable, or failed tensor operation. This is distinct from an estimate that is finite but not converged.
_Avoid_: Non-convergence

**NTU truncation error**:
The relative norm error reported in the neighbourhood tensor update metric when a PEPS bond is truncated. It is not a Schmidt discarded weight.
_Avoid_: Discarded weight, Schmidt weight

**PEPS SVD tolerance**:
The singular-value cutoff supplied to each local PEPS SVD. Together with the PEPS bond-dimension cap, it controls which singular values the update may retain; it is not a threshold applied to the reported NTU truncation error and is not a convergence certificate.
_Avoid_: NTU error tolerance, PEPS discarded-weight tolerance

**NTU iteration tolerance**:
The stopping threshold for the change in the local NTU truncation-error objective during one bond optimization. Reaching this threshold or the local iteration cap only ends that optimization; neither outcome certifies or rejects the full PEPS estimate.
_Avoid_: PEPS convergence tolerance, physical-error tolerance

**PEPS numerical control**:
A public resource or approximation parameter that a caller varies across runs, such as a time step, bond dimension, SVD cutoff, or iteration budget. It does not include YASTN-specific algorithm variants fixed by `ryd_gate`.
_Avoid_: Arbitrary engine option, YASTN kwargs passthrough

**PEPS grid register**:
A `Register` created by `Register.chain`, `Register.rectangle`, or `Register.square`, carrying private grid-shape provenance for the current YASTN square-lattice adapter. Direct arbitrary coordinates and triangular registers belong outside this backend even when their coordinates could be inferred as a grid.
_Avoid_: PEPS geometry input, inferred arbitrary-coordinate grid

**PEPS graph neighbour**:
Two sites adjacent along exactly one index of the validated open Cartesian PEPS lattice. This topology is distinct from selecting a shell by Euclidean distance; physical interaction terms are allowed only on a subset of these graph edges.
_Avoid_: Nearest-distance shell, automatic PEPS cutoff

**Unit-disk-graph tensor network**:
A future tensor-network algorithm whose graph comes from arbitrary two-dimensional Rydberg-atom coordinates and a distance-defined edge rule. It is a separate backend design, not an extension mode or geometry option of the current YASTN square-lattice PEPS adapter.
_Avoid_: Arbitrary-geometry PEPS mode

**PEPS contraction error**:
The maximum of the zipper discarded ratio and the absolute final variational-overlap change already produced across one deterministic boundary contraction. It is a dimensionless heuristic summary, not an error bound or convergence gate.
_Avoid_: Certified amplitude error, discarded Schmidt weight

**PEPS validity slack**:
The private complex128 roundoff scale `sqrt(machine epsilon) * max(1, |value|)` used only to decide whether a mathematically real quantity is real up to floating-point noise. It is independent of user convergence controls.
_Avoid_: Environment tolerance, convergence threshold

**Imaginary-time stage**:
One entry in a dimensionless PEPS imaginary-time schedule: a step size and a maximum number of updates applied to the normalized snapshot Hamiltonian.
_Avoid_: Physical-time segment

**Final refinement stage**:
The stage with the smallest imaginary-time step. It supplies the finest estimate in a multi-stage schedule but does not automatically certify convergence.
_Avoid_: Converged stage

**Hamiltonian scale**:
The deterministic local angular-frequency scale used to normalize a ground-state snapshot Hamiltonian without changing its ground-state eigenvectors.
_Avoid_: Physical imaginary time
