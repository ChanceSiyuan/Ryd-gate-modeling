# Quantum Optimal Control

This context describes numerical optimization of finite-dimensional pulse coordinates through caller-owned scalar loss functions.

## Language

**Pulse parameterization**:
A family of admissible control pulses selected by a finite coordinate vector. It may describe an analytic function with named coefficients or a piecewise waveform with finitely many control values; one concrete coordinate vector determines one fully specified pulse.
_Avoid_: Initial protocol, runtime optimizer protocol

**Parameter space**:
The conceptual set of named scalar and array coordinate blocks available to a pulse search. Its public representation is an ordinary mapping from names to real scalars or real arrays; it does not determine how those values generate a physical pulse.
_Avoid_: `ParameterSpace` class, flat optimizer vector, pulse parameterization

**Named parameter mapping**:
A flat `Mapping[str, float | ndarray]` used for the initial candidate, every call to the scalar loss, and the best candidate in the result. Array shapes are preserved across private packing and unpacking.
_Avoid_: Parameter wrapper objects, positional parameter list, packed coordinates

**Named bounds mapping**:
An optional mapping from parameter names to `(lower, upper)` pairs. Scalar limits broadcast across an array parameter, array limits may constrain its elements individually, and a parameter omitted from the mapping is unbounded.
_Avoid_: Positional bounds list, mandatory bounds entry, silent clipping

**Named scales mapping**:
An optional mapping of positive numerical scales with the same names and broadcast rules as the parameters. It changes only the solver's private dimensionless coordinates; the loss, bounds, and returned parameters retain their original physical units. Missing scales equal one and are never inferred.
_Avoid_: Automatic scaling, unit conversion, scaled loss input

**Optimization method name**:
A stable lower-case string selecting one registered solver implementation. The base interface uses method names plus a method-specific options mapping rather than public solver classes.
_Avoid_: Solver object, backend-native method identifier

**Optimization result**:
A QOC-owned record containing the best named parameters and scalar loss together with method-independent solver status and evaluation counts. Its success flag means numerical solver success only and carries no physical acceptance claim.
_Avoid_: SciPy result, physical validation result, gradient result

**Single-start solve**:
One invocation of one numerical solver from one named initial candidate. Multistart search, duration continuation, warm-start sequencing, and physical validation are study orchestration built from repeated single-start solves.
_Avoid_: Optimization campaign, implicit continuation, automatic validation

**Evaluation callback**:
An optional caller function invoked after each successful finite scalar-loss evaluation with the evaluation index, named parameters, current loss, and best loss so far. It externalizes logging and checkpoint policy without making complete history or physical diagnostics part of the optimization result.
_Avoid_: Built-in trajectory archive, physical result callback

**Loss evaluation**:
One actual invocation of the caller-owned scalar loss requested by the solver. QOC does not deduplicate or cache repeated candidates; a study may explicitly wrap a deterministic loss when reuse is valid.
_Avoid_: Unique candidate, cached evaluation

**Packed coordinates**:
The private flat numerical representation of one set of parameter values used by a numerical solver. Packing and unpacking must preserve the public names and block shapes.
_Avoid_: Public parameter API, historical x-vector layout

**Scalar loss**:
A caller-supplied function that maps one named parameter candidate to exactly one finite real scalar. The caller owns protocol construction, system binding, initial states, observables, simulation, result interpretation, and physical fidelity inside this function; `qoc` neither sees nor reconstructs those operations. Exceptions and invalid values fail the solve unless the caller explicitly converts a known condition into a finite penalty.
_Avoid_: Evolution oracle, result reducer, physical objective object

**Explicit loss penalty**:
A finite scalar deliberately returned by the caller-owned loss for a recognized undesirable candidate. QOC never creates such a penalty from an exception or non-finite result.
_Avoid_: Silent exception conversion, implicit NaN penalty

**Scoring-only variable**:
A value that changes how one fixed evolution result is compared with a target but does not change the physical pulse or evolution. It is eliminated inside the caller-owned scalar loss and is never a QOC parameter coordinate.
_Avoid_: Pulse parameter, optimizer parameter

**Profiled loss**:
A caller-owned scalar loss obtained after eliminating scoring-only variables from one candidate's existing physical results. The selected scoring values are study diagnostics and are not returned through the base QOC loss interface.
_Avoid_: Joint pulse-and-score optimization

**Fixed-duration solve**:
One pulse-parameter optimization whose caller-owned scalar loss captures a prescribed evolution duration that remains constant for every candidate in that solve.
_Avoid_: Free-duration optimization

**Duration continuation**:
A sequence of fixed-duration solves in which the study rebuilds its scalar loss at each duration and uses one solution to initialize the next.
_Avoid_: Duration as an ordinary pulse coordinate, independent duration sweep

**Constraint violation**:
A named, nonnegative measure of how far one pulse candidate lies outside an admissible condition. Its physical meaning is supplied by research code, while the chosen method determines whether it is prevented, penalized, or only reported.
_Avoid_: Guaranteed hard constraint, search loss

**Constraint guarantee**:
The enforcement level a particular pulse-search method provides for a constraint: hard bounds, admissible reparameterization, penalty-based encouragement, or nonlinear-program feasibility.
_Avoid_: Constraint definition

**Differentiable loss**:
A scalar loss whose derivative is available to a gradient method. The base minimize signature does not take a derivative argument and the result does not return one; a gradient method receives the derivative through its method-specific options, and derivative provenance still determines whether a method may be called GRAPE.
_Avoid_: Base-interface `jac`, returned gradient, automatically GRAPE

**Black-box pulse optimization**:
A pulse search that selects candidates solely from scalar loss evaluations, without using derivatives or the internal structure of quantum evolution.
_Avoid_: GRAPE, direct trajectory optimization

**Generic gradient pulse optimization**:
A pulse search whose solver uses loss derivatives without the forward-state/backward-costate structure required by GRAPE. In the base interface, any numerical derivative is an internal solver detail.
_Avoid_: GRAPE

**GRAPE**:
An indirect pulse-search method that constructs time-slice control gradients from a forward physical trajectory and a backward costate trajectory, while intermediate quantum states remain determined by rollout.
_Avoid_: Any gradient-based optimizer, finite-difference pulse search

**Local dynamics oracle**:
An external operation that evaluates a one-step dynamics residual, and the required local derivatives, for arbitrary intermediate state and control values. Unlike a closed scalar loss, it can evaluate deliberately infeasible trajectories during direct trajectory optimization.
_Avoid_: Full rollout, sampled trajectory

**Direct trajectory optimization**:
A specialized constrained optimization in which intermediate quantum states and controls are decision variables and local dynamics equations are enforced as constraints. It requires a separate interface for defects and Jacobians and cannot be reconstructed from the base scalar loss.
_Avoid_: Scalar-loss minimization, endpoint rollout

**Bilinear control model**:
The plain numerical statement of one linearly driven control problem: a Hermitian drift matrix, named Hermitian control operators entering with real per-slice coefficients, and named initial-state vectors, all as bare arrays and mappings. QOC neither knows nor asks which physical package produced it.
_Avoid_: Physics-aware model class, `ryd_gate` system input

**Control channel**:
One named real control coordinate multiplying one Hermitian operator in a bilinear control model. A physically complex drive appears as two quadrature channels; nonlinear physical parameterizations reach channels only through the caller's control map.
_Avoid_: Complex control coefficient, nonlinear channel

**Control map**:
The caller-supplied pair of operations taking named parameters to per-slice channel values and pulling per-slice channel gradients back to named parameter gradients. It owns every nonlinearity between parameters and channels; zero correction structure and endpoint conventions stay with the caller.
_Avoid_: QOC-side pulse parameterization, exported Jacobian matrix

**Terminal objective**:
The caller-supplied callable mapping final state vectors to exactly one scalar and the terminal costates. The physical meaning of the scalar stays with the caller.
_Avoid_: QOC-owned fidelity, physical objective object

**Discrete adjoint gradient**:
The machine-precision derivative of the piecewise-constant slice propagation the search actually evaluates, built from forward states, backward costates through the same slice propagators, and exact slice-exponential derivatives. It is consistent with the discrete loss by construction.
_Avoid_: Continuous costate integration, integrator-tolerance gradient

**GRAPE engine**:
The QOC-owned forward/backward costate propagation over one bilinear control model. Its presence is what entitles a QOC method to the name GRAPE under the naming rule; it performs no physical validation.
_Avoid_: Physics simulator, `ryd_gate` adapter, candidate validator
