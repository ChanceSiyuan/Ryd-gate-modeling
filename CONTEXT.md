# Rydberg Simulation

This context describes the physical and numerical language used by `ryd_gate` to distinguish quantum-simulation results from evidence about their tensor-network approximation.

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
