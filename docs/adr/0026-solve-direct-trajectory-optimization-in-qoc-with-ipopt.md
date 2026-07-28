# Solve direct trajectory optimization in qoc with IPOPT

GRAPE (ADR-0024) compresses all time slices into one endpoint sensitivity
and can honor hardware limits only through penalties. The direct method
reserved by ADR-0007 — intermediate states as decision variables plus local
dynamics constraints — is implemented in `qoc.direct` over the same exported
bilinear control model arrays as GRAPE: knot unitaries `isovec(U_k)` and
per-channel control chains `u -> du -> ddu` are the decision variables, the
slice dynamics `U_k = expm(-i H(u) dt) U_{k-1}` are per-slice defect
constraints with exact `expm_frechet` Jacobians, and every hardware limit
(amplitude, slew rate, curvature, endpoint pinning) is a plain variable
bound. The double-integrator chain makes piecewise-linear waveforms and
their derivative bounds native instead of penalized, and the optimizer may
start from and traverse infeasible state trajectories.

The nonlinear program is solved by IPOPT through the optional `cyipopt`
dependency (extra `qoc-direct`), imported lazily inside `optimize` so the
base package stays dependency-free. The terminal objective remains a
caller-supplied callback in the grape costate convention (`G =
dL/d(conj(U_N))`); `qoc` still never imports a physics package. Durations
stay fixed within one solve (ADR-0005); this knot-point formulation was
chosen because it extends to free knot durations without restructuring.

## Considered options

- scipy-only solvers (trust-constr, hand-rolled augmented Lagrangian) —
  rejected: at the target scale (~10^4 variables with as many equality
  constraints) their convergence is the dominant project risk, and an
  augmented-Lagrangian schedule would become our code to tune and maintain.
- Piccolo.jl — rejected: introduces a Julia toolchain into a Python
  repository and bypasses the qoc package boundary entirely.
- Defect constraints on state vectors instead of unitaries — deferred: the
  first consumer scores the full propagator; a state-trajectory variant can
  be added beside this one when an ensemble objective needs it.
