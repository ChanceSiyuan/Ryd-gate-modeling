---
status: superseded by ADR-0025
---

# Keep gradients out of the base minimize interface

The base `qoc.minimize(loss, x0, ...)` interface accepts only a scalar loss. It
does not accept `jac`, `grad`, or `value_and_grad`, and an optimization result
does not expose a gradient. Derivative-free solvers use only loss evaluations;
solvers such as L-BFGS-B may approximate derivatives internally when selected.

True GRAPE will use a separate future interface that has access to the
forward-state/backward-costate structure required to calculate its gradient.
That gradient remains internal to the GRAPE implementation and is not a value
the ordinary caller must construct or consume. A finite-difference scalar-loss
search is never labeled GRAPE.
