# Route caller gradients through method-specific options

ADR-0017 kept every derivative out of the base `minimize` interface and
promised GRAPE "a separate future interface" whose gradient the ordinary
caller would never construct or consume. ADR-0024 then delivered that separate
interface as the qoc GRAPE engine (`grape.value_and_grad`) and made the study
the only glue between the engine and the optimizer — so a study now
legitimately holds a named gradient and needs a sanctioned way to hand it to a
gradient method.

The base signature is unchanged and still accepts only a scalar loss: no
`jac` argument, and no gradient in the optimization result. This ADR adds
that a method's method-specific options mapping (ADR-0018) may carry two
reserved keys:

- `"gradient"` (gradient methods only): a callable from named parameters to a
  named gradient mapping with the same names and shapes.
- `"iteration_callback"`: a callable invoked with the named parameters after
  each accepted solver iteration — the per-iteration channel that the
  ADR-0022 per-evaluation callback deliberately cannot express, because
  line-search evaluations are not optimizer iterations.

Naming discipline is unchanged (ADR-0007): a solver fed a caller-supplied
gradient is generic gradient optimization; GRAPE remains the engine that
constructs the gradient from forward states and backward costates, and a
finite-difference scalar-loss search is still never labeled GRAPE.
