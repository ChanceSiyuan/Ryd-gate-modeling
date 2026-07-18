# Use plain named parameter mappings

The base QOC interface accepts initial parameters as a flat
`Mapping[str, float | ndarray]`. Scalar entries represent individual pulse
coordinates; array entries represent finite coefficient blocks or sampled
controls. The caller-owned loss receives candidates in the same named shapes,
and an optimization result returns its best candidate in that representation.

`qoc` privately validates, copies, packs, and unpacks this mapping for numerical
solvers. It will not require a public `ParameterSpace`, `Parameter`, scalar
wrapper, array wrapper, or positional optimizer vector. This one representation
therefore covers finite analytic pulse parameters, spline coefficients, and
piecewise pulse values without exposing solver bookkeeping to the caller.

Bounds use an optional mapping keyed by the same parameter names. Each bound is
a `(lower, upper)` pair whose entries may be scalars broadcast over a parameter
block or arrays broadcastable to that block's shape; omitted parameter names
are unbounded.

Numerical scales use another optional mapping with the same broadcasting
rules. Every supplied scale must be positive; omitted names use scale one.
Solvers operate on coordinates divided by these scales, while the loss, bounds,
and returned best parameters always use the caller's original physical units.
`qoc` never guesses scales from initial values or bounds.
