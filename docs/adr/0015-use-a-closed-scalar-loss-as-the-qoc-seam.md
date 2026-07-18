# Use a closed scalar loss as the QOC seam

The base `qoc` interface will be numerical parameter optimization:

```python
qoc.minimize(loss, x0, method=...)
```

`loss` receives one candidate in the public named representation and returns
exactly one scalar. The caller owns the complete physical evaluation inside
that function: constructing a protocol, binding it to a system, selecting
initial states and observables, calling `simulate`, reading final results, and
reducing them to the physical loss. Scoring-only quantities such as the CZ
local-Z phase are profiled inside that caller-owned loss and never enter the
QOC parameter space.

Consequently, the base `qoc` package will not expose rollout cases, evolution
oracles, result reducers, physical objectives, or a `ryd_gate` adapter. A study
imports both packages and passes its closed loss to `qoc`; neither `qoc` nor
`ryd_gate` depends on the other.

The public candidate representation is an ordinary named mapping of real
scalars and real arrays. An analytic pulse, spline pulse, and piecewise pulse
differ only in how the caller turns those finite coordinates into a protocol.
Solvers privately pack the values into flat coordinates and return the best
candidate with the original names and shapes.

The scalar-loss interface covers black-box methods and solvers that perform
their own numerical differentiation. A method is called GRAPE only when it
constructs gradients through the required forward-state/backward-costate
calculation, and Direct uses a separate future interface for local dynamics
constraints and Jacobians; neither requirement will enlarge the base
scalar-loss interface.

The loss remains strictly scalar. `qoc` records parameters, scalar losses, and
solver status; a study computes physical diagnostics such as profiled phases,
populations, and observables separately from the selected final parameters.
