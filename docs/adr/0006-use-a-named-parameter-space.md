---
status: superseded by ADR-0015
---

# Use a named parameter space at the QOC boundary

Pulse coordinates exposed by `qoc` will be named scalar or vector blocks carrying their initial values, bounds, and numerical scales. Solvers may privately pack those blocks into a flat numerical vector, but evolution oracles and results use the named representation. This supports analytic pulse families and piecewise control nodes without preserving fragile positional layouts such as undocumented `x[i]` conventions.
