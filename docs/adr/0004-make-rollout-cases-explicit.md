---
status: superseded by ADR-0015
---

# Make rollout initial states explicit

A rollout problem in `qoc` will explicitly contain its pulse-parameter space, duration, evolution oracle, ordered initial-state set, and a user-supplied result reducer. `qoc` may therefore coordinate batching, caching, and per-state diagnostics while treating every state and evolution result as opaque; a fully closed `loss(parameters)` callback remains usable as an adapter but is not the primary problem model because it hides the repeated quantum evolutions.
