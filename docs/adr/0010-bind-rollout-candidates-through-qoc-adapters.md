---
status: superseded by ADR-0015
---

# Bind rollout candidates through QOC adapters

The `qoc` core will expose a narrow rollout capability whose `bind(parameters, duration)` operation creates one candidate-bound rollout session. `qoc.adapters.ryd_gate` will implement that capability by using a user-supplied protocol factory, `RydbergSystem.with_protocol()`, and the public `ryd_gate.simulate()` function; it will provide a batch fast path so all logical inputs share one candidate binding and backend compilation. `ryd_gate` remains unchanged and never imports `qoc`, while GRAPE and direct methods use separate, stronger capabilities rather than enlarging the rollout interface.
