---
status: superseded by ADR-0003
---

# Put QOC methods behind a system-neutral control model

GRAPE, direct optimal control, objectives, and constraints in `qoc` will depend on a system-neutral control model rather than on `RydbergSystem` or concrete Rydberg protocols. Physical simulators enter through adapters that translate their dynamics and pulse families into this model and translate optimized controls back into native protocols; the first adapter will target `ryd_gate`.
