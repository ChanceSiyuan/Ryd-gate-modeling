---
status: superseded by ADR-0015
---

# Keep quantum optimal control outside `ryd_gate`

Reusable quantum-optimal-control research code will live in the `qoc` Python package beside `ryd_gate` under `src/`, rather than inside the simulator package. This preserves `ryd_gate` as the protocol-bound physical simulation library while allowing optimization methods and their research-facing abstractions to evolve independently; `qoc` may consume public `ryd_gate` APIs, but `ryd_gate` must not depend on `qoc`.
