---
status: superseded by ADR-0015
---

# Keep physical loss definitions outside the QOC core

The `qoc` core will define the result-reducer contract and generic scalar composition, regularization, and penalty tools, but it will not embed CZ, phase-gate, level-population, or model-specific fidelity formulas. Research code owns the choice of initial states and the physical interpretation of opaque evolution results, so the seven-level CZ reducer remains an example or study component rather than a dependency of the optimization engine.
