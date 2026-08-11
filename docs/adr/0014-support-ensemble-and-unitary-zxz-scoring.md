---
status: superseded by ADR-0015
---

# Support ensemble and full-unitary ZXZ scoring

`qoc.objectives.zxz` will support two explicit ways to score the target
transformation $\exp(-i\tau H_{\mathrm{ZXZ}})$:

1. mean final-state fidelity over a supplied, fixed ensemble of initial states,
   including the Haar-state evaluation used by the GRAPE comparison in
   `hu2025universal` in `.knowledge/references.bib`; and
2. fidelity between the complete candidate and target unitaries.

The study chooses the scoring mode explicitly. Neither mode belongs to GRAPE,
Direct, or another optimizer, so both can be reused by any method whose
capabilities provide the required evolution data.
