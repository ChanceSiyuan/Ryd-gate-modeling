---
status: superseded by ADR-0015
---

# Group target transformations as objectives

Reusable, system-independent definitions of desired quantum transformations and
their scoring mathematics will live together under `qoc.objectives`. A CZ gate
and the target evolution
$\exp(-i\tau H_{\mathrm{ZXZ}})$ are peer control objectives, for example
`qoc.objectives.cz` and `qoc.objectives.zxz`; they are not split merely because
one is conventionally named as a gate and the other through a Hamiltonian.

`qoc.core` remains unaware of physical targets, and `qoc.adapters` remains
unaware of objective formulas. A study reducer extracts the quantities required
by an objective from an external simulator's opaque results. The native
Hamiltonian that actually generates a candidate rollout remains the
responsibility of the external system and its adapter, not the objective.
