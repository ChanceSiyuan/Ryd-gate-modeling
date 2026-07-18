---
status: superseded by ADR-0015
---

# Restrict the CZ objective to the symmetric two-atom problem

The initial `qoc.objectives.cz` objective supports only the exchange-symmetric
two-atom CZ problem used by the seven-level study. It evaluates the three
logical representatives $|00\rangle$, $|01\rangle$, and $|11\rangle$, treats
$|10\rangle$ as equivalent to $|01\rangle$, and profiles one shared local-Z
phase from those rollout results.

The study wiring is responsible for supplying exchange-symmetric system
parameters and controls. A future independently addressed or otherwise
asymmetric two-atom problem must use a distinct full-basis objective rather
than silently reusing this reduction.
