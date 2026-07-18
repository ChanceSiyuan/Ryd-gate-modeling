---
status: superseded by ADR-0015
---

# Profile scoring-only variables inside the result reducer

Variables that change only how a fixed evolution result is compared with a target, such as the local-Z phase `theta` in the seven-level CZ score, will not appear in the QOC parameter space or any pulse optimizer's coordinates. The result reducer must eliminate such variables using the already-computed rollout results and may report the selected value as a diagnostic; changing a scoring convention must never trigger another physical evolution.
