---
status: superseded by ADR-0015
---

# Keep duration fixed within a rollout solve

Each rollout problem will optimize pulse coordinates at one prescribed evolution duration. Time-optimal studies will run an outer duration continuation that warm-starts a new fixed-duration problem from a preceding solution; only a future direct-trajectory problem may make knot times or interval durations optimization variables. This prevents an ordinary pulse candidate from improving its loss merely by silently extending its own evolution.
