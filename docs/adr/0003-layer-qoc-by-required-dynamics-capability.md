---
status: superseded by ADR-0015
---

# Layer QOC by required dynamics capability

The base `qoc` problem will depend only on an evolution oracle that maps pulse parameters, an opaque initial state, and a duration to an opaque evolution result. Methods may require stronger optional capabilities: efficient gradient methods require a differentiable loss, while direct trajectory optimization requires a local dynamics oracle that can evaluate residuals and derivatives at arbitrary intermediate states. This keeps every method independent of level structure without pretending that an endpoint rollout alone can implement GRAPE or direct collocation.
