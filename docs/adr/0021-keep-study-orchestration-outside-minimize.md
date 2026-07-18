# Keep study orchestration outside minimize

One call to `qoc.minimize` performs one optimization from one named initial
candidate. It does not perform multistart search, duration continuation,
protocol-family selection, candidate ranking, or physical validation.

A study composes those workflows with ordinary control flow. It may call
`minimize` repeatedly, pass one result's `best_parameters` as the next call's
`x0`, and validate selected parameters through `ryd_gate`. This keeps research
policy outside the numerical optimizer and makes every expensive solve
explicit.
