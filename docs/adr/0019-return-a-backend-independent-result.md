# Return a backend-independent optimization result

`qoc.minimize` returns one QOC-owned result with these common fields:

- `best_parameters`
- `best_loss`
- `success`
- `message`
- `method`
- `n_iterations`
- `n_evaluations`

The best parameters retain the public named scalar and array shapes. `success`
reports numerical solver termination only; it never asserts that a physical
gate, state, or Hamiltonian-synthesis acceptance criterion passed. The result
does not expose an underlying SciPy result, gradients, simulator results, or
physical diagnostics.
