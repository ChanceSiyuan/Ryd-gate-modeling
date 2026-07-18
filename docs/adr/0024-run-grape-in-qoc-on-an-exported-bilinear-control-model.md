# Run GRAPE in qoc on an exported bilinear control model

GRAPE needs the forward-state/backward-costate structure that a closed scalar
loss cannot expose, so it does not fit the ADR-0015 seam. We keep both package
boundaries intact by exchanging plain arrays instead of capabilities:
`ryd_gate` gains one narrow export, `bilinear_control_model(system, states=...)`,
that returns the already-compiled bilinear form of a protocol-bound system —
the Hermitian drift matrix, one Hermitian control operator per real control
channel (complex drives are split into two quadrature channels), and the
requested initial-state vectors — as bare ndarrays and plain mappings. The
GRAPE engine lives in `qoc` and consumes only those arrays plus caller-supplied
callbacks: a control map with its pullback (named parameters to per-slice
channel values and back) and a terminal objective (final states to one scalar
and the terminal costates). `qoc` still never imports `ryd_gate` or interprets
physical systems; a study carries the arrays across.

The gradient is the discrete adjoint: forward piecewise-constant slice
propagators, backward costate propagation through the same propagators, and
exact Frechet derivatives of each slice exponential, so the gradient is exact
to machine precision for the discrete loss the optimizer actually evaluates.
Continuous-time acceptance still comes from independent exact-ODE validation
through `simulate`, which is unchanged.

The export carries no metadata fields. Basis ordering, angular-frequency
units, and quadrature conventions are pinned by a parity test that propagates
the exported model with constant controls and matches `simulate` on the same
protocol. The export uses the bound protocol only for channel structure,
refuses systems bound to noise realizations, and is dense-only until a use
case requires more.

## Considered options

- A gradient mode or state-retention option on `simulate` — rejected: it
  grows optimization-only parameters on the physics entry point, and states
  retained from the adaptive integrator cannot feed a consistent discrete
  adjoint.
- An adjoint engine inside `ryd_gate` — rejected: everything GRAPE-specific
  would sit in `ryd_gate` while the `qoc` method reduced to generic gradient
  descent, contradicting the ADR-0007 naming discipline.
- `qoc` calling `ryd_gate`'s compiler directly — rejected: it saves no code
  (the same translation layer must exist somewhere), adds a qoc→ryd_gate
  dependency on private modules, and forfeits `qoc`'s system neutrality
  (ADR-0002).
