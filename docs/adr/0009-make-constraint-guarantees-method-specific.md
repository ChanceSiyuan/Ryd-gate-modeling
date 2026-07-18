# Make constraint guarantees method-specific

`qoc` will not present every constraint as equally hard. Rollout methods guarantee parameter bounds and may incorporate other user-defined violations as penalties; GRAPE may use bounds, admissible reparameterizations, or penalties and must report residual violations; direct trajectory optimization exposes supported equality and inequality functions as nonlinear-program constraints. Physical meanings such as Rabi amplitude or slew rate remain outside the core, which handles only named values and violations.
