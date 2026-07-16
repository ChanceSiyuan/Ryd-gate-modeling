# Changelog

## 0.1.0 — Unreleased

Initial public release of `ryd-gate`, a continuous-time simulator for
interacting Rydberg-atom arrays.

### Public API

- Expose the six-name top-level API: `Register`, `RydbergSystem`, `NoiseModel`,
  `level_structure`, `simulate`, and `simulate_ensemble`.
- Provide immutable two-dimensional registers, protocol-bound systems,
  interaction cutoffs, and composable observable expressions.
- Provide the `1r`, `01r`, `rb87_7_mp`, `rb87_7_pm`, and
  `rb87_297_clock_4` level-structure presets.
- Export eight protocols from `ryd_gate.protocols`: `SweepProtocol`,
  `DigitalAnalogProtocol`, `CZProtocol`, `TOProtocol`, `ARProtocol`,
  `Direct297PiProtocol`, `Direct297CZProtocol`, and `Direct297TOProtocol`,
  together with `blackman_pulse` and `phase_from_chirp`.
- Provide forward atomic-physics helpers in `ryd_gate.physics` for laser Rabi
  frequencies, Zeeman shifts, and ARC pair-C6 calculations.

### Simulation and results

- Unify time evolution through `simulate()` with the exact ODE, MPS,
  Cartesian PEPS, and arbitrary-geometry graph-PEPS backends. Tensor-network
  backends support the effective `1r` and `01r` presets.
- Support product and logical-plus initial states, batched product states,
  explicit measurement grids, and named Hermitian observables.
- Return raw immutable physical readouts through `EvolutionResult`,
  `GroundStateResult`, and `EnsembleResult`: expectations, final amplitudes,
  samples, and PEPS numerical evidence where applicable.
- Provide DMRG, Cartesian PEPS imaginary-time, and graph-PEPS imaginary-time
  ground-state searches through `RydbergSystem.ground_state()`.
- Provide reproducible quasi-static Gaussian laser and position-noise ensembles
  through `NoiseModel` and `simulate_ensemble()`.

### Scope and packaging

- Keep all library Hamiltonians Hermitian. Gate metrics, decay budgets,
  optimization, plotting of results, persistence, and report generation remain
  explicit responsibilities of scripts and notebooks.
- Document the frozen model, simulation, and gate-workflow surfaces in
  `docs/model.md`, `docs/simulation.md`, and `docs/gates.md`.
- Support Python 3.10+, ship `py.typed`, and provide optional dependencies for
  MPS, Cartesian PEPS, graph-PEPS, and CUDA execution.
