# Changelog

## Unreleased (0.1.0 development line)

Product-API refactor. The staged refactor specs (`stageplans/`, Decision Log
D1–D13) have been retired now that the protocol-only surface has landed; the
sections below summarize the history.

### Fluent system builder (replaces `from_lattice`)
- `RydbergSystem.from_lattice(...)` is removed in favour of a three-step builder
  that separates the previously-conflated concerns:
  `RydbergSystem.set_atom_level(level_structure, param_set=..., **flags)` →
  `.set_atom_geom(geometry, interaction=...)` (adds the Rydberg `H_vdw`) →
  `.set_protocol(protocol)` (or `.build()` for an undriven system;
  `set_atom_geom` is optional and defaults to a single atom).
- The 420/1013 nm laser parameters (`Delta_Hz`, `rabi_420_Hz`, `rabi_1013_Hz`)
  now travel on the drive protocol (e.g. `DoubleARPProtocol(..., Delta_Hz=...)`)
  via `Protocol.laser_kwargs()`, and are baked into the operating point when
  `set_protocol` materializes the system.

### API ergonomics reframe
- `EvolutionResult` gained result-side accessors — `final_state`,
  `expectation(name)` / `expectations`, `probabilities()`, and
  `sample(n_shots)` — with the measuring system attached, so results read and
  sample themselves instead of threading the state back through the system.
- `simulate(...)` takes an optional `x` (only the CZ-gate protocols need a
  parameter vector) and an `observables=` argument, unified across the exact
  (final-state values) and tensor-network (per-time series) backends.
- Precision fixes: `analysis.addressing.default_sweep_x` reads physical
  parameters from metadata (previously an `AttributeError`); the documented
  `analysis.observables` helpers (`measure_observables`, `measure_trajectory`,
  `state_overlap`, `norm_squared`) are now exported from `ryd_gate.analysis`.
- De-duplicated the TO/AR Blackman drive; added `scripts/api_walkthrough.py`,
  a runnable end-to-end tour of the public API.

### Preset cleanup (Decision D13)
- Removed the symbolic `ger` level-structure preset (zero workflow users);
  `analog_3` is the only built-in three-level ladder. Custom symbolic
  three-level models are hand-built `LevelStructureSpec` instances passed to
  `RydbergSystem.set_atom_level`.

### Surface streamlining — Protocol-only simulator (Decision D12)
- Removed the Pulser-parity Sequence product surface: `Sequence`,
  `simulate_sequence`, `DeviceSpec`/`ChannelSpec`, product `Waveform`/`Pulse`,
  `ObservableConfig`, `SimulationResult`/state handles, the
  `sequence_from_protocol` bridge, and the Pulser abstract-repr interop
  module. Continuous-time protocols bound to `RydbergSystem` are the single
  control surface; `simulate(...)` returns the kernel `EvolutionResult`.
- Frozen `v1` schemas reduced to the reproducibility set: `register`,
  `register-layout`, `level-structure`, `noise`, `cz-gate-report`
  (six Sequence-face schemas removed). The `interop` extra is gone;
  `schema` remains.
- `ryd_gate.pulse` is now the kernel Blackman-envelope module only
  (`blackman_window` / `blackman_pulse` / `blackman_pulse_sqrt`).
- The gate line (CZ protocols, `CZGateReport`, gate metrics) and the noise
  layer (`NoiseModel`, exact Monte Carlo) are unchanged.

### Stage 1 — API foundation
- `Register` / `RegisterLayout` replace the old lattice factories in place
  (chain/square/rectangle/triangular/from_coordinates, stable atom ids,
  sublattice conventions preserved).
- `LevelStructureSpec` extended into the user-facing atom model; presets
  `01`/`1r`/`01r`/`ger`/`analog_3`/`rb87_7` with `supports_backend` truth
  table.
- `DeviceSpec` / `ChannelSpec` hardware constraints as validating data;
  `Waveform` / `Pulse` (integer ns, rad/µs); `ValidationIssue` +
  `raise_for_errors`; schema-tagged `ryd-gate/<kind>/v1` serialization.

### Stage 2 — Sequence + exact results
- `Sequence` (append-only, device-validated, replay-based `from_dict`,
  `draw()` via the kernel protocol plotter).
- `SequenceProtocol` kernel lowering (amp = Ω/2, det = −Δ, rad/s);
  `simulate_sequence`; lazy `SimulationResult` + `ExactStateHandle`
  (expectations, populations, multinomial sampling).

### Stage 3 — Backend-native result handles
- Capability-aware state handles (`QuantumStateHandle` protocol);
  `MPSStateHandle` with TeNPy-native expectations and guarded statevector
  materialization; `simulate_sequence(backend="mps")`.

### Stage 4 — NoiseModel
- Declarative `NoiseModel` (Pulser-aligned names + microscopic extensions),
  `configure_monte_carlo_runner` with exact unit conversions onto the
  existing exact Monte Carlo runner; decay flags at construction time.
- Kernel bug fix: `DenseODEBackend` corrupted scipy-sparse IR terms via
  `np.asarray`.

### Stage 5 — Gate library and error budgets
- `ryd_gate.gates` namespace; `CZGateReport` / `cz_gate_report` over a
  shared single-solve overlap core in `analysis.gate_metrics`.
- Benchmark pins (TO dark ≈ 7.8e-7 infidelity; AR / Double-ARP path pins);
  AR re-optimization workflow in `scripts/optimize_ar_cz.py`.

### Stage 6 — Serialization freeze + Pulser interop
- Frozen v1 JSON Schemas for all 11 payload kinds (shipped in-package;
  optional `jsonschema` via `schema` / `interop` extras).
- Pulser abstract-repr subset bridge (`ryd_gate.interop.pulser`) with typed,
  path-aware `PulserInteropError`; `RegisterLayout.define_register` (D10);
  `ObservableConfig` streaming schedules on the TeNPy measurement path.

### Stage 8 — Surface convergence
- Sequence pulse phase with Pulser virtual-Z semantics (`phase_rad` +
  accumulated `post_phase_shift_rad`), lowered to complex drive
  coefficients; exact backend only, typed refusal on TN backends.
- Local channels: `Sequence.target(...)` + replayable `TargetOp`
  (per-site compiler keys; works on exact and mps); `rydberg_local` gains
  `1r` channel maps; additive `"target"` op in the sequence/v1 schema.
- `simulate_sequence` accepts `gputn` / `peps` (non-native states expose
  `raw` + `UnsupportedStateHandle`); error code renamed to
  `simulate_sequence.backend_unsupported`.
- `sequence_from_protocol`: explicit, lossy Protocol → Sequence
  discretization bridge with loss metadata and typed refusals.
- Docs: the `SequenceProtocol` convergence point documented in
  fundamentals; capability matrix regenerated.

### Stage 7 — Docs, examples, packaging
- Sphinx product docs (getting started, fundamentals, how-tos, generated
  capability matrix, autodoc API reference); README quickstart rewritten
  around the Sequence and gate-report examples.
- Executable `examples/` demos; notebooks migrated to the product API
  (`Register.*`, `analog_3` preset); nbconvert-based gated notebook runner.
- Packaging: `py.typed`, scoped mypy gate, `docs` extra, repo-wide ruff
  clean; D11 cleanup — bare `"ger"` is symbolic regardless of `param_set`.
