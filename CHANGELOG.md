# Changelog

## Unreleased (0.1.0 development line)

Product-API refactor, staged per `stageplans/` (binding specs; Decision Log
D1–D11 in `stageplans/README.md`).

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
