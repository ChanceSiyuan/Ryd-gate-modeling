# Changelog

## Unreleased (0.1.0 development line)

### `rb87_297_clock_4`: 297 nm single-photon clock → nP₃/₂ model
- New atom-level preset **`rb87_297_clock_4`** (`("0", "1", "r", "r_garb")`):
  direct σ⁻ 297 nm excitation from the clock-like `|F=2, mF=0⟩` ground state
  `|1⟩` to the 53P₃/₂ target (`mⱼ=-3/2`) and garbage (`mⱼ=-1/2`) Zeeman
  branches; the logical `|0⟩` (`|F=1, mF=0⟩`) is a dark spectator carrying only
  the static clock hyperfine energy (`h[0,0] = -ω_hf`, as in the seven-level
  model). Unit-Rabi drive blocks carry the branch dipole ratio (1/√3);
  atom-level kwargs are `enable_rydberg_decay`, `magnetic_field_G`,
  `ryd_level`. The `|r_garb⟩` detuning is the Δmⱼ=+1 excited-state nP₃/₂
  Zeeman splitting relative to the shared low-field clock-state ground energy;
  nP₃/₂ decay (`-iΓ/2` on both Rydberg levels) comes from ARC lifetimes (300 K
  total, 0 K radiative → RD/BBR split in metadata).
- New **`Direct297PiProtocol(power_at_atoms_w, beam_area_um2, ...)`**: Blackman
  π-pulse whose target Rabi is `single_photon_rabi(...)/√2` (clock-state
  factor) from the beam power *at the atoms* (no optics-loss factor inside);
  the Rydberg level is read from the bound `rb87_297_clock_4` system, and
  `t_gate=None` auto-calibrates the target pulse area to π.
- New **`Direct297CZProtocol(t_gate=..., A_297=..., phi_297=..., ...)`** — the
  single-beam CZ pulse container (297 analog of `CZProtocol`): drive
  `Ω_297_max · A_297(s) · e^{-iφ_297(s)}` on the `"297"` ratio group, with
  `Ω_297_max` from power/area and the bound system's `ryd_level`;
  `pulse_traces` exposes `Ω_297` and the chirp `dφ_297/dt`. New
  **`Direct297TOProtocol(power, area, blackman=True)`** builder with
  `x = [A, ω/Ω_297, φ0, δ/Ω_297, θ, T/T_297]`, `T_297 = 2π/Ω_297`
  (`theta_index=4`, `t_gate_index=5`).
- New four-level 297 gate metrics in `analysis.gate_metrics` (also exported
  from `ryd_gate.gates`), explicitly named to keep the seven-level API intact:
  `average_gate_infidelity_297` (evolves |00⟩,|01⟩,|10⟩,|11⟩; four-state
  Nielsen formula that reduces to the seven-level three-state formula when
  `a01 == a10`; residuals report `r`/`r_garb`/`logical_loss`),
  `population_evolution_297` (`sum_n_r`/`sum_n_r_garb` time series), and
  `error_budget_297` (flat budget, no XYZ/AL/LG branching:
  `p_ryd_decay = Γ·∫(n_r+n_r_garb)dt` split into target/garb, plus final
  residuals and `p_total`; default inputs `["01", "10", "11"]`), plus
  `project_theta_297` (bounded 1-D refit of the single-qubit Z `theta` alone —
  the 297 counterpart of `optimize_cz_parameters`' theta-projection warm
  start, returning a `Theta297Projection` that never scores worse than the
  seed; also exported from `ryd_gate.analysis`). All `*_297` functions reject
  non-`rb87_297_clock_4` systems.
- New physics helpers: `zeeman_shift_rad_s(B, l=..., j=..., delta_mj=...)`
  (generalizes the nS₁/₂ helper), `direct_297_rabis`, and
  `arc_pair_c6_rad_s_um6` — ARC perturbative pair C6 converted to this repo's
  `V = +C6/R⁶` sign convention, with max-overlap eigenchannel selection in the
  degenerate mⱼ manifold (warns when the bare-channel overlap is < 0.5) and
  per-orientation caching.
- New `vdw_couplings_from_c6_function(coords, c6_fn, quantization_axis=z)`:
  anisotropic `C6(θ, φ)/R⁶` pair couplings; the 297 model uses it by default
  (2D registers lie in the xy plane, B ⊥ plane → θ=π/2 for all pairs), with the
  single target-channel C6 applied to *all* r/r_garb pair projectors (coarse
  first-version approximation). An explicit `InteractionSpec(C6=...)` still
  overrides with the isotropic scalar.

Product-API refactor. The staged refactor specs (`stageplans/`, Decision Log
D1–D13) have been retired now that the protocol-only surface has landed; the
sections below summarize the history.

### rb87_7 split into `rb87_7_mp` / `rb87_7_pm` (manifold tags; `param_set` removed)
- The Rb87 seven-level model is now selected by manifold/polarization tag, not a
  `param_set`: **`rb87_7_mp`** (σ⁻/σ⁺, was `param_set="our"`) and **`rb87_7_pm`**
  (σ⁺/σ⁻, was `"lukin"`). The bare `"rb87_7"` tag and the `param_set` kwarg are
  removed (breaking) — both raise a clear error pointing to the new tags.
- Static atom/manifold values are explicit `set_atom_level` kwargs: `Delta_Hz`,
  `ryd_level`, `C6_rad_s_um6`, `t_rise`, `detuning_sign`, `enable_*`. No laser
  Rabi amplitude enters at `set_atom_level` (the drive blocks stay unit-Rabi).
- CZ/TO/AR protocols own the 420/1013 Rabis: `omega_*_max` default to the fixed
  σ⁻/σ⁺ (`rb87_7_mp`) canonical values — a protocol constant, never inferred from
  the system; pass them explicitly for the `rb87_7_pm` manifold.

### Fluent system builder (replaces `from_lattice`)
- `RydbergSystem.from_lattice(...)` is removed in favour of a three-step builder
  that separates the previously-conflated concerns:
  `RydbergSystem.set_atom_level(level_structure, **flags)` →
  `.set_atom_geom(geometry, interaction=...)` (adds the Rydberg `H_vdw`) →
  `.set_protocol(protocol)`. Every step returns a fully materialized, usable
  system, so `set_atom_geom` (defaults to a single atom) and `set_protocol`
  (undriven otherwise) are both optional.
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
  `01`/`1r`/`01r`/`ger`/`analog_3`/`rb87_7_mp`/`rb87_7_pm` with `supports_backend` truth
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
