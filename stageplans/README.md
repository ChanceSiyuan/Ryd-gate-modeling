# Stage Plans for the Product API Refactor

This directory contains review-first implementation plans. These files are not implementation notes; they are **binding task specifications**. If an implementation step needs to deviate from a stage plan, update the stage plan first and get it reviewed. The matching user-facing API contract for Stage 1 is [docs/stage1_api.md](../docs/stage1_api.md); stage plans own engineering (current-code refactor steps, file ownership, tests, acceptance), the API doc owns behavior (purpose/input/output/failure per callable).

`Plan_stage1.md` at the repo root is the **original vision document** and is kept for history. Where it conflicts with the stage plans (it predates several design decisions), the stage plans win — see the Decision Log below for exactly which of its proposals were superseded and why.

## Product Goal and Competitive Positioning

Target: a neutral-atom simulation SDK competitive with [Pulser](https://docs.pasqal.com/pulser/) (Pasqal) and credible next to Bloqade (QuEra), while keeping this repo's research kernel as the engine.

**Table stakes we match (Pulser parity targets):**

- declarative data layer: `Register`/`RegisterLayout`, `DeviceSpec`/`ChannelSpec` (device-as-validator), `Waveform`/`Pulse`, `Sequence` with declared channels — Stages 1–2;
- everything drawable (`Register.draw`, `Sequence.draw`) and everything serializable (schema-tagged dicts from day one, frozen JSON Schemas later) — Stages 1–2, 6;
- declarative `NoiseModel` with Pulser-compatible parameter names where semantics coincide — Stage 4;
- uniform results across backends with explicit capabilities instead of silent densification — Stage 3.

**Where we differentiate (Pulser cannot do these today):**

1. **Gate-level modeling.** Pulser has no two-qubit-gate concept: no CZ protocol library, no gate fidelity, no error budgets. This repo already has time-optimal/amplitude-robust/double-ARP CZ protocols, `average_gate_infidelity`, and `error_budget` — Stage 5 productizes them as the flagship.
2. **Multilevel atomic realism.** Pulser is 2–3 levels plus one generic leakage state with phenomenological rates. We ship `rb87_7` (7-level Rb87 with intermediate-state scattering, branching ratios via ARC, garbage levels) and an open `LevelStructureSpec` for custom models — noise *derived from laser/atomic parameters*, not just dialed-in rates.
3. **Backend fleet.** Exact sparse/dense, TeNPy MPS (DMRG/TDVP), cuQuantum GPU-TN, YASTN PEPS, PEPSKit — broader than Pulser's QuTiP + emu-mps/emu-sv lineup, behind one `simulate()` dispatcher.
4. **Two front doors, one kernel.** The `Sequence` API (hardware-shaped, integer-ns, validated) for Pulser-style programs, and the continuous-time `Protocol` API (gate protocols, callable schedules) for optimal-control research. Both lower to the same `RydbergSystem → HamiltonianIR → backend` path, so results are comparable by construction.

**What we deliberately do not chase yet:** cloud/QPU submission, EOM/DMM/SLM/XY hardware features (no kernel counterpart), and full Pulser feature parity. A Pulser-subset *import/export* (Stage 6) lowers switching costs without chasing parity.

## Decision Log (supersedes Plan_stage1.md where they conflict)

| # | Decision | Supersedes | Why |
|---|---|---|---|
| D1 | **No `src/ryd_gate/api/` package.** Product objects live in domain modules (`lattice/geometry.py`, `core/level_structures.py`, `devices.py`, `pulse.py`, `sequence.py`, `results.py`). | Plan_stage1.md's `api/` + `compiler/` tree | A separate api/ layer is a wrapper layer: every object would exist twice (api.Register vs LatticeGeometry). In-place refactor keeps one implementation per concept. |
| D2 | **No `AtomModel` class.** `LevelStructureSpec` is extended in place as the user-facing atom model. | Plan_stage1.md API 3 | `AtomModel.to_level_structure()` is a permanent sync burden between two near-identical classes; the kernel object can simply grow the user-facing fields. |
| D3 | **`Register` replaces `LatticeGeometry` in place** (rename + extend; same kernel field names `N/coords/sublattice/spacing_um`; no `to_geometry()`). | Plan_stage1.md API 1 migration strategy | The kernel consumes attribute names, not the class identity — renaming is nearly free and avoids a forwarding class forever. |
| D4 | **Top-level `ryd_gate` exports are the product surface** (Pulser-style flat namespace). | StagePlan_02 draft's "no top-level re-exports" | Product users should not memorize module paths; domain imports remain for library code. |
| D5 | **Serialization is a Stage 1 acceptance criterion** (schema-tagged plain dicts on every object), not a late stage. | Original stage ordering | Retrofitted serialization always misses fields; the tag format (`ryd-gate/<kind>/v1`) future-proofs files written before the Stage 6 schema freeze. |
| D6 | **Geometry conventions are preserved exactly** through the migration: chain sublattice `(-1)**i`, rectangle checkerboard `(-1)**(row+col)`, `make_geometry_from_coords` call sites migrate with `center=False`. | Earlier stage1_api draft (all-zero sublattice) | Staggered-magnetization analysis and basis site order are physics contracts; the refactor must be behavior-identical. |
| D7 | **`SequenceProtocol` is a kernel `Protocol` subclass**; `Sequence.draw()` reuses the existing `Protocol.plot()`/`pulse_traces` machinery; backends never see `Sequence`. | (new) | One scheduling interface and one plotting implementation in the repo. |
| D8 | **Capability-gated results** (`QuantumStateHandle` with explicit `capabilities`, typed `UnsupportedResultQuery`/`StateMaterializationError`) instead of best-effort densification. | (new) | TN backends must never silently materialize 2^N vectors; refusals must be scriptable. |
| D9 | **Kernel Blackman helpers stay in `ryd_gate.pulse`, soft-closed** (gate protocols import them by name): top-level re-exports removed, excluded from `ryd_gate.pulse.__all__`, docstring-marked internal/unstable, declared unsupported for user code in docs/stage1_api.md. Hard-rename to `_blackman_*` **rejected** — it would force edits to `gate_cz_*.py` and the legacy backends, which are on Stage 1's do-not-modify list, for no user benefit. Deprecation **rejected** — the helpers are permanent kernel functions, not a migration path. | Earlier draft ("stop treating as API" read as deletion) | `gate_cz_to.py`/`gate_cz_ar.py` are live consumers; the product `Waveform` shares the window formula but differs in shape (pure window vs flat-top), units (ns + rad/µs vs seconds), and contract (serializable/validated vs internal). The user-facing distinction is documented in stage1_api.md ("Choosing a Control Surface" + Blackman use/don't-use notes). |
| D10 | **Classmethod register constructors never auto-attach layouts** (`Register.square(...).layout is None`); a `RegisterLayout` is explicit provenance only (direct constructor or `dataclasses.replace`), and layout↔register consistency validation is deferred to the future `define_register` mapping API (Stage 6 roadmap). | Suggestion to auto-synthesize a matching layout in every classmethod — rejected | Layout presence must stay meaningful ("built from a trap pattern", the semantics behind Pulser's `requires_layout`) rather than decorative; auto-attach would serialize every register's coordinates twice for zero information; and `None`-default → auto-attach later is non-breaking, while removing auto-layouts later would break stored files. |
| D11 | **Preset names encode Hamiltonian construction semantics; `param_set` encodes numbers-only variants.** `ger` (symbolic: protocol-driven coefficients, no static `H_1013`) and `analog_3` (physical Rb87 blocks: static coupling, analog default interaction) stay two presets; `rb87_7` keeps `param_set="our"|"lukin"` because those switch numbers under identical construction. Merging into one `ger` + `param_set` axis **rejected** — it overloads the tag with semantics, and a forgotten kwarg would silently swap physics engines; `analog_3` as a deprecated alias **rejected** (no-shim rule); deleting symbolic `ger` **rejected** — it breaks protocol-driven workflows and a pinned kernel test. The legacy internal branch `("ger", param_set="analog_3")` survives for frozen notebooks and is removed in Stage 7. | Suggestion to merge `ger`/`analog_3` into one name + `param_set` tag, or to keep only the physical preset | Verified in code: `factories.py::_physical_model_for` (line 169) gives bare `ger` a different compile path (symbolic) than `analog_3` (physical `_apply_analog_3_lattice_blocks` + different default interaction); `tests/core/test_rydberg_system_model.py:195` pins it. This is a semantics difference, not a numbers difference — the opposite of `our`/`lukin`. |

## Existing Repository State (kernel that must remain intact)

```text
src/ryd_gate/core/system.py            RydbergSystem (+ from_lattice)
src/ryd_gate/core/level_structures.py  LevelStructureSpec / InteractionSpec / presets
src/ryd_gate/lattice/geometry.py       geometry kernel (becomes Register in Stage 1)
src/ryd_gate/protocols/                Protocol ABC + gate/sweep/lattice protocols
src/ryd_gate/ir/                       HamiltonianIR, EvolutionResult
src/ryd_gate/backends/                 exact, tenpy_mps, gputn, peps2d, pepskit
src/ryd_gate/simulate.py               unified dispatcher simulate(system, x, psi0, backend=...)
src/ryd_gate/analysis/                 gate metrics, error budgets, lattice observables
```

The data flow `RydbergSystem + Protocol → compile_hamiltonian_ir → backend → EvolutionResult` is the engine. Stages add product surfaces that compile **into** it, never around it.

## Stage List

| Stage | File | Goal | Backend changes |
|---|---|---|---|
| 1 | `StagePlan_01_API_Foundation.md` | Product data layer refactored in place: `Register` (replaces `LatticeGeometry`), extended `LevelStructureSpec`, `DeviceSpec`/`ChannelSpec`, `Waveform`/`Pulse`, validation primitive, serialization, drawing | None |
| 2 | `StagePlan_02_Sequence_Exact_Result.md` | `Sequence` → `SequenceProtocol` (kernel `Protocol`) → existing exact backend; `simulate_sequence`; lazy `SimulationResult` + `ExactStateHandle`; `Sequence.draw()` | None (additive `simulate_sequence` only) |
| 3 | `StagePlan_03_Backend_Native_Result_Handles.md` | Capability-aware state handles; MPS-native expectations/materialization guards; `simulate_sequence(backend="mps")` | Controlled, result-plumbing only |
| 4 | (future — outline below) | `NoiseModel` → existing Monte Carlo / non-Hermitian decay | Noise plumbing only |
| 5 | (future — outline below) | Gate library + fidelity/error-budget productization (flagship) | None |
| 6 | (future — outline below) | JSON Schema freeze + Pulser subset import/export; observable configs | None |
| 7 | (future — outline below) | Docs site, executed tutorials, capability matrix, packaging/CI polish | None |

A stage starts only after the previous stage's acceptance (including the full fast test suite) is green.

**Status (2026-06-11):** Stages 1 and 2 implemented and accepted. Stage 1: 117 new tests, in-place data layer, committed (`6f762b3`, with a dedup pass: `blackman_pulse_sqrt`→`blackman_pulse`, `Waveform`→`blackman_window`, factories→`Register.distance_pairs`). Stage 2: `Sequence`/`SequenceProtocol`/`results.py`/`simulate_sequence`, 45 new tests including a π-pulse physics check through the exact solver; full fast suite 354 passed; `backends/` and `ir/` diffs empty. Known pre-existing issue (not stage scope): `app/pages/2_lattice_simulator.py` imports several names from `ryd_gate.lattice` that died in the May 2026 refactor (`build_hamiltonian`, `evolve_sweep`, …); only its `make_square_lattice` usage was migrated. Stage 3 not started.

## Future Stage Outlines

These are scoping paragraphs, not yet binding specs; each becomes a full StagePlan file (same rigor as 01–03: current-state → refactor steps → no-wrapper rules → tests → acceptance) before implementation.

### Stage 4 — NoiseModel

Declarative `NoiseModel` frozen dataclass in `src/ryd_gate/noise.py`, compiled onto the **existing** noise machinery (`backends/exact/monte_carlo_runner.py`'s `setup_detuning_noise/setup_amplitude_noise/...` and the non-Hermitian decay flags in `local_blocks.py`) — the runner keeps doing the work; `NoiseModel` is the data that configures it. Parameter names align with Pulser where semantics match (`state_prep_error`, `p_false_pos`, `p_false_neg`, `temperature_uK`, `laser_waist_um`, `amp_sigma`, `detuning_sigma_rad_per_us`, `runs`/`n_trajectories`), plus the microscopic extensions Pulser lacks: `rydberg_decay`/`intermediate_decay` with ARC-derived branching, position noise, local RIN. Includes `noise_types` inference, `summary()`, `to_dict`/`from_dict` (`ryd-gate/noise/v1`), and per-backend/per-level-structure capability validation with `ValidationIssue` codes. Acceptance pins: new API reproduces existing `MonteCarloRunner.setup_*` results exactly on the CZ benchmark notebook's parameters.

### Stage 5 — Gate Library and Error Budgets (flagship differentiator)

Productize what already works: a documented `ryd_gate.gates` namespace re-exporting `TOProtocol`/`ARProtocol`/`DoubleARPProtocol` with curated literature-referenced docstrings; promote `analysis.gate_metrics.average_gate_infidelity` and `error_budget` to documented API with stable signatures; a `CZGateReport` result (fidelity, phase error, per-channel error budget, serialization) produced from one call. Optimization *workflows* stay in `scripts/` (repo convention: only truly reusable functions enter `src/`). Acceptance pins: reproduce the published benchmark values currently checked in `tests/protocols/test_cz_gate_phase.py` and the CZ validation notebook through the new entry points. No Pulser equivalent exists for any of this — docs must lead with it.

### Stage 6 — Serialization Freeze and Pulser Interop

Freeze `v1` JSON Schemas for every `ryd-gate/<kind>/v1` payload (schema files shipped in-package; validation optional extra, mirroring Pulser's `abstract_repr` approach). Then `src/ryd_gate/interop/pulser.py`: import a Pulser abstract-repr subset (Register; Constant/Ramp/Blackman/Interpolated/Custom waveforms; pulses; global Rydberg channel sequences; NoiseModel fields with matching semantics) into native objects, and export the same subset; every unsupported feature raises a typed error listing the offending construct. Pulser layout import brings the trap→register mapping with it: `RegisterLayout.define_register(trap_ids, qubit_ids)` lands here (see D10). Also: `EmulationConfig`-style observables-at-times for TN backends (extends the Stage 3 `measure_mps_observable` path). Acceptance: round-trip a Pulser AFM tutorial sequence and match QuTiP results within tolerance.

### Stage 7 — Docs, Examples, Packaging

Sphinx site (docs/ already has `conf.py`/`index.rst`): Getting Started, Fundamentals (units/conventions/Hamiltonian), How-Tos per feature, auto-generated API reference, a **capability matrix** page (level structure × backend × noise — generated from `supports_backend` so it cannot rot), and the gate-modeling tutorial as the flagship example. Executed-notebook CI (Pulser's `check-notebooks` pattern) for `examples/`. Notebook migration to the product API happens here, and with it the removal of the legacy `("ger", param_set="analog_3")` inference branch in `factories.py` (D11) — after this stage, bare `ger` is symbolic regardless of `param_set`. Packaging polish: `py.typed`, ruff + mypy gates, SemVer + CHANGELOG, optional-dependency matrix documented (`tn`, `tn-2d`, `gputn-cu12`, `app`), PyPI readiness. README quickstart rewritten around the Stage 2 sequence example plus the Stage 5 gate example.

## Global Non-Negotiable Rules

1. Do not delete or rewrite `RydbergSystem`, `Protocol`, or `compile_hamiltonian_ir`; product surfaces compile into them.
2. No wrapper layers: one class per concept, refactored in place (Decision Log D1–D3, D7). A new class is justified only by genuinely new behavior, and must state that justification in its stage plan.
3. `Register` stays geometry-only (no level structure, interaction strengths, pulses, or backend options). `DeviceSpec` stays data (no backend, no job state). `Sequence` never builds Hamiltonian matrices.
4. No silent dense statevectors for TN backends — capability gates and typed errors only.
5. No new mandatory dependencies through Stage 4 (stdlib + existing deps; jsonschema arrives as an *optional* extra in Stage 6).
6. Existing `simulate(system, x, ...)` signature and behavior stay bit-identical until a stage plan explicitly changes them.
7. Every new public object ships `to_dict`/`from_dict` with a `ryd-gate/<kind>/v1` tag and a round-trip test, in the same stage it is introduced.
8. Validation methods return `list[ValidationIssue]` and never raise; `raise_for_errors` is the only raise boundary; issue codes are API and tests assert on them.
9. The **full fast suite** (`uv run pytest -m "not slow" -q`) must be green at the end of every stage — new-tests-only acceptance is not acceptance. TN-touching test runs use `OMP_NUM_THREADS=1` (repo convention for this machine).
10. Notebooks (`*.ipynb`) are untouched until Stage 7; `examples/`, `scripts/`, and `app/` are migrated whenever a stage removes a name they import (the stage's call-site table must list them).
11. Removed names are removed — no compatibility aliases, no deprecation shims (pre-1.0 repo, internal call sites are migrated in the same change).

## How to Execute a Stage

1. Read the stage plan and its referenced sections of `docs/stage1_api.md`; resolve any contradiction by editing the documents first (review-first rule).
2. Implement in the plan's step order; each step ends with its verify command green.
3. Finish with the stage's acceptance block, including the full fast suite and the grep checks.
4. Update this README's stage table status and, if scope changed, the Decision Log.
