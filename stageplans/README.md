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
| D12 | **Pulser parity abandoned; the product surface is the Protocol-only TFIM/Rydberg simulator.** The Sequence surface (`Sequence`, `simulate_sequence`, `DeviceSpec`/`ChannelSpec`, product `Waveform`/`Pulse`, `ObservableConfig`, `SimulationResult` + state handles, `sequence_from_protocol`, the Pulser abstract-repr interop) is removed, along with its six frozen schemas (sequence/device/channel/pulse/waveform/observable-config). Frozen `v1` serialization survives only for the reproducibility set: register, register-layout, level-structure, noise, cz-gate-report. Kernel Blackman helpers stay in `ryd_gate.pulse` (per D9). | Stages 1–3, 6, 8 Sequence/interop commitments (D7, D8 retired with the surface) | Every research workflow in `scripts/` and `scripts/notebooks/` uses the Protocol path; the Sequence surface only appeared in its own docs/tests/examples. Maintaining two narratives cost docs/tests/API surface in the repo's main line (TFIM/quench/lattice dynamics + CZ gate physics) for no research benefit. |
| D13 | **The symbolic `ger` preset is removed; `analog_3` is the only built-in three-level ladder.** Custom symbolic three-level models remain supported as hand-built `LevelStructureSpec` instances passed to `RydbergSystem.from_lattice` (the generic factory registers the same `drive_420`/`H_1013` symbolic blocks); the D11 names-carry-semantics rule is unchanged — custom specs never mount physical blocks, regardless of `param_set`. | D11 (the `ger`/`analog_3` preset pair; the rule survives, the preset does not) | `ger` had zero notebook/script/example users — only the D11 split tests referenced it. It sat between `analog_3` (more physical) and `rb87_7` (more complete) while adding a third semantics to explain. The escape hatch for custom symbolic models is the spec class itself, not a preset name. |

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
| 4 | `StagePlan_04_NoiseModel.md` | `NoiseModel` → existing Monte Carlo / non-Hermitian decay | Noise plumbing only |
| 5 | `StagePlan_05_Gate_Library_and_Error_Budgets.md` | Gate library + fidelity/error-budget productization (flagship) | None |
| 6 | `StagePlan_06_Serialization_Pulser_Interop.md` | JSON Schema freeze + Pulser subset import/export; observable configs | None |
| 7 | `StagePlan_07_Docs_Examples_Packaging.md` | Docs site, executed tutorials, capability matrix, packaging/CI polish | None |
| 8 | `StagePlan_08_Convergence.md` | Surface convergence: sequence phase (virtual-Z) + local targeting + gputn/peps on the sequence path; `sequence_from_protocol` lossy bridge; convergence docs | Controlled (sequence-layer codes renamed/retired: `*_not_stage2`, `backend_not_stage3`) |

A stage starts only after the previous stage's acceptance (including the full fast test suite) is green.

**Status (2026-06-11):** Stages 1, 2, and 3 implemented and accepted. Stage 1: 117 new tests, in-place data layer, committed (`6f762b3`, with a dedup pass: `blackman_pulse_sqrt`→`blackman_pulse`, `Waveform`→`blackman_window`, factories→`Register.distance_pairs`). Stage 2: `Sequence`/`SequenceProtocol`/`results.py`/`simulate_sequence`, 45 new tests including a π-pulse physics check through the exact solver; committed (`a6d0398`). Cleanup: exact legacy backend, Streamlit app, and D4 symmetry side branch removed (`24d6a38`). Stage 3: capability-aware result handles, `simulate_sequence(..., backend="mps")`, MPS-native expectations, guarded statevector materialization, and unsupported handles for non-native backends; full fast suite 324 passed. Stage 4: declarative `NoiseModel` + `configure_monte_carlo_runner` over the existing exact Monte Carlo runner and construction-time decay flags, 47 new tests; includes a kernel bug fix (`DenseODEBackend` silently corrupted scipy-sparse IR terms via `np.asarray`); full fast suite 371 passed. Stage 5: `ryd_gate.gates` namespace + `CZGateReport`/`cz_gate_report` over the shared `_cz_overlaps` helper (single-solve refactor of `average_gate_infidelity`), benchmark pins (TO dark 7.8e-7 is the only true high-fidelity point; the legacy X_AR and Saffman-our Double-ARP configurations are deterministic path pins — an AR re-optimization workflow was added at `scripts/optimize_ar_cz.py`); full fast suite 391 passed. Stage 6: frozen `v1` JSON Schemas for all 11 payload kinds (`src/ryd_gate/schemas/`, optional `jsonschema` via the `schema`/`interop` extras), Pulser abstract-repr subset bridge (`interop/pulser.py`, typed `PulserInteropError` codes, no Pulser dependency), `RegisterLayout.define_register` (D10), and `ObservableConfig` streaming schedules lowered onto the Stage 3 TeNPy measurement path; full fast suite 461 passed. Stage 7: Sphinx product docs (guides + generated capability matrix + autodoc reference), README/CHANGELOG rewrite, four new bounded-runtime examples, all 11 notebooks migrated to the product API (gated execution via nbconvert: `docs/_scripts/run_notebooks.py`; cz + ac-stark notebooks execute clean), D11 legacy branch removed (bare `ger` is symbolic), `py.typed` + scoped mypy gate green, repo-wide ruff clean, `docs` extra split out; full fast suite 469 passed. **All seven stages complete.** Stage 8 (convergence, `StagePlan_08_Convergence.md`): sequence pulse phase with virtual-Z accumulation (complex coefficients, exact backend; typed TN refusal), `Sequence.target(...)`/`TargetOp` local addressing over the existing per-site channel-lowering rules (exact + mps), `gputn`/`peps` wired onto the sequence path (verified end-to-end with the installed yastn stack), and the explicit lossy `sequence_from_protocol` bridge; the `SequenceProtocol` convergence point is now documented in `docs/fundamentals.md`.

## Remaining Stage Specs

Stages 4-7 now have binding specs. The short notes below are only an index; the linked files are the implementation contracts.

### Stage 4 — NoiseModel

Binding spec created: [StagePlan_04_NoiseModel.md](StagePlan_04_NoiseModel.md).

### Stage 5 — Gate Library and Error Budgets (flagship differentiator)

Binding spec created: [StagePlan_05_Gate_Library_and_Error_Budgets.md](StagePlan_05_Gate_Library_and_Error_Budgets.md).

### Stage 6 — Serialization Freeze and Pulser Interop

Binding spec created: [StagePlan_06_Serialization_Pulser_Interop.md](StagePlan_06_Serialization_Pulser_Interop.md).

### Stage 7 — Docs, Examples, Packaging

Binding spec created: [StagePlan_07_Docs_Examples_Packaging.md](StagePlan_07_Docs_Examples_Packaging.md).

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
10. Notebooks (`*.ipynb`) are untouched until Stage 7; `examples/` and `scripts/` are migrated whenever a stage removes a name they import (the stage's call-site table must list them).
11. Removed names are removed — no compatibility aliases, no deprecation shims (pre-1.0 repo, internal call sites are migrated in the same change).

## How to Execute a Stage

1. Read the stage plan and its referenced sections of `docs/stage1_api.md`; resolve any contradiction by editing the documents first (review-first rule).
2. Implement in the plan's step order; each step ends with its verify command green.
3. Finish with the stage's acceptance block, including the full fast suite and the grep checks.
4. Update this README's stage table status and, if scope changed, the Decision Log.
