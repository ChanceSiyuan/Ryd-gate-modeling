# Stage Plans for Product API Refactor

This directory contains review-first implementation plans. These files are not implementation notes; they are binding task specifications. If a future implementation step needs to deviate from a stage plan, update the relevant stage plan first and get it reviewed.

## Existing Repository State

Do not overwrite or reinterpret `Plan_stage1.md`. That file currently contains a broad product API plan. The files in this directory split that broad plan into deterministic implementation stages.

Current known working core must remain intact:

- `src/ryd_gate/lattice/geometry.py`
- `src/ryd_gate/core/level_structures.py`
- `src/ryd_gate/core/system.py`
- `src/ryd_gate/protocols/`
- `src/ryd_gate/ir/`
- `src/ryd_gate/backends/`
- `src/ryd_gate/simulate.py`

## Stage Boundary Decision

The key decision is:

**Stage 1 does not implement MPS, PEPS, GPUTN, or backend-native result handles.**

Reason:

- Stage 1's purpose is to establish the obvious user-facing API data layer.
- `MPSStateHandle`, `PEPSStateHandle`, sequential MPS sampling, PEPS contraction, and backend-native state retention require changes to algorithm backend contracts.
- Those changes are deeper than API construction and must be isolated in a later stage to avoid mixing public API design with numerical backend refactoring.

Stage 1 may mention result and state-handle names only as reserved vocabulary. It must not implement them.

## Stage List

| Stage | File | Goal | Backend Changes |
|---|---|---|---|
| 1 | `StagePlan_01_API_Foundation.md` | Add deterministic user-facing data structures: `Register`, `AtomModel`, `DeviceSpec`, `ChannelSpec`, `Waveform`, `Pulse`, validation primitives | None |
| 2 | `StagePlan_02_Sequence_Exact_Result.md` | Add `Sequence`, compile it to existing `RydbergSystem + Protocol`, run exact backend, return lazy exact result wrapper | Exact only, through existing `simulate()` |
| 3 | `StagePlan_03_Backend_Native_Result_Handles.md` | Add backend-native state-handle interface and implement exact/MPS capability-aware lazy queries | Yes, controlled backend contract changes |

Future stages not yet specified in this directory:

- Stage 4: `NoiseModel` and existing Monte Carlo integration.
- Stage 5: Pulser subset import/export.
- Stage 6: docs, capability matrix, and examples.

## Global Non-Negotiable Rules

1. Do not delete or rewrite `RydbergSystem`.
2. Do not delete or rewrite `Protocol`.
3. Do not bypass `compile_hamiltonian_ir`.
4. Do not make `Register` contain `AtomModel`, pulse, interaction strength, or backend options.
5. Do not make `DeviceSpec` represent a backend or QPU job.
6. Do not make `Sequence` generate dense Hamiltonian matrices directly.
7. Do not materialize dense statevectors for TN backends by default.
8. Do not add new mandatory dependencies in Stage 1 or Stage 2.
9. Do not modify notebooks in these stages.
10. Do not change existing public behavior of `ryd_gate.simulate(system, x, ...)` until a stage plan explicitly allows it.

