# Fundamentals: Units, Conventions, Lowering

## Units

| layer | time | amplitude / detuning |
|---|---|---|
| product (`Waveform`, `Pulse`, `Sequence`) | integer **ns** | **rad/µs** |
| kernel (`Protocol`, Hamiltonian IR, backends) | seconds | rad/s |

The conversion is exact: a waveform value `v` rad/µs enters the kernel as
`v * 1e6` rad/s; a duration `d` ns becomes `d * 1e-9` s.

## Coefficient lowering (Sequence → Hamiltonian)

A `Sequence` never reaches a backend directly. It compiles to a kernel
`SequenceProtocol`, which owns the single lowering convention:

- the **amplitude** waveform Ω(t) drives the channel's amplitude compiler
  channel with coefficient **Ω(t)/2** (rad/s);
- the **detuning** waveform Δ(t) drives the detuning compiler channel with
  coefficient **−Δ(t)** (rad/s).

Channel maps live on `ChannelSpec` (`amplitude_channels` /
`detuning_channels`, keyed by level-structure name). For
`DeviceSpec.virtual_rb87()`:

| level structure | amplitude channel | detuning channel |
|---|---|---|
| `1r` | `global_X` | `global_n` |
| `01r` | `drive_R` | `delta_R` |

## Level structures: names carry semantics

Preset *names* encode Hamiltonian construction; `param_set` only switches
numerical parameter sets within identical semantics:

- `ger` — symbolic three-level ladder, blocks driven by protocols;
- `analog_3` — *physical* Rb87 analog three-level blocks (static `H_1013`,
  decay-capable `H_const`);
- `rb87_7` — full seven-level Rb87 gate model (`param_set="our"` or
  `"lukin"`).

Since Stage 7 (Decision Log D11), bare `"ger"` is symbolic regardless of
`param_set`.

## Data flow

```
Register + LevelStructureSpec (+ InteractionSpec)
    └─ RydbergSystem.from_lattice(..., protocol=...)
         └─ compile_hamiltonian_ir → backend (exact / mps / gputn / peps)
              └─ EvolutionResult  (kernel)  →  SimulationResult (product, lazy)
```

Geometry conventions are preserved from the original lattice factories:
chain sublattice signs `(-1)**i`, rectangle `(-1)**(row+col)`, triangular
all-zero; `Register.from_coordinates` infers spacing from the smallest
positive sorted-x difference.

## Serialization

Every product object writes a plain dict tagged
`"schema": "ryd-gate/<kind>/v1"` and reconstructs via `from_dict`. The `v1`
payloads are frozen as JSON Schema files in `ryd_gate/schemas/` (see the
[interop how-to](how_to_interop.md)).
