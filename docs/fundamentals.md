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

## One kernel, two surfaces: the SequenceProtocol convergence point

There is exactly **one** simulation system. A `Sequence` never reaches a
backend: it compiles to `SequenceProtocol` — an ordinary kernel `Protocol`
subclass — and from there shares every stage of the pipeline with the
research protocols:

```
Sequence ──► SequenceProtocol ─┐
                               ├─► RydbergSystem (+ bound Protocol)
TOProtocol / SweepProtocol ────┘        └─ compile_hamiltonian_ir
TFIMQuenchProtocol / ...                     └─ backend (exact / mps / gputn / peps)
                                                  └─ EvolutionResult (kernel)
                                                       └─ SimulationResult (product, lazy)
```

The two surfaces differ only in what they are *made of*:

- a `Sequence` is **data** — device-validated, integer-ns, serializable to
  frozen `v1` payloads, Pulser-interoperable;
- a `Protocol` is **code** — continuous-time functions plus an optimization
  parameter vector `x`, resolved against the physical system at compile
  time.

A Sequence is therefore always expressible as a Protocol (that is the
lowering); the reverse is lossy, and the explicit bridge for it is
`sequence_from_protocol` (see below).

## Phase and local addressing (Stage 8)

Pulse phases lower to complex amplitude coefficients:
`(Ω(t)/2)·e^{−iφ}` with `φ = phase_rad + Σ post_phase_shift_rad` of earlier
pulses on the channel (Pulser virtual-Z semantics). The TN profile lowering
keeps only real parts, so phase-modulated sequences run on the exact backend
and raise `sequence.phase_backend_unsupported` elsewhere — never silently
dropped. Local channels (`Sequence.target(...)`) emit per-site compiler keys
`f"{base}_{site}"`, resolved by the same kernel channel-lowering rules
(`split_site_channel` / `block_name_for_drive_channel`) the research
protocols use.

## Protocol → Sequence (explicit, lossy)

```python
from ryd_gate import sequence_from_protocol

seq = sequence_from_protocol(system, x, dt_ns=1)   # samples the bound protocol
```

The bridge inverts the lowering exactly on the sampled grid and stamps
`{discretized_from, dt_ns, t_gate_s, x}` into the pulse metadata. Protocols
that drive channels beyond one device channel's (amp, det) pair — the
two-photon `drive_420` gate set, per-site addressing functions — are refused
with `discretize.channel_not_representable`: gate-grade fidelity (~1e-7)
lives on the continuous path by design.

Geometry conventions are preserved from the original lattice factories:
chain sublattice signs `(-1)**i`, rectangle `(-1)**(row+col)`, triangular
all-zero; `Register.from_coordinates` infers spacing from the smallest
positive sorted-x difference.

## Serialization

Every product object writes a plain dict tagged
`"schema": "ryd-gate/<kind>/v1"` and reconstructs via `from_dict`. The `v1`
payloads are frozen as JSON Schema files in `ryd_gate/schemas/` (see the
[interop how-to](how_to_interop.md)).
