# How-To: Build and Run Sequence Programs

## Build

```python
import numpy as np
from ryd_gate import DeviceSpec, Pulse, Register, Sequence, Waveform

register = Register.chain(4, spacing_um=6.0)        # or square/rectangle/triangular/from_coordinates
device = DeviceSpec.virtual_rb87()

seq = Sequence(register, device, "1r")               # validates register + model against the device
seq.declare_channel("ryd", "rydberg_global")         # user alias -> device channel

seq.add(Pulse.constant_detuning(Waveform.blackman(800, area=np.pi), 0.0), "ryd")
seq.delay(200, "ryd")
seq.add(Pulse.constant(400, 2.0, -1.0), "ryd")       # square amplitude + detuning
seq.measure("rydberg")                               # locks the sequence
```

Construction fails fast: invalid registers, unsupported level structures,
channel collisions, and channel-limit violations raise `ValueError` with
stable codes (`register.min_distance`, `sequence.channel_model_mismatch`,
`pulse.amplitude_limit`, ...). Hyperfine channels stay gated
(`sequence.hyperfine_not_stage2`).

## Phase and local addressing (Stage 8)

```python
# Ramsey: two pi/2 pulses, second one phase-shifted (virtual-Z semantics)
seq.add(Pulse.constant_detuning(Waveform.blackman(1000, area=np.pi/2), 0.0), "ryd")
seq.add(Pulse.constant_detuning(Waveform.blackman(1000, area=np.pi/2), 0.0,
                                phase_rad=np.pi), "ryd")     # echoes back to |1>

# Local channel: target, pulse, retarget
seq2 = Sequence(Register.chain(2, 6.0), device, "1r")
seq2.declare_channel("loc", "rydberg_local")
seq2.target("q0", "loc")
seq2.add(Pulse.constant_detuning(Waveform.blackman(1000, area=np.pi), 0.0), "loc")
seq2.target("q1", "loc")                                      # replayable TargetOp
```

A pulse's effective phase is `phase_rad` plus the accumulated
`post_phase_shift_rad` of earlier pulses on the same channel. Phase lowers
to complex drive coefficients and is exact-backend only
(`sequence.phase_backend_unsupported` on TN backends — never silently
dropped). Local targeting works on exact and mps.

## Run

```python
from ryd_gate import simulate_sequence

result = simulate_sequence(seq)                      # exact backend (default)
result.expectation("sum_nr")                          # cached lazy queries
result.populations("r")                               # per-site, register order
result.sample(1000, basis="rydberg", seed=7)          # multinomial bitstrings
result.statevector()                                  # dense state (exact only)
result.raw                                            # kernel EvolutionResult
```

On the MPS backend the result holds a **native TeNPy state** — expectations
are computed in MPS form, dense materialization is guarded:

```python
result = simulate_sequence(seq, backend="mps", backend_options={"chi_max": 64, "dt": 2.5e-7})
result.expectation("sum_nr")                          # native MPS expectation
result.statevector(max_dim=4096)                      # explicit opt-in, tiny systems only
result.capabilities                                   # frozenset of supported queries
```

Streaming observables during MPS evolution use
[`ObservableConfig`](how_to_interop.md):

```python
from ryd_gate import ObservableConfig

config = ObservableConfig(("sum_nr",), every_ns=200)
result = simulate_sequence(seq, backend="mps", observables=config,
                           backend_options={"chi_max": 64, "dt": 2.5e-7})
result.raw.metadata["obs"]["sum_nr"]                  # values at 0, 200, ... ns
```

## Serialize and draw

```python
data = seq.to_dict()                                  # "ryd-gate/sequence/v1"
seq2 = Sequence.from_dict(data)                       # replays through full validation
seq.draw()                                            # renders via the kernel protocol plotter
```
