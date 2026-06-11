# Getting Started

`ryd-gate` simulates neutral-atom (Rydberg) quantum systems with two control
surfaces over one exact/tensor-network kernel:

- the **Sequence API** — discrete, device-validated pulse schedules
  (Pulser-style: registers, channels, waveforms in rad/µs and integer ns);
- the **Protocol API** — continuous-time gate and many-body protocols for
  research workflows (time-optimal CZ gates, TFIM quenches, sweeps).

## Install

```bash
pip install -e .                 # base (exact backend)
pip install -e ".[tn]"           # + TeNPy MPS backend
pip install -e ".[schema]"       # + JSON Schema validation
pip install -e ".[dev,docs]"     # development / documentation tooling
```

## First simulation (Sequence API)

```python
import numpy as np
from ryd_gate import DeviceSpec, Pulse, Register, Sequence, Waveform, simulate_sequence

seq = Sequence(Register.chain(1, 20.0), DeviceSpec.virtual_rb87(), "1r")
seq.declare_channel("ryd", "rydberg_global")
seq.add(Pulse.constant_detuning(Waveform.blackman(1000, area=np.pi), 0.0), "ryd")

result = simulate_sequence(seq)
print(result.populations("r"))     # ~[1.0]: a pi pulse drives |1> -> |r>
print(result.sample(1000, basis="rydberg", seed=1))
```

## First gate report (Protocol API)

```python
from ryd_gate import Register, RydbergSystem
from ryd_gate.gates import TOProtocol, cz_gate_report

system = RydbergSystem.from_lattice(
    Register.chain(2, spacing_um=3.0), "rb87_7", param_set="our",
    blackmanflag=True, detuning_sign=1,
)
report = cz_gate_report(system, TOProtocol(), X_TO_DARK, include_error_budget=False)
print(report.fidelity)             # ~0.9999992 for the benchmark parameters
```

Continue with [Fundamentals](fundamentals.md) for units and conventions, the
how-to guides for each feature, and the [Capability Matrix](capability_matrix.md)
for what runs where.
