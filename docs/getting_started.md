# Getting Started

`ryd-gate` is a Rydberg neutral-atom many-body simulator. Continuous-time
pulse **protocols** are the single control surface: a protocol bound to a
`RydbergSystem` lowers to a unified Hamiltonian IR and runs on the exact
state-vector backend or the tensor-network backends (MPS / PEPS / gputn).

## Install

```bash
pip install -e .                 # base (exact backend)
pip install -e ".[tn]"           # + TeNPy MPS backend
pip install -e ".[schema]"       # + JSON Schema validation
pip install -e ".[dev,docs]"     # development / documentation tooling
```

## First simulation (TFIM quench)

```python
import numpy as np
from ryd_gate import Register, RydbergSystem, TFIMQuenchProtocol, simulate

protocol = TFIMQuenchProtocol(hx=2 * np.pi * 1e6, t_gate=0.5e-6)
system = RydbergSystem.from_lattice(
    Register.square(2, spacing_um=9.0), "1r", protocol=protocol,
)
result = simulate(system, psi0="all_1", observables=["sum_nr"])
print(result.expectation("sum_nr"))   # Rydberg population (read off the result)
```

Swap `backend="exact"` for `"mps"`, `"peps"`, or `"gputn"` to run the same
system on a tensor-network engine — see the
[Capability Matrix](capability_matrix.md) for what runs where. The research
notebooks under `scripts/notebooks/` cover TFIM critical fields, quench
benchmarks, and 2D lattice dynamics end to end.

## First gate report (CZ line)

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
