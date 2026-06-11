# How-To: CZ Gate Reports and Error Budgets

Gate-level Rydberg modeling is this package's flagship: two-qubit CZ
protocols, Nielsen average gate fidelity, and microscopic XYZ/AL/LG error
budgets — none of which has a Pulser equivalent.

## The gate namespace

```python
from ryd_gate.gates import (
    ARProtocol, DoubleARPProtocol, TOProtocol,      # the kernel protocol classes
    average_gate_infidelity, error_budget,           # the analysis functions
    CZGateReport, cz_gate_report,                    # the stable report surface
)
```

`ryd_gate.gates` re-exports kernel objects — it is a documented namespace,
not a second implementation.

## One-call gate report

```python
from ryd_gate import Register, RydbergSystem

X_TO_DARK = [-0.6989301339711643, 1.0296229082590798, 0.3759232324550267,
             1.5710180991068543, 1.4454279613697887, 1.3406239758422793]

system = RydbergSystem.from_lattice(
    Register.chain(2, spacing_um=3.0), "rb87_7", param_set="our",
    blackmanflag=True, detuning_sign=1,
)
report = cz_gate_report(system, TOProtocol(), X_TO_DARK, include_error_budget=False)

report.infidelity        # ~7.8e-7 (benchmark pin)
report.fidelity          # 1 - infidelity (derived, never stored)
report.phase_error_rad   # residual conditional-phase error, wrapped to [-pi, pi]
report.theta_rad         # the single-qubit Rz phase the protocol targets
report.residuals         # mean leakage populations: e1/e2/e3/ryd/ryd_garb
report.to_dict()         # "ryd-gate/cz-gate-report/v1"
```

The report evolves |00⟩, |01⟩, |11⟩ **once** through the same machinery as
`average_gate_infidelity` (shared overlap core); it never calls a backend
directly.

## Error budgets

With decay enabled at construction time, `error_budget` decomposes the
infidelity by physical source and decay channel class:

```python
system_decay = RydbergSystem.from_lattice(
    Register.chain(2, spacing_um=3.0), "rb87_7", param_set="our",
    blackmanflag=True, detuning_sign=1, enable_rydberg_decay=True,
)
report = cz_gate_report(system_decay, TOProtocol(), X_TO_DARK)
report.error_budget["rydberg_decay"]    # {"total": ..., "XYZ": ..., "AL": ..., "LG": ...}
```

## Benchmark status (pinned in tests/gates/test_cz_benchmark_pins.py)

- **TO dark** (`X_TO_DARK`, blackman flat-top, detuning_sign=+1): infidelity
  ≈ 7.8e-7 — the validated high-fidelity operating point.
- The legacy AR parameter set and the Saffman-constants Double-ARP
  configuration are **not** high-fidelity optima under the current protocol
  conventions; their pins guard the computation path, not gate quality.
  Re-optimization lives in `scripts/optimize_ar_cz.py`.

Noisy gate statistics combine this with the [NoiseModel](how_to_noise.md)
Monte Carlo runner.
