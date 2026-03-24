# Solver Validation

This document describes the validation test suite that verifies the correctness of the `ryd_gate` implementation.

## Overview

| Solver | Module | Hilbert Space | Method |
|--------|--------|---------------|--------|
| **Schrodinger** | `ryd_gate.solvers.schrodinger` | 49-dim (7-level x 2 atoms) | SciPy `solve_ivp` (DOP853) |

## Test Organization

Tests are split into **fast** (default) and **slow** (ODE-based) categories:

```bash
uv run pytest              # Fast tests only (~1s)
uv run pytest -m slow      # Slow ODE tests only (~15min)
uv run pytest -m ""        # All tests
```

### Fast Tests (no ODE solves)

| Class | What it validates |
|-------|-------------------|
| `TestCZGateSimulatorInit` | Constructor parameters, param_set/strategy dispatch, error flags |
| `TestHamiltonianConstruction` | Matrix shapes (49x49), Hermiticity, coupling structure |
| `TestMonteCarloSimulation` (partial) | `MonteCarloResult` dataclass, constructor validation, vdW operator |
| `TestMonteCarloJax` | NotImplementedError for AR strategy |
| `TestNominalDistance` | Nominal distance values |
| `TestDiagnoseEdgeCases` | ValueError for missing initial_state |
| `TestMonteCarloResultSerialization` | Save/load roundtrip for MC results |
| `TestBuildSSSStateMap` | 12 SSS state labels, normalization |
| `TestOptimizeDispatch` | Invalid strategy/fid_type errors |
| `TestSetupProtocolAR` | Parameter storage workflow |
| `TestPlotBlochDispatch` | AR strategy prints message |
| `TestDiagnosePlotDispatch` | Invalid strategy at init raises |
| `test_init.py` | Package imports and `__all__` exports |
| `test_blackman.py` | Blackman window/pulse math |

### Slow Tests (ODE-based)

| Class | What it validates |
|-------|-------------------|
| `TestFidelityCalculation` | Fidelity bounds, return types, Lukin params |
| `TestStateEvolution` | State vector shapes, norm preservation |
| `TestDiagnosticMethods` | diagnose_run output shapes, all initial states |
| `TestStoredParameterWorkflow` | setup_protocol -> gate_fidelity workflow |
| `TestIndependentErrorFlags` | Each error flag independently, Hermiticity, perfect gate |
| `TestBranchingRatios` | Branching sums, error budget structure, population evolution |
| `TestMonteCarloWithBranching` | MC branching decomposition, residuals, save/load |
| `TestStateInfidelity` | Per-state infidelity for labels and ndarray inputs |
| `TestGetGateResult` | get_gate_result for TO/AR with and without t_eval |
| `TestFidelityTypes` | Bell, SSS, and AR non-average fidelity types |
| `TestARFidelityResiduals` | AR return_residuals support |
| `TestMCProgressPrint` | MC progress indicator output |
| `test_cz_gate_phase.py` | CZ phase relation, single-qubit symmetry, basis fidelity |

## SSS State Construction

The Symmetric State Set (SSS) consists of 12 specific two-qubit input states:

```python
SSS-0: 0.5(|00> + |01> + |10> + |11>)   # Equal superposition
SSS-1: 0.5(|00> - |01> - |10> + |11>)   # Bell-like
SSS-4: |00>                              # Computational basis
SSS-5: |11>                              # Computational basis
# ... and 8 more with various phase factors
```

**Validated properties:**
- Exactly 12 SSS states plus 4 computational basis states
- Each state normalized to 1
- States only occupy |0>, |1> computational basis

## Numerical Tolerances

| Property | Tolerance | Justification |
|----------|-----------|---------------|
| State normalization | 1e-10 | Direct computation |
| Fidelity bounds | +/-1e-10 | Floating point precision |
| ODE integration | rtol=1e-8, atol=1e-12 | DOP853 adaptive stepping |

## How to Run Tests

```bash
# Fast tests (default, ~1 second)
uv run pytest

# All tests (~15 minutes)
uv run pytest -m ""

# With coverage
uv run pytest -m "" --cov=ryd_gate --cov-report=html

# Specific file
uv run pytest tests/test_ideal_cz.py -v -m ""
```

## References

- [PRX Quantum: Benchmarking and Fidelity Response Theory of High-Fidelity Rydberg Entangling Gates](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.6.010331)
