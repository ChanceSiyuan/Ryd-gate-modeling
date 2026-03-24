# ryd-gate

[![CI](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simulation and optimization of Rydberg-atom entangling gates for neutral-atom quantum computing.

This package provides tools for modelling two-photon Rydberg excitation in 87Rb, including hyperfine structure, Rydberg blockade, spontaneous decay, and AC Stark shifts. It uses a SciPy-based Schrodinger-equation solver with pulse-shape optimisation routines for time-optimal (TO) and amplitude-robust (AR) CZ gate protocols.

## Features

- **7-level Schrodinger solver** in a 49-dimensional two-atom Hilbert space
- **Modular architecture**: `core/`, `protocols/`, `solvers/`, `analysis/` subpackages
- **Strategy pattern**: TO and AR protocols share a unified ODE solver
- **Full error model** including spontaneous decay (Rydberg + intermediate), dephasing, and position errors
- **Monte Carlo noise analysis** for Rydberg dephasing and 3D position fluctuations
- **Local addressing simulation** with 784nm pinning laser and sweep protocols
- **Pulse optimization** for TO and AR gate protocols

## Installation

```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# Using pip
pip install -e ".[dev]"
```

## Quickstart

```python
from ryd_gate import CZGateSimulator

sim = CZGateSimulator(param_set='our', strategy='TO')

# Time-optimal pulse parameters: [A, w/Omega_eff, phi_0, delta/Omega_eff, theta, T/T_scale]
X_TO = [0.1122, 1.0431, -0.72565603, 0.0, 0.452, 1.219096]
infidelity = sim.gate_fidelity(X_TO)
print(f"Gate infidelity: {infidelity:.2e}")
```

### Direct module imports (recommended for new code)

```python
from ryd_gate.core.atomic_system import create_atomic_system
from ryd_gate.protocols.gate_cz_to import TOProtocol
from ryd_gate.solvers.schrodinger import solve_gate
from ryd_gate.analysis.gate_metrics import average_gate_infidelity

system = create_atomic_system(param_set="our")
protocol = TOProtocol()
infidelity = average_gate_infidelity(system, protocol, X_TO)
```

## Package Layout

```
src/ryd_gate/
  core/
    atomic_system.py       # AtomicSystem dataclass, Hamiltonian builders, branching ratios
  protocols/
    base.py                # Protocol ABC (CZ gates) + SweepProtocol ABC (addressing)
    gate_cz_to.py          # TOProtocol (cosine phase modulation, 6 params)
    gate_cz_ar.py          # ARProtocol (dual-sine phase modulation, 8 params)
    local_sweep.py         # SweepAddressingProtocol (linear sweep + local pinning)
  solvers/
    schrodinger.py         # solve_gate() for CZ gates + evolve() for generic H(t)
    monte_carlo.py         # MonteCarloEngine (CZ) + AddressingMCEngine (local addressing)
  analysis/
    gate_metrics.py        # average_gate_infidelity, error_budget, state_infidelity
    addressing_metrics.py  # AddressingEvaluator (pinning error, crosstalk, leakage)
  ideal_cz.py             # Backward-compatible CZGateSimulator facade
  blackman.py             # Blackman window pulse shaping
```

## Physical Model

The solver models two 87Rb atoms with:

| Parameter | Value | Description |
|-----------|-------|-------------|
| n | 70 | Rydberg principal quantum number |
| Omega_eff | 2pi x 7 MHz | Effective two-photon Rabi frequency |
| Delta | 2pi x 9.1 GHz | Intermediate state detuning |
| d | 3 um | Interatomic distance |
| C6 | 2pi x 874 GHz um^6 | van der Waals coefficient |

## Testing

```bash
# Run fast tests only (default, ~1 second)
uv run pytest

# Run all tests including slow ODE-based tests (~15 min)
uv run pytest -m ""

# Run only slow tests
uv run pytest -m slow

# Run with coverage
uv run pytest -m "" --cov=ryd_gate --cov-report=html
```

### Test Files

| Test File | Description |
|-----------|-------------|
| `test_ideal_cz.py` | Schrodinger solver: initialization, Hamiltonian, fidelity, Monte Carlo |
| `test_cz_gate_phase.py` | CZ gate phase and SSS state verification |
| `test_blackman.py` | Blackman pulse shaping functions |
| `test_init.py` | Package-level imports and exports |

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting_started.md) | Installation and basic usage |
| [Schrodinger Solver](docs/schrodinger_solver.md) | 7-level model theory and API |
| [Error Budget](docs/error_budget_methodology.md) | Error decomposition methodology |
| [Validation](docs/validation.md) | Test suite documentation |

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/calibration_sensitivity.py` | Calibration sensitivity analysis |
| `scripts/error_deterministic.py` | Deterministic error budget calculation |
| `scripts/error_monte_carlo.py` | Monte Carlo error analysis |
| `scripts/generate_mc_data.py` | Generate Monte Carlo datasets |
| `scripts/generate_si_tables.py` | Generate SI tables for the paper |
| `scripts/opt_bright.py` | Optimize bright-detuning parameters |
| `scripts/opt_dark.py` | Optimize dark-detuning parameters |
| `scripts/plot_mid_pop.py` | Plot intermediate state populations |
| `scripts/plot_population_evolution_sch.py` | Plot population evolution |
| `scripts/verify_cz_dark.py` | Verify CZ gate with dark-detuning parameters |
| `scripts/run_addressing_sim.py` | Local addressing pinning error heatmap |

## References

* Evered *et al.*, "High-fidelity parallel entangling gates on a neutral-atom quantum computer", Nature **622**, 268 (2023).
* Ma *et al.*, "Benchmarking and fidelity response theory of high-fidelity Rydberg entangling gates", PRX Quantum **6**, 010331 (2025).

## License

MIT
