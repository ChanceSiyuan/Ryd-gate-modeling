# ryd-gate

[![CI](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Universal Rydberg-atom simulation toolkit for neutral-atom quantum computing.

Models two-photon excitation in ⁸⁷Rb (hyperfine structure, Rydberg blockade, spontaneous decay, AC Stark shifts) and N-atom lattice systems. Provides time-optimal (TO) and amplitude-robust (AR) CZ gate protocols, adiabatic sweep protocols, Monte Carlo noise analysis, local addressing simulation, and an optional tensor-network path for large lattices.

## Installation

```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# With optional tensor-network support
uv pip install -e ".[dev,tn]"
```

## Quickstart

### Two-atom CZ gate

```python
import ryd_gate as rg

# 1. Create system
system = rg.create_our_system()

# 2. Choose protocol
protocol = rg.TOProtocol()

# 3. Run simulation
import numpy as np
psi0 = np.zeros(49, dtype=complex)
psi0[8] = 1.0  # |11> state

X_TO = [0.1122, 1.0431, -0.72565603, 0.0, 0.452, 1.219096]
result = rg.simulate(system, protocol, X_TO, psi0)

# 4. Analyze
infidelity = rg.average_gate_infidelity(system, protocol, X_TO)
print(f"Gate infidelity: {infidelity:.2e}")
```

### N-atom lattice sweep

```python
import ryd_gate as rg
import numpy as np
from ryd_gate.lattice import ground_state

# 1. Create lattice system (4-atom chain)
system = rg.create_lattice_system(N=4, geometry="chain", spacing_um=5.0)

# 2. Sweep protocol
protocol = rg.SweepProtocol()

# 3. Simulate (psi0 = ground state |gggg>)
psi0 = ground_state(N=4)
params = [2 * np.pi * -40e6, 2 * np.pi * 40e6, 10e-6]  # [delta_start, delta_end, t_sweep]
result = rg.simulate(system, protocol, params, psi0)
```

## Package Layout

```
src/ryd_gate/
├── __init__.py          # Public API (import everything from here)
├── pulse.py             # Pulse shaping: blackman_window, blackman_pulse, blackman_pulse_sqrt
├── core/                # Atomic physics primitives
│   ├── atomic_system.py     # AtomicSystem, LatticeSystem dataclasses + factory functions
│   ├── system_model.py      # SystemModel abstract base
│   ├── basis.py             # BasisSpec: Hilbert space structure
│   ├── blocks.py            # BlockRegistry: named Hamiltonian operator blocks
│   ├── observables.py       # ObservableRegistry: named measurement operators
│   ├── operators.py         # Low-level operator builders (occupation, projectors)
│   ├── hamiltonian_builders.py  # Precomputed 49×49 two-atom Hamiltonians
│   ├── ac_stark.py          # AC Stark shift (Grimm formula, 784nm calibration)
│   ├── branching.py         # Radiative decay branching ratios (ARC)
│   ├── registry.py          # Protocol-system compatibility registry
│   ├── rb87_two_atom.py     # Rb87TwoAtomModel: 7-level two-atom SystemModel
│   ├── analog_3level.py     # Analog3LevelModel: 3-level two-atom SystemModel
│   └── lattice_2level.py    # Lattice2LevelModel: N-atom 2-level SystemModel
├── protocols/           # Pulse protocols
│   ├── gate_cz_to.py        # TOProtocol: time-optimal CZ gate (6 params)
│   ├── gate_cz_ar.py        # ARProtocol: amplitude-robust CZ gate (8 params)
│   ├── sweep.py             # SweepProtocol: adiabatic detuning sweep (3 params)
│   ├── base.py              # Protocol abstract base class
│   └── channels.py          # Drive channel name constants
├── solvers/             # Simulation engine
│   ├── dispatch.py          # simulate(): high-level entry point
│   ├── base.py              # SolverBackend ABC, EvolutionResult
│   ├── dense_ode.py         # DenseODEBackend: adaptive ODE (scipy DOP853)
│   ├── sparse_expm.py       # SparseExpmBackend: piecewise-constant expm_multiply
│   ├── schrodinger.py       # solve_gate(), evolve() — direct legacy solvers
│   ├── monte_carlo.py       # MonteCarloEngine: quasi-static noise (legacy)
│   ├── monte_carlo_runner.py # MonteCarloRunner: noise via compiler + backend
│   ├── ir.py                # HamiltonianIR, HamiltonianTerm: solver-agnostic IR
│   ├── compiler_base.py     # Compiler abstract base
│   ├── dense_atomic.py      # DenseAtomicCompiler: two-atom → dense IR
│   └── sparse_lattice.py    # SparseLatticeCompiler: lattice → sparse IR
├── lattice/             # N-atom lattice simulation
│   ├── geometry.py          # SquareLattice, LatticeGeometry
│   ├── operators.py         # Hamiltonian builders (VdW, drive, 3-level)
│   ├── states.py            # product_state, af_config, domain_config, ground_state
│   ├── evolution.py         # evolve_constant_H, evolve_sweep
│   ├── solver.py            # solve_lattice: unified lattice evolution wrapper
│   ├── observables.py       # Rydberg occupation, staggered magnetization
│   └── plotting.py          # Spatial density and population evolution plots
├── analysis/            # Post-processing and metrics
│   ├── gate_metrics.py      # average_gate_infidelity, error_budget, sss_infidelity
│   ├── observable_metrics.py # measure_observables, measure_trajectory
│   ├── addressing_metrics.py # AddressingEvaluator: pinning, crosstalk, leakage
│   ├── local_addressing.py  # Default sweep parameters, evaluate_addressing()
│   └── coarsening.py        # Domain identification, boundary masks
├── tn/                  # Optional tensor-network path (requires tenpy)
│   ├── lattice_spec.py      # TNLatticeSpec: MPS lattice specification
│   ├── simulate.py          # simulate_tn(): MPS time evolution
│   └── ...
└── legacy/              # Backward-compatible CZGateSimulator facade (deprecated)
```

## API Reference

All commonly used symbols are available from the top-level package:

```python
import ryd_gate as rg
```

### Systems

| Symbol | Description |
|--------|-------------|
| `rg.create_our_system()` | Two-atom 87Rb, 7-level model ("our" params) |
| `rg.create_lukin_system()` | Two-atom 87Rb, 7-level model (Lukin group params) |
| `rg.create_analog_system()` | Two-atom 3-level analog model |
| `rg.create_lattice_system(N, ...)` | N-atom 2-level lattice |
| `rg.AtomicSystem` | Two-atom system dataclass |
| `rg.LatticeSystem` | N-atom lattice system dataclass |

### Protocols

| Symbol | Description |
|--------|-------------|
| `rg.TOProtocol()` | Time-optimal CZ gate — cosine phase, 6 params |
| `rg.ARProtocol()` | Amplitude-robust CZ gate — dual-sine phase, 8 params |
| `rg.SweepProtocol()` | Adiabatic detuning sweep — 3 params |

### Simulation

| Symbol | Description |
|--------|-------------|
| `rg.simulate(system, protocol, x, psi0)` | Compile + evolve (auto-selects backend) |
| `rg.EvolutionResult` | Result dataclass: psi_final, times, states |
| `rg.HamiltonianIR` | Solver-agnostic Hamiltonian IR |

### Analysis

| Symbol | Description |
|--------|-------------|
| `rg.average_gate_infidelity(system, protocol, x)` | Nielsen average gate infidelity |
| `rg.error_budget(system, protocol, x)` | Error decomposition by channel |
| `rg.AddressingEvaluator` | Local addressing error analysis |

### Pulse Utilities

| Symbol | Description |
|--------|-------------|
| `rg.blackman_pulse(t, t_rise, t_gate)` | Blackman-windowed flat-top pulse |
| `rg.blackman_pulse_sqrt(t, t_rise, t_gate)` | Square-root Blackman envelope |
| `rg.blackman_window(t, t_rise)` | Raw Blackman window function |

## Physical Model

The two-atom solver models two ⁸⁷Rb atoms with:

| Parameter | Value | Description |
|-----------|-------|-------------|
| n | 70 | Rydberg principal quantum number |
| Ω_eff | 2π × 7 MHz | Effective two-photon Rabi frequency |
| Δ | 2π × 9.1 GHz | Intermediate state detuning |
| d | 3 μm | Interatomic distance |
| C₆ | 2π × 874 GHz·μm⁶ | van der Waals coefficient |

The 7-level basis per atom: `|0⟩, |1⟩, |e₁⟩, |e₂⟩, |e₃⟩, |r⟩, |r_garb⟩`
Two-atom Hilbert space: 7² = 49 dimensions.

## Scripts

| Script | Description |
|--------|-------------|
| `error_deterministic.py` | Deterministic error budget for TO/AR protocols |
| `error_monte_carlo.py` | Monte Carlo noise analysis (dephasing, amplitude) |
| `generate_si_tables.py` | Generate supplemental tables for publication |
| `verify_cz_dark.py` | Verify CZ gate with dark-detuning parameters |
| `run_cz_simulation.py` | CZ gate simulation with TO/AR protocols |
| `plot_population_evolution_sch.py` | Plot Schrödinger population evolution |
| `sensitivity_gaussian_iso.py` | Sensitivity analysis for Gaussian position errors |
| `run_local_sweep.py` | Local addressing sweep visualization (3-level model) |
| `demo_local_addressing.py` | Local addressing pinning error demonstration |
| `demo_local_addressing_tn.py` | Local addressing demonstration with TN backend |
| `scan_local_addressing.py` | 2D grid scan over addressing parameters |
| `scan_local_addressing_singlequbit.py` | Single-qubit addressing scan |
| `diagnose_adiabaticity.py` | Diagnose adiabaticity quality for sweep protocols |
| `plot_ac_stark_landscape.py` | AC Stark shift landscape visualization |
| `run_3level_lattice.py` | 3-level lattice simulation |
| `addressing_enlevel.py` | Energy level diagram for local addressing |

## Testing

```bash
# Run fast tests only (~1 second)
uv run pytest

# Run all tests including slow ODE-based tests (~15 min)
uv run pytest -m ""

# Run with coverage
uv run pytest -m "" --cov=ryd_gate --cov-report=html
```

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting_started.md) | Installation and basic usage |
| [Schrodinger Solver](docs/schrodinger_solver.md) | 7-level model theory and API |
| [Error Budget](docs/error_budget_methodology.md) | Error decomposition methodology |
| [Validation](docs/validation.md) | Test suite documentation |

## References

* Evered *et al.*, "High-fidelity parallel entangling gates on a neutral-atom quantum computer", *Nature* **622**, 268 (2023).
* Ma *et al.*, "Benchmarking and fidelity response theory of high-fidelity Rydberg entangling gates", *PRX Quantum* **6**, 010331 (2025).

## License

MIT
