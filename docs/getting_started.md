# Getting Started with ryd_gate

## Overview

`ryd_gate` is a Python package for simulating and optimizing Rydberg-atom entangling gates in neutral-atom quantum computers. It provides:

- **7-level Schrodinger solver** for two-atom CZ gate dynamics
- **49-dimensional Hilbert space** (7 levels x 2 atoms) with hyperfine structure
- **Modular architecture**: physics, protocols, solvers, and analysis are separated
- **Rydberg blockade**, spontaneous decay, and AC Stark shift modeling
- **Monte Carlo noise analysis** for dephasing and position errors
- **Local addressing simulation** with sweep protocols and 784nm pinning
- **Pulse optimization** for time-optimal (TO) and amplitude-robust (AR) protocols

## Installation

```bash
# From source (recommended for development)
git clone https://github.com/ChanceSiyuan/Ryd-gate-modeling.git
cd Ryd-gate-modeling
uv pip install -e ".[dev]"

# Dependencies installed automatically:
# - qutip (Bloch sphere visualization)
# - arc (atomic calculations for Rb)
# - numpy, scipy, matplotlib
```

## Architecture

The codebase is organized into four subpackages:

| Subpackage | Description |
|------------|-------------|
| `ryd_gate.core` | `AtomicSystem` dataclass, Hamiltonian builders, branching ratios |
| `ryd_gate.protocols` | `Protocol` ABC for CZ gates, `SweepProtocol` ABC for addressing |
| `ryd_gate.solvers` | `solve_gate()` for CZ, `evolve()` for generic H(t), Monte Carlo engines |
| `ryd_gate.analysis` | Gate fidelity metrics, error budget, addressing evaluator |
| `ryd_gate.ideal_cz` | Backward-compatible `CZGateSimulator` facade |

### Key Exports

```python
from ryd_gate import (
    CZGateSimulator,       # Backward-compatible facade
    MonteCarloResult,       # MC result dataclass
    AtomicSystem,           # Physical system container
    Protocol,               # CZ gate protocol ABC
    SweepProtocol,          # Sweep protocol ABC
    TOProtocol,             # Time-optimal CZ protocol
    ARProtocol,             # Amplitude-robust CZ protocol
    SweepAddressingProtocol,# Local addressing sweep
    AddressingEvaluator,    # Pinning/crosstalk metrics
)
```

## CZGateSimulator (Facade)

The `CZGateSimulator` facade preserves the original API while delegating to modular subpackages.

### Constructor

```python
CZGateSimulator(
    param_set: Literal['our', 'lukin'] = 'our',
    strategy: Literal['TO', 'AR'] = 'AR',
    blackmanflag: bool = True,
    detuning_sign: Literal[1, -1] = 1,
    *,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
    enable_0_scattering: bool = True,
    enable_rydberg_dephasing: bool = False,
    enable_position_error: bool = False,
    enable_polarization_leakage: bool = False,
    sigma_detuning: float | None = None,
    sigma_pos_xyz: tuple[float, float, float] | None = None,
    n_mc_shots: int = 100,
    mc_seed: int | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param_set` | `str` | `'our'` | Physical parameter configuration (`'our'` = n=70, `'lukin'` = n=53) |
| `strategy` | `str` | `'AR'` | Pulse optimization strategy (`'TO'` or `'AR'`) |
| `blackmanflag` | `bool` | `True` | Apply Blackman envelope for smooth pulses |
| `detuning_sign` | `int` | `1` | Sign of intermediate detuning (+1 bright, -1 dark) |
| `enable_rydberg_decay` | `bool` | `False` | Include Rydberg state decay as imaginary energy shifts |
| `enable_intermediate_decay` | `bool` | `False` | Include intermediate state decay as imaginary energy shifts |
| `enable_0_scattering` | `bool` | `True` | Include |0> AC Stark shift and scattering loss |
| `enable_rydberg_dephasing` | `bool` | `False` | Enable Monte Carlo T2* dephasing noise |
| `enable_position_error` | `bool` | `False` | Enable Monte Carlo 3D position fluctuations |
| `enable_polarization_leakage` | `bool` | `False` | Include coupling to unwanted Rydberg state |r'> |
| `sigma_detuning` | `float\|None` | `None` | Dephasing noise std dev (Hz). Required when dephasing enabled |
| `sigma_pos_xyz` | `tuple\|None` | `None` | Position noise std dev (sigma_x, sigma_y, sigma_z) in meters |
| `n_mc_shots` | `int` | `100` | Number of Monte Carlo shots |
| `mc_seed` | `int\|None` | `None` | RNG seed for reproducibility |

### Key Methods

| Method | Description |
|--------|-------------|
| `setup_protocol(x)` | Store pulse parameters for repeated use |
| `optimize(x_initial, fid_type)` | Run pulse parameter optimization (Nelder-Mead) |
| `gate_fidelity(x, fid_type)` | Average gate infidelity. Returns `(mean, std)` when MC enabled |
| `error_budget(x, initial_states)` | XYZ/AL/LG error decomposition by source |
| `state_infidelity(initial_state, x)` | Infidelity for a specific initial state |
| `run_monte_carlo_simulation(x, ...)` | Full quasi-static MC simulation |
| `diagnose_plot(x, initial_state)` | Plot population evolution |
| `diagnose_run(x, initial_state)` | Return population time series (mid, ryd, ryd_garb) |
| `plot_bloch(x, save=True)` | Generate Bloch sphere plots (TO only) |

## Use Cases

### 1. Basic Fidelity Calculation

```python
from ryd_gate import CZGateSimulator

sim = CZGateSimulator(param_set='our', strategy='TO')
X_TO = [0.1122, 1.0431, -0.72565603, 0.0, 0.452, 1.219096]
infidelity = sim.gate_fidelity(X_TO)
print(f"Gate infidelity: {infidelity:.2e}")
```

### 2. Direct Module Usage (Recommended for New Code)

```python
from ryd_gate.core.atomic_system import create_atomic_system
from ryd_gate.protocols.gate_cz_to import TOProtocol
from ryd_gate.analysis.gate_metrics import average_gate_infidelity

system = create_atomic_system(param_set="our")
protocol = TOProtocol()
X_TO = [0.1122, 1.0431, -0.72565603, 0.0, 0.452, 1.219096]
infidelity = average_gate_infidelity(system, protocol, X_TO)
```

### 3. Monte Carlo Noise Analysis

```python
sim = CZGateSimulator(
    param_set='our', strategy='TO',
    enable_rydberg_dephasing=True,
    enable_position_error=True,
    sigma_detuning=50e3,
    sigma_pos_xyz=(30e-9, 30e-9, 30e-9),
    n_mc_shots=200,
    mc_seed=42,
)
mean_infid, std_infid = sim.gate_fidelity(X_TO)
print(f"Infidelity: {mean_infid:.2e} +/- {std_infid:.2e}")
```

### 4. Local Addressing Simulation

```python
import numpy as np
from ryd_gate.core.atomic_system import create_atomic_system, build_sss_state_map
from ryd_gate.protocols.local_sweep import SweepAddressingProtocol
from ryd_gate.solvers.monte_carlo import AddressingMCEngine
from ryd_gate.analysis.addressing_metrics import AddressingEvaluator

system = create_atomic_system(param_set="our")
protocol = SweepAddressingProtocol(
    omega=system.rabi_eff,
    delta_start=-2 * np.pi * 15e6,
    delta_end=+2 * np.pi * 15e6,
    t_gate=1.5e-6,
    local_detuning_A=-2 * np.pi * 12e6,
)

engine = AddressingMCEngine(system, protocol, sigma_detuning=100e3, sigma_local_rin=0.01)
states = build_sss_state_map()
final_states = engine.run(states["11"], n_shots=50, seed=42)

evaluator = AddressingEvaluator(final_states)
print(f"Pinning error: {evaluator.pinning_error():.6f}")
print(f"Crosstalk: {evaluator.crosstalk_error():.6f}")
```

### 5. Including Decay Effects

```python
sim_decay = CZGateSimulator(
    param_set='our', strategy='AR',
    enable_rydberg_decay=True,
    enable_intermediate_decay=True,
    enable_polarization_leakage=True,
)
X_AR = [0.85973359, 0.39146974, 0.99181418, 0.1924498,
        -1.17123748, -0.00826712, 1.67429728, 0.28527346]
infid = sim_decay.gate_fidelity(X_AR)
print(f"Infidelity with decay: {infid:.6f}")
```

---

## See Also

- `examples/run_cz_simulation.py` -- Complete simulation example
- `docs/schrodinger_solver.md` -- Detailed solver documentation
- `docs/error_budget_methodology.md` -- Error decomposition methodology
