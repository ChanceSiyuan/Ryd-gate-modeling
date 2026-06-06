# Getting Started with ryd_gate

## Overview

`ryd_gate` is a Python package for simulating and optimizing Rydberg-atom entangling gates in neutral-atom quantum computers. It provides:

- **7-level Schrodinger solver** for two-atom CZ gate dynamics
- **49-dimensional Hilbert space** (7 levels x 2 atoms) with hyperfine structure
- **Modular architecture**: symbolic models, protocols, compilers, backends, and analysis are separated
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

The codebase is organized into focused subpackages:

| Subpackage | Description |
|------------|-------------|
| `ryd_gate.model` | `RydbergSystem`, symbolic blocks, observables, and lattice state helpers |
| `ryd_gate.protocols` | Protocol classes for CZ gates, sweep protocols, and digital-analog schedules |
| `ryd_gate.backends.exact` | Backend-specific compilation from symbolic systems to IR |
| `ryd_gate.backends.exact` | Dense ODE, sparse expm, Monte Carlo, and future solver backends |
| `ryd_gate.simulate` | High-level compile + evolve dispatch (default exact) |
| `ryd_gate.analysis` | Gate fidelity metrics, error budget, addressing evaluator |
| `ryd_gate.backends.exact.legacy` | Historical CZ simulator and Monte Carlo implementations |

### Key Exports

```python
from ryd_gate import (
    RydbergSystem,          # Symbolic system container
    Protocol,               # CZ gate protocol ABC
    SweepProtocol,          # Sweep protocol
    TOProtocol,             # Time-optimal CZ protocol
    ARProtocol,             # Amplitude-robust CZ protocol
    simulate,               # Compile + evolve
    AddressingEvaluator,    # Pinning/crosstalk metrics
)
```

## New Simulation API

Bind the protocol to `RydbergSystem`, then call `simulate(system, x, psi0)`.

## Use Cases

### 1. Basic Fidelity Calculation

```python
import numpy as np
from ryd_gate.backends.exact import simulate
from ryd_gate import RydbergSystem, TOProtocol
from ryd_gate.lattice import make_chain

protocol = TOProtocol()
system = RydbergSystem.from_lattice(
    make_chain(2, spacing_um=3.0),
    "rb87_7",
    param_set="our",
    protocol=protocol,
)
psi0 = np.zeros(49, dtype=complex)
psi0[8] = 1.0
X_TO = [0.1122, 1.0431, -0.72565603, 0.0, 0.452, 1.219096]
result = simulate(system, X_TO, psi0)
print(f"Final norm: {np.linalg.norm(result.psi_final):.6f}")
```

### 2. Direct Compiler Usage

```python
from ryd_gate import RydbergSystem, SweepProtocol, compile_hamiltonian_ir
from ryd_gate.backends.exact import compile_expm_ir
from ryd_gate.lattice import make_chain

delta_start, delta_end, t_sweep = -3.0, 3.0, 10.0
protocol = SweepProtocol(
    t_gate=t_sweep,
    omega_half_fn=lambda t: 0.5,
    delta_fn=lambda t: delta_start + (delta_end - delta_start) * t / t_sweep,
)
system = RydbergSystem.from_lattice(make_chain(4), "1r", protocol=protocol)
params = system.unpack_params([])
hamiltonian = compile_hamiltonian_ir(system, params)
exact_ir = compile_expm_ir(hamiltonian)
```

### 3. Monte Carlo Noise Analysis

```python
from ryd_gate.backends.exact import MonteCarloRunner

runner = MonteCarloRunner(system, [])
runner.setup_detuning_noise(50e3)
shots = runner.run_states([system.ground_state()], n_shots=50, seed=42)
```

### 4. Local Addressing Simulation

```python
import numpy as np
from ryd_gate.backends.exact import simulate
from ryd_gate import RydbergSystem, SweepProtocol
from ryd_gate.lattice import make_chain

delta_start = -2 * np.pi * 15e6
delta_end = 2 * np.pi * 15e6
t_sweep = 1.5e-6

system = RydbergSystem.from_lattice(
    make_chain(2, spacing_um=3.0),
    "1r",
)
protocol = SweepProtocol(
    t_gate=t_sweep,
    omega_half_fn=lambda t: np.pi * 4e6,
    delta_fn=lambda t: delta_start + (delta_end - delta_start) * t / t_sweep,
    address_fn=lambda t, i: -2 * np.pi * 12e6 if i == 0 else 0.0,
)
result = simulate(system.with_protocol(protocol), [], "all_ground")
```

### 5. Including Decay Effects

```python
from ryd_gate.backends.exact.legacy.ideal_cz import CZGateSimulator

legacy_sim = CZGateSimulator(param_set="our", strategy="AR")
```

---

## See Also

- `examples/run_cz_simulation.py` -- Complete simulation example
- `docs/schrodinger_solver.md` -- Detailed solver documentation
- `docs/error_budget_methodology.md` -- Error decomposition methodology
