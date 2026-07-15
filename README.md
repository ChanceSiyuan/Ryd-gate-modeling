# ryd-gate

[![CI](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`ryd-gate` simulates Rydberg neutral-atom systems with continuous-time pulse
protocols. The same protocol-bound system can use the exact ODE, MPS, or PEPS
backend when that backend supports its level structure and geometry.

The library returns raw physical readouts—observable expectations, final basis
amplitudes, and samples. Gate metrics, error budgets, result-analysis plots,
optimization, and persistence stay in the calling scripts and notebooks; each
concrete protocol's `.plot()` remains available for inspecting input pulses.

## Install

```bash
uv pip install -e .                 # exact_ode
uv pip install -e ".[tn]"          # + MPS / DMRG
uv pip install -e ".[tn-2d]"       # + PEPS on CPU
uv pip install -e ".[tn-2d,torch]" # + PEPS on CUDA
uv pip install -e ".[dev]"         # tests and development tools
```

## Quick start

```python
import numpy as np

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.protocols import SweepProtocol

system = RydbergSystem(
    level_structure=level_structure("1r", ryd_level=70),
    register=Register.square(2, spacing_um=9.0),
    protocol=SweepProtocol(
        t_gate_s=0.5e-6,
        omega_half_rad_s=lambda t: 2 * np.pi * 1e6,
        detuning_rad_s=lambda t: 0.0,
    ),
)

obs = system.observables
n_r = sum(obs.n("r", i) for i in range(system.N))
result = simulate(system, observables={"n_r": n_r})

print(result.times)
print(result.expectation("n_r"))
print(result.sample(shots=1000, seed=0).most_common(3))
```

The unit and level-label conventions are defined once in
[Model](docs/model.md#units-and-labels).

## Documentation

- [Model](docs/model.md): geometry, level structures, interactions,
  Hamiltonians, and protocols.
- [Simulation](docs/simulation.md): initial states, observables, backends,
  results, ground states, PEPS evidence, and noise ensembles.
- [Gates](docs/gates.md): CZ/TO/AR and 297-nm workflows, gate metrics, and
  first-order error budgets.

Public signatures can be inspected with an IDE or `help(...)`; constructors
remain the authority for validation and error messages. Runnable examples are
in [examples](examples/README.md); complete research workflows are in `scripts/`
and `scripts/notebooks/`.

## Development

```bash
OMP_NUM_THREADS=1 uv run pytest -q
uv run ruff check src tests examples
uv run mypy src/ryd_gate
uv run python scripts/check_notebooks.py
```

## References

- Evered *et al.*, “High-fidelity parallel entangling gates on a neutral-atom
  quantum computer,” *Nature* **622**, 268 (2023).
- Ma *et al.*, “Benchmarking and fidelity response theory of high-fidelity
  Rydberg entangling gates,” *PRX Quantum* **6**, 010331 (2025).

## License

MIT
