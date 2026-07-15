# ryd-gate

[![CI](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Rydberg neutral-atom many-body simulator: TFIM quenches, lattice dynamics,
and microscopic gate physics on one exact + tensor-network kernel.
Continuous-time pulse **protocols** are the single control surface — a
protocol bound to a `RydbergSystem` lowers to a unified Hamiltonian IR and
runs on any backend:

- **many-body / TFIM line** — 2D quenches, annealing, critical behavior on
  `1r`/`01r` lattices, with exact state-vector, TeNPy MPS (DMRG/TDVP), and
  YASTN 2D PEPS backends;
- **gate line** — microscopic ⁸⁷Rb CZ gate modeling (7-level two-photon
  structure, blockade, spontaneous decay, AC Stark shifts) with time-optimal /
  amplitude-robust / adiabatic pulse protocols; Nielsen fidelities and
  per-channel error budgets are computed in scripts/notebooks from the
  evolved states.

## Installation

```bash
uv pip install -e .               # base: exact state-vector backend
uv pip install -e ".[tn]"         # + TeNPy MPS backend (DMRG/TDVP)
uv pip install -e ".[dev]"        # + test/lint/type tooling
```

## Quickstart 1 — TFIM quench on a Rydberg lattice

```python
import numpy as np
from ryd_gate import Register, RydbergSystem, TFIMQuenchProtocol, level_structure, simulate

protocol = TFIMQuenchProtocol(hx=2 * np.pi * 1e6, t_gate=0.5e-6)
system = RydbergSystem(level_structure=level_structure("1r"),
                       register=Register.square(2, spacing_um=9.0),
                       protocol=protocol)
# default initial state: every site in the preset's initial level (|1> here).
# Observables are named expressions built from the system's factory; without
# t_eval they are recorded at t_gate only (a shape-(1,) complex array).
result = simulate(system, observables={"n_r": system.observables.level_sum("r")})
n_r = result.expectation("n_r")[0].real            # endpoint value, real part
assert 0.0 < n_r < system.N                        # quench excites Rydberg population
print(f"<n_r> after the quench: {n_r:.3f}")
print(result.sample(1000, seed=0).most_common(3))  # sampled measurement outcomes
```

The same system runs on the tensor-network backends
(`simulate(system, backend="mps")`, `"peps"`) — see the
[capability matrix](docs/capability_matrix.qmd).

## Quickstart 2 — CZ gate fidelity

The library evolves states; the gate metric is yours to write. Evolve the CZ
basis states, phase-correct the overlaps, and score them with the Nielsen
formula:

```python
import numpy as np
from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.protocols import TOProtocol

# historical layout [A, w, phi0, d, theta, T]; theta is a scoring parameter
X_TO_DARK = [-0.6894097925886826, 1.040962607910546, 0.3277877211544321,
             1.5639989822346387, 0.6689846026179691, 1.3407418093368753]

system = RydbergSystem(
    level_structure=level_structure("rb87_7_mp", detuning_sign=1),
    register=Register.chain(2, spacing_um=3.0))
pulse = TOProtocol(phase_amplitude=X_TO_DARK[0], frequency_ratio=X_TO_DARK[1],
                   phase_offset=X_TO_DARK[2], detuning_ratio=X_TO_DARK[3],
                   duration_ratio=X_TO_DARK[5])   # fully specified at construction
theta = X_TO_DARK[4]                       # ideal single-qubit Rz phase

results = simulate(system.with_protocol(pulse),
                   [["0", "0"], ["0", "1"], ["1", "1"]])
s0, s1 = np.eye(7, dtype=complex)[0], np.eye(7, dtype=complex)[1]
a00 = np.vdot(np.kron(s0, s0), results[0].final_state)
a01 = np.exp(-1j * theta) * np.vdot(np.kron(s0, s1), results[1].final_state)
a11 = np.exp(-2j * theta - 1j * np.pi) * np.vdot(np.kron(s1, s1), results[2].final_state)
# Nielsen average gate fidelity (d = 4; |10> folded into |01> by symmetry)
fidelity = (1 / 20) * (abs(a00 + 2 * a01 + a11) ** 2
                       + abs(a00) ** 2 + 2 * abs(a01) ** 2 + abs(a11) ** 2)
infidelity = 1.0 - fidelity
assert infidelity < 1e-4                   # ~6e-5 on current atomic data
print(fidelity)
```

The three 7-level evolutions run on the adaptive `exact_ode` solver, which
resolves the GHz optical phases with error control — this block takes about
3 minutes single-threaded.

## Documentation

The Quarto site under `docs/` (after `uv sync --extra docs`, build with
`cd docs && uv run quartodoc build && quarto render`; output lands in
`docs/_build/html`):

[Getting Started](docs/getting_started.qmd) ·
[Fundamentals (units/conventions)](docs/fundamentals.qmd) ·
[Hamiltonians & notation](docs/hamiltonians.qmd) ·
[NoiseModel](docs/how_to_noise.qmd) ·
[CZ gates](docs/how_to_gates.qmd) ·
[Capability matrix](docs/capability_matrix.qmd)

Runnable demos live in [`examples/`](examples/README.md); research notebooks
in `scripts/notebooks/` (execute the gated set with
`uv run python docs/_scripts/run_notebooks.py`).

## Optional dependencies

| extra | contents |
|---|---|
| *(base)* | numpy, scipy, qutip, matplotlib, ARC (exact backend) |
| `dev` | pytest, pytest-cov, ruff, mypy, nbconvert/nbclient/ipykernel |
| `docs` | quartodoc + griffe (API reference; site built with Quarto) |
| `tn` | physics-tenpy (MPS DMRG/TDVP backend) |
| `tn-2d` | physics-tenpy, yastn (2D PEPS backend) |
| `torch` | PyTorch 2.5.1 (optional YASTN PEPS-on-CUDA path) |

## Project structure

```
src/ryd_gate/
   core/           RydbergSystem, level structures, operators
   protocols/      continuous-time pulse protocols (TFIM quench/anneal,
                   sweeps, digital-analog, CZ gate protocols)
   backends/       exact state-vector + MPS / PEPS engines
   ir.py           unified Hamiltonian IR + EvolutionResult
   lattice.py      Register, plotting
   simulate.py     simulate(system, initial_state=None, ...) dispatcher
   noise.py        NoiseModel + simulate_ensemble (raw shot ensembles)
   physics.py      AC Stark shifts, ARC decay branching
tests/             pytest suite (fast suite: `uv run pytest -q`)
scripts/           optimization workflows + research notebooks
docs/              Quarto site + generated capability matrix
```

## Development

```bash
OMP_NUM_THREADS=1 uv run pytest -q          # fast suite
uv run pytest -m ""                          # including slow solver tests
uv run ruff check src tests docs examples   # lint
uv run mypy src/ryd_gate                    # scoped type gate
uv run python docs/_scripts/build_capability_matrix.py --check
```

See `CHANGELOG.md` for the stage-by-stage history.

## References

* Evered *et al.*, "High-fidelity parallel entangling gates on a neutral-atom quantum computer", *Nature* **622**, 268 (2023).
* Ma *et al.*, "Benchmarking and fidelity response theory of high-fidelity Rydberg entangling gates", *PRX Quantum* **6**, 010331 (2025).

## License

MIT
