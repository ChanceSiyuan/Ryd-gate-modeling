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
  `1r`/`01r` lattices, with exact state-vector, TeNPy MPS (DMRG/TDVP),
  YASTN 2D PEPS, and cuTensorNet GPU backends;
- **gate line** — microscopic ⁸⁷Rb CZ gate modeling (7-level two-photon
  structure, blockade, spontaneous decay, AC Stark shifts), time-optimal /
  amplitude-robust / double-ARP protocols, Nielsen fidelities, and
  per-channel error budgets.

## Installation

```bash
uv pip install -e .               # base: exact state-vector backend
uv pip install -e ".[tn]"         # + TeNPy MPS backend (DMRG/TDVP)
uv pip install -e ".[dev]"        # + test/lint/type tooling
```

## Quickstart 1 — TFIM quench on a Rydberg lattice

```python
import numpy as np
from ryd_gate import Register, RydbergSystem, TFIMQuenchProtocol, simulate

protocol = TFIMQuenchProtocol(hx=2 * np.pi * 1e6, t_gate=0.5e-6)
system = (RydbergSystem.set_atom_level("1r")
          .set_atom_geom(Register.square(2, spacing_um=9.0)).set_protocol(protocol))
# x is optional (this protocol carries its own schedule); request observables.
result = simulate(system, psi0="all_1", observables=["sum_nr"])
n_r = result.expectation("sum_nr")                 # read straight off the result
assert 0.0 < n_r < system.N                        # quench excites Rydberg population
print(f"<n_r> after the quench: {n_r:.3f}")
print(result.sample(1000, seed=0).most_common(3))  # sampled measurement outcomes
```

The same system runs on the tensor-network backends
(`simulate(system, backend="mps")`, `"peps"`) — see the
[capability matrix](docs/capability_matrix.qmd).

## Quickstart 2 — CZ gate report

```python
from ryd_gate import Register, RydbergSystem
from ryd_gate.gates import TOProtocol, cz_gate_report

X_TO_DARK = [-0.6894097925886826, 1.040962607910546, 0.3277877211544321,
             1.5639989822346387, 0.6689846026179691, 1.3407418093368753]

system = (RydbergSystem
          .set_atom_level("rb87_7", param_set="our", detuning_sign=1)
          .set_atom_geom(Register.chain(2, spacing_um=3.0)).build())
report = cz_gate_report(system, TOProtocol(), X_TO_DARK, include_error_budget=False)
assert report.infidelity < 1e-4                    # benchmark point: ~3.7e-7
print(report.fidelity, report.phase_error_rad)
```

## Documentation

The Quarto site under `docs/` (after `uv sync --extra docs`, build with
`cd docs && uv run quartodoc build && quarto render`; output lands in
`docs/_build/html`):

[Getting Started](docs/getting_started.qmd) ·
[Fundamentals (units/conventions)](docs/fundamentals.qmd) ·
[Hamiltonians & notation](docs/hamiltonians.qmd) ·
[NoiseModel](docs/how_to_noise.qmd) ·
[Gate reports](docs/how_to_gates.qmd) ·
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
| `schema` | jsonschema (frozen-payload validation) |
| `tn` | physics-tenpy (MPS DMRG/TDVP backend) |
| `tn-2d` | physics-tenpy, yastn (2D PEPS backend) |
| `torch` | PyTorch 2.5.1 (optional YASTN PEPS-on-CUDA path) |

## Project structure

```
src/ryd_gate/
   core/           RydbergSystem, level structures, operators, serialization
   protocols/      continuous-time pulse protocols (TFIM quench/anneal,
                   sweeps, digital-analog, CZ gate protocols)
   backends/       exact state-vector + MPS / PEPS engines
   ir.py           unified Hamiltonian IR + EvolutionResult
   lattice.py      Register, RegisterLayout, plotting
   simulate.py     simulate(system, x=(), ...) backend dispatcher
   noise.py        NoiseModel -> exact Monte Carlo / decay flags
   gates.py        CZ gate library: CZGateReport, cz_gate_report
   analysis/       gate metrics, lattice observables, domain analysis
   physics.py      AC Stark shifts, ARC decay branching
   schemas/        frozen ryd-gate/<kind>/v1 JSON Schemas
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
