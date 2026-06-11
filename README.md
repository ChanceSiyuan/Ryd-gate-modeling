# ryd-gate

[![CI](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/ChanceSiyuan/Ryd-gate-modeling/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Rydberg-atom simulation toolkit for neutral-atom quantum computing, with two
control surfaces over one exact + tensor-network kernel:

- a **Pulser-style Sequence API** — device-validated registers, channels,
  waveforms (integer ns, rad/µs), serializable to frozen
  `ryd-gate/<kind>/v1` payloads, with a Pulser abstract-repr interop bridge;
- a **gate/protocol API** — the flagship: microscopic ⁸⁷Rb CZ gate modeling
  (7-level two-photon structure, blockade, spontaneous decay, AC Stark
  shifts), time-optimal / amplitude-robust / double-ARP protocols, Nielsen
  fidelities, and per-channel error budgets — none of which has a Pulser
  equivalent.

## Installation

```bash
uv pip install -e .               # base: exact state-vector backend
uv pip install -e ".[tn]"         # + TeNPy MPS backend (DMRG/TDVP)
uv pip install -e ".[dev]"        # + test/lint/type tooling
```

## Quickstart 1 — run a pulse sequence

```python
import numpy as np
from ryd_gate import DeviceSpec, Pulse, Register, Sequence, Waveform, simulate_sequence

seq = Sequence(Register.chain(1, 20.0), DeviceSpec.virtual_rb87(), "1r")
seq.declare_channel("ryd", "rydberg_global")
seq.add(Pulse.constant_detuning(Waveform.blackman(1000, area=np.pi), 0.0), "ryd")

result = simulate_sequence(seq)
assert result.populations("r")[0] > 0.999          # pi pulse: |1> -> |r>
print(result.sample(1000, basis="rydberg", seed=1))
```

The same sequence runs on the MPS backend
(`simulate_sequence(seq, backend="mps")`) with backend-native result handles
— see the [capability matrix](docs/capability_matrix.md).

## Quickstart 2 — CZ gate report

```python
from ryd_gate import Register, RydbergSystem
from ryd_gate.gates import TOProtocol, cz_gate_report

X_TO_DARK = [-0.6989301339711643, 1.0296229082590798, 0.3759232324550267,
             1.5710180991068543, 1.4454279613697887, 1.3406239758422793]

system = RydbergSystem.from_lattice(
    Register.chain(2, spacing_um=3.0), "rb87_7", param_set="our",
    blackmanflag=True, detuning_sign=1,
)
report = cz_gate_report(system, TOProtocol(), X_TO_DARK, include_error_budget=False)
assert report.infidelity < 1e-4                    # benchmark point: ~7.8e-7
print(report.fidelity, report.phase_error_rad)
```

## Documentation

The Sphinx site under `docs/` (build with
`uv run sphinx-build -b html docs docs/_build/html` after
`uv sync --extra docs`):

[Getting Started](docs/getting_started.md) ·
[Fundamentals (units/conventions)](docs/fundamentals.md) ·
[Sequences](docs/how_to_sequences.md) ·
[NoiseModel](docs/how_to_noise.md) ·
[Gate reports](docs/how_to_gates.md) ·
[Serialization & Pulser interop](docs/how_to_interop.md) ·
[Capability matrix](docs/capability_matrix.md)

Runnable demos live in [`examples/`](examples/README.md); research notebooks
in `scripts/notebooks/` (execute the gated set with
`uv run python docs/_scripts/run_notebooks.py`).

## Optional dependencies

| extra | contents |
|---|---|
| *(base)* | numpy, scipy, qutip, matplotlib, ARC (exact backend) |
| `dev` | pytest, pytest-cov, ruff, mypy, nbconvert/nbclient/ipykernel |
| `docs` | sphinx, sphinx-rtd-theme, myst-parser |
| `schema` | jsonschema (frozen-payload validation) |
| `interop` | jsonschema (Pulser abstract-repr bridge checks) |
| `tn` | physics-tenpy (MPS DMRG/TDVP backend) |
| `tn-2d` | physics-tenpy, yastn, cotengra, autoray (2D PEPS) |
| `gputn-cu12` | cuQuantum / cuPy stack (GPU tensor networks, sm_70+) |

## Project structure

```
src/ryd_gate/
   lattice/        Register, RegisterLayout, plotting
   pulse.py        Waveform / Pulse (product) + kernel Blackman helpers
   devices.py      DeviceSpec (hardware constraints as data)
   sequence.py     Sequence -> SequenceProtocol -> kernel
   results.py      SimulationResult + capability-aware state handles
   noise.py        NoiseModel -> exact Monte Carlo / decay flags
   gates/          CZ gate library: protocols, metrics, CZGateReport
   observables.py  ObservableConfig streaming schedules
   interop/        Pulser abstract-repr subset bridge
   schemas/        frozen ryd-gate/<kind>/v1 JSON Schemas
   core/, ir/, protocols/, backends/, analysis/   the kernel
tests/             pytest suite (fast suite: `uv run pytest -q`)
scripts/           optimization workflows + research notebooks
docs/              Sphinx site + generated capability matrix
```

## Development

```bash
OMP_NUM_THREADS=1 uv run pytest -q          # fast suite
uv run pytest -m ""                          # including slow solver tests
uv run ruff check src tests docs examples   # lint
uv run mypy src/ryd_gate                    # scoped type gate
uv run python docs/_scripts/build_capability_matrix.py --check
```

See `CHANGELOG.md` for the stage-by-stage history and `stageplans/` for the
binding refactor specifications.

## References

* Evered *et al.*, "High-fidelity parallel entangling gates on a neutral-atom quantum computer", *Nature* **622**, 268 (2023).
* Ma *et al.*, "Benchmarking and fidelity response theory of high-fidelity Rydberg entangling gates", *PRX Quantum* **6**, 010331 (2025).

## License

MIT
