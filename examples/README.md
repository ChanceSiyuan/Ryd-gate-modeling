# Examples

Runnable, self-contained demos — a good place to start. Run them with `uv run`:

```bash
uv run python examples/demo_sequence_exact.py
OMP_NUM_THREADS=1 uv run python examples/demo_noise_model.py
OMP_NUM_THREADS=1 uv run python examples/demo_cz_gate_report.py
uv run python examples/demo_pulser_interop.py
uv run python examples/demo_local_addressing.py --Lx 2 --Ly 2 --experiment domain
uv run python examples/demo_local_addressing_tn.py        # needs the tn extra
```

| Example | What it shows | Runtime |
|---|---|---|
| `demo_sequence_exact.py` | Sequence API end to end: build, validate, run exactly, sample, serialize. | seconds |
| `demo_noise_model.py` | Declarative `NoiseModel` onto the exact Monte Carlo runner + decay flags. | seconds |
| `demo_cz_gate_report.py` | The flagship TO dark CZ benchmark as a one-call `CZGateReport`. | ~15 s |
| `demo_pulser_interop.py` | Import/run/export a Pulser abstract-repr payload; typed refusal demo. | seconds |
| `demo_local_addressing.py` | Exact local-addressing experiments (domain shrinking, Higgs mode). | heavy at 4×4 — pass `--Lx 2 --Ly 2 --n-steps 20` |
| `demo_local_addressing_tn.py` | The same workflow through the tensor-network backend (`tn` extra). | depends on size |

Every demo either runs in the base environment or states the optional extra
it needs. Experimental / batch workflows live under `scripts/`.
