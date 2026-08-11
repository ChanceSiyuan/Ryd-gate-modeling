# Examples

Runnable, self-contained demos — a good place to start. Run them with `uv run`:

```bash
OMP_NUM_THREADS=1 uv run python examples/demo_noise_model.py
OMP_NUM_THREADS=1 uv run python examples/demo_cz_gate.py
uv run python examples/demo_local_addressing.py --Lx 2 --Ly 2 --experiment domain
uv run python examples/demo_local_addressing_tn.py        # needs the tn extra
OMP_NUM_THREADS=1 uv run python scripts/api_walkthrough.py # extended API tour
```

| Example | What it shows | Runtime |
|---|---|---|
| `demo_noise_model.py` | Declarative per-laser `NoiseModel` (amplitude + frequency) through `simulate_ensemble`. | ~5 min |
| `demo_cz_gate.py` | The flagship TO dark CZ benchmark — Nielsen fidelity and phase diagnostics computed inline. | ~3 min (adaptive exact_ode solver) |
| `demo_local_addressing.py` | Exact local-addressing experiments (domain shrinking, Higgs mode). | heavy at 4×4 — pass `--Lx 2 --Ly 2 --n-steps 20` |
| `demo_local_addressing_tn.py` | The same workflow through the tensor-network backend (`tn` extra). | depends on size |
| `scripts/api_walkthrough.py` | Extended public-API tour: quench, noise ensemble, and CZ fidelity. | ~3 min, dominated by CZ exact ODE |

Every demo either runs in the base environment or states the optional extra
it needs. The extended walkthrough stays under `scripts/` because it combines
several studies; experimental and batch workflows live there as well.
