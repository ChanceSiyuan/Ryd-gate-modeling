# Examples

Runnable, self-contained demos — a good place to start. Run them with `uv run`:

```bash
uv run python examples/demo_local_addressing.py --Lx 2 --Ly 2 --experiment domain
uv run python examples/demo_local_addressing_tn.py
```

| Example | What it shows |
|---|---|
| `demo_local_addressing.py` | Exact state-vector local-addressing experiments (domain shrinking, Higgs mode) on a small Rydberg array. |
| `demo_local_addressing_tn.py` | The same local-addressing workflow run through the tensor-network backend (`simulate_tn`). |

`demo_local_addressing.py` defaults to a 4×4 exact simulation, which is heavy;
pass `--Lx 2 --Ly 2 --n-steps 20` for a quick run.

Experimental / batch / plotting scripts live under `scripts/`.
