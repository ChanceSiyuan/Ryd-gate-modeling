# Research workflows

`scripts/` contains study orchestration, reproducibility entry points, and
plotting utilities. It is not part of the stable `ryd_gate`/`qoc` API: reusable
model and solver behavior belongs under `src/`, while a concrete scientific
campaign stays here.

Numerical payloads produced by these workflows live only in the local
`results/` tree. Open the linked result report before rerunning a study; it
records the physical model, artifact schema, provenance, and cheapest replay
command.

## Study map

| Study | Main entry points | Report |
|---|---|---|
| Smooth two-atom `01r` control | `adiabatic_01r_optimization.py`, `one_r_control.py` | [`results/01r_adiabatic_optimization`](../results/01r_adiabatic_optimization/) |
| 297 nm calibration and pair spectra | `calibrate_to_297.py`, `check_297_garb_leg.py`, `check_297_pair_channels.py` | [`results/297_to_calibration`](../results/297_to_calibration/) |
| 297 nm laser noise | `laser_noise_psd.py`, `phase_noise_summary.py`, `phase_noise_mc_check.py`, `laser_noise_band_analysis.py`, `intensity_noise_band_analysis.py` | [`results/297_laser_noise`](../results/297_laser_noise/) |
| Maximum-leakage sweeps | `max_leakage_297_sweep.py`, `max_leakage_ode_sweep.py` | [`results/max_leakage_297`](../results/max_leakage_297/), [`results/max_leakage_ode`](../results/max_leakage_ode/) |
| TFIM annealing | `anneal_sweep.py`, `calibrate_anneal_3x3.py`, `anneal_model.py` | [`results/anneal_sweep`](../results/anneal_sweep/) |
| CZ GRAPE seam validation | `cz_grape_e2e_validation.py`, `one_r_control.py` | [`results/cz_grape_e2e`](../results/cz_grape_e2e/) |
| Error-budget maps | `error_budget_sweep.py`, `gen_error_budget_g20.py`, `error_budget_model.py` | [`results/error_budget`](../results/error_budget/) |
| Direct ZXZ control and transfer | `zxz_direct_qoc.py`, `zxz_transfer_test.py`, `plot_transfer_decay.py` | [`results/zxz_direct_qoc`](../results/zxz_direct_qoc/) |

The complete directory index, including notebook-owned studies, is
[`results/README.md`](../results/README.md).

## Supporting entry points

- `api_walkthrough.py` is the runnable public-API tour.
- `check_notebooks.py` executes the CPU-gated notebook set in temporary files.
- `bench_quench_check.py` compares tensor-network refactor baselines.
- `run_peps_10x10.py` is the maintained large PEPS validation case.
- `sweeplib/` owns the shared axes, solver, append-only store, campaign runner,
  and plotting machinery used by the two maximum-leakage sweeps.
- `notebooks/` contains interactive studies and replay notebooks. Keep their
  machine-local paths and generated scratch data out of Git.

Use a script's module docstring and `--help` output as the command-line
authority. When a run creates or changes `results/<study>/`, update that
directory's `README.md`, update the results index when necessary, and run:

```bash
python .agents/skills/results-report/validate.py
```
