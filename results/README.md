# results/ — index

Every directory below holds one study's artifacts and a `README.md` that answers three
questions: what the study does, what its core numbers and figures are, and which script
produced it. Start there; this table is only the map.

**Figures are not tracked in git** (`.gitignore` excludes `*.png`/`*.pdf`), so image
links render only on a checkout that has the data. Each README gives the command that
regenerates derived figures. Measured input images are also local-only; their owning
report records their provenance and identifies inputs that cannot be regenerated.

| Directory | What it answers | Produced by | State |
|---|---|---|---|
| [`01r_adiabatic_optimization`](01r_adiabatic_optimization/) | How short can a smooth two-atom `01r` computational-return pulse get and still pass `L_max ≤ 1e-4`, `C_φ ≥ 0.5`? | `scripts/adiabatic_01r_optimization.py` | complete — 48 stages, 46 accepted, rerun-verified bit-exact |
| [`297_laser_noise`](297_laser_noise/) | What do the two candidate lasers' measured frequency-noise spectra cost the 297 nm CZ, how much laser power does it take, and does intensity noise (RIN) matter? | `scripts/laser_noise_psd.py`, `scripts/max_leakage_297_sweep.py`, `scripts/phase_noise_summary.py`, `scripts/intensity_noise_band_analysis.py` | complete — filter grid 2026-08-01; RIN ruled out (ε ≤ 3.3e-6), 2026-08-06 |
| [`297_to_calibration`](297_to_calibration/) | Best time-optimal phase-family CZ for the 297 nm single-photon model, plus all four 53P Zeeman pair spectra and 70S benchmark curves versus $B$, $R$, and direction | `scripts/calibrate_to_297.py`, `scripts/check_297_pair_channels.py` | calibration artifacts complete; 2276D audit invalidates the scalar model, and 105-case pair-potential scan is complete, 2026-08-10 |
| [`ac_stark_addressing`](ac_stark_addressing/) | Which wavelength/power addresses one atom with least pinning leak, crosstalk and scatter? | `scripts/notebooks/02_ac_stark_addressing.ipynb` | complete |
| [`anneal_sweep`](anneal_sweep/) | Does the TFIM anneal protocol reach the target order, and is the PEPS bond dimension converged? | `scripts/anneal_sweep.py`, `scripts/calibrate_anneal_3x3.py` | complete — D=10 converged, D=6 has a documented artifact |
| [`cz_gate`](cz_gate/) | Best achievable CZ infidelity for the time-optimal and adiabatic-return families | `scripts/notebooks/01_cz_gate.ipynb` | complete — `our` AR 9.46e-6; TO mp 1.40e-5, pm 7.67e-5 |
| [`cz_grape_e2e`](cz_grape_e2e/) | Is the `qoc` discrete-adjoint GRAPE seam trustworthy end to end? | `scripts/cz_grape_e2e_validation.py` | complete — all acceptance gates passed |
| [`error_budget`](error_budget/) | Where in (Δe, K_eff, D_sweep, laser power) space is CZ leakage smallest? | `scripts/error_budget_sweep.py`, `scripts/gen_error_budget_g20.py` | complete — cost model since superseded, data still current |
| [`max_leakage_297`](max_leakage_297/) | Same map for the 297 nm **single-photon** CZ, across five lattice spacings | `scripts/max_leakage_297_sweep.py` | a3.0 full grid; a4–a10 scatter tier |
| [`max_leakage_ode`](max_leakage_ode/) | Coherent-leakage map for the two-photon `rb87_7_mp` CZ, across five lattice spacings | `scripts/max_leakage_ode_sweep.py` | a3.0 full grid; a4–a10 scatter tier |
| [`zxz_direct_qoc`](zxz_direct_qoc/) | Can IPOPT direct transcription synthesise a three-atom ZXZ gate, beat GRAPE, and transfer to larger lattices (arXiv:2508.19075 Fig. 3)? | `scripts/zxz_direct_qoc.py`, `scripts/zxz_transfer_test.py` | complete — F 0.9276/0.9922 vs GRAPE medians ≤0.66; short pulse transfers on 1D (~0.906/atom), long pulse and 2D do not |

## Conventions

- **Sweep stores** (`max_leakage_*`) use one sub-store per lattice spacing, `a<spacing>.0/`,
  each with `manifest.json`, `chunks/`, `scatter/`, `trajectories/`, `reports/`, `plots/`,
  `exports/`, `logs/`. `logs/store.lock` is an `fcntl.flock` file whose PID record is kept
  deliberately; it holds no lock once the process exits.
- **Cached-replay pattern.** Several studies write one JSON/NPZ artifact that the matching
  notebook reloads, so re-running the notebook costs no optimiser, ODE or ARC work.
- Anything not answered here belongs in the directory's own `README.md`, not this table.
