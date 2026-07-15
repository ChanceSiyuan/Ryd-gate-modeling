# find_phase perturbation check

Date: 2026-07-07

Question: does the current `scripts/notebooks/find_phase.ipynb` experiment sit in the perturbative regime assumed by `Rydberg_sim.tex`, and is Claude's explanation for the phase mismatch supported?

## Sources

- `Rydberg_sim.tex` Theorem 1 proof defines the one-step single-atom reduction with `R_a = (H_Q - E_a)^-1` and explicitly assumes the resolvents exist, `||V_QP|| ||R_a|| << 1`, and the local large-detuning approximation is valid.
- `Rydberg_sim.tex` Theorem 2 makes the same second-order SW assumption for the interacting two-atom `P/Q` split.
- `scripts/notebooks/find_phase.ipynb` currently uses `SPACING_UM = 3.0`, `T = 1.0e-6`, `N_STEPS = 3000`, `Delta = 40.1e3 * MHz`, `OPTICS_LOSS = 0.8`, powers `p420_w = 6*(1-OPTICS_LOSS)` and `p1013_w = 100*(1-OPTICS_LOSS)`, and a Stark-compensated chirp `base_chirp + Dr_nom*b*b - D1_nom*a*a`.
- The notebook output records `omega_420/2pi = 3590.906 MHz`, `omega_1013/2pi = 692.432 MHz`, and `omega_eff/2pi = 15.502 MHz`.
- `docs/how_to_gates.qmd` already documents the intended interpretation: the converter is exact against instantaneous tex-frame resolvent reduction, but quantitative agreement with full seven-level dynamics is controlled only while `Omega_420/Delta_e` is small; at the full find_phase drive, dropped fourth-order optical terms can move phases by `O(1 rad)`.

## Checks run

I reconstructed the current notebook pulse and compared three evolutions:

- Full seven-level `rb87_7_mp` two-atom simulation.
- Theorem 1 single-atom `lower_cz_to_effective_01r` simulation on `01r`.
- Theorem 2 pair model `lower_cz_to_effective_pair` evolved as a dense `9x9` Hamiltonian.

I also computed a direct tex-assumption diagnostic over the pulse:

`eta(t) = ||V_QP(t)||_2 * max_a ||(H_Q(t) - E_a(t))^-1||_2`

This is a sufficient, conservative perturbative small parameter matching the form of the assumption in `Rydberg_sim.tex`.

## Results

Current full notebook drive:

```text
scale=1.0 Omega420/2pi=3590.9MHz Omega1013/2pi=692.4MHz ret_min=0.992758
  full   theta1=-1.822981 ZZ=-2.515369
  single theta1=+2.945866 ZZ=-3.124496 dtheta=1.514e+00 dZZ=6.091e-01
  pair   theta1=+2.683328 ZZ=-0.787482 dtheta=1.777e+00 dZZ=1.728e+00
```

Same pulse shape, 0.3x drive:

```text
scale=0.3 Omega420/2pi=1077.3MHz Omega1013/2pi=207.7MHz ret_min=0.569558
  full   theta1=+0.066955 ZZ=+0.949628
  single theta1=+0.071062 ZZ=+0.963596 dtheta=4.107e-03 dZZ=1.397e-02
  pair   theta1=+0.075341 ZZ=+0.948782 dtheta=8.386e-03 dZZ=8.463e-04
```

Tex perturbative diagnostic:

```text
scale=0.3
  Omega420/Delta=0.0269
  max ||V_PQ||/2pi=768.8 MHz  min sigma(HQ-Ea)/2pi=26.5 MHz
  max eta=28.9920  median eta=16.9856

scale=1.0
  Omega420/Delta=0.0895
  max ||V_PQ||/2pi=2562.6 MHz  min sigma(HQ-Ea)/2pi=6.2 MHz
  max eta=359.6829  median eta=46.6038
```

The large `eta` values are driven by the `r'` component of `H_Q`: `Rydberg_sim.tex` puts the eliminated-block entry at `delta_r - Delta_add`, and the current Stark-compensated chirp can bring that denominator close to the kept subspace. This norm diagnostic is conservative because the near-small-denominator eigenvector has limited overlap with the direct `P -> e_F` couplings, but it still shows the strict sufficient condition in the tex is not satisfied.

## Conclusion

Claude's main claim is supported: the current `find_phase.ipynb` full-power experiment is not quantitatively inside the second-order perturbative regime of `Rydberg_sim.tex`.

The strongest evidence is the scaling check. With the same pulse construction at `0.3x` drive, both Theorem 1 and Theorem 2 effective evolutions match the full seven-level phases at the `10^-2 rad` level. At the current full drive, the same comparison gives `O(1 rad)` phase errors. That is exactly the behavior expected when the instantaneous second-order reduction is implemented correctly but the accumulated phase is sensitive to omitted higher-order optical terms and near-small `r'` resolvent denominators.

So the phase mismatch in the notebook should not be treated as an implementation bug in `lower_cz_to_effective_01r` or `lower_cz_to_effective_pair`. It is a regime issue: the experiment is using a strong, Stark-compensated pulse where the tex's second-order perturbative assumptions are not small enough for quantitative phase prediction.
