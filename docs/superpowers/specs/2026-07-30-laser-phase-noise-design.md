# Laser phase noise from a measured PSD — design

Date: 2026-07-30
Status: approved in conversation.

## Goal

Add PSD-driven laser phase noise to `ryd_gate`, following Jiang, Scott, Friesen and
Saffman, *Sensitivity of quantum gate fidelity to laser phase and intensity noise*,
PRA **107**, 042611 (2023) (arXiv:2210.11007), and use it to re-render the
`results/max_leakage_297/a3.0` map family once per measured laser, with a
power↔Rabi conversion table under each figure so the minimum required 297 nm laser
power can be read off directly.

Two measured spectra drive the campaign, both in `results/297_laser_noise/`:

| | file | role |
|---|---|---|
| ECDL | `ECDL_phasenoise.png` | higher phase noise, full power |
| seed | `seed_lasernoise.png` | low phase noise, roughly half the power |

Both are one-sided frequency-noise amplitude spectral densities `sqrt(S_dnu)` in
Hz/sqrt(Hz) over 1 Hz – 1 MHz, measured on the **1180/1187 nm fundamental**.

## Settled decisions

1. **Fourth-harmonic conversion.** 297 nm is the fourth harmonic of the measured
   fundamental, so the optical phase is multiplied by 4 and
   `S_dnu(297) = 16 * S_dnu(fundamental)`. The harmonic is an explicit parameter,
   not a baked-in constant.
2. **No servo bump.** Neither measured trace shows one, and the campaign assumes
   none exists above the measurement edge. The analytic Gaussian servo-bump form
   (paper Eq. 39) is still implemented, because 420/1013 have no measured spectrum
   and because a future locked-laser measurement will need it.
3. **Extrapolation above 1 MHz is an explicit, bracketing parameter.** Nothing is
   measured above 1 MHz, but `f ~ Omega/2pi = 9–18 MHz` is where the gate is most
   sensitive. Two policies bracket it:
   - `flat` — hold `S_dnu` at its 1 MHz value (conservative; the headline)
   - `power` — continue the power law fitted to the last measured decade
     (ASD ~ f^-0.46 for ECDL, f^-0.55 for seed; optimistic)
4. **Full-grid errors come from a filter function, not Monte Carlo.** Monte Carlo
   over all 32832 grid points costs weeks per noise model; the filter function
   costs one pass, carries no statistical noise, and is reusable across all four
   (laser × extrapolation) combinations. Monte Carlo is the *validator*.
5. **Metric.** The phase-noise channel is the ensemble-mean *increase* in the same
   terminal nonlogical leakage the existing maps already report, so the new map
   composes with the existing family:
   `eps_phase = max_s <Delta L_s>`, `s` over the logical inputs 00/01/10/11.
6. **Store era.** A new sibling series inside the existing
   `results/max_leakage_297/a3.0` store. The coherent and scatter chunks are not
   recomputed and not touched.

## Physics

### Where the noise enters

The 297 drive contributes `H_drive = c(t) B + c*(t) B^dag` with
`c(t) = Omega(t) exp(-i phi(t))` (`scripts/max_leakage_297_sweep.py:429`). Phase
noise adds `phi -> phi + phi_n`. Because `B` lowers the Rydberg-excitation number
`N_r` by one, the frame transformation `V = exp(+i phi_n(t) N_r)` removes the noise
from the drive exactly and leaves

```
H(t) = H_0(t) + 2*pi*dnu(t) * N_r ,          phi_n_dot = 2*pi*dnu
```

with `H_0` the noiseless Hamiltonian in rad/s and `N_r` the diagonal operator
counting atoms in the Rydberg manifold — here **both** `r` and `r_garb`, since one
laser drives both legs. `V` is diagonal and acts as the identity on the logical
subspace, so neither the leakage nor the logical amplitudes are affected by the
residual frame factor at `T`; the transformation is exact, not an approximation.

The general rule, needed once 420/1013 are simulated: a laser group's phase noise
couples to the number operator of the levels *above* its transition. For the
420/1013 ladder that is `N_e + N_r` and `N_r` respectively, and their phase noises
add in the adiabatically eliminated two-level description (paper Eq. 93/95).

### Trace generation (Monte Carlo path)

Paper Eq. 104, rewritten for the one-sided densities this repository stores:

```
phi_n(t) = 2*pi*dnu_0*t + sum_j sqrt(2 * S_phi(f_j) * df_j) * cos(2*pi*f_j*t + psi_j)
S_phi(f) = S_dnu(f) / f**2 ,     psi_j ~ U[0, 2*pi)
```

(the paper's `2*sqrt(S^2s * df)` with `S^2s = S^1s / 2`).

The frequency grid is **hybrid**, because a uniform grid from 1 Hz to 72 MHz needs
1e8 terms:

- **below `1/t_gate`** the noise is frozen over the gate. Its whole effect is one
  quasi-static frequency offset `dnu_0 ~ N(0, sigma**2)` with
  `sigma**2 = int_{f_min}^{1/t_gate} S_dnu df`, contributing the linear term above.
- **above `1/t_gate`** a logarithmic grid, default 40 points per decade up to
  `f_max = 4 * Omega/2pi`, with `df_j` the bin width. About 350 terms.

`f_min` is an explicit modelling parameter, default **1 Hz** (the measurement
edge). It is not cosmetic: `S_dnu` rises as `f^-2.5` at the low end while the gate's
response to a static detuning is finite, so the error integral is infrared
divergent and `f_min` is physically the inverse of the calibration/relock timescale.
Every reported number carries its `f_min`, and the deliverable reports the
sensitivity of `eps_phase` to moving it to 10 Hz.

Traces are returned as a callable (dense presample + cubic spline) suitable for the
existing `phase_297_rad` / `phase_420_rad` / `phase_1013_rad` protocol callbacks.
Independent `psi_j` are drawn per laser group.

### Filter function (full-grid path)

First-order perturbation in `dnu` on the noiseless trajectory gives the terminal
state correction `chi_1(T) = -2*pi*i * int_0^T dnu(t) A(t) dt` with
`A(t) = U_0(T,t) N_r psi_0(t)`. The term linear in `dnu` averages to zero, so with
`G(f) = int_0^T A(t) exp(-2*pi*i*f*t) dt`:

```
<Delta L> = 2*pi**2 * int_{-inf}^{inf} S_dnu(|f|) * ||Q G(f)||**2 df
```

`Q` projects onto the nonlogical subspace — the same projector the existing
`max_leakage` uses. This is exact to second order in the noise, which the paper
validates against direct simulation in its Figs. 6–9 for exactly this weak-noise
regime (`2*pi*dnu / Omega ~ 0.01` here).

**Evaluation.** Only the projected components of `G` are needed, so the propagator
is never formed. Writing `Q = sum_q |q><q|` over the 12 nonlogical basis states,

```
<q|A_s(t)> = <phi_q(t)| N_r |psi_s(t)> ,     |phi_q(t)> = U_0(t,T)|q>
```

so the run needs one **backward** solve of the 12 `|q>` from `T` to `0` and one
forward solve of the logical inputs — 15 columns against the current 3, i.e. ~5x
the per-point cost. The backward leg is the same RHS evaluated at `T - tau` with a
flipped sign, integrated forward in `tau`; the envelope breakpoints are symmetric,
so the segment structure is unchanged.

This formulation is also what keeps the sampled integrand free of GHz content, even
though the Rydberg pair interaction and the `|0>` hyperfine offset are GHz-scale:
`phi_q` and `psi_s` obey the *same* equation, `N_r` is diagonal, and the product
runs over a single basis index `i`, so the `exp(-i D_i t)` factors cancel pointwise
and only drive-scale (<~50 MHz) structure survives. Forming `A(t)` from a
propagator instead would reintroduce `exp(i (D_j - D_s) t)` cross terms and demand
~70 ps sampling. The integrator itself is untouched; `n_t = 4096` samples suffice,
enforced by a `dt`-halving convergence check rather than assumed.

`G` is then evaluated by direct quadrature on a **logarithmic** frequency grid over
`[f_min, f_max] = [1 Hz, 200 MHz]` — not by FFT: the FFT grid spacing `1/T ~ 1 MHz`
cannot represent the low-frequency band at all, and the direct sum is one BLAS
matmul. `K(f)` carries fringe structure on the `1/T` scale, so it is evaluated at
200 points per decade and **integrated** into 30-points-per-decade storage bins,
`K_b = int_bin (||Q G(f)||**2 + ||Q G(-f)||**2) df`; the smooth `S_dnu` is then
sampled at bin centres.

**Reusability.** `||Q G(f)||**2` does not depend on the PSD. It is binned in
frequency and stored once; all four (laser × extrapolation) models are then a
reweighted sum over the stored bins, at no further solver cost.

## Components

### 1. `src/ryd_gate/phase_noise.py` (new; nothing existing is modified)

`docs/simulation.md:515` already rules that time-dependent correlated noise is
expressed by explicitly constructing protocol realizations in the research script,
not as a `NoiseModel` mode. This module is therefore a *pulse-construction* helper,
not an extension of `NoiseModel`/`simulate_ensemble`.

- `PhaseNoisePSD` — the spectrum. Constructed either from measured samples
  (`from_csv`, log-log interpolation, `harmonic`, `extrapolation` in
  `{"flat", "power"}`) or from the analytic white-plus-Gaussian-servo-bump form of
  paper Eq. 39. Public surface: `s_dnu(f)`, `s_phi(f)`, `sigma_nu(f_lo, f_hi)`.
- `phase_trace(psd, t_gate, *, seed, f_min, f_max, points_per_decade)` — the hybrid
  grid above; returns a `PhaseTrace` exposing `__call__(t)`, its `times`/`values`
  arrays, and the drawn `dnu_0`.

Reached as `ryd_gate.phase_noise`, an expert module in the same position as
`ryd_gate.physics` — deliberately **not** a top-level export, because
`src/ryd_gate/__init__.py` documents its namespace as "exactly the seven names
below" and that contract is not worth breaking for this.

### 2. `scripts/laser_noise_psd.py` (exists; extended)

Already digitizes both PNGs to `results/297_laser_noise/psd_{ECDL,seed}.csv` and
renders `psd_model.png/pdf`. Extended to emit the `PhaseNoisePSD` construction
parameters (fitted power-law exponent, 1 MHz edge value) as a small JSON so the
sweep and the plots read one authoritative source.

### 3. `scripts/max_leakage_297_sweep.py` — new `filter` subcommand

A third append-only series beside `chunks/` and `scatter/`, reusing `Store`,
`Runner`, `CostModel`, the manifest hash gates and resume unchanged:

```
results/max_leakage_297/a3.0/filter/filter_NNNNNN.npz
```

Each record holds, per point and per logical input, the binned kernel
`K_b = sum_{f in bin b} ||Q G(f)||**2 df` on a fixed logarithmic frequency grid,
plus the grid itself and the convergence-check residual. `eps_phase` for any PSD is
then `2*pi**2 * sum_b S_dnu(f_b) K_b`.

Scope: the `filter` fast path is 297-specific (one laser, one phase generator).
420/1013 are covered by the general Monte Carlo path through `phase_noise.py`.

### 4. Validation

- **Literature check (unit).** Two-level resonant Rabi with white noise: Monte Carlo
  via `phase_trace`, the filter function, and the paper's closed form
  `eps = pi**3 * h0 * N / Omega_0` (Eq. 79) must agree. This validates the whole
  chain against a published result. Note the paper's `h0` is **two-sided**
  (its Sec. II C says so explicitly), so against this repository's one-sided
  densities the target is `pi**3 * (h0_onesided / 2) * N / Omega_0`. This test is
  what pins that factor; the order-of-magnitude figures quoted while scoping the
  campaign did not apply it and are correspondingly a factor 2 high.
- **Trace statistics (unit).** Welch PSD of many generated traces recovers the input
  `S_dnu`; `var(phi)` matches `int S_phi df` over the resolved band.
- **Grid check (slow, deselected).** ~20 grid points, 200 shots each, direct Monte
  Carlo through the sweep solver against the `filter` prediction; acceptance is
  agreement within the Monte Carlo standard error.

### 5. Deliverable figures

Per laser, rendered through the existing `render_panel_grid`:

- `max_leakage`, `p_ryd`, `p_r_garb` — unchanged (noise-independent)
- `eps_phase` — new
- `total_error` = coherent leakage + `p_ryd` + `p_r_garb` + `eps_phase` — changed

Headline set uses the conservative `flat` extrapolation; `eps_phase` and
`total_error` are additionally rendered with `power`. 2 x 5 + 2 x 2 = 14 figures,
written to `results/max_leakage_297/a3.0/plots/phase_noise/<laser>/`.

**Power table.** Under every figure, 8 rows (`n`) x 6 columns
(`Omega_297/2pi` = 9, 11, 13.5, 15, 16.5, 18 MHz), each cell giving
*power at the atoms / nominal power* in W, with the caption stating
`beam_area = 420 um**2`, `optics loss = 0.8`, and the scaling rule `P ~ A`. The
per-`n` ARC Rabi-per-watt values are computed once and cached to npz so plotting
never touches ARC.

## Cost

| stage | estimate |
|---|---|
| `filter` pass, full 13x13 grid | ~5x the existing per-point cost (12 backward adjoint + 3 forward columns vs 3) -> ~6 h at 20 workers, **once** |
| all four noise models from the stored kernels | seconds |
| Monte Carlo validation, 20 points x 200 shots | ~1 h |

## Out of scope

Servo-bump fitting to a future locked-laser measurement; intensity (RIN) noise;
a `filter` fast path for the 420/1013 ladder; re-running the coherent or scatter
series; spacings other than 3.0 um.
