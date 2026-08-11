---
status: accepted
document-role: design-record
---

# 297 nm single-photon leakage sweep — design

Date: 2026-07-24
Status: approved in conversation; implementation plan to follow.

## Goal

A standalone counterpart of `scripts/max_leakage_ode_sweep.py` for the 297 nm
single-photon CZ: `scripts/max_leakage_297_sweep.py`, writing the same kind of
resumable store (chunks + scatter + exports + plots, hash-gated, nested inner
grids) to `results/max_leakage_297/a{spacing:.1f}/`, and rendering the same
8×9 map family. Single-photon excitation has no intermediate state, so the
scattering budget has no `p_mid` channel and the `total_error` map is expected
to differ qualitatively from the two-photon
`results/max_leakage_ode/legacy_c6-874/plots/total_error_8x9.pdf`.

## Decisions (settled with the user)

1. **Model**: existing preset `rb87_297_clock_4` (clock encoding, levels
   `0/1/r/r_garb`; garbage/target leg ratio 1/√3 ≈ 0.577, n-independent) +
   `Direct297CZProtocol`. No stretched-encoding variant in this scan.
2. **Waveform**: same adiabatic family as the two-photon scan — quintic
   smoothstep envelope (ramp fraction 0.15) on Ω₂₉₇, phase =
   ∫chirp with chirp(t) = −D_sweep·cos(2πt/T), no offset, no Stark
   compensation term (none exists for single photon).
3. **Panel axes (8×9)**: Rydberg level n ∈ {50, 53, 56, 60, 64, 68, 71, 73}
   (nP₃/₂) × t_gate_us ∈ {1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5} —
   identical to the two-photon scan's T axis.
   Per-n panel operators carry their own ARC lifetime
   (`includeLevelsUpTo = n+27`, 300 K), channel-resolved P-state C6
   (θ=π/2 geometry as in the preset), Ω(n) matrix elements, and Zeeman data.
4. **Inner axes (nested 4→7→13→25 grids)**: single-photon Rabi Ω₂₉₇/2π
   anchors {9, 12, 15, 18} MHz (target clock leg, swept directly — power,
   optics loss and beam area are deliberately NOT model inputs because they
   are poorly known; 1–3 W laser power at the notebook's nominal
   0.8-loss/420 µm² optics corresponds to ≈9.6–16.6 MHz at 53P, which the
   anchor range covers with margin) × D_sweep anchors {2, 10, 20, 30} MHz
   with the 20 MHz hardware-cap annotation — identical D_sweep structure to
   the two-photon scan for comparability. The garbage leg is Ω·(1/√3) via
   the preset's internal leg ratio; only `omega_297_max_rad_s` reaches the
   protocol.
5. **Field/geometry**: B = 20 G (matches the two-photon scan; supersedes the
   100 G in the external 53P₃/₂ decision memo), spacing default 3.0 µm with
   the `--spacing-um`/`a{spacing:.1f}` sub-store convention carried over.
6. **Scattering series**: channels `p_ryd` (target nP₃/₂ decay) and
   `p_r_garb` only; `total_error = coherent leakage + p_ryd + p_r_garb`.
   Plot metrics: `max_leakage, p_ryd, p_r_garb, p_loss_total, total_error`
   (five maps; no `p_mid`). No per-panel PNGs.
7. **Era**: current `main` (ARC-computed, channel-resolved C6). No legacy
   compatibility constraints — this is a fresh store family.

## Architecture: fork, not shared core

`scripts/max_leakage_297_sweep.py` is created by copying
`scripts/max_leakage_ode_sweep.py` and swapping the physics layer. A
shared-core refactor is explicitly deferred: the original script is actively
driving the five-spacing campaign, and its store/solver/Runner machinery is
battle-tested — forking freezes it; the two scripts may be unified later.

Physics-layer diffs (everything else — Store, hashes, resume, Runner,
CostModel, batching, export, plot scaffolding, CLI shape — stays):

- `ScanConfig`: `delta_e_ghz` axis → `ryd_n` axis; `omega_anchors_mhz` →
  `p297_anchors_w`; add `optics_loss = 0.8`, `beam_area_um2 = 420.0`; drop
  two-photon-only fields (`p1013_nominal_w`, `beam_factor`, `detuning_sign`
  if unused). `spacing_um`, tolerances, `ramp_frac`, `n_eval_trajectory`
  carry over.
- `warm_and_build`: per-n `PanelOperators` from
  `level_structure("rb87_297_clock_4", ryd_level=n, magnetic_field_G=20)`
  + `Register.chain(2, spacing_um=…)`; model_hash = SHA256 over the
  aggregated per-n operator bytes (same recipe). Setup checks: H equivalence
  vs `backend="exact_ode"` on one panel, error-norm seam, swap symmetry —
  same gates as the original.
- Pulse layer: Ω₂₉₇(P, n) from `rb87_297_clock_rabi_frequencies`; envelope
  and `phase_from_chirp` identical in form; `pulse_hash` sampled the same
  way.
- Scatter pass: Γ from the preset's `decay_rates_per_s` (`r`, `r_garb`,
  equal by construction); 301-sample trapezoid per input, same
  trajectory-equivalence gate.
- Two-atom dimension 16 (vs 49) and no GHz optical scales → points are
  expected 5–20× cheaper; batching/tolerances unchanged until the pilot
  measures real throughput.

## Store layout

```
results/max_leakage_297/
└── a3.0/                    # default; same sub-store convention as max_leakage_ode
    └── manifest.json chunks/ scatter/ trajectories/ exports/ plots/ reports/ logs/
```

## Testing

New slim `tests/test_max_leakage_297_sweep.py` (importlib load, alias
`mls297`), following the existing file's patterns: axis sizes/nesting/20 MHz
node, pulse envelope-integral and phase-is-integral-of-chirp checks, key
canonicalization, manifest/chunk provenance guards (parametrized reuse),
parser coverage, plot smoke on a synthetic mini-store asserting the five
metrics and no `panel_*.png`, physics-hash sensitivity to `ryd_n` axis and
`spacing_um`. One slow ARC equivalence test vs `backend="exact_ode"`
(deselected by default) mirroring the original's.

## Execution plan (separate decision)

After (or throttled alongside) the running five-spacing campaign:
`pilot → run --target-level 13 → scatter --level 13 → export → plot × 5`
on the DGX from main. Expected 13×13 wall-time well under the two-photon
figure given the 16-dim model; pilot measures it.

## What this scan feeds back to the 53P₃/₂ decision

The n axis prices "staying at 53" (blockade-limited → τ-limited transition);
the P₂₉₇ axis locates the safe Ω window under the blockade wall at n=53; the
D_sweep axis sets the frequency-control tolerance for the locked laser chain;
the per-channel decomposition isolates the garbage-branch cost of the clock
encoding.

## Out of scope

Stretched-encoding model variant; shared-core refactor of the two sweep
scripts; running the campaign; hyperfine-resolved absolute laser setpoints
(documented in the decision memo, irrelevant to the rotating-frame model).
