---
status: accepted
document-role: design-record
---

# Zeeman-Resolved 53P Pair-Potential Figures

## Goal

Extend the existing 10 GHz pair-potential study so that its scheme-1
seven-angle figures cover all four Zeeman levels of

\[
53P_{3/2},\qquad m_j\in\{-3/2,-1/2,+1/2,+3/2\},
\]

at 20, 40, and 60 G. Simplify those figures by removing the ineffective
overlap color bar and replacing it with an explicit marker-size legend. Remove
the scheme-2 angular heatmap and \(W_{\rm weak}(R)\) summary figures, while
retaining the underlying numerical spectrum and weak-weight arrays in the
JSON sidecar.

The existing 70S scheme-1 benchmark remains in scope. This change is a
pair-spectrum comparison only; it does not change the four-level gate model,
laser couplings, or pulse optimization.

## Physical quantity and state definitions

For each selected Zeeman level, the doorway state is

\[
|rr;m_j\rangle
=|53P_{3/2},m_j;53P_{3/2},m_j\rangle,
\]

and every plotted local eigenstate is measured by

\[
\Delta_k/h=[E_k-\epsilon_{rr}(B)]/h,
\qquad
p_k=|\langle rr;m_j|\Psi_k\rangle|^2.
\]

The calculation keeps the existing fields, seven values of \(\theta\),
\(\phi=0\), 41-point \(R=2.5\text{--}8.0\,\mu\mathrm m\) grid, 10 GHz energy
window, \(n\pm3\), \(l_{\max}=2\), dipole-dipole coupling, ARC linear Zeeman
model, full diagonalization, branch count, anchor, and numerical acceptance
criteria. Each Zeeman level is a separate ARC doorway calculation; changing
\(m_j\) is not treated as a relabeling of the existing spectrum.

At zero field, opposite \(m_j\) values are related by time reversal. At fixed
positive field they are not assumed to have identical spectra. The generated
data, rather than symmetry inference, are authoritative for the plotted
20/40/60 G cases.

## Data model and naming

Keep the existing `53P3_2` key for \(m_j=-3/2\) so existing references and
filenames remain stable. Add three manifolds:

- `53P3_2_mj_m1_2` for \(m_j=-1/2\);
- `53P3_2_mj_p1_2` for \(m_j=+1/2\);
- `53P3_2_mj_p3_2` for \(m_j=+3/2\).

Each record stores an explicit `mj` and a display label containing that value.
The existing `70S1_2` record remains unchanged. The resulting study contains
five manifolds, three fields, and seven angles, for 105 complete cases.

The sidecar structure does not change: every case still stores the complete
local bright spectrum above the existing cutoff, tracked branches,
`spectrum_rr_overlap`, and `weak_shift_weight`. Adding manifold records is an
extension of the existing schema, not a new interpretation of its fields.
Strict configuration validation remains in place, so the production artifact
is regenerated rather than silently resuming a sidecar with a different
manifold fingerprint.

## Scheme-1 figure design

Each state/field figure retains the current seven data panels plus the eighth
legend panel. It shows two layers:

1. The complete retained local spectrum at every \(R\), drawn as neutral gray
   points. This layer remains because branches selected among the five largest
   anchor overlaps can leave substantial doorway weight in untracked states
   away from \(R=3\,\mu\mathrm m\).
2. Up to five continuously tracked anchor-ranked branches, drawn with the
   existing categorical line colors and sparse colored markers.

Both layers use one shared marker-area function of \(p_k\). The gray spectrum
therefore retains quantitative overlap information without a second color
encoding. The eighth panel contains:

- the existing five line-color handles for anchor rank; and
- gray marker-only handles labeled \(p_k=0.1,0.5,1.0\).

Remove the magma mapping, `ScalarMappable`, and figure color bar. The legend
must state that marker *area* encodes \(p_k\); line color continues to encode
anchor rank only.

All four 53P Zeeman manifolds use one shared symmetric energy limit computed
from their complete plotted spectra and tracked branches. This makes visual
comparisons across \(m_j\) honest. The 70S benchmark keeps its own shared
limit across magnetic fields so its larger scale does not compress the 53P
figures.

Expected filenames are the state key followed by the field, including:

- `pair_potential_53P3_2_B20G.png` for the existing \(-3/2\) case;
- `pair_potential_53P3_2_mj_m1_2_B20G.png`;
- `pair_potential_53P3_2_mj_p1_2_B20G.png`;
- `pair_potential_53P3_2_mj_p3_2_B20G.png`;

with corresponding 40 G and 60 G files, plus the existing three 70S files.

## Removal of scheme 2

Delete the scheme-2-only spectral-density and field-summary rendering
functions. `render_pair_potential_figures` returns only the fifteen scheme-1
paths and no longer creates:

- `pair_potential_summary_B20G.png`;
- `pair_potential_summary_B40G.png`;
- `pair_potential_summary_B60G.png`.

Remove those three generated PNG files from the results directory and remove
their prose and image references from the README. Do not remove
`weak_shift_weight` or the complete local spectra from
`pair_potential_curves.json`: they remain useful diagnostics and support the
README's compact Zeeman comparison.

## README and provenance

Update `results/297_to_calibration/README.md` to:

- define all four 53P doorway states and distinguish them from laser
  accessibility in the current \(\sigma^-\) gate;
- explain the gray complete spectrum, categorical branch colors, and common
  overlap-size legend;
- embed the twelve Zeeman-resolved 53P scheme-1 figures and retain the three
  70S scheme-1 figures;
- remove the complete scheme-2 subsection and summary-image references;
- add a compact generated-value table comparing
  \(W_{\rm weak}(R=3\,\mu\mathrm m)\) for the four \(m_j\) values at each field
  and representative \(\theta=0^\circ,45^\circ,90^\circ\);
- update case counts, runtime, file counts, reproduction text, and provenance
  from the regenerated artifact without inventing values.

The README must state that only \(m_j=-3/2\) is the target `r` state of the
current \(\sigma^-\) gate. The other three calculations compare possible
Zeeman doorway pair spectra; they are not gate-error predictions and do not
include the Rabi frequencies, unwanted optical legs, or pulse reoptimization
needed to make those states gate targets.

## Tests and verification

Use test-driven changes. Fast tests must first fail and then pass for:

1. the four explicit 53P `mj` definitions plus the unchanged 70S benchmark;
2. one common marker-area mapping used by gray spectrum and tracked markers;
3. a size legend containing \(p_k=0.1,0.5,1.0\) and no figure color bar;
4. renderer output containing exactly fifteen scheme-1 filenames and no
   `pair_potential_summary_*` files;
5. shared 53P energy limits and a separate 70S limit;
6. configuration validation and expected 105-case completion accounting.

After fast tests pass, run the existing slow ARC test, regenerate the full
sidecar with single-threaded BLAS, render all figures, inspect representative
negative- and positive-\(m_j\) images, and verify `--plot-only` does not rewrite
JSON. Check that the three obsolete summary PNGs are absent. Finally update
and validate the mandatory results README with the repository's
`results-report` validator.

## Non-goals

- No mixed \(m_{j1}\ne m_{j2}\) scheme-1 curves.
- No new magnetic fields, angles, distances, or basis-convergence scan.
- No change to the 30 GHz single-distance `pair_channels.json` audit.
- No change to laser polarization, Rabi calibration, four-level dynamics, or
  gate optimization.
- No replacement of the complete local spectrum by only five tracked lines.
