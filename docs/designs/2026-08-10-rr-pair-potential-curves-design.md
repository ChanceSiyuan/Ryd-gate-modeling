---
status: accepted
document-role: design-record
---

# `rr` Pair-Potential Curves at 20, 40, and 60 G

## Goal

Extend the explicit ARC pair-basis audit so that it shows how the dressed
pair-state energies coupled to

$$
|rr\rangle
=|53P_{3/2},m_j=-3/2;53P_{3/2},m_j=-3/2\rangle
$$

change with interatomic distance. Produce one figure at each of 20, 40, and
60 G. Each figure must show six continuously tracked eigenvalue branches and
their instantaneous `rr` overlap. Keep the existing 160 G single-distance
audit.

The requested “largest” states mean the six eigenstates with the largest
$|rr\rangle$ overlap at the working distance $R=3\ \mu\mathrm m$, not the six
algebraically largest eigenvalues of the truncated 2276-dimensional matrix.
The latter are set by the 30 GHz basis cutoff and are not blockade channels.

## Physics quantity

For fixed $B$ and $R$, diagonalize the existing truncated Hamiltonian

$$
H_{\rm pair}(B,R,\theta)
=\sum_\nu\epsilon_\nu(B)|\nu\rangle\langle\nu|
+V_{dd}(R,\theta),
$$

with the existing settings $\theta=\pi/2$, $\phi=0$, $n_i=48\ldots58$,
$l_i\le2$, zero-field pair-defect window 30 GHz, and dipole-dipole coupling
only. Report the frequency shift relative to the uncoupled `rr` basis energy,

$$
\Delta_k(B,R)/h=[E_k(B,R)-\epsilon_{rr}(B)]/h,
$$

and the instantaneous spectral weight

$$
p_k(B,R)=|\langle\Psi_k(B,R)|rr\rangle|^2.
$$

The curves are an explicit truncated-pair result, not an unqualified exact
atomic spectrum. All existing Zeeman, diamagnetic, hyperfine, angular-momentum,
and basis-truncation qualifications remain in force.

## Distance grid and branch definition

Use 81 distances from 2.5 to 8.0 micrometres. Construct the grid in two pieces
so that it contains the working distance exactly: nine points from 2.5 to 3.0
inclusive and 73 points from 3.0 to 8.0 inclusive, dropping the duplicated
3.0 point.

At $R=3\ \mu\mathrm m$, select the six eigenstates with the largest $p_k$.
These are the branch seeds. Track each seed toward increasing and decreasing
$R$ separately. Between adjacent points form

$$
O_{ij}=|\langle\Psi_i(R_a)|\Psi_j(R_b)\rangle|^2
$$

and use a Hungarian assignment that maximizes total overlap. This follows
continuous adiabatic eigenvectors through avoided crossings. It deliberately
does not reselect the six largest instantaneous `rr` weights at every point,
which would splice different eigenstates into discontinuous curves.

At each distance, obtain a local candidate eigenspace around
$\epsilon_{rr}(B)$ with deterministic shift-invert `eigsh`. Start from 64
candidates and double up to 256 until the captured `rr` weight is at least
0.995 and every assigned branch has adjacent eigenvector overlap at least
0.25. Store the adjacent-state match at every grid point, using 1.0 at the
anchor, so all branch arrays have length 81. If the maximum candidate count
still misses either threshold, fail rather than draw a misidentified curve.

The six curves define numerical eigenvector branches. At an exact degeneracy,
individual vectors inside the degenerate subspace are basis-dependent; the
README will state this rather than assigning extra physical meaning to the
branch label.

## Figure design

Create three PNG files in `results/297_to_calibration/`:

- `pair_rr_potential_B20G.png`
- `pair_rr_potential_B40G.png`
- `pair_rr_potential_B60G.png`

Each image has two vertically aligned panels and uses the same axes and branch
colours as the other images.

The upper panel plots $\Delta_k/h$ against $R$. Use a symmetric-logarithmic
energy axis so that weak shifts and the more distant strongly shifted branch
remain visible together. Shade the diagnostic weak window
$|\Delta/h|<83.07$ MHz, draw a zero-energy guide, and mark $R=3\ \mu$m.
Plot each tracked eigenvalue as a fixed-colour line and overlay sparse markers
whose area encodes $p_k$.

The lower panel plots $p_k(R)$ for the same six colours on a linear 0-to-1
axis. This makes transfer of `rr` character at avoided crossings explicit.
The legend is ordered by the seed overlap at 3 micrometres and gives, for each
branch, its anchor shift, anchor `rr` overlap, and largest anchor basis-state
component. Detailed top-four anchor components remain available in the JSON
and are summarized in the README rather than crowded into the plot.

The three plots use global energy limits computed from all 18 tracked curves,
so visual differences between fields are not produced by autoscaling.

## Data and code changes

Keep `scripts/check_297_pair_channels.py` as the sole producer.

- Expand the full-pair field set to 20, 40, 60, and 160 G.
- Add a plot-field set containing only 20, 40, and 60 G.
- Reuse each field's ARC basis and sparse matrix parts across all 81 distances;
  only the $R^{-3}$ coefficient changes.
- Add `rr_potential_curves` under each plotted
  `full_pair.fields.<B>` record. Store the distance array, six branches of
  shifts and overlaps, anchor metadata, candidate-count diagnostics, and
  adjacent-match diagnostics.
- Increment `schema_version` from 2 to 3 because the artifact now contains a
  distance-dependent branch-tracking schema, not only single-distance spectra.
- Keep `effective_c6_comparison` explicitly non-authoritative; extending it to
  40 and 60 G is acceptable because the field loop is shared, but it is not
  plotted.
- Add `--plot-only`, which reads the existing JSON and regenerates the three
  figures without ARC diagonalization. The default command recomputes data,
  writes JSON, and writes the figures.

The PNG files remain untracked under the repository's results policy. The
JSON, script, tests, README, results index, design, and implementation plan are
tracked.

## TDD and verification

Add tests before production changes for:

1. the 81-point grid containing both endpoints and exactly 3 micrometres;
2. anchor selection by descending `rr` overlap;
3. Hungarian matching preserving branch identity in a synthetic avoided
   crossing where energy-order matching would swap it;
4. curve records containing six equal-length shift, overlap, and continuity
   arrays;
5. output fields containing 20, 40, 60, and 160 G while plots use only 20,
   40, and 60 G;
6. a synthetic `--plot-only`/render smoke test producing the three expected
   image names without invoking ARC.

Run the fast tests through a visible RED-to-GREEN cycle, then run the real ARC
integration test. Run the production calculation once, inspect all three
figures, exercise `--plot-only`, validate the JSON diagnostics, update the
README appendix and `results/README.md`, and run the mandatory results-report
validator. Preserve the user's current uncommitted README edits while adding
the new subsection and figure references.

## README interpretation

The appendix will define the plotted shift and overlap, explain the adiabatic
matching rule, embed all three figures, and give a compact anchor table for the
six branches at each field. It will distinguish three concepts:

- eigenvalue-branch continuity from adjacent eigenvector overlap;
- instantaneous laser relevance from $p_k(R)$;
- the heuristic blockade window from $|\Delta/h|<83.07$ MHz.

It will not reinterpret $p_k$ as a transition probability or gate error, and
will not claim that branch tracking removes the need for a multi-channel gate
dynamics calculation.

## Non-goals

- No `r_rgarb` potential curves.
- No sweep over angle, principal quantum number, or basis cutoff. No
  potential-curve field beyond 20, 40, and 60 G; 160 G remains a
  single-distance audit only.
- No change to the four-level gate dynamics or pulse optimization.
- No algebraically largest cutoff-edge eigenvalues.
- No signed eigenvector-coefficient visualization; the physically invariant
  overlap and basis probabilities are reported instead.
