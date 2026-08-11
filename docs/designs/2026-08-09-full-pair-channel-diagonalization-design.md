---
status: accepted
document-role: design-record
---

# Full Pair-Channel Diagonalization Design

## Goal

Replace the final interaction diagnosis in
`scripts/check_297_pair_channels.py` with an explicit ARC pair-basis
diagonalization of

\[
H_{\rm pair}(B,R,\theta)=H_A(B)+H_B(B)+V_{dd}(R,\theta),
\]

while retaining the existing zero-field second-order \(C_6\) calculation only
as a labeled comparison. Update
`results/297_to_calibration/README.md` and `pair_channels.json` so their claims
match the model actually run.

## Scope and physical boundary

The authoritative calculation uses ARC 3.10.2
`PairStateInteractions.defineBasis(..., Bz=...)` with:

- \(53P_{3/2}+53P_{3/2}\) as the reference pair state;
- \(\theta=\pi/2\), \(\phi=0\), and \(R=3\,\mu\mathrm m\);
- `nRange=5`, so \(48\le n_1,n_2\le58\);
- `lrange=2`, so the retained orbital basis has \(l\le2\);
- `energyDelta=30e9` Hz;
- `interactionsUpTo=1`, so the explicit off-diagonal coupling is dipole-dipole;
- \(B=20\) G and \(160\) G, passed to ARC as tesla.

ARC applies the linear paramagnetic Zeeman shift to every retained pair-basis
state. The model does not include the diamagnetic term, hyperfine structure, or
magnetic mixing between different \(j\) manifolds. It is therefore an explicit
diagonalization of a documented truncated ARC pair basis, not an unqualified
exact atomic Hamiltonian.

## Alternatives considered

1. **Authoritative full-pair result plus an effective-\(C_6\) comparison
   (selected).** This directly answers whether the old approximation changes
   the blockade diagnosis while preserving an auditable comparison.
2. **Remove the effective model entirely.** This is smaller, but discards the
   quantitative explanation of why the previous result changed.
3. **Reimplement ARC angular and radial matrix construction locally.** This
   offers more control but duplicates mature Wigner-algebra code and creates a
   large validation burden outside the requested scope.

## Computation

For each magnetic field, the script builds one ARC basis and assembles the
sparse Hamiltonian in GHz:

```python
hamiltonian = calc.matDiagonal + calc.matR[0] / (R_um * 1e-6) ** 3
```

The same Hamiltonian is used for both diagnostic bare channels:

- `rr`: \(|53P_{3/2},-3/2;53P_{3/2},-3/2\rangle\);
- `r_rgarb`: \(|53P_{3/2},-3/2;53P_{3/2},-1/2\rangle\).

The bare reference energy for each channel is its corresponding diagonal entry
in `calc.matDiagonal`. Reported shifts are eigenenergies minus that reference.

The full 2276-dimensional matrix need not be densely diagonalized. For each
bare channel, deterministic sparse shift-invert diagonalization extracts states
near its bare reference energy. The requested eigenpair count grows until:

1. the returned spectrum brackets the full \(\pm5\Omega_{\max}\) weak-blockade
   window; and
2. either the requested captured bare-state overlap is reached or the declared
   eigenpair cap is reached.

The calculation records whether the weak window was bracketed and the total
captured overlap. `weak_shift_weight` is reported only from a bracketed window;
failure to bracket it is an error rather than a partial number presented as
complete. A fixed starting vector makes the sparse solve deterministic.

Each retained eigenstate records its shift, overlap with the diagnostic bare
channel, total weight in the \(53P_{3/2}+53P_{3/2}\) manifold, and its leading
explicit pair-basis components.

## Effective-model comparison

The existing `getC6perturbatively(..., degeneratePerturbation=True)` path is
kept under `effective_c6_comparison`. It contains the zero-field \(16\times16\)
\(C_6\) eigensystem, the reconstructed \(C_6/R^6+H_Z^{PP}\) spectra, and the
effective exchange matrix element. None of these finite-field spectra is used
as the final physical conclusion.

The current hand-written intermediate-state inventory is renamed
`radial_defect_ranking`. Its values remain
\((R_1R_2)^2/|\delta|\), without angular factors or denominator sign, and the
README describes it only as a radial-plus-defect screening diagnostic.

## Output contract

`pair_channels.json` receives a schema version and four top-level sections:

- `params`: shared geometry, truncations, Rabi threshold, ARC version, and
  stated approximations;
- `full_pair`: authoritative field-resolved basis metadata and channel spectra;
- `effective_c6_comparison`: the previous perturbative calculation;
- `radial_defect_ranking`: the explicitly non-quantitative channel inventory.

All energies stored in spectra are labeled in MHz. Basis-building and
diagonalization timings are provenance, not physics claims.

## Tests

Fast tests use small synthetic sparse Hamiltonians to verify:

- ARC distance scaling and GHz-to-MHz conversion;
- channel-specific reference-energy subtraction;
- overlap sorting and weak-window summation;
- adaptive eigenpair extraction brackets the requested window;
- failure is explicit when a requested window cannot be bracketed.

One `slow` test builds a deliberately reduced ARC pair basis and verifies that
`Bz` changes more than the reference \(53P+53P\) diagonal alone, guarding
against regression to “zero-field \(C_6\) plus hand-added initial-manifold
Zeeman.”

## Documentation and result validation

After tests pass, run the script for 20 G and 160 G to regenerate
`results/297_to_calibration/pair_channels.json`. Rewrite the README appendix
from those generated values, distinguish the authoritative and comparison
models, update the runtime/provenance entry, and document all truncations.
Because this writes data under `results/`, apply the repository's mandatory
`results-report` validation before delivery.
