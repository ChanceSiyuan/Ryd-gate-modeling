# Theory

This directory contains repository-owned derivations that define or validate
implemented physical models. It does not contain downloaded paper sources.

## Derivations

- `derivations/rydberg-simulation.tex` — a LaTeX chapter fragment covering the
  full single- and two-photon Hamiltonians, effective reductions, and pulse
  optimization. The theorem numbering is referenced by
  `src/ryd_gate/core/effective_theory.py` and its physics tests.

The fragment is not currently a standalone LaTeX document: it expects a parent
preamble and references `pic/GRAPE.png`, which is not present in this checkout.
That missing figure is a known source gap, not a reproducible project artifact.
External literature metadata and notes live in `.knowledge/`; ignored full-text
sources live in `.knowledge/.raw/`.
