# Project literature notes

## Optimal control

- The GRAPE baseline and time-optimal blockade-gate conventions come from
  `jandura2022timeoptimal` [@jandura2022timeoptimal].
- The direct trajectory-optimization design and ZXZ reproduction target refer
  to the direct-control section and appendix of `hu2025universal`
  [@hu2025universal].

## Dynamics simulation

- The annealing scripts use the long-range-observable and convergence caveats
  associated with `sun2026quantumclassical` [@sun2026quantumclassical].

## Local source policy

External full-text TeX is ignored under `.knowledge/.raw/`; it is a local
reading aid, not a maintained manuscript. Repository-owned derivations remain
tracked with their project manuscript under `manuscripts/` and are cited by
stable path from code and tests.

## Dynamic detuning for MWIS encodings (surveyed 2026-08-11)

### Field landscape

- Arbitrary-connectivity Rydberg MWIS encodings replace graph crossings by
  crossing and crossing-with-edge (CWE) gadgets [@nguyen2023quantum].
- In the ordered regime, the CWE low-energy manifold is an L-shaped
  two-domain-wall configuration graph.  Its corner supports a geometrically
  localized state, and sweeping a distant logical endpoint through this state
  creates an exponentially small avoided crossing [@bombieri2025quantum].
- Corner-induced bound states and their exponentially decaying lead tails are
  standard quantum-waveguide phenomena [@dauge2012quantum]; graph Agmon
  estimates provide a discrete localization language [@steinerberger2022agmon].
- Site-resolved light shifts are experimentally available and have already
  been used to encode weighted non-unit-disk graphs on Rydberg hardware
  [@oliveira2025demonstration].  This makes diagonal, time-dependent controls a
  physical path deformation rather than a purely formal catalyst.
- Diagonal catalysts and optimized schedule paths can improve gaps, but their
  benefit is instance- and bias-dependent; a poorly aligned catalyst can make
  an anneal worse [@albash2021diagonal; @zeng2016schedule].  A mere
  reparameterization of a fixed Hamiltonian path changes runtime allocation,
  not the instantaneous minimum gap [@jeong2026enhanced].

### Open problems

- Characterize the column space of the physical-to-configuration-space
  detuning map for networks containing many gadgets, including its locality
  and rank after quotienting out constant energy shifts.
- Find solution-agnostic delocalizing potentials that work simultaneously for
  all logical corridors, rather than selecting a corridor using knowledge of a
  preferred local outcome.
- Transfer effective-model polynomial-gap certificates to the full Rydberg
  Hamiltonian at fixed experimental ratios of Rabi frequency, detuning,
  blockade strength, and interaction-tail compensation.
- Determine whether a path with a small instantaneous gap is a true runtime
  bottleneck or admits a controlled diabatic bypass; hard Rydberg instances
  can benefit from deliberately nonadiabatic protocols [@schiffer2024circumventing].

### Practical bottlenecks and cautions

- Time dependence does not enlarge the instantaneous span of diagonal
  operators.  If the desired configuration-space potential is outside the
  occupation-matrix column space, dynamic scheduling alone cannot synthesize
  it exactly.
- A worst-case Weyl bound requires the residual potential to be smaller than
  the target polynomial gap, which can be far stricter than experimental
  calibration.  Structure-aware no-bound-state or conductance certificates
  are therefore preferable to a raw least-squares residual.
- Endpoint constraints, amplitude and slew-rate limits, and leakage out of the
  domain-wall manifold must be imposed during optimization.  Spatially
  inhomogeneous driving can change transition order in other annealing models
  [@susa2018exponential], but that observation does not by itself certify the
  CWE control path.
