# Project literature notes

## Optimal control

- The GRAPE baseline and time-optimal blockade-gate conventions come from
  `jandura2022timeoptimal` [@jandura2022timeoptimal].
- The direct trajectory-optimization design and ZXZ reproduction target refer
  to the direct-control section and appendix of `hu2025universal`
  [@hu2025universal]. The pulse is optimized on three atoms; applying the same
  waveform to a longer chain is numerical transfer, not an error-controlled
  consequence of the direct-transcription optimizer.
- The global and local Hilbert--Schmidt compilation costs used to separate
  whole-unitary accuracy from single-site channel accuracy originate in QAQC
  [@khatri_2019_quantum].

## Local-to-global compilation and certification

- Huang et al. reconstruct a promised shallow circuit from local inversions
  and sew them with ancillas and SWAPs; their diamond-distance error is the sum
  of the local inversion errors [@huang_2024_learning]. The theorem assumes
  access to the large-system unitary and a shallow-circuit/QCA promise. It does
  not infer an unknown large-system unitary from its three-site restriction.
- Mizuta et al. give the closest positive transfer theorem for the present
  control problem: a local variational cost evaluated on a subsystem covering
  the Lieb--Robinson cone bounds the corresponding cost on a larger system
  [@mizuta_2022_local]. Local-observable accuracy can use a system-size
  independent window at fixed time and tolerance; global fidelity requires a
  window that grows with system size.
- The resulting control prescription is configuration-aware rather than a
  fixed-waveform transfer: map the measured positions and target graph to a
  new global waveform, minimize the worst strong local residual over every
  causal-window environment, and accept it only after a Lieb--Robinson and
  Duhamel error audit. The three-atom pulse is only a continuation warm start.
  A finite control net makes the search complete with margin inside a fixed
  compact knot family, while exact post-checks make any returned certificate
  sound.
- Before that search, the distinct windows form a finite block-diagonal
  ensemble driven by the same controls. Its dynamical Lie algebra gives a
  structural feasibility test: the reachable group must intersect the
  product of zero-local-residual target cosets. A disjoint intersection gives
  a positive residual floor and proves that more global control generators or
  local addressing are necessary [@agrachev_2016_ensemble].
- Haah et al. provide a constructive Lieb--Robinson decomposition of local
  Hamiltonian evolution into overlapping finite blocks and make explicit why
  the total error budget is shared among an extensive number of blocks
  [@haah_2018_quantum].
- Else et al. supply the power-law Lieb--Robinson bound relevant to the
  Rydberg tail. In one dimension, the van der Waals interaction
  (1/r^6=1/r^{D+\alpha}) has (D=1,\alpha=5), so fixed-time local
  certification is possible, while a global error guarantee needs a growing
  buffer [@else_2020_improved].
- Bravyi, Parham, and Tran show that shallow-circuit distance to the identity
  can be bounded from light-cone-separated local commutators with a constant
  multiplicative factor [@bravyi_2026_identity]. This is a scalable
  certification tool, not a replacement for the missing local data.
- Trivedi, Franco Rubio, and Cirac formalize the system-size independent
  stability of fixed-time local observables under extensive simulator errors
  [@trivedi_2024_quantum]. This explains why local (Z) profiles may remain
  useful even when global state or process fidelity decays with atom number.

## Key open problems

- Derive useful numerical constants, rather than only asymptotic exponents, in
  a Lieb--Robinson bound for the time-dependent (1/r^6) Rydberg Hamiltonian
  at the experimental pulse amplitudes and duration.
- Decide whether the local inversions can be implemented with the available
  global Ω/Δ controls. Huang et al.'s exact sewing circuit additionally uses
  local SWAPs, ancillas, and a coloring schedule.
- Build a control objective on a five-site target light cone plus a physical
  propagation buffer, and optimize several boundary classes jointly. A
  three-site full-unitary objective cannot see overlapping ZXZ terms or
  connected effective errors supported on four or more sites.

## Key bottlenecks

- The three-site restriction is non-identifying: arbitrary connected
  four-site-and-larger errors vanish on three atoms but alter every larger
  chain.
- Global channel error is extensive. A fixed local error density generally
  gives a global infidelity that grows with system size, even when every fixed
  local observable has a bounded error.
- Long pulse duration enlarges the causal cone and resolves weak long-range
  couplings, so a pulse with excellent three-site fidelity can transfer worse
  than a shorter, less accurate pulse.
- A change of geometry or coordination number changes both the native
  interaction graph and the target light-cone classes; one-dimensional
  training does not certify a two-dimensional deployment.

## Dynamics simulation

- The annealing scripts use the long-range-observable and convergence caveats
  associated with `sun2026quantumclassical` [@sun2026quantumclassical].

## Local source policy

External full-text TeX is ignored under `.knowledge/.raw/`; it is a local
reading aid, not a maintained manuscript. Repository-owned derivations remain
tracked with their project manuscript under `manuscripts/` and are cited by
stable path from code and tests.
