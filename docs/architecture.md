# Architecture

`ryd_gate` separates the **physics model** (systems, protocols, and a unified
Hamiltonian IR) from the **algorithm backends** that evolve it. The core never
materializes algorithm-specific matrices; each backend lowers the same IR into
its own representation.

## Data flow

```
        RydbergSystem            Protocol
   (geometry + level structure)  (pulse schedule)
                 \                  /
                  \                /
                   v              v
            ryd_gate.compile_hamiltonian_ir(system, params)
                          |
                          v
                   HamiltonianIR
        (static_terms, drive_terms, basis, geometry, ...)
                          |
            +-------------+-------------------------------+
            |                                             |
            v                                             v
   exact backend                              tensor-network backends
   (ExactSparseCompiler ->                    (TNCompiler -> TNEvolutionIR ->
    SparseExpmBackend /                        tenpy / itensor / ttn /
    DenseODEBackend)                           gputn / 2dtn / nqs)
            |                                             |
            +----------------------+----------------------+
                                   v
                            EvolutionResult
                     (psi_final, times, states, metadata)
```

The friendly entry point `ryd_gate.simulate(system, x, psi0, backend=...)`
dispatches across these: `backend="exact"` calls
`ryd_gate.backends.exact.simulate`, and any tensor-network name lowers the
system to a TN lattice spec and calls `ryd_gate.backends.tn_common.simulate_tn`.

## Backends

All backends live under `ryd_gate.backends`. Heavy/optional engines are pulled in
through the extras shown below (`pip install ryd-gate[<extra>]`).

| Backend (`ryd_gate.backends.*`) | `backend=` names | Extra | Representation | Practical scale |
|---|---|---|---|---|
| `exact` | `exact`, `sparse`, `sparse_expm` | — (core) | dense/sparse state vector | ≲ 14–16 qubits (2-level); 2 atoms × 7 levels |
| `tenpy_mps` | `tenpy`, `tn`, `mps` | `tn` | MPS (DMRG/TDVP) | 1D / quasi-1D chains, tens–hundreds of sites |
| `itensor` | `itensors` | (Julia ITensors) | MPS/TEBD via Julia | 1D chains |
| `ttn` | `ttn` | `tn-ttn` | tree tensor network (vendored PyTreeNet) | tree-structured / moderate 2D |
| `gputn` | `gputn`, `gpu` | `gputn-cu12` | CUDA / cuQuantum MPS | GPU-accelerated 1D/2D |
| `peps2d` / `tn_common` 2dtn | `2dtn`, `peps` | `tn-2d` | 2D PEPS / belief propagation | 2D lattices |
| `nqs` | `nqs` | `nqs` | neural quantum states (NetKet/jVMC) | 2D, variational |

`tn_common` holds the shared tensor-network IR (`TNCompiler`, `TNEvolutionIR`),
the lattice spec, and the `simulate_tn` dispatcher used by every TN backend;
`ryd_gate._vendor` holds vendored third-party kernels (PyTreeNet) — see
`src/ryd_gate/_vendor/NOTICE.md`.

## Core layout

| Module | Responsibility |
|---|---|
| `ryd_gate.core.system` | `RydbergSystem` (geometry + level structure + protocol) |
| `ryd_gate.core.level_structures` | level/transition/interaction specs and presets |
| `ryd_gate.core.rb87_params` | Rb87 seven-level physical parameter sets |
| `ryd_gate.core.local_blocks` | single-atom Hamiltonian matrix blocks |
| `ryd_gate.core.factories` | `from_lattice` construction |
| `ryd_gate.protocols` | pulse protocols (CZ gates, sweeps, TFIM dynamics) |
| `ryd_gate.ir` | `HamiltonianIR`, `EvolutionResult`, `compile_hamiltonian_ir` |
| `ryd_gate.analysis` | gate metrics, observables, coarsening/domain analysis |
