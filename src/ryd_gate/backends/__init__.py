"""Simulation backends for ryd_gate.

Each subpackage lowers the unified Hamiltonian IR into a specific solver:

- ``exact``      — exact state-vector evolution (sparse expm / dense ODE)
- ``tenpy_mps``  — TeNPy MPS DMRG/TDVP
- ``tn_common``  — shared tensor-network IR, lattice spec, and ``simulate_tn`` dispatch
- ``itensor``    — ITensors / ITensorNetworks / TNQS Julia bridges
- ``ttn``        — tree tensor-network TDVP (vendored pytreenet)
- ``gputn``      — CUDA / cuQuantum tensor-network
- ``gputtn``     — public alias for ITensorNetworks.jl GPU TTN-TDVP
- ``nqs``        — neural quantum states (NetKet / jVMC)
- ``peps2d``     — 2D PEPS / belief propagation

The two entry points are ``exact.simulate`` and ``tn_common.simulate_tn``; the
convenience wrapper :func:`ryd_gate.simulate` dispatches across them by name.
"""
