"""Simulation backends for ryd_gate.

Each subpackage lowers the unified Hamiltonian IR into a specific solver:

- ``exact``      — exact state-vector evolution (sparse expm / dense ODE)
- ``tenpy_mps``  — TeNPy MPS DMRG/TDVP
- ``tn_common``  — shared tensor-network IR, lattice spec, and ``simulate_tn`` dispatch
- ``peps2d``     — YASTN finite PEPS

The two entry points are ``exact.simulate`` and ``tn_common.simulate_tn``; the
convenience wrapper :func:`ryd_gate.simulate` dispatches across them by name.
"""
