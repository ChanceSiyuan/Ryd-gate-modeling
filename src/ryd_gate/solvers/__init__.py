"""Simulation engine: ODE / matrix-exponential backends and Monte Carlo runner.

Contents
--------
- ``dispatch``          — ``simulate()``: high-level entry point
- ``base``              — SolverBackend ABC, EvolutionResult dataclass
- ``dense_ode``         — DenseODEBackend: adaptive ODE via scipy DOP853
- ``sparse_expm``       — SparseExpmBackend: piecewise-constant matrix exponential
- ``ir``                — HamiltonianIR, HamiltonianTerm: solver-agnostic IR
- ``monte_carlo_runner``— MonteCarloRunner: noise sampling via system + backend
"""
