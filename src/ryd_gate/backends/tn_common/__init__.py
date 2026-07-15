"""Shared tensor-network lowering and dispatch (private).

The one public seam is :func:`ryd_gate.backends.tn_common.simulate.simulate_tn`,
reached through :func:`ryd_gate.simulate`.  Concrete kernels live in sibling
packages: ``tenpy_mps`` (MPS) and ``peps`` (PEPS).
"""

from .simulate import simulate_tn

__all__ = ["simulate_tn"]
