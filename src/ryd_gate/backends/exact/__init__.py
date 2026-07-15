"""Exact state-vector backend: adaptive DOP853 ODE integration (E03)."""

from .compiler import compile_exact
from .simulate import simulate, simulate_states

__all__ = ["compile_exact", "simulate", "simulate_states"]
