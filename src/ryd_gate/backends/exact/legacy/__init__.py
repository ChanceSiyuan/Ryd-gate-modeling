"""Historical CZ and Monte Carlo implementations.

New code should import from the modular subpackages directly. ``solve_gate`` is
re-exported as the public entry for evolving a legacy ``AtomicSystem`` so callers
need not reach into private modules.
"""

from ._solve_gate import solve_gate

__all__ = ["solve_gate"]
