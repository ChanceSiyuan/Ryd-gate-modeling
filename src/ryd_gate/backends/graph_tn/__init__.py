"""Private quimb graph-PEPS backend for ARBITRARY 2D atom geometry.

Importing this package imports only the dependency-free contract layer
(:mod:`_options`, :mod:`_graph`); it never imports quimb or :mod:`_engine`. The
dispatcher runs every dependency-free preflight through the re-exports below, then
calls a lazy wrapper that imports :mod:`_engine`, whose ``_load_quimb`` performs
the actual third-party import. Nothing here is public.

Unlike the YASTN ``peps`` backend (Cartesian chain/rectangle/square only), this
backend accepts ANY register — triangular, random coordinates, grids — and
contracts observables on the arbitrary interaction graph via exact / cluster /
belief-propagation environments.
"""

from __future__ import annotations

from ryd_gate.backends.graph_tn._graph import build_graph
from ryd_gate.backends.graph_tn._options import (
    GraphTNError,
    validate_ground_options,
    validate_real_time_options,
)

__all__ = [
    "GraphTNError",
    "build_graph",
    "validate_real_time_options",
    "validate_ground_options",
    "evolve_graph_tn",
    "solve_graph_tn_ground",
]


def evolve_graph_tn(*args, **kwargs):
    """Lazy wrapper: imports :mod:`_engine` (and thus quimb) only when called."""
    from ryd_gate.backends.graph_tn._engine import evolve_graph_tn as _impl

    return _impl(*args, **kwargs)


def solve_graph_tn_ground(*args, **kwargs):
    """Lazy wrapper: imports :mod:`_engine` (and thus quimb) only when called."""
    from ryd_gate.backends.graph_tn._engine import solve_graph_tn_ground as _impl

    return _impl(*args, **kwargs)
