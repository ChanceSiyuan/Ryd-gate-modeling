"""Private YASTN finite-PEPS backend package.

Importing this package imports only the dependency-free contract layer
(:mod:`_options`, :mod:`_layout`, :mod:`_numerics`); it never imports YASTN or
:mod:`_engine`. The dispatcher runs every dependency-free preflight through the
re-exports below, then calls a lazy wrapper that imports :mod:`_engine`, whose
``_load_yastn`` performs the actual third-party import. Nothing here is public.
"""

from __future__ import annotations

from ryd_gate.backends.peps._layout import peps_lattice_spec, validate_and_map_pairs
from ryd_gate.backends.peps._options import validate_ground_options, validate_real_time_options

__all__ = [
    "peps_lattice_spec",
    "validate_and_map_pairs",
    "validate_real_time_options",
    "validate_ground_options",
    "evolve_peps",
    "solve_peps_ground_state",
]


def evolve_peps(*args, **kwargs):
    """Lazy wrapper: imports :mod:`_engine` (and thus YASTN) only when called."""
    from ryd_gate.backends.peps._engine import evolve_peps as _impl

    return _impl(*args, **kwargs)


def solve_peps_ground_state(*args, **kwargs):
    """Lazy wrapper: imports :mod:`_engine` (and thus YASTN) only when called."""
    from ryd_gate.backends.peps._engine import solve_peps_ground_state as _impl

    return _impl(*args, **kwargs)
