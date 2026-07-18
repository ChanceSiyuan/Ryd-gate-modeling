"""ryd_gate — Rydberg neutral-atom gate & many-body simulator.

Build a 2D :class:`Register`, obtain a built-in :func:`level_structure` preset,
construct one fully specified protocol, bind it in a :class:`RydbergSystem`, then
:func:`simulate` (or :func:`simulate_ensemble` with a :class:`NoiseModel`)::

    from ryd_gate import Register, RydbergSystem, level_structure, simulate
    from ryd_gate.protocols import SweepProtocol

    system = RydbergSystem(
        level_structure=level_structure("01r", ryd_level=70),
        register=Register.chain(4, spacing_um=5.0),
        protocol=SweepProtocol(t_gate_s=1e-6, omega_half_rad_s=..., detuning_rad_s=...),
    )
    result = simulate(system)                       # backend="exact_ode" | "mps" | "peps"

The top-level namespace is exactly the seven names below. Protocols live in
``ryd_gate.protocols``, forward physics helpers in ``ryd_gate.physics``, and the
returned result types in ``ryd_gate.results``. :func:`bilinear_control_model`
is the search-side export of a system's compiled bilinear form (ADR-0024).
"""

__version__ = "0.1.0"

from .bilinear import bilinear_control_model
from .core.level_structures import level_structure
from .core.system import RydbergSystem
from .lattice import Register
from .noise import NoiseModel, simulate_ensemble
from .simulate import simulate

__all__ = [
    "Register",
    "RydbergSystem",
    "NoiseModel",
    "bilinear_control_model",
    "level_structure",
    "simulate",
    "simulate_ensemble",
]
