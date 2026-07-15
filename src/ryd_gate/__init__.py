"""ryd_gate — Rydberg neutral-atom many-body simulator.

TFIM / quench / lattice dynamics and gate physics on one kernel:
continuous-time pulse protocols are the single control surface, lowered to a
unified Hamiltonian IR and evolved by exact state-vector or tensor-network
backends.

Typical workflow
----------------
1. **Choose a pulse protocol**::

       from ryd_gate import TFIMQuenchProtocol, SweepProtocol
       from ryd_gate.protocols import TOProtocol, ARProtocol, CZProtocol

       protocol = TFIMQuenchProtocol(...)     # 2D TFIM / g-r lattice quench
       protocol = SweepProtocol(...)          # function-defined Rydberg sweep
       protocol = TOProtocol(...)             # time-optimal CZ pulse (normalized fields)
       protocol = ARProtocol(...)             # amplitude-robust CZ pulse
       protocol = CZProtocol(...)             # direct 420/1013 laser pulse (e.g. adiabatic)

2. **Create a quantum system with the protocol bound**::

       system = RydbergSystem(
           level_structure=level_structure("01r"),
           register=Register.chain(4),
           protocol=protocol,
       )

3. **Simulate**::

       result = simulate(system)   # exact_ode; or backend="mps" / "peps"

Public API
----------
The top-level namespace stays small. Specialized surfaces live in submodules:

- ``ryd_gate.protocols`` — the full protocol collection (incl. ``DigitalAnalogProtocol``)
- ``ryd_gate.ir``        — Hamiltonian IR (``HamiltonianIR``, ``compile_hamiltonian_ir``) and result containers
- ``ryd_gate.core``      — symbolic systems, blocks, observable expressions
- ``ryd_gate.backends``  — exact state-vector + MPS/PEPS engines
"""

__version__ = "0.1.0"

# --- Systems & geometry ---
from .core.level_structures import (
    InteractionSpec,
    level_structure,
)
from .core.system import RydbergSystem
from .ir import EvolutionResult
from .lattice import Register

# --- Noise layer ---
from .noise import EnsembleResult, NoiseModel, simulate_ensemble

# --- Protocols (most common; full collection in ryd_gate.protocols) ---
from .protocols.lattice_dynamics import TFIMAnnealProtocol, TFIMQuenchProtocol
from .protocols.sweep import SweepProtocol

# --- Unified simulation entry point ---
from .simulate import simulate

__all__ = [
    "Register",
    "RydbergSystem",
    "InteractionSpec",
    "level_structure",
    "NoiseModel",
    "EvolutionResult",
    "EnsembleResult",
    "simulate",
    "simulate_ensemble",
    "SweepProtocol",
    "TFIMQuenchProtocol",
    "TFIMAnnealProtocol",
]
