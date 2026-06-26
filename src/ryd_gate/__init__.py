"""ryd_gate — Rydberg neutral-atom many-body simulator.

TFIM / quench / lattice dynamics and gate physics on one kernel:
continuous-time pulse protocols are the single control surface, lowered to a
unified Hamiltonian IR and evolved by exact state-vector or tensor-network
backends.

Typical workflow
----------------
1. **Choose a pulse protocol**::

       from ryd_gate import TFIMQuenchProtocol, SweepProtocol
       from ryd_gate.gates import TOProtocol, ARProtocol, DoubleARPProtocol

       protocol = TFIMQuenchProtocol(...)     # 2D TFIM / g-r lattice quench
       protocol = SweepProtocol(...)         # function-defined Rydberg sweep
       protocol = TOProtocol()               # time-optimal CZ gate (6 params)
       protocol = ARProtocol()               # amplitude-robust CZ gate (8 params)
       protocol = DoubleARPProtocol(...)      # double-ARP CZ gate

2. **Create a quantum system with the protocol bound**::

       system = (RydbergSystem.set_atom_level("01r")
                 .set_atom_geom(Register.chain(4)).set_protocol(protocol))

3. **Simulate**::

       result = simulate(system, x, backend="exact_dense")   # or "exact_sparse" / "mps" / ...

Public API
----------
The top-level namespace stays small. Specialized surfaces live in submodules:

- ``ryd_gate.gates``     — CZ gate protocols + ``cz_gate_report`` / ``CZGateReport``
- ``ryd_gate.protocols`` — the full protocol collection (incl. ``Protocol``, ``DigitalAnalogProtocol``)
- ``ryd_gate.analysis``  — gate metrics (``average_gate_infidelity``, ``error_budget``), observables, domain analysis
- ``ryd_gate.ir``        — Hamiltonian IR (``HamiltonianIR``, ``compile_hamiltonian_ir``) and result containers
- ``ryd_gate.core``      — symbolic systems, blocks, observable registries
- ``ryd_gate.backends``  — exact state-vector + MPS/PEPS engines
"""

__version__ = "0.1.0"

# --- Systems & geometry ---
from .core.level_structures import (
    DEFAULT_C6,
    InteractionSpec,
    LevelStructureSpec,
    level_structure,
)
from .core.system import RydbergSystem
from .ir import EvolutionResult
from .lattice import Register, RegisterLayout

# --- Noise layer ---
from .noise import NoiseModel, configure_monte_carlo_runner

# --- Protocols (most common; full collection in ryd_gate.protocols) ---
from .protocols.lattice_dynamics import TFIMAnnealProtocol, TFIMQuenchProtocol
from .protocols.sweep import SweepProtocol

# --- Unified simulation entry point ---
from .simulate import simulate

__all__ = [
    # Systems & geometry
    "RydbergSystem",
    "Register",
    "RegisterLayout",
    "InteractionSpec",
    "LevelStructureSpec",
    "level_structure",
    "DEFAULT_C6",
    # Noise layer
    "NoiseModel",
    "configure_monte_carlo_runner",
    # Lattice-dynamics protocols
    "SweepProtocol",
    "TFIMQuenchProtocol",
    "TFIMAnnealProtocol",
    # Simulation
    "simulate",
    "EvolutionResult",
]
