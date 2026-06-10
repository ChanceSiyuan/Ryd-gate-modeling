"""ryd_gate — Rydberg atom systems, pulse protocols, and Hamiltonian IR.

Typical workflow
----------------
1. **Choose a pulse protocol**::

       protocol = TOProtocol()               # time-optimal CZ gate (6 params)
       protocol = ARProtocol()               # amplitude-robust CZ gate (8 params)
       protocol = DoubleARPProtocol(...)      # double-ARP CZ gate
       protocol = SweepProtocol(...)         # function-defined Rydberg sweep
       protocol = TFIMQuenchProtocol(...)     # 2D TFIM / g-r lattice quench

2. **Create a quantum system with the protocol bound**::

       from ryd_gate.lattice import make_chain

       system = RydbergSystem.from_lattice(make_chain(4), "01r", protocol=protocol)

3. **Choose an algorithm package**::

       from ryd_gate.backends.exact import simulate              # exact state-vector
       from ryd_gate.backends.tn_common import simulate_tn       # tensor-network dispatch

Subpackages
-----------
- ``ryd_gate.core``      — Symbolic Rydberg systems, blocks, observables
- ``ryd_gate.protocols`` — Pulse protocols: TOProtocol, ARProtocol, SweepProtocol
- ``ryd_gate.ir``        — Unified Hamiltonian and result representations
- ``ryd_gate.lattice``   — N-atom lattice: geometry, operators, states, evolution
- ``ryd_gate.analysis``  — Post-processing: gate metrics, observables, domain analysis
"""

__version__ = "0.1.0"

# --- Systems ---
from .core.basis import BasisSpec
from .core.blocks import BlockRegistry
from .core.level_structures import (
    DEFAULT_C6,
    InteractionSpec,
    LevelStructureSpec,
    TransitionSpec,
    level_structure,
)
from .core.observables import Observable, ObservableRegistry
from .core.system import RydbergSystem

# --- Advanced / new-arch primitives ---
from .core.system_model import SystemModel
from .ir import EvolutionResult, HamiltonianIR, HamiltonianTerm, compile_hamiltonian_ir

# --- Protocols ---
from .protocols.base import Protocol
from .protocols.digital_analog import DigitalAnalogProtocol
from .protocols.gate_cz_ar import ARProtocol
from .protocols.gate_cz_double_arp import DoubleARPProtocol
from .protocols.gate_cz_to import TOProtocol
from .protocols.lattice_dynamics import (
    TFIMAnnealProtocol,
    TFIMQuenchProtocol,
    TFIMRydbergControls,
    interaction_longitudinal_shifts,
    tfim_to_rydberg_controls,
)
from .protocols.sweep import SweepProtocol

# --- Analysis (convenience re-exports) ---
# --- Pulse utilities ---
from .pulse import blackman_pulse, blackman_pulse_sqrt, blackman_window

# --- Unified simulation entry point ---
from .simulate import simulate


def __getattr__(name: str):
    """Lazy exports for optional/heavy physics helpers."""
    if name == "compute_shift_scatter":
        from .physics.ac_stark import compute_shift_scatter

        return compute_shift_scatter
    if name in {"average_gate_infidelity", "error_budget"}:
        from .analysis import gate_metrics

        return getattr(gate_metrics, name)
    if name == "AddressingEvaluator":
        from .analysis.addressing_metrics import AddressingEvaluator

        return AddressingEvaluator
    raise AttributeError(f"module 'ryd_gate' has no attribute {name!r}")


__all__ = [
    # Systems
    "RydbergSystem",
    "LevelStructureSpec",
    "TransitionSpec",
    "InteractionSpec",
    "DEFAULT_C6",
    "level_structure",
    "compute_shift_scatter",
    # Protocols
    "Protocol",
    "TOProtocol",
    "ARProtocol",
    "DoubleARPProtocol",
    "SweepProtocol",
    "TFIMAnnealProtocol",
    "TFIMQuenchProtocol",
    "TFIMRydbergControls",
    "tfim_to_rydberg_controls",
    "interaction_longitudinal_shifts",
    "DigitalAnalogProtocol",
    # IR
    "EvolutionResult",
    "HamiltonianIR",
    "HamiltonianTerm",
    "compile_hamiltonian_ir",
    # Analysis
    "average_gate_infidelity",
    "error_budget",
    "AddressingEvaluator",
    # Simulation
    "simulate",
    # Pulse utilities
    "blackman_pulse",
    "blackman_pulse_sqrt",
    "blackman_window",
    # Advanced primitives
    "SystemModel",
    "BasisSpec",
    "BlockRegistry",
    "ObservableRegistry",
    "Observable",
]
