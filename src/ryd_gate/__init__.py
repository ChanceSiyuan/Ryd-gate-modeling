"""ryd_gate — Rydberg neutral-atom many-body simulator.

TFIM / quench / lattice dynamics and gate physics on one kernel:
continuous-time pulse protocols are the single control surface, lowered to a
unified Hamiltonian IR and evolved by exact state-vector or tensor-network
backends.

Typical workflow
----------------
1. **Choose a pulse protocol**::

       protocol = TFIMQuenchProtocol(...)     # 2D TFIM / g-r lattice quench
       protocol = SweepProtocol(...)         # function-defined Rydberg sweep
       protocol = TOProtocol()               # time-optimal CZ gate (6 params)
       protocol = ARProtocol()               # amplitude-robust CZ gate (8 params)
       protocol = DoubleARPProtocol(...)      # double-ARP CZ gate

2. **Create a quantum system with the protocol bound**::

       system = RydbergSystem.from_lattice(Register.chain(4), "01r", protocol=protocol)

3. **Simulate**::

       result = simulate(system, x, backend="exact")   # or "mps" / "peps" / "gputn"

Subpackages
-----------
- ``ryd_gate.core``      — Symbolic Rydberg systems, blocks, observables
- ``ryd_gate.protocols`` — Pulse protocols: TFIMQuenchProtocol, SweepProtocol, gate CZ protocols
- ``ryd_gate.ir``        — Unified Hamiltonian and result representations
- ``ryd_gate.lattice``   — N-atom lattice geometry (Register)
- ``ryd_gate.backends``  — exact state-vector + MPS/PEPS/gputn engines
- ``ryd_gate.analysis``  — Post-processing: gate metrics, observables, domain analysis
"""

__version__ = "0.1.0"

# --- Systems ---
from .core.level_structures import (
    DEFAULT_C6,
    InteractionSpec,
    LevelStructureSpec,
    TransitionSpec,
    level_structure,
)
from .core.model import (
    BasisSpec,
    BlockRegistry,
    Observable,
    ObservableRegistry,
    SystemModel,
)
from .core.serialization import ValidationIssue, raise_for_errors
from .core.system import RydbergSystem
from .ir import EvolutionResult, HamiltonianIR, HamiltonianTerm, compile_hamiltonian_ir
from .lattice import Register, RegisterLayout

# --- Noise layer ---
from .noise import NoiseModel, configure_monte_carlo_runner

# --- Protocols ---
from .protocols.base import Protocol
from .protocols.digital_analog import DigitalAnalogProtocol
from .protocols.gate_cz import ARProtocol, DoubleARPProtocol, TOProtocol
from .protocols.lattice_dynamics import (
    TFIMAnnealProtocol,
    TFIMQuenchProtocol,
    TFIMRydbergControls,
    interaction_longitudinal_shifts,
    tfim_to_rydberg_controls,
)
from .protocols.sweep import SweepProtocol

# --- Unified simulation entry point ---
from .simulate import simulate


def __getattr__(name: str):
    """Lazy exports for optional/heavy physics helpers."""
    if name == "compute_shift_scatter":
        from .physics import compute_shift_scatter

        return compute_shift_scatter
    if name in {"average_gate_infidelity", "error_budget"}:
        from .analysis import gate_metrics

        return getattr(gate_metrics, name)
    if name in {"CZGateReport", "cz_gate_report"}:
        from . import gates

        return getattr(gates, name)
    if name == "AddressingEvaluator":
        from .analysis.addressing import AddressingEvaluator

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
    # Lattice geometry
    "Register",
    "RegisterLayout",
    "ValidationIssue",
    "raise_for_errors",
    # Noise layer
    "NoiseModel",
    "configure_monte_carlo_runner",
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
    # Gate library
    "CZGateReport",
    "cz_gate_report",
    # Simulation
    "simulate",
    # Advanced primitives
    "SystemModel",
    "BasisSpec",
    "BlockRegistry",
    "ObservableRegistry",
    "Observable",
]
