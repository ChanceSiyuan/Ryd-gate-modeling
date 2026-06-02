"""ryd_gate — Rydberg-atom quantum gate and lattice simulation toolkit.

Typical workflow
----------------
1. **Choose a pulse protocol**::

       protocol = TOProtocol()               # time-optimal CZ gate (6 params)
       protocol = ARProtocol()               # amplitude-robust CZ gate (8 params)
       protocol = SweepProtocol()            # adiabatic detuning sweep (lattice)

2. **Create a quantum system with the protocol bound**::

       system = RydbergSystem.from_preset("01r", N=4, protocol=protocol)

3. **Run the simulation**::

       result = simulate(system, params, psi0)

4. **Analyze results**::

       final_norm = abs(result.psi_final.conj() @ result.psi_final)

Subpackages
-----------
- ``ryd_gate.model``     — Symbolic Rydberg systems, blocks, observables
- ``ryd_gate.protocols`` — Pulse protocols: TOProtocol, ARProtocol, SweepProtocol
- ``ryd_gate.compilers`` — Backend-specific compilers from symbolic systems to IR
- ``ryd_gate.ir``        — Intermediate representations
- ``ryd_gate.backends``  — ODE, sparse-expm, and future TN backends
- ``ryd_gate.simulate``  — High-level compile + evolve dispatch
- ``ryd_gate.lattice``   — N-atom lattice: geometry, operators, states, evolution
- ``ryd_gate.analysis``  — Post-processing: gate metrics, observables, domain analysis
- ``ryd_gate.tn``        — Optional tensor-network path (requires tenpy)
- ``ryd_gate.legacy``    — Historical CZ and Monte-Carlo implementations
"""

__version__ = "0.1.0"

# --- Systems ---
from .model.system import (
    DEFAULT_C6,
    InteractionSpec,
    LevelStructureSpec,
    RydbergSystem,
    TransitionSpec,
    level_structure,
)

# --- Protocols ---
from .protocols.base import Protocol
from .protocols.gate_cz_ar import ARProtocol
from .protocols.gate_cz_to import TOProtocol
from .protocols.sweep import SweepProtocol
from .protocols.digital_analog import DigitalAnalogProtocol, Segment

# --- Simulation ---
from .simulate import simulate
from .backends import EvolutionResult, SolverBackend
from .ir import HamiltonianIR, HamiltonianTerm
from .compilers import ExactSparseCompiler, compile_expm_ir

# --- Analysis (convenience re-exports) ---
# --- Pulse utilities ---
from .pulse import blackman_pulse, blackman_pulse_sqrt, blackman_window

# --- Advanced / new-arch primitives ---
from .model.system_model import SystemModel
from .model.basis import BasisSpec
from .model.blocks import BlockRegistry
from .model.observables import Observable, ObservableRegistry


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
    "SweepProtocol",
    "DigitalAnalogProtocol",
    "Segment",
    # Simulation
    "simulate",
    "EvolutionResult",
    "SolverBackend",
    "HamiltonianIR",
    "HamiltonianTerm",
    "ExactSparseCompiler",
    "compile_expm_ir",
    # Analysis
    "average_gate_infidelity",
    "error_budget",
    "AddressingEvaluator",
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
