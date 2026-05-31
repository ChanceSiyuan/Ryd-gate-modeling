"""ryd_gate — Rydberg-atom quantum gate and lattice simulation toolkit.

Typical workflow
----------------
1. **Create a quantum system**::

       system = RydbergSystemModel.from_preset("01r", N=4)

2. **Choose a pulse protocol**::

       protocol = TOProtocol()               # time-optimal CZ gate (6 params)
       protocol = ARProtocol()               # amplitude-robust CZ gate (8 params)
       protocol = SweepProtocol()            # adiabatic detuning sweep (lattice)

3. **Run the simulation**::

       result = simulate(system, protocol, params, psi0)

4. **Analyze results**::

       from ryd_gate.analysis import average_gate_infidelity, error_budget
       infidelity = average_gate_infidelity(system, protocol, params)

Subpackages
-----------
- ``ryd_gate.core``      — Rydberg system models, Hamiltonian blocks, operators
- ``ryd_gate.protocols`` — Pulse protocols: TOProtocol, ARProtocol, SweepProtocol
- ``ryd_gate.solvers``   — Simulation engine: ODE backends, Monte Carlo, Hamiltonian IR
- ``ryd_gate.lattice``   — N-atom lattice: geometry, operators, states, evolution
- ``ryd_gate.analysis``  — Post-processing: gate metrics, observables, domain analysis
- ``ryd_gate.tn``        — Optional tensor-network path (requires tenpy)
- ``ryd_gate.legacy``    — Backward-compatible ``CZGateSimulator`` facade (deprecated)
"""

__version__ = "0.1.0"

import warnings as _warnings

# --- Systems ---
from .physics.ac_stark import compute_shift_scatter
from .core.rydberg_system import (
    DEFAULT_C6,
    InteractionSpec,
    LevelStructureSpec,
    RydbergSystem,
    RydbergSystemModel,
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
from .solvers.dispatch import simulate
from .solvers.base import EvolutionResult, SolverBackend
from .solvers.ir import HamiltonianIR, HamiltonianTerm

# --- Analysis (convenience re-exports) ---
from .analysis.gate_metrics import average_gate_infidelity, error_budget
from .analysis.addressing_metrics import AddressingEvaluator

# --- Pulse utilities ---
from .pulse import blackman_pulse, blackman_pulse_sqrt, blackman_window

# --- Advanced / new-arch primitives ---
from .core.system_model import SystemModel
from .core.basis import BasisSpec
from .core.blocks import BlockRegistry
from .core.observables import Observable, ObservableRegistry

# --- Backward-compatible legacy facade (deprecated) ---
with _warnings.catch_warnings():
    _warnings.simplefilter("ignore", DeprecationWarning)
    from .legacy.ideal_cz import CZGateSimulator, MonteCarloResult

__all__ = [
    # Systems
    "RydbergSystem",
    "RydbergSystemModel",
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
    # Legacy (deprecated)
    "CZGateSimulator",
    "MonteCarloResult",
]
