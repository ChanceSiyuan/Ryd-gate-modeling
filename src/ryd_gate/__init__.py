"""ryd_gate – Simulation and optimisation of Rydberg-atom entangling gates."""

__version__ = "0.1.0"

import warnings as _warnings

from .analysis.addressing_metrics import AddressingEvaluator
from .blackman import blackman_pulse, blackman_pulse_sqrt, blackman_window
from .core.atomic_system import (
    PROTOCOL_REGISTRY,
    AtomicSystem,
    LatticeSystem,
    compatible_protocols,
    compute_shift_scatter,
    create_analog_system,
    create_lattice_system,
    create_lukin_system,
    create_our_system,
)

# Import from legacy facade (suppress DeprecationWarning for this internal import)
with _warnings.catch_warnings():
    _warnings.simplefilter("ignore", DeprecationWarning)
    from .ideal_cz import CZGateSimulator, MonteCarloResult

from .protocols.base import Protocol
from .protocols.gate_cz_ar import ARProtocol
from .protocols.gate_cz_to import TOProtocol
from .protocols.sweep import SweepProtocol

# New architecture exports
from .compilers.ir import HamiltonianIR, HamiltonianTerm
from .core.basis import BasisSpec
from .core.blocks import BlockRegistry
from .core.observables import Observable, ObservableRegistry
from .core.system_model import SystemModel
from .solvers.base import EvolutionResult, SolverBackend
from .solvers.dispatch import simulate

__all__ = [
    # Legacy (backward-compatible)
    "CZGateSimulator",
    "MonteCarloResult",
    "AtomicSystem",
    "LatticeSystem",
    "create_our_system",
    "create_lukin_system",
    "create_analog_system",
    "create_lattice_system",
    "compatible_protocols",
    "PROTOCOL_REGISTRY",
    "compute_shift_scatter",
    "Protocol",
    "TOProtocol",
    "ARProtocol",
    "SweepProtocol",
    "AddressingEvaluator",
    "blackman_pulse",
    "blackman_pulse_sqrt",
    "blackman_window",
    # New architecture
    "SystemModel",
    "BasisSpec",
    "BlockRegistry",
    "ObservableRegistry",
    "Observable",
    "HamiltonianIR",
    "HamiltonianTerm",
    "SolverBackend",
    "EvolutionResult",
    "simulate",
]
