"""ryd_gate – Simulation and optimisation of Rydberg-atom entangling gates."""

__version__ = "0.1.0"

from .analysis.addressing_metrics import AddressingEvaluator
from .blackman import blackman_pulse, blackman_pulse_sqrt, blackman_window
from .core.atomic_system import AtomicSystem
from .ideal_cz import CZGateSimulator, MonteCarloResult
from .protocols.base import Protocol, SweepProtocol
from .protocols.gate_cz_ar import ARProtocol
from .protocols.gate_cz_to import TOProtocol
from .protocols.local_sweep import SweepAddressingProtocol

__all__ = [
    "CZGateSimulator",
    "MonteCarloResult",
    "AtomicSystem",
    "Protocol",
    "SweepProtocol",
    "TOProtocol",
    "ARProtocol",
    "SweepAddressingProtocol",
    "AddressingEvaluator",
    "blackman_pulse",
    "blackman_pulse_sqrt",
    "blackman_window",
]
