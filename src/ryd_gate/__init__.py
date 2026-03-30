"""ryd_gate – Simulation and optimisation of Rydberg-atom entangling gates."""

__version__ = "0.1.0"

from .analysis.addressing_metrics import AddressingEvaluator
from .blackman import blackman_pulse, blackman_pulse_sqrt, blackman_window
from .core.atomic_system import (
    PROTOCOL_REGISTRY,
    AtomicSystem,
    compatible_protocols,
    compute_shift_scatter,
    create_analog_system,
    create_lukin_system,
    create_our_system,
)
from .ideal_cz import CZGateSimulator, MonteCarloResult
from .protocols.base import Protocol
from .protocols.gate_cz_ar import ARProtocol
from .protocols.gate_cz_to import TOProtocol
from .protocols.local_sweep import SweepAddressingProtocol

__all__ = [
    "CZGateSimulator",
    "MonteCarloResult",
    "AtomicSystem",
    "create_our_system",
    "create_lukin_system",
    "create_analog_system",
    "compatible_protocols",
    "PROTOCOL_REGISTRY",
    "compute_shift_scatter",
    "Protocol",

    "TOProtocol",
    "ARProtocol",
    "SweepAddressingProtocol",
    "AddressingEvaluator",
    "blackman_pulse",
    "blackman_pulse_sqrt",
    "blackman_window",
]
