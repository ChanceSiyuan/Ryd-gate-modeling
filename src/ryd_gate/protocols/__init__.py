"""Pulse protocols for Rydberg gate and lattice sweep simulations.

Exactly the eight user-selectable protocol classes plus the two pulse
construction helpers are public (P32). The ``Protocol`` ABC, the private
resolved-drive IR, and every internal pulse builder stay in their defining
modules.

Every protocol is fully specified at construction; binding it to a system
resolves its physical scales through the single private seam
``protocol._resolve(system)`` and exposes the duration as ``system.t_gate``.
"""

from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol
from ryd_gate.protocols.direct_297 import (
    Direct297CZProtocol,
    Direct297PiProtocol,
    Direct297TOProtocol,
)
from ryd_gate.protocols.gate_cz import ARProtocol, CZProtocol, TOProtocol
from ryd_gate.protocols.pulses import blackman_pulse, phase_from_chirp
from ryd_gate.protocols.sweep import SweepProtocol

__all__ = [
    "CZProtocol",
    "TOProtocol",
    "ARProtocol",
    "SweepProtocol",
    "DigitalAnalogProtocol",
    "Direct297PiProtocol",
    "Direct297CZProtocol",
    "Direct297TOProtocol",
    "blackman_pulse",
    "phase_from_chirp",
]
