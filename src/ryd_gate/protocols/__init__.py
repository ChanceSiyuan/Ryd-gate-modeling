"""Pulse protocols for Rydberg gate and lattice sweep simulations.

Contents
--------
- ``gate_cz_to`` — TOProtocol: time-optimal CZ gate (cosine phase, 6 params)
- ``gate_cz_ar`` — ARProtocol: amplitude-robust CZ gate (dual-sine phase, 8 params)
- ``gate_cz_double_arp`` — DoubleARPProtocol: double rapid-adiabatic-passage CZ gate
- ``sweep``      — SweepProtocol: function-defined global Rydberg sweep
- ``lattice_dynamics`` — TFIMQuenchProtocol / TFIMAnnealProtocol for 2D lattice dynamics
- ``base``       — Protocol abstract base class
- ``channels``   — Drive channel name constants (e.g. ``"drive_420"``, ``"global_n"``)

All protocol classes implement:
- ``n_params``: number of real-valued parameters
- ``unpack_params(x, system)``: convert parameter vector to named dict
- ``get_drive_coefficients(t, params)``: time-dependent Hamiltonian coefficients
"""

from ryd_gate.protocols.base import Protocol
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol
from ryd_gate.protocols.gate_cz_ar import ARProtocol
from ryd_gate.protocols.gate_cz_double_arp import DoubleARPProtocol
from ryd_gate.protocols.gate_cz_to import TOProtocol
from ryd_gate.protocols.lattice_dynamics import (
    TFIMAnnealProtocol,
    TFIMQuenchProtocol,
    TFIMRydbergControls,
    interaction_longitudinal_shifts,
    tfim_to_rydberg_controls,
)
from ryd_gate.protocols.sequence_protocol import SequenceProtocol, compile_sequence_to_system
from ryd_gate.protocols.sweep import SweepProtocol

__all__ = [
    "Protocol",
    "TOProtocol",
    "ARProtocol",
    "DoubleARPProtocol",
    "SweepProtocol",
    "SequenceProtocol",
    "compile_sequence_to_system",
    "TFIMAnnealProtocol",
    "TFIMQuenchProtocol",
    "TFIMRydbergControls",
    "tfim_to_rydberg_controls",
    "interaction_longitudinal_shifts",
    "DigitalAnalogProtocol",
]
