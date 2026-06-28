"""Pulse protocols for Rydberg gate and lattice sweep simulations.

Contents
--------
- ``gate_cz``   — CZ gate protocols: CZProtocol (the laser-domain container),
  TOProtocol / ARProtocol builders, and the ``phase_from_chirp`` pulse helper
- ``sweep``      — SweepProtocol: function-defined global Rydberg sweep
- ``lattice_dynamics`` — TFIMQuenchProtocol / TFIMAnnealProtocol for 2D lattice dynamics
- ``digital_analog`` — DigitalAnalogProtocol: piecewise digital-analog schedules
- ``base``       — Protocol abstract base class

All protocol classes implement:
- ``n_params``: number of real-valued parameters
- ``unpack_params(x, system)``: convert parameter vector to named dict
- ``get_drive_coefficients(t, params)``: time-dependent Hamiltonian coefficients
"""

from ryd_gate.protocols.base import Protocol
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol
from ryd_gate.protocols.gate_cz import (
    ARProtocol,
    CZProtocol,
    EffectiveCZProtocol,
    TOProtocol,
    phase_from_chirp,
)
from ryd_gate.protocols.lattice_dynamics import (
    TFIMAnnealProtocol,
    TFIMQuenchProtocol,
    TFIMRydbergControls,
    interaction_longitudinal_shifts,
    tfim_to_rydberg_controls,
)
from ryd_gate.protocols.sweep import SweepProtocol

__all__ = [
    "Protocol",
    "TOProtocol",
    "ARProtocol",
    "CZProtocol",
    "EffectiveCZProtocol",
    "phase_from_chirp",
    "SweepProtocol",
    "TFIMAnnealProtocol",
    "TFIMQuenchProtocol",
    "TFIMRydbergControls",
    "tfim_to_rydberg_controls",
    "interaction_longitudinal_shifts",
    "DigitalAnalogProtocol",
]
