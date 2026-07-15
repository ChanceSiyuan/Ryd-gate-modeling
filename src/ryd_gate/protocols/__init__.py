"""Pulse protocols for Rydberg gate and lattice sweep simulations.

Contents
--------
- ``gate_cz``   — CZ gate protocols: CZProtocol (the laser-domain container)
  and the TOProtocol / ARProtocol phase families
- ``direct_297`` — 297 nm single-photon σ⁻ protocols: Direct297PiProtocol
  (π-pulse), Direct297CZProtocol (CZ pulse container), Direct297TOProtocol
  (Time-Optimal phase family)
- ``sweep``      — SweepProtocol: function-defined global Rydberg sweep
- ``lattice_dynamics`` — TFIMQuenchProtocol / TFIMAnnealProtocol for 2D lattice dynamics
- ``digital_analog`` — DigitalAnalogProtocol: the full local 3x3 ``{0,1,r}``
  Hamiltonian as five independent control functions (matrix-element convention)
- ``base``       — Protocol abstract base class

This package exports exactly the user-selectable protocol classes.  The
``Protocol`` ABC and pulse-construction helpers stay in their defining
modules (``ryd_gate.protocols.base.Protocol``,
``ryd_gate.protocols.gate_cz.phase_from_chirp``,
``ryd_gate.protocols.lattice_dynamics.tfim_to_rydberg_controls`` /
``interaction_longitudinal_shifts``).

Every protocol is fully specified at construction; binding it to a system
resolves its normalized quantities against that system's physical scales
(the resolved duration is ``system.t_gate``).
"""

from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol
from ryd_gate.protocols.direct_297 import (
    Direct297CZProtocol,
    Direct297PiProtocol,
    Direct297TOProtocol,
)
from ryd_gate.protocols.gate_cz import (
    ARProtocol,
    CZProtocol,
    TOProtocol,
)
from ryd_gate.protocols.lattice_dynamics import (
    TFIMAnnealProtocol,
    TFIMQuenchProtocol,
)
from ryd_gate.protocols.sweep import SweepProtocol

__all__ = [
    "CZProtocol",
    "TOProtocol",
    "ARProtocol",
    "Direct297PiProtocol",
    "Direct297CZProtocol",
    "Direct297TOProtocol",
    "SweepProtocol",
    "TFIMQuenchProtocol",
    "TFIMAnnealProtocol",
    "DigitalAnalogProtocol",
]
