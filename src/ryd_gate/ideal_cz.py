"""Backward-compatible shim — imports from ryd_gate.legacy.ideal_cz.

.. deprecated::
    Import from ``ryd_gate.legacy.ideal_cz`` or use the modular subpackages directly.
"""

import warnings as _warnings

_warnings.warn(
    "ryd_gate.ideal_cz is deprecated; "
    "use ryd_gate.legacy.ideal_cz or import from subpackages directly.",
    DeprecationWarning,
    stacklevel=2,
)

from ryd_gate.legacy.ideal_cz import CZGateSimulator, MonteCarloResult  # noqa: F401,E402

__all__ = ["CZGateSimulator", "MonteCarloResult"]
