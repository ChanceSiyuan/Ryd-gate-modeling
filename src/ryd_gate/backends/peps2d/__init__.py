"""2D PEPS / belief-propagation external-solver boundary.

Thin boundary package: a 2D PEPS/BP run is dispatched through
``ryd_gate.backends.tn_common.simulate_tn(spec, protocol, x, backend="2dtn")``,
which builds the TN IR and hands it to :class:`External2DTNBPBackend`. That class
is re-exported here for discovery; shared TN IR/compiler types live in
``ryd_gate.backends.tn_common``.
"""

from ryd_gate.backends.tn_common.external_backends import External2DTNBPBackend

from .quimb_backend import Quimb2DTNBackend, Quimb2DTNError
from .yastn_backend import YASTN2DTNBackend, YASTN2DTNError

__all__ = [
    "External2DTNBPBackend",
    "Quimb2DTNBackend",
    "Quimb2DTNError",
    "YASTN2DTNBackend",
    "YASTN2DTNError",
]
