"""Neural quantum state (tVMC) external-solver boundary.

Thin boundary package: an NQS run is dispatched through
``ryd_gate.backends.tn_common.simulate_tn(spec, protocol, x, backend="nqs")``,
which builds the TN IR and hands it to :class:`ExternalNQSTVMCBackend`. That
class is re-exported here for discovery; shared TN IR/compiler types live in
``ryd_gate.backends.tn_common``.
"""

from ryd_gate.backends.tn_common.external_backends import ExternalNQSTVMCBackend

__all__ = ["ExternalNQSTVMCBackend"]
