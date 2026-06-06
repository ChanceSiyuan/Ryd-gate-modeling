"""Neural quantum state (tVMC) backends.

An NQS run is dispatched through
``ryd_gate.backends.tn_common.simulate_tn(spec, protocol, x, backend="nqs")``,
which builds the TN IR and hands it to :class:`ExternalNQSTVMCBackend`.  The
default package target is NetKet, implemented by :class:`NetKetNQSTVMCBackend`.
Shared TN IR/compiler types live in ``ryd_gate.backends.tn_common``.
"""

from ryd_gate.backends.tn_common.external_backends import ExternalNQSTVMCBackend

from .netket_backend import NetKetNQSError, NetKetNQSTVMCBackend

__all__ = ["ExternalNQSTVMCBackend", "NetKetNQSError", "NetKetNQSTVMCBackend"]
