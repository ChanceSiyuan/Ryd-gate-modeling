"""Self-written finite-PEPS engine for 2D Rydberg lattice dynamics.

A dense, fully-controlled replacement for the YASTN ``fpeps`` path: 2nd-order
Trotter evolution with NTU bond truncation and a boundary-MPS contraction for all
measurements.  Selected via ``backend="peps"`` (engine_package="rydtn", the
default); the YASTN path remains available via ``engine_package="yastn"``.
"""

from __future__ import annotations

from .tensors import ArrayBackend, RydTNError, resolve_backend

__all__ = ["ArrayBackend", "RydTNError", "resolve_backend"]


def __getattr__(name):
    # Lazily expose the backend class so importing the subpackage stays cheap
    # and does not require the full engine to be present during partial builds.
    if name == "RydTNPEPSBackend":
        from .backend import RydTNPEPSBackend

        return RydTNPEPSBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
