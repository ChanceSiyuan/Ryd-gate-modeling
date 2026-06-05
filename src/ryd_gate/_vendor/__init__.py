"""Vendored third-party packages used by selected backends.

Currently this holds a copy of PyTreeNet (used by the ``ttn`` backend). See
``NOTICE.md`` for upstream provenance and license. Prefer the helper below over
importing the vendored package directly.
"""

from __future__ import annotations

from importlib import import_module


def import_pytreenet():
    """Return the vendored PyTreeNet module, falling back to an installed copy."""
    try:
        return import_module("ryd_gate._vendor.pytreenet")
    except ImportError:
        return import_module("pytreenet")
