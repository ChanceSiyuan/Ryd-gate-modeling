"""Vendored optional dependencies used by selected backends."""

from __future__ import annotations

from importlib import import_module


def import_pytreenet():
    """Return the vendored PyTreeNet module, falling back to site-packages."""
    try:
        return import_module("ttn.vendor.pytreenet")
    except ImportError:
        return import_module("pytreenet")
