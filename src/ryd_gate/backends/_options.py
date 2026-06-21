"""Shared helper for backend option objects.

Backends historically take ``backend_options`` as a loose ``dict``. They now also
accept typed option dataclasses (``ExactOptions``, ``TenpyOptions``).
:func:`as_backend_options` normalizes either form into the plain ``dict`` the
backend constructors expect, dropping unset (``None``) fields so each backend's
own defaults still apply.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass


def as_backend_options(backend_options) -> dict:
    """Return a plain options dict from ``None``, a dict, or an option dataclass."""
    if backend_options is None:
        return {}
    if is_dataclass(backend_options) and not isinstance(backend_options, type):
        return {k: v for k, v in asdict(backend_options).items() if v is not None}
    return dict(backend_options)
