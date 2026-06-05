"""Typed options for the exact state-vector backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExactOptions:
    """Options for :func:`ryd_gate.backends.exact.simulate`.

    ``None`` means "use the backend default".
    """

    n_steps: int | None = None
