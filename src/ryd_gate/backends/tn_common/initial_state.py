"""Shared initial-state label resolution for tensor-network backends.

Both the MPS (TeNPy) and PEPS (YASTN) backends turn a user ``initial_state`` -- a
named pattern, a 0/1 occupation array, or an explicit per-site label list -- into a
list of generic level labels (``ground``/``"0"``/``"1"``/``"r"``) in 2D site order.
They differ only in post-processing (MPS remaps to TeNPy site labels; PEPS uses the
labels as-is), so the shared resolution lives here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ryd_gate.backends.tn_common.lattice_spec import TNLatticeSpec


def validate_level_labels(spec: TNLatticeSpec, labels: Sequence[str]) -> None:
    """Raise ``ValueError`` if any label is not a level of ``spec``."""
    allowed = set(spec.level_spec.levels)
    unknown = sorted(set(labels) - allowed)
    if unknown:
        raise ValueError(f"Unknown level label(s) for {spec.level_structure}: {unknown}.")


def named_level_labels(spec: TNLatticeSpec, name: str) -> list[str]:
    """Generic per-site level labels for a named pattern (validated against ``spec``)."""
    ground = spec.level_spec.initial_level_or_default()
    if name == "all_ground":
        labels = [ground] * spec.N
    elif name == "all_1":
        labels = ["1"] * spec.N
    elif name in {"all_0", "all_zero"}:
        labels = ["0"] * spec.N
    elif name == "all_r":
        labels = ["r"] * spec.N
    elif name == "af1":
        labels = ["r" if s > 0 else ground for s in spec.sublattice]
    elif name == "af2":
        labels = ["r" if s < 0 else ground for s in spec.sublattice]
    else:
        raise ValueError(f"Unknown initial-state string: {name!r}.")
    validate_level_labels(spec, labels)
    return labels


def level_labels(spec: TNLatticeSpec, config) -> list[str]:
    """Generic per-site level labels in 2D site order (validated against ``spec``).

    ``config`` is a named-pattern string (see :func:`named_level_labels`), a 0/1
    occupation array (``1`` -> ``|r>``, ``0`` -> the non-Rydberg reference level),
    or an explicit per-site label list.
    """
    if isinstance(config, str):
        return named_level_labels(spec, config)

    arr = np.asarray(config)
    if arr.shape != (spec.N,):
        raise ValueError(f"initial_state must have shape ({spec.N},), got {arr.shape}.")

    if arr.dtype.kind in {"U", "S", "O"}:
        labels = [str(x) for x in arr]
    else:
        ground = spec.level_spec.initial_level_or_default()
        labels = ["r" if int(c) == 1 else ground for c in arr.astype(int)]
    validate_level_labels(spec, labels)
    return labels
