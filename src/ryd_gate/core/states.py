"""Initial-state normalization and the private label→dense-state compiler.

The public ``initial_state`` input is a flat physical level-label list (one
product state), a nested list of them (a batch), the string ``"plus"``, or
``None`` (which is exactly ``["1"] * N`` for every retained preset, E27).
Backend-native states and arbitrary dense vectors are rejected. The
label→amplitude/index conversion is a private state-compiler seam (S05/S11).
"""

from __future__ import annotations

import numpy as np


def normalize_initial_state(initial_state, n_sites: int):
    """Normalize the public ``initial_state`` input into ``(kind, value)``.

    - ``kind == "single"``: ``value`` is a per-site label list or the literal
      string ``"plus"``.
    - ``kind == "batch"``: ``value`` is a list of per-site label lists.

    ``None`` maps to ``["1"] * n_sites`` (E27). Dense vectors / backend-native
    states are rejected.
    """
    if initial_state is None:
        return "single", ["1"] * n_sites
    if isinstance(initial_state, str):
        if initial_state == "plus":
            return "single", "plus"
        raise ValueError(
            f"unknown initial_state string {initial_state!r}; allowed: None, 'plus', "
            "a flat level-label sequence, or a nested sequence of them."
        )
    if isinstance(initial_state, (list, tuple)):
        if len(initial_state) == 0:
            raise ValueError("initial_state must not be an empty sequence.")
        if all(isinstance(entry, (list, tuple)) for entry in initial_state):
            for entry in initial_state:
                _require_label_list(entry, n_sites)
            return "batch", [list(entry) for entry in initial_state]
        _require_label_list(initial_state, n_sites)
        return "single", list(initial_state)
    raise TypeError(
        "initial_state must be None, 'plus', a level-label sequence, or a nested "
        f"sequence of level-label sequences; got {type(initial_state).__name__}."
    )


def _require_label_list(entry, n_sites: int) -> None:
    if len(entry) != n_sites:
        raise ValueError(
            f"initial_state has {len(entry)} per-site labels but the system has {n_sites} sites."
        )
    for label in entry:
        if not isinstance(label, str):
            raise TypeError(
                f"initial_state entries must be level-label strings; got {type(label).__name__}."
            )


# ── private label → dense state (exact backend; site 0 is most significant) ──


def product_index(labels, basis) -> int:
    """Dense computational-basis index of a per-site label list."""
    idx = 0
    d = basis.local_dim
    for label in labels:
        idx = idx * d + basis.level_index(label)
    return idx


def dense_product_state(labels, basis) -> np.ndarray:
    """Dense product state ``|labels>`` in this basis."""
    psi = np.zeros(basis.total_dim, dtype=complex)
    psi[product_index(labels, basis)] = 1.0
    return psi


def _plus_local_amplitudes(levels) -> np.ndarray:
    labels = tuple(str(level) for level in levels)
    if "0" not in labels or "1" not in labels:
        raise ValueError(f"'plus' state requires '0' and '1' levels; got {labels}.")
    v = np.zeros(len(labels), dtype=complex)
    v[labels.index("0")] = 1.0 / np.sqrt(2.0)
    v[labels.index("1")] = 1.0 / np.sqrt(2.0)
    return v


def dense_plus_state(basis) -> np.ndarray:
    """Uniform ``|+> = (|0>+|1>)/√2`` product state across all sites."""
    v = _plus_local_amplitudes(basis.local_levels)
    psi = v
    for _ in range(basis.n_sites - 1):
        psi = np.kron(psi, v)
    return psi
