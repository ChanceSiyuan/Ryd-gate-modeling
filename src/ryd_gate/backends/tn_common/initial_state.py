"""Initial-state resolution for the tensor-network backends (E17).

``simulate`` hands the TN backends a single normalized initial state: either a
flat physical level-label list (one product state) or the literal string
``"plus"`` (the uniform ``(|0>+|1>)/sqrt(2)`` product state, matching
``ryd_gate.core.states.dense_plus_state``).  The old named aliases
(``all_ground`` / ``af1`` / ``af2`` / ...) are gone — callers pass explicit
labels.  This module turns that input into per-site local amplitude vectors in
the system's level order.
"""

from __future__ import annotations

import numpy as np


def initial_local_amplitudes(terms, initial_state) -> np.ndarray:
    """Return per-site local amplitude vectors ``(N, d)`` in level order.

    ``terms`` is the compiled :class:`~ryd_gate.backends.tn_common.compiler.TNTerms`
    (it carries ``levels`` and ``n_sites``).
    """
    levels = terms.levels
    n, d = terms.n_sites, terms.local_dim

    if isinstance(initial_state, str):
        if initial_state != "plus":
            raise ValueError(
                f"unknown initial_state string {initial_state!r}; the TN backends "
                "accept a flat level-label list or 'plus'."
            )
        if "0" not in levels or "1" not in levels:
            raise ValueError(f"'plus' state requires '0' and '1' levels; got {levels}.")
        vec = np.zeros(d, dtype=complex)
        vec[levels.index("0")] = 1.0 / np.sqrt(2.0)
        vec[levels.index("1")] = 1.0 / np.sqrt(2.0)
        return np.tile(vec, (n, 1))

    labels = list(initial_state)
    if len(labels) != n:
        raise ValueError(
            f"initial_state has {len(labels)} per-site labels but the system has {n} sites."
        )
    amps = np.zeros((n, d), dtype=complex)
    for i, label in enumerate(labels):
        if label not in levels:
            raise ValueError(f"unknown level label {label!r}; levels are {levels}.")
        amps[i, levels.index(label)] = 1.0
    return amps


def validate_labels(terms, labels) -> list[str]:
    """Validate a flat per-site label list for ``amplitude`` / ``phase_reference``."""
    labels = list(labels)
    if len(labels) != terms.n_sites:
        raise ValueError(
            f"expected {terms.n_sites} per-site labels, got {len(labels)}."
        )
    for label in labels:
        if label not in terms.levels:
            raise ValueError(f"unknown level label {label!r}; levels are {terms.levels}.")
    return labels
