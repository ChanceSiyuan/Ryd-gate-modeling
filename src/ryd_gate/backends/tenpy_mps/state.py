"""MPS initial-state builder and the private final-state reader (ER04/ER09)."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ryd_gate.backends.tn_common.initial_state import (
    initial_local_amplitudes,
    validate_labels,
)


def build_initial_mps(terms, initial_state, site):
    """Product-state (or ``"plus"``) MPS from per-site local amplitude vectors."""
    from tenpy.networks.mps import MPS

    amps = initial_local_amplitudes(terms, initial_state)  # (N, d)
    local_states = [amps[i] for i in range(terms.n_sites)]
    return MPS.from_product_state(
        [site] * terms.n_sites, local_states, bc="finite", dtype=complex,
        unit_cell_width=terms.n_sites,
    )


class MPSReader:
    """Lazy ``amplitude`` / ``sample`` over the private final MPS (ER04/ER09).

    ``amplitude`` is a product-state overlap contraction; ``sample`` is TeNPy's
    sequential conditional MPS measurement.  Both keep the state's complex
    normalization / global phase, so amplitude phases are physical.
    """

    __slots__ = ("_psi", "_terms", "_site")

    def __init__(self, psi, terms, site) -> None:
        self._psi = psi
        self._terms = terms
        self._site = site

    def amplitude(self, labels) -> complex:
        labels = validate_labels(self._terms, labels)
        bra = build_initial_mps(self._terms, labels, self._site)
        return complex(bra.overlap(self._psi))

    def sample(self, shots: int, seed: int) -> Counter:
        rng = np.random.default_rng(seed)
        levels = self._terms.levels
        counts: Counter = Counter()
        for _ in range(int(shots)):
            sigmas, _amp = self._psi.sample_measurements(rng=rng)
            counts[tuple(levels[int(s)] for s in sigmas)] += 1
        return counts
