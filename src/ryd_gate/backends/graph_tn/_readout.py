"""Lazy amplitude/sample readers over a final graph-PEPS state. No quimb import.

Every method operates on the gauge-absorbed ``quimb`` state object handed in by
the engine (``isel``/``contract``/``to_dense`` are methods on that object), so no
third-party import is needed here. Amplitudes are exact computational-basis
coefficients (feasible for small ``N``); sampling draws from the exact Born
distribution obtained by densifying the state, so both are limited to small
systems (``local_dim ** n_sites`` must fit in memory).
"""

from __future__ import annotations

import threading
from collections import Counter

import numpy as np

from ryd_gate.backends.graph_tn._options import GraphTNError
from ryd_gate.backends.tn_common.initial_state import validate_labels

_PHASE_REF_TOL = 1e-9
_MAX_DENSE_DIM = 1 << 20  # cap the exact statevector used for sampling


def _to_host_complex(value) -> complex:
    """Bring a possibly-device (NumPy or Torch-CUDA) scalar to a host Python ``complex``."""
    if hasattr(value, "item"):
        return complex(value.item())
    return complex(np.asarray(value).reshape(()))


def _to_host_array(value) -> np.ndarray:
    """Bring a possibly-device (NumPy or Torch-CUDA) array to a host NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class _BaseReader:
    """Shared exact-amplitude cache (normalized coefficients) under one lock."""

    __slots__ = ("_psi", "_terms", "_lock", "_norm", "_dense")

    def __init__(self, psi, terms) -> None:
        self._psi = psi
        self._terms = terms
        self._lock = threading.RLock()
        self._norm: float | None = None
        self._dense: np.ndarray | None = None

    def _index(self, labels) -> list[int]:
        levels = self._terms.levels
        return [levels.index(x) for x in labels]

    def _coefficient(self, labels) -> complex:
        """Normalized computational-basis amplitude ``<labels | psi> / ||psi||``."""
        psi = self._psi
        sites = sorted(psi.sites)
        idx = self._index(labels)
        sel = {psi.site_ind_id.format(s): idx[k] for k, s in enumerate(sites)}
        raw = _to_host_complex(psi.isel(sel).contract(output_inds=()))
        if self._norm is None:
            self._norm = _to_host_complex(psi.H @ psi).real ** 0.5
        return raw / self._norm

    def _statevector(self) -> np.ndarray:
        if self._dense is None:
            psi = self._psi
            sites = sorted(psi.sites)
            n, d = self._terms.n_sites, self._terms.local_dim
            if d ** n > _MAX_DENSE_DIM:
                raise GraphTNError(
                    f"sample() densifies the state ({d}**{n} amplitudes), which exceeds the "
                    f"{_MAX_DENSE_DIM} limit; graph-PEPS sampling is only available for small systems."
                )
            v = _to_host_array(psi.to_dense([psi.site_ind_id.format(s) for s in sites])).reshape(-1)
            nrm = np.linalg.norm(v)
            if not np.isfinite(nrm) or nrm == 0.0:
                raise GraphTNError(f"final state has a non-finite/zero norm {nrm!r}; cannot sample.")
            self._dense = v / nrm
        return self._dense

    def _sample(self, shots: int, seed: int) -> Counter:
        v = self._statevector()
        n, d = self._terms.n_sites, self._terms.local_dim
        levels = self._terms.levels
        probs = np.abs(v) ** 2
        probs = probs / probs.sum()
        rng = np.random.default_rng(seed)
        draws = rng.choice(len(probs), size=int(shots), p=probs)
        counts: Counter = Counter()
        for flat in draws:
            digits = []
            x = int(flat)
            for _ in range(n):
                digits.append(x % d)
                x //= d
            digits.reverse()  # big-endian: site 0 is the most significant factor
            counts[tuple(levels[k] for k in digits)] += 1
        return counts


class _RealTimeReader(_BaseReader):
    """Lazy amplitude/sample over the final real-time graph-PEPS."""

    __slots__ = ()

    def amplitude(self, labels) -> complex:
        labels = validate_labels(self._terms, labels)
        with self._lock:
            return self._coefficient(labels)

    def sample(self, shots: int, seed: int) -> Counter:
        with self._lock:
            return self._sample(shots, seed)


class _GroundStateReader(_BaseReader):
    """Lazy phase-referenced amplitude/sample over the graph-PEPS ground state."""

    __slots__ = ()

    def amplitude(self, labels, phase_reference) -> complex:
        target = validate_labels(self._terms, labels)
        reference = validate_labels(self._terms, phase_reference)
        with self._lock:
            ref_coeff = self._coefficient(reference)
            if abs(ref_coeff) <= _PHASE_REF_TOL:
                raise GraphTNError(
                    f"phase_reference amplitude is numerically zero ({abs(ref_coeff):.2e}); "
                    "choose a reference basis state with nonzero amplitude."
                )
            target_coeff = ref_coeff if list(target) == list(reference) else self._coefficient(target)
            return target_coeff / (ref_coeff / abs(ref_coeff))

    def sample(self, shots: int, seed: int) -> Counter:
        with self._lock:
            return self._sample(shots, seed)
