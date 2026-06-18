"""Trotter gate construction for the rydtn engine (pure NumPy, complex128).

Gates are ``exp(-coeff * H)`` for Hermitian ``H`` (mirror
``yastn.tn.fpeps.gates.gate_local_exp`` / ``gate_nn_exp``).  Coefficient
convention is exactly YASTN's ``exp(-coeff * H)``:

* real-time local half-step  ``coeff = 0.5j * dt``
* real-time nn full step      ``coeff = 1j * dt``
* imaginary-time local half   ``coeff = 0.5 * dtau``
* imaginary-time nn full       ``coeff = dtau``

The gates are tiny (``d <= 3``), so we always build them in complex128 and let
the caller cast to the working dtype when applying them to PEPS tensors.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def gate_local_exp(coeff: complex, H: np.ndarray, *, hermitian: bool = True) -> np.ndarray:
    """Single-site gate ``G[a, b] = (exp(-coeff*H))[a, b]``.

    Applied as ``A'[..., a] = sum_b G[a, b] A[..., b]`` on the physical leg.
    ``hermitian=True`` (1r/01r) uses an eigendecomposition; ``hermitian=False``
    (analog_3, decay-ready) uses a general matrix exponential.
    """
    H = np.asarray(H, dtype=np.complex128)
    if not hermitian:
        return expm(-coeff * H)
    w, S = np.linalg.eigh(H)
    return (S * np.exp(-coeff * w)) @ S.conj().T


def gate_nn_exp(coeff: complex, Hnn4: np.ndarray) -> np.ndarray:
    """Two-site gate ``exp(-coeff*H)`` as a 4-leg tensor ``(s0o, s0i, s1o, s1i)``.

    ``Hnn4`` carries the same leg order (e.g. ``PEPSOps.nn_hamiltonian()`` scaled by
    the bond strength).  Matches ``gate_nn_exp(...)`` up to its SVD split.
    """
    H = np.asarray(Hnn4, dtype=np.complex128)
    d = H.shape[0]
    # fuse to a matrix over (out=(s0o,s1o), in=(s0i,s1i))
    M = H.transpose(0, 2, 1, 3).reshape(d * d, d * d)
    w, S = np.linalg.eigh(M)
    Mexp = (S * np.exp(-coeff * w)) @ S.conj().T
    return Mexp.reshape(d, d, d, d).transpose(0, 2, 1, 3)
