"""Initial state constructors for many-body Rydberg lattice systems.

2-level (|g>, |r>) helpers:
- :func:`product_state`     — product state from a 0/1 bit configuration
- :func:`af_config`         — antiferromagnetic 0/1 config from sublattice signs
- :func:`domain_config`     — AF1 bulk with an AF2 square domain

3-level (g, e, r) cascade helpers:
- :func:`product_state_3level`  — product state from a {0,1,2} trit configuration
- :func:`ground_state`          — all atoms in |g>
- :func:`checkerboard_rydberg`  — |r> on one sublattice, |g> on the other

Lives in ``core/`` (not ``lattice/``) because these constructors assume
specific local Hilbert dimensions / level labels; ``lattice/`` is reserved
for pure geometry.  Index conventions match
:func:`ryd_gate.core.operators.embed_site_op`.
"""

from __future__ import annotations

import numpy as np

from ryd_gate.lattice import is_in_domain

# ─────────────────────────────────────────────────────────────────────
# 2-level
# ─────────────────────────────────────────────────────────────────────


def product_state(config, N):
    """Create a computational basis product state.

    Parameters
    ----------
    config : array-like of 0/1
        Per-site occupation: 0 = |g>, 1 = |r>.
    N : int
        Number of atoms.
    """
    psi = np.zeros(2 ** N, dtype=complex)
    idx = 0
    for bit in config:
        idx = 2 * idx + int(bit)
    psi[idx] = 1.0
    return psi


def af_config(sublattice, which=1):
    """Antiferromagnetic configuration.

    Parameters
    ----------
    sublattice : ndarray
        Checkerboard signs (+1/-1) from lattice geometry.
    which : {1, 2}
        AF1: sublattice +1 sites excited.
        AF2: sublattice -1 sites excited.
    """
    if which == 1:
        return (sublattice > 0).astype(int)
    else:
        return (sublattice < 0).astype(int)


def domain_config(coords, sublattice, domain_center, domain_radius):
    """AF1 bulk with AF2 domain in a square region around domain_center."""
    config = af_config(sublattice, which=1)
    cx, cy = domain_center
    for i, (ix, iy) in enumerate(coords):
        if is_in_domain(ix, iy, cx, cy, domain_radius):
            config[i] = 1 if sublattice[i] < 0 else 0
    return config


# ─────────────────────────────────────────────────────────────────────
# 3-level (g, e, r cascade)
# ─────────────────────────────────────────────────────────────────────


def product_state_3level(config, N):
    """Create a 3-level computational basis product state.

    Parameters
    ----------
    config : array-like of {0, 1, 2}
        Per-site level: 0=|g⟩, 1=|e⟩, 2=|r⟩.
    N : int
        Number of atoms.
    """
    config = np.asarray(config, dtype=int)
    dim = 3 ** N
    idx = 0
    for i in range(N):
        idx = idx * 3 + config[i]
    psi = np.zeros(dim, dtype=complex)
    psi[idx] = 1.0
    return psi


def ground_state(N):
    """All atoms in |g⟩ (3-level)."""
    return product_state_3level([0] * N, N)


def checkerboard_rydberg(sublattice, which=1):
    """Checkerboard with |r⟩ on one sublattice, |g⟩ on the other (3-level)."""
    N = len(sublattice)
    if which == 1:
        config = np.where(sublattice > 0, 2, 0)
    else:
        config = np.where(sublattice < 0, 2, 0)
    return product_state_3level(config, N)


# ─────────────────────────────────────────────────────────────────────
# Product superpositions (uniform per-site superposition states)
# ─────────────────────────────────────────────────────────────────────


def product_superposition_state(local_amps, N):
    """Create a uniform product-superposition state vector ``⊗ᵢ local_amps``.

    Parameters
    ----------
    local_amps : array-like of complex, shape (d,)
        Single-site amplitudes in level order, placed identically on every site.
    N : int
        Number of sites.

    The index convention matches :func:`product_state_3level` and
    :meth:`ryd_gate.core.system.RydbergSystem.product_state`
    (``idx = Σ level_index · d^(N-1-i)``; site 0 is most significant), so the
    returned dense vector is consistent with the exact backend's basis ordering.
    """
    v = np.asarray(local_amps, dtype=complex)
    psi = v
    for _ in range(int(N) - 1):
        psi = np.kron(psi, v)
    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("local_amps must have nonzero norm.")
    return psi / norm


def plus_local_amplitudes(levels):
    """Single-site ``|+> = (|0>+|1>)/√2`` amplitudes for the given level labels.

    Requires both ``"0"`` and ``"1"`` levels (e.g. the ``01r`` structure); the
    Rydberg level (and any others) get zero amplitude.
    """
    labels = tuple(str(level) for level in levels)
    if "0" not in labels or "1" not in labels:
        raise ValueError(f"'plus' state requires '0' and '1' levels; got {labels}.")
    v = np.zeros(len(labels), dtype=complex)
    v[labels.index("0")] = 1.0 / np.sqrt(2.0)
    v[labels.index("1")] = 1.0 / np.sqrt(2.0)
    return v
