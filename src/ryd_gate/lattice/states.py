"""Initial state preparation for Rydberg atom arrays."""

import numpy as np

from .geometry import is_in_domain


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
        Checkerboard signs (+1/-1) from SquareLattice.
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


# ── 3-level state preparation ────────────────────────────────────────


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
    dim = 3**N
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
