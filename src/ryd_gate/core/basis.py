"""Basis specification for quantum systems with symbolic level labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class BasisSpec:
    """Describes the Hilbert space structure of a multi-site quantum system.

    Parameters
    ----------
    site_labels : tuple of str
        Labels for each site/atom (e.g. ("A", "B") for two-atom).
    local_levels : tuple of str
        Labels for each single-site energy level (e.g. ("0", "1", "e1", "e2", "e3", "r", "r_garb")).
    local_dim : int
        Number of levels per site (must equal len(local_levels)).
    total_dim : int
        Full Hilbert space dimension (local_dim ** n_sites).
    """

    site_labels: tuple[str, ...]
    local_levels: tuple[str, ...]
    local_dim: int
    total_dim: int

    def __post_init__(self):
        if self.local_dim != len(self.local_levels):
            raise ValueError(
                f"local_dim={self.local_dim} != len(local_levels)={len(self.local_levels)}"
            )
        expected_dim = self.local_dim ** len(self.site_labels)
        if self.total_dim != expected_dim:
            raise ValueError(
                f"total_dim={self.total_dim} != local_dim^n_sites={expected_dim}"
            )

    @property
    def n_sites(self) -> int:
        return len(self.site_labels)

    def level_index(self, label: str) -> int:
        """Return the integer index for a level label. Raises ValueError if not found."""
        try:
            return self.local_levels.index(label)
        except ValueError:
            raise ValueError(f"Level '{label}' not in {self.local_levels}") from None

    def site_index(self, label: str) -> int:
        """Return the integer index for a site label."""
        try:
            return self.site_labels.index(label)
        except ValueError:
            raise ValueError(f"Site '{label}' not in {self.site_labels}") from None

    def projector(self, site: str, level: str) -> NDArray[np.complexfloating]:
        """Build |level><level| on the given site, tensored with identity on other sites.

        Returns a total_dim x total_dim dense matrix.
        """
        site_idx = self.site_index(site)
        level_idx = self.level_index(level)

        sq = np.zeros((self.local_dim, self.local_dim), dtype=np.complex128)
        sq[level_idx, level_idx] = 1.0

        # Build tensor product: I^{site_idx} x sq x I^{n_sites - site_idx - 1}
        result = sq
        for i in range(self.n_sites - 1, -1, -1):
            if i == site_idx:
                continue
            if i > site_idx:
                result = np.kron(result, np.eye(self.local_dim, dtype=np.complex128))
            else:
                result = np.kron(np.eye(self.local_dim, dtype=np.complex128), result)
        return result
