"""Foundational data structures for quantum system models.

- :class:`BasisSpec` — Hilbert space structure with symbolic level labels
"""

from __future__ import annotations

from dataclasses import dataclass


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
