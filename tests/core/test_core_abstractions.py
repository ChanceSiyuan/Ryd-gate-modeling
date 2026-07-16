"""Tests for the Phase 0 core abstractions: BasisSpec."""

import pytest

from ryd_gate.core.model import BasisSpec

# ---------------------------------------------------------------------------
# BasisSpec
# ---------------------------------------------------------------------------

LEVELS_7 = ("0", "1", "e1", "e2", "e3", "r", "r_garb")
SITES_2 = ("A", "B")


def _make_basis() -> BasisSpec:
    return BasisSpec(
        site_labels=SITES_2,
        local_levels=LEVELS_7,
        local_dim=7,
        total_dim=49,
    )


class TestBasisSpec:
    def test_level_index(self):
        bs = _make_basis()
        assert bs.level_index("r") == 5
        assert bs.level_index("0") == 0

    def test_level_index_invalid(self):
        bs = _make_basis()
        with pytest.raises(ValueError, match="Level 'x'"):
            bs.level_index("x")

    def test_site_index(self):
        bs = _make_basis()
        assert bs.site_index("B") == 1
        assert bs.site_index("A") == 0

    def test_site_index_invalid(self):
        bs = _make_basis()
        with pytest.raises(ValueError, match="Site 'C'"):
            bs.site_index("C")

    def test_validation_local_dim_mismatch(self):
        with pytest.raises(ValueError, match="local_dim"):
            BasisSpec(
                site_labels=("A",),
                local_levels=("0", "1"),
                local_dim=3,
                total_dim=3,
            )

    def test_validation_total_dim_mismatch(self):
        with pytest.raises(ValueError, match="total_dim"):
            BasisSpec(
                site_labels=("A", "B"),
                local_levels=("0", "1"),
                local_dim=2,
                total_dim=8,
            )
