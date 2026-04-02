"""Tests for TN lattice spec and snake-order mapping."""

import numpy as np
import pytest

from ryd_gate.tn.lattice_spec import (
    TNLatticeSpec,
    create_tn_lattice_spec,
    snake_order_mapping,
)


class TestSnakeOrderMapping:
    def test_roundtrip(self):
        """inv_snake[snake_to_2d[i]] == i for all i."""
        for Lx, Ly in [(2, 2), (3, 3), (4, 4), (5, 3), (16, 16)]:
            s2d, inv = snake_order_mapping(Lx, Ly)
            N = Lx * Ly
            np.testing.assert_array_equal(inv[s2d], np.arange(N))
            np.testing.assert_array_equal(s2d[inv], np.arange(N))

    def test_is_permutation(self):
        """Snake order is a permutation of [0, N)."""
        s2d, inv = snake_order_mapping(4, 4)
        assert set(s2d) == set(range(16))
        assert set(inv) == set(range(16))

    def test_first_row_left_to_right(self):
        """Row 0 goes left-to-right."""
        Ly = 5
        s2d, _ = snake_order_mapping(3, Ly)
        # First Ly entries should be 0,1,2,...,Ly-1
        np.testing.assert_array_equal(s2d[:Ly], np.arange(Ly))

    def test_second_row_right_to_left(self):
        """Row 1 goes right-to-left."""
        Ly = 4
        s2d, _ = snake_order_mapping(3, Ly)
        row1 = s2d[Ly:2*Ly]
        expected = np.array([1*Ly + (Ly-1-j) for j in range(Ly)])
        np.testing.assert_array_equal(row1, expected)


class TestCreateTNLatticeSpec:
    def test_basic_properties(self):
        spec = create_tn_lattice_spec(Lx=3, Ly=3, V_nn=24.0)
        assert spec.N == 9
        assert spec.Lx == 3
        assert spec.Ly == 3
        assert spec.V_nn == 24.0
        assert spec.bc == "open"
        assert len(spec.snake_to_2d) == 9
        assert len(spec.inv_snake) == 9

    def test_sublattice_consistency(self):
        """Sublattice matches ryd_gate.lattice convention."""
        from ryd_gate.lattice.geometry import make_square_lattice

        spec = create_tn_lattice_spec(Lx=4, Ly=4)
        sq = make_square_lattice(4, 4)
        np.testing.assert_array_equal(spec.sublattice, sq.sublattice)
        np.testing.assert_array_equal(spec.coords, sq.coords)

    def test_vdw_pairs_consistency(self):
        """VdW pairs match ryd_gate.lattice convention."""
        from ryd_gate.lattice.geometry import make_square_lattice

        spec = create_tn_lattice_spec(Lx=3, Ly=3)
        sq = make_square_lattice(3, 3)
        assert spec.vdw_pairs == sq.vdw_pairs

    def test_frozen(self):
        """Spec is immutable."""
        spec = create_tn_lattice_spec(Lx=2, Ly=2)
        with pytest.raises(AttributeError):
            spec.Lx = 5
