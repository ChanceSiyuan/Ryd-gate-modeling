"""Tests for coarsening post-processing functions."""

import numpy as np
import pytest

from ryd_gate.analysis.coarsening import (
    build_neighbor_lists,
    coarsegrained_boundary_mask,
    correct_single_spin_flips,
    domain_area_distribution,
    identify_domains,
)


# --- Fixtures ---

@pytest.fixture
def lattice_4x4():
    """4x4 square lattice geometry."""
    Lx, Ly = 4, 4
    coords = np.array([(ix, iy) for ix in range(Lx) for iy in range(Ly)])
    sublattice = np.array([(-1) ** (ix + iy) for ix, iy in coords])
    nn, nnn = build_neighbor_lists(coords)
    return Lx, Ly, coords, sublattice, nn, nnn


@pytest.fixture
def af1_occ(lattice_4x4):
    """Perfect AF1 occupation: sublattice +1 sites excited."""
    _, _, _, sublattice, _, _ = lattice_4x4
    return (sublattice > 0).astype(float)


@pytest.fixture
def af2_occ(lattice_4x4):
    """Perfect AF2 occupation: sublattice -1 sites excited."""
    _, _, _, sublattice, _, _ = lattice_4x4
    return (sublattice < 0).astype(float)


# --- build_neighbor_lists ---

class TestBuildNeighborLists:
    def test_corner_atom_has_2_nn(self, lattice_4x4):
        _, _, _, _, nn, nnn = lattice_4x4
        # Corner (0,0) has 2 NN: (1,0) and (0,1)
        assert len(nn[0]) == 2

    def test_interior_atom_has_4_nn(self, lattice_4x4):
        Lx, Ly, coords, _, nn, _ = lattice_4x4
        # Find an interior atom (1,1) -> index = 1*Ly + 1 = 5
        idx = 5  # (1,1) in row-major
        assert len(nn[idx]) == 4

    def test_corner_atom_has_1_nnn(self, lattice_4x4):
        _, _, _, _, _, nnn = lattice_4x4
        # Corner (0,0) has 1 NNN: (1,1)
        assert len(nnn[0]) == 1

    def test_symmetric(self, lattice_4x4):
        _, _, _, _, nn, nnn = lattice_4x4
        for i, neighbors in enumerate(nn):
            for j in neighbors:
                assert i in nn[j], f"NN asymmetry: {i} in nn[{j}]?"
        for i, neighbors in enumerate(nnn):
            for j in neighbors:
                assert i in nnn[j], f"NNN asymmetry: {i} in nnn[{j}]?"


# --- correct_single_spin_flips ---

class TestCorrectSingleSpinFlips:
    def test_no_flip_in_perfect_af1(self, lattice_4x4, af1_occ):
        _, _, _, sublattice, nn, nnn = lattice_4x4
        corrected = correct_single_spin_flips(af1_occ, sublattice, nn, nnn)
        np.testing.assert_array_equal(corrected, af1_occ)

    def test_no_flip_in_perfect_af2(self, lattice_4x4, af2_occ):
        _, _, _, sublattice, nn, nnn = lattice_4x4
        corrected = correct_single_spin_flips(af2_occ, sublattice, nn, nnn)
        np.testing.assert_array_equal(corrected, af2_occ)

    def test_single_defect_corrected(self, lattice_4x4, af1_occ):
        _, _, _, sublattice, nn, nnn = lattice_4x4
        occ = af1_occ.copy()
        # Flip an interior site (1,1) = index 5 in AF1 bulk
        defect_idx = 5
        occ[defect_idx] = 1.0 - occ[defect_idx]
        corrected = correct_single_spin_flips(occ, sublattice, nn, nnn)
        np.testing.assert_array_equal(corrected, af1_occ)

    def test_domain_wall_not_corrected(self, lattice_4x4):
        """Atoms at a domain wall have some same-type neighbors, so they
        should NOT be identified as single-spin flips."""
        Lx, Ly, _, sublattice, nn, nnn = lattice_4x4
        # Left half AF1, right half AF2
        occ = np.zeros(Lx * Ly)
        for i in range(Lx * Ly):
            ix, iy = divmod(i, Ly)
            if iy < Ly // 2:
                occ[i] = 1.0 if sublattice[i] > 0 else 0.0  # AF1
            else:
                occ[i] = 1.0 if sublattice[i] < 0 else 0.0  # AF2
        corrected = correct_single_spin_flips(occ, sublattice, nn, nnn)
        # Should be unchanged -- domain wall atoms have same-type neighbors
        np.testing.assert_array_equal(corrected, occ)

    def test_batch_processing(self, lattice_4x4, af1_occ):
        _, _, _, sublattice, nn, nnn = lattice_4x4
        batch = np.stack([af1_occ, af1_occ])
        corrected = correct_single_spin_flips(batch, sublattice, nn, nnn)
        assert corrected.shape == batch.shape
        np.testing.assert_array_equal(corrected, batch)


# --- coarsegrained_boundary_mask ---

class TestCoarseGrainedBoundaryMask:
    def test_perfect_af1_interior_bulk(self, lattice_4x4, af1_occ):
        Lx, Ly, _, sublattice, _, _ = lattice_4x4
        C, is_boundary = coarsegrained_boundary_mask(af1_occ, Lx, Ly)
        # Interior excited atoms in perfect AF should have C=0
        # Interior ground atoms should have C=4
        # (Edge atoms may differ due to boundary fill)
        for i in range(Lx * Ly):
            ix, iy = divmod(i, Ly)
            if 0 < ix < Lx - 1 and 0 < iy < Ly - 1:
                if af1_occ[i] == 1:
                    assert C[i] == 0, f"Interior excited site ({ix},{iy}) has C={C[i]}"
                else:
                    assert C[i] == 4, f"Interior ground site ({ix},{iy}) has C={C[i]}"

    def test_perfect_af_interior_is_bulk(self, lattice_4x4, af1_occ):
        Lx, Ly, _, _, _, _ = lattice_4x4
        _, is_boundary = coarsegrained_boundary_mask(af1_occ, Lx, Ly)
        # Interior atoms of perfect AF should not be boundaries
        for i in range(Lx * Ly):
            ix, iy = divmod(i, Ly)
            if 0 < ix < Lx - 1 and 0 < iy < Ly - 1:
                assert not is_boundary[i], f"Interior site ({ix},{iy}) wrongly marked as boundary"

    def test_all_excited_has_boundaries(self, lattice_4x4):
        Lx, Ly, _, _, _, _ = lattice_4x4
        occ = np.ones(Lx * Ly)
        C, is_boundary = coarsegrained_boundary_mask(occ, Lx, Ly)
        # All excited: C = count of neighbors. Excited with C != 0 is boundary
        # Interior atoms have C=4, so they're boundary (n=1, C!=0)
        for i in range(Lx * Ly):
            ix, iy = divmod(i, Ly)
            if 0 < ix < Lx - 1 and 0 < iy < Ly - 1:
                assert C[i] == 4
                assert is_boundary[i]

    def test_batch_processing(self, lattice_4x4, af1_occ):
        Lx, Ly, _, _, _, _ = lattice_4x4
        batch = np.stack([af1_occ, af1_occ])
        C, is_boundary = coarsegrained_boundary_mask(batch, Lx, Ly)
        assert C.shape == batch.shape
        assert is_boundary.shape == batch.shape


# --- identify_domains ---

class TestIdentifyDomains:
    def test_single_domain_in_perfect_af(self, lattice_4x4, af1_occ):
        _, _, _, sublattice, nn, _ = lattice_4x4
        labels = identify_domains(af1_occ, sublattice, nn)
        # Perfect AF1: all same type -> 1 domain
        assert len(np.unique(labels)) == 1

    def test_two_domains(self, lattice_4x4):
        """Left half AF1, right half AF2 -> at least 2 domains."""
        Lx, Ly, _, sublattice, nn, _ = lattice_4x4
        occ = np.zeros(Lx * Ly)
        for i in range(Lx * Ly):
            ix, iy = divmod(i, Ly)
            if iy < Ly // 2:
                occ[i] = 1.0 if sublattice[i] > 0 else 0.0
            else:
                occ[i] = 1.0 if sublattice[i] < 0 else 0.0
        labels = identify_domains(occ, sublattice, nn)
        assert len(np.unique(labels)) >= 2

    def test_all_sites_labeled(self, lattice_4x4, af1_occ):
        _, _, _, sublattice, nn, _ = lattice_4x4
        labels = identify_domains(af1_occ, sublattice, nn)
        assert np.all(labels >= 0)


# --- domain_area_distribution ---

class TestDomainAreaDistribution:
    def test_single_domain(self, lattice_4x4, af1_occ):
        _, _, _, sublattice, nn, _ = lattice_4x4
        labels = identify_domains(af1_occ, sublattice, nn)
        areas, weights = domain_area_distribution(labels)
        assert len(areas) == 1
        assert areas[0] == 16  # 4x4 = 16 sites
        np.testing.assert_allclose(weights.sum(), 1.0)

    def test_weights_sum_to_one(self, lattice_4x4):
        Lx, Ly, _, sublattice, nn, _ = lattice_4x4
        occ = np.zeros(Lx * Ly)
        for i in range(Lx * Ly):
            ix, iy = divmod(i, Ly)
            if iy < Ly // 2:
                occ[i] = 1.0 if sublattice[i] > 0 else 0.0
            else:
                occ[i] = 1.0 if sublattice[i] < 0 else 0.0
        labels = identify_domains(occ, sublattice, nn)
        _, weights = domain_area_distribution(labels)
        np.testing.assert_allclose(weights.sum(), 1.0)
