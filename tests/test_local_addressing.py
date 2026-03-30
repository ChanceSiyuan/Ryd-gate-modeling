"""Tests for the ryd_gate.lattice subpackage (many-body Rydberg array)."""

import numpy as np
import pytest


# ---- Fixtures (session-scoped to avoid rebuilding operators) ----

@pytest.fixture(scope="session")
def small_lattice():
    from ryd_gate.lattice import make_square_lattice
    return make_square_lattice(3, 3)


@pytest.fixture(scope="session")
def small_ops(small_lattice):
    from ryd_gate.lattice import build_operators
    return build_operators(small_lattice.N, small_lattice.vdw_pairs, V_nn=24.0)


@pytest.fixture(scope="session")
def small_bit_masks(small_lattice):
    from ryd_gate.lattice import precompute_bit_masks
    return precompute_bit_masks(small_lattice.N)


# ---- Geometry ----

class TestGeometry:
    def test_lattice_dimensions(self, small_lattice):
        assert small_lattice.N == 9
        assert small_lattice.Lx == 3
        assert small_lattice.Ly == 3
        assert small_lattice.coords.shape == (9, 2)

    def test_sublattice_checkerboard(self, small_lattice):
        for i, (ix, iy) in enumerate(small_lattice.coords):
            assert small_lattice.sublattice[i] == (-1) ** (ix + iy)

    def test_sublattice_balanced(self, small_lattice):
        # 3x3: 5 sites of one sign, 4 of the other
        counts = np.bincount(((small_lattice.sublattice + 1) // 2).astype(int))
        assert set(counts) == {4, 5}

    def test_neighbor_count(self, small_lattice):
        # 3x3 lattice: 12 NN pairs + 8 NNN pairs = 20 pairs total
        nn = sum(1 for _, _, v in small_lattice.vdw_pairs if abs(v - 1.0) < 0.01)
        nnn = sum(1 for _, _, v in small_lattice.vdw_pairs
                  if abs(v - 1.0 / 8) < 0.01)
        assert nn == 12  # 3x3 grid nearest neighbors
        assert nnn == 8   # diagonal pairs

    def test_is_in_domain(self):
        from ryd_gate.lattice import is_in_domain
        assert is_in_domain(1, 1, 1.0, 1.0, 0.8)
        assert not is_in_domain(0, 0, 1.0, 1.0, 0.8)


# ---- Operators ----

class TestOperators:
    def test_dimensions(self, small_ops, small_lattice):
        dim = 2 ** small_lattice.N
        assert small_ops['sum_X'].shape == (dim, dim)
        assert small_ops['H_vdw'].shape == (dim, dim)
        assert len(small_ops['n_list']) == small_lattice.N

    def test_hermiticity(self, small_ops):
        for key in ['sum_X', 'sum_n', 'H_vdw']:
            H = small_ops[key].toarray()
            np.testing.assert_allclose(H, H.conj().T, atol=1e-14)

    def test_hamiltonian_hermitian(self, small_ops, small_lattice):
        from ryd_gate.lattice import build_hamiltonian
        H = build_hamiltonian(1.0, 2.0, np.zeros(small_lattice.N), small_ops)
        H_dense = H.toarray()
        np.testing.assert_allclose(H_dense, H_dense.conj().T, atol=1e-14)


# ---- States ----

class TestStates:
    def test_product_state_normalized(self, small_lattice):
        from ryd_gate.lattice import product_state
        psi = product_state([0] * small_lattice.N, small_lattice.N)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-14

    def test_all_ground_is_first_basis(self, small_lattice):
        from ryd_gate.lattice import product_state
        psi = product_state([0] * small_lattice.N, small_lattice.N)
        assert psi[0] == 1.0
        assert np.sum(np.abs(psi) ** 2) == pytest.approx(1.0)

    def test_af_configs_complementary(self, small_lattice):
        from ryd_gate.lattice import af_config
        af1 = af_config(small_lattice.sublattice, which=1)
        af2 = af_config(small_lattice.sublattice, which=2)
        # AF1 and AF2 should be complementary
        np.testing.assert_array_equal(af1 + af2, np.ones(small_lattice.N))


# ---- Observables ----

class TestObservables:
    def test_af1_staggered_magnetization(self, small_lattice, small_ops,
                                          small_bit_masks):
        from ryd_gate.lattice import product_state, af_config, measure_from_states
        config = af_config(small_lattice.sublattice, which=1)
        psi = product_state(config, small_lattice.N)
        ms, n_mean, occ = measure_from_states(
            psi, small_bit_masks, small_lattice.sublattice)
        assert ms == pytest.approx(1.0, abs=1e-10)

    def test_af2_staggered_magnetization(self, small_lattice, small_ops,
                                          small_bit_masks):
        from ryd_gate.lattice import product_state, af_config, measure_from_states
        config = af_config(small_lattice.sublattice, which=2)
        psi = product_state(config, small_lattice.N)
        ms, _, _ = measure_from_states(
            psi, small_bit_masks, small_lattice.sublattice)
        assert ms == pytest.approx(-1.0, abs=1e-10)

    def test_all_ground_magnetization(self, small_lattice, small_bit_masks):
        from ryd_gate.lattice import product_state, measure_from_states
        psi = product_state([0] * small_lattice.N, small_lattice.N)
        ms, n_mean, _ = measure_from_states(
            psi, small_bit_masks, small_lattice.sublattice)
        assert n_mean == pytest.approx(0.0, abs=1e-10)

    def test_batch_vs_single(self, small_lattice, small_bit_masks):
        from ryd_gate.lattice import product_state, af_config, measure_from_states
        config = af_config(small_lattice.sublattice, which=1)
        psi = product_state(config, small_lattice.N)
        # Single
        ms1, _, _ = measure_from_states(psi, small_bit_masks,
                                         small_lattice.sublattice)
        # Batch of 1
        ms_batch, _, _ = measure_from_states(psi[np.newaxis, :], small_bit_masks,
                                              small_lattice.sublattice)
        assert ms1 == pytest.approx(ms_batch[0], abs=1e-14)


# ---- Evolution ----

class TestEvolution:
    def test_norm_preserved(self, small_lattice, small_ops):
        from ryd_gate.lattice import (
            product_state, build_hamiltonian, evolve_constant_H,
        )
        psi0 = product_state([0] * small_lattice.N, small_lattice.N)
        H = build_hamiltonian(1.0, 0.5, np.zeros(small_lattice.N), small_ops)
        _, states = evolve_constant_H(psi0, H, 2.0, 20)
        norms = np.linalg.norm(states, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)
