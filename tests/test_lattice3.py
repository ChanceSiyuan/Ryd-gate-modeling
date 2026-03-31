"""Tests for the ryd_gate.lattice3 subpackage (N-atom 3-level systems)."""

import numpy as np
import pytest

from ryd_gate.lattice import (
    LatticeGeometry,
    ThreeLevelOps,
    build_3level_operators,
    checkerboard_rydberg,
    evolve_3level_sweep,
    ground_state,
    make_3level_square_lattice,
    make_geometry_from_coords,
    measure_rydberg_occupation,
    precompute_trit_masks,
    product_state_3level,
    staggered_magnetization,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def small_geom():
    return make_3level_square_lattice(2, 2, spacing_um=3.0)


@pytest.fixture(scope="session")
def small_ops(small_geom):
    return build_3level_operators(
        small_geom,
        Delta=2 * np.pi * 9.1e9,
        Omega_1013=2 * np.pi * 491e6,
        Omega_420=2 * np.pi * 491e6,
    )


@pytest.fixture(scope="session")
def small_masks(small_geom):
    return precompute_trit_masks(small_geom.N)


# ── Geometry ──────────────────────────────────────────────────────────

class TestGeometry:
    def test_square_lattice_dims(self, small_geom):
        assert small_geom.N == 4
        assert small_geom.coords.shape == (4, 2)

    def test_physical_spacing(self, small_geom):
        dists = []
        for i in range(small_geom.N):
            for j in range(i + 1, small_geom.N):
                dists.append(np.linalg.norm(
                    small_geom.coords[i] - small_geom.coords[j]))
        assert min(dists) == pytest.approx(3.0, rel=1e-10)

    def test_sublattice_checkerboard(self, small_geom):
        for i, (x, y) in enumerate(small_geom.coords):
            expected = (-1) ** (round(x / 3.0) + round(y / 3.0))
            assert small_geom.sublattice[i] == expected

    def test_vdw_all_pairs(self, small_geom):
        n_pairs = small_geom.N * (small_geom.N - 1) // 2
        assert len(small_geom.vdw_couplings) == n_pairs

    def test_custom_coords(self):
        coords = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]])
        geom = make_geometry_from_coords(coords)
        assert geom.N == 3
        assert len(geom.vdw_couplings) == 3


# ── Operators ─────────────────────────────────────────────────────────

class TestOperators:
    def test_dimensions(self, small_ops):
        dim = 3**4
        assert small_ops.dim == dim
        assert small_ops.H_const.shape == (dim, dim)
        assert small_ops.H_1013.shape == (dim, dim)

    def test_h_const_diagonal(self, small_ops):
        H = small_ops.H_const.toarray()
        off_diag = H - np.diag(np.diag(H))
        assert np.allclose(off_diag, 0, atol=1e-14)

    def test_h1013_hermiticity(self, small_ops):
        H = small_ops.H_1013 + small_ops.H_1013_dag
        dense = H.toarray()
        np.testing.assert_allclose(dense, dense.conj().T, atol=1e-14)

    def test_h420_hermiticity(self, small_ops):
        H = small_ops.H_420_uniform + small_ops.H_420_uniform_dag
        dense = H.toarray()
        np.testing.assert_allclose(dense, dense.conj().T, atol=1e-14)

    def test_per_atom_ops_count(self, small_ops):
        assert len(small_ops.H_420_per_atom) == small_ops.N
        assert len(small_ops.n_r_list) == small_ops.N

    def test_nonuniform_rabi(self, small_geom):
        omega_arr = np.array([1e6, 2e6, 3e6, 4e6]) * 2 * np.pi
        ops = build_3level_operators(
            small_geom, Delta=2 * np.pi * 9.1e9,
            Omega_1013=2 * np.pi * 491e6, Omega_420=omega_arr,
        )
        assert ops.H_420_uniform.nnz > 0


# ── States ────────────────────────────────────────────────────────────

class TestStates:
    def test_ground_state_normalized(self):
        psi = ground_state(4)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-14

    def test_ground_state_first_element(self):
        psi = ground_state(3)
        assert psi[0] == 1.0
        assert np.sum(np.abs(psi)**2) == pytest.approx(1.0)

    def test_product_state_rydberg(self):
        psi = product_state_3level([2, 0, 2], 3)
        # |r,g,r⟩ = index 2*9 + 0*3 + 2 = 20
        assert psi[20] == 1.0

    def test_checkerboard_config(self, small_geom):
        psi = checkerboard_rydberg(small_geom.sublattice, which=1)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-14


# ── Observables ───────────────────────────────────────────────────────

class TestObservables:
    def test_ground_state_zero_rydberg(self, small_masks):
        psi = ground_state(4)
        occ = measure_rydberg_occupation(psi, small_masks)
        np.testing.assert_allclose(occ, 0.0, atol=1e-14)

    def test_checkerboard_magnetization(self, small_geom, small_masks):
        psi = checkerboard_rydberg(small_geom.sublattice, which=1)
        occ = measure_rydberg_occupation(psi, small_masks)
        ms = staggered_magnetization(occ, small_geom.sublattice)
        assert ms == pytest.approx(1.0, abs=1e-10)

    def test_batch_measurement(self, small_masks):
        psi1 = ground_state(4)
        psi2 = product_state_3level([2, 2, 2, 2], 4)
        batch = np.array([psi1, psi2])
        occ = measure_rydberg_occupation(batch, small_masks)
        assert occ.shape == (2, 4)
        np.testing.assert_allclose(occ[0], 0.0, atol=1e-14)
        np.testing.assert_allclose(occ[1], 1.0, atol=1e-14)

    def test_trit_mask_completeness(self, small_masks):
        total = small_masks['g'] + small_masks['e'] + small_masks['r']
        np.testing.assert_allclose(total, 1.0, atol=1e-14)


# ── Two-atom consistency ──────────────────────────────────────────────

class TestTwoAtomConsistency:
    def test_vdw_matches_analog_system(self):
        """VdW for 2 atoms at 3 μm should match create_analog_system's v_ryd."""
        from ryd_gate.core.atomic_system import create_analog_system
        sys2 = create_analog_system()

        geom = make_3level_square_lattice(1, 2, spacing_um=3.0)
        ops = build_3level_operators(
            geom, Delta=sys2.Delta,
            Omega_1013=sys2.rabi_1013, Omega_420=sys2.rabi_420,
        )

        # Extract VdW energy: |rr⟩ diagonal element
        psi_rr = product_state_3level([2, 2], 2)
        vdw_lattice3 = np.real(psi_rr.conj() @ ops.H_const.toarray() @ psi_rr)
        vdw_analog = np.real(psi_rr.conj() @ sys2.tq_ham_const @ psi_rr)

        assert vdw_lattice3 == pytest.approx(vdw_analog, rel=1e-6)


# ── Evolution (slow) ─────────────────────────────────────────────────

class TestCheckerboard:
    def test_2x2_adiabatic_half_filling(self):
        """Adiabatic sweep on 2×2 lattice produces checkerboard (half-filling).

        On a balanced 2×2 lattice, AF1 and AF2 are degenerate, so the
        noiseless simulation produces (|AF1⟩+|AF2⟩)/√2 with m_s=0 but
        each atom at P_r ≈ 0.5 (half-filling).
        """
        geom = make_3level_square_lattice(2, 2, spacing_um=5.0)
        ops = build_3level_operators(
            geom, Delta=2 * np.pi * 9.1e9,
            Omega_1013=2 * np.pi * 491e6,
            Omega_420=2 * np.pi * 491e6,
        )
        psi0 = ground_state(4)
        result = evolve_3level_sweep(
            psi0, ops,
            delta_start=-2 * np.pi * 40e6,
            delta_end=2 * np.pi * 40e6,
            t_sweep=1.5e-6, n_steps=300,
        )
        masks = precompute_trit_masks(4)
        occ = measure_rydberg_occupation(result.psi_final, masks)
        # Half-filling: mean P_r ≈ 0.5 indicates checkerboard superposition
        assert np.mean(occ) == pytest.approx(0.5, abs=0.1)
