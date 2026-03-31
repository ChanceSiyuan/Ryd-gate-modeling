"""Tests for the LatticeSystem + SweepProtocol + solve_lattice stack."""

import numpy as np
import pytest

from ryd_gate.core.atomic_system import LatticeSystem, create_lattice_system
from ryd_gate.lattice import (
    af_config,
    measure_from_states,
    precompute_bit_masks,
    product_state,
    solve_lattice,
)
from ryd_gate.protocols.sweep import SweepProtocol


@pytest.fixture(scope="module")
def system_3x3():
    return create_lattice_system(Lx=3, Ly=3, V_nn=24.0)


@pytest.fixture(scope="module")
def masks_3x3(system_3x3):
    return precompute_bit_masks(system_3x3.N)


# ── System ────────────────────────────────────────────────────────────

class TestLatticeSystem:
    def test_creation(self, system_3x3):
        assert system_3x3.param_set == "lattice"
        assert system_3x3.N == 9
        assert system_3x3.Lx == 3
        assert system_3x3.Ly == 3

    def test_operators_shape(self, system_3x3):
        dim = 2**9
        assert system_3x3.sum_X.shape == (dim, dim)
        assert system_3x3.H_vdw.shape == (dim, dim)
        assert len(system_3x3.n_list) == 9

    def test_hermiticity(self, system_3x3):
        H = system_3x3.sum_X.toarray()
        np.testing.assert_allclose(H, H.conj().T, atol=1e-14)

    def test_sublattice(self, system_3x3):
        for i, (ix, iy) in enumerate(system_3x3.coords):
            assert system_3x3.sublattice[i] == (-1) ** (ix + iy)


# ── Protocol ──────────────────────────────────────────────────────────

class TestSweepProtocol:
    def test_n_params(self):
        proto = SweepProtocol()
        assert proto.n_params == 3

    def test_validate(self):
        proto = SweepProtocol()
        proto.validate_params([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            proto.validate_params([1.0, 2.0])

    def test_unpack(self, system_3x3):
        proto = SweepProtocol()
        params = proto.unpack_params([-10.0, 10.0, 5.0], system_3x3)
        assert params["delta_start"] == -10.0
        assert params["delta_end"] == 10.0
        assert params["t_gate"] == 5.0

    def test_addressing(self, system_3x3):
        proto = SweepProtocol(addressing={0: 50.0})
        deltas = proto.get_pin_deltas(system_3x3.N)
        assert deltas[0] == 50.0
        assert deltas[1] == 0.0


# ── Registry ──────────────────────────────────────────────────────────

class TestRegistry:
    def test_lattice_in_registry(self):
        from ryd_gate.core.atomic_system import compatible_protocols
        assert "SweepProtocol" in compatible_protocols("lattice")


# ── Solver ────────────────────────────────────────────────────────────

class TestSolveLattice:
    def test_norm_preserved(self, system_3x3):
        proto = SweepProtocol(n_steps=50)
        psi0 = product_state([0] * 9, 9)
        psi_f = solve_lattice(system_3x3, proto, [-10.0, 10.0, 5.0], psi0)
        assert abs(np.linalg.norm(psi_f) - 1.0) < 1e-6

    def test_sweep_excites_atoms(self, system_3x3):
        """After sweep from negative to positive Δ, Rydberg population > 0."""
        proto = SweepProtocol(n_steps=100)
        psi0 = product_state([0] * 9, 9)
        psi_f = solve_lattice(system_3x3, proto, [-10.0, 10.0, 5.0], psi0)
        masks = precompute_bit_masks(9)
        _, n_mean, _ = measure_from_states(psi_f, masks, system_3x3.sublattice)
        assert n_mean > 0.1

    def test_checkerboard_ordering(self, system_3x3, masks_3x3):
        """Adiabatic sweep should produce approximate checkerboard (|m_s| > 0.5)."""
        proto = SweepProtocol(n_steps=400, omega_ramp_frac=0.05)
        psi0 = product_state([0] * 9, 9)
        psi_f = solve_lattice(system_3x3, proto, [-10.0, 10.0, 30.0], psi0)
        ms, _, _ = measure_from_states(psi_f, masks_3x3, system_3x3.sublattice)
        assert abs(ms) > 0.5

    def test_pinning_suppresses_atom(self, system_3x3, masks_3x3):
        """Pinning atom 0 should suppress its Rydberg excitation."""
        proto = SweepProtocol(addressing={0: 50.0}, n_steps=100)
        psi0 = product_state([0] * 9, 9)
        psi_f = solve_lattice(system_3x3, proto, [-10.0, 10.0, 5.0], psi0)
        _, _, occ = measure_from_states(psi_f, masks_3x3, system_3x3.sublattice)
        assert occ[0] < 0.1
