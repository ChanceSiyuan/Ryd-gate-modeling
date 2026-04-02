"""Tests for Phase 4 SystemModel wrappers: Rb87TwoAtomModel, Lattice2LevelModel, Analog3LevelModel."""

import numpy as np
import pytest

from ryd_gate.core.atomic_system import (
    AtomicSystem,
    LatticeSystem,
    create_lattice_system,
    create_analog_system,
)
from ryd_gate.core.system_model import SystemModel
from ryd_gate.core.models.rb87_two_atom import Rb87TwoAtomModel
from ryd_gate.core.models.lattice_2level import Lattice2LevelModel
from ryd_gate.core.models.analog_3level import Analog3LevelModel


# ---------------------------------------------------------------------------
# Rb87TwoAtomModel
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rb87_model():
    return Rb87TwoAtomModel.from_param_set("our")


class TestRb87TwoAtomModel:
    def test_is_system_model(self, rb87_model):
        assert isinstance(rb87_model, SystemModel)

    def test_param_set(self, rb87_model):
        assert rb87_model.param_set == "our"

    def test_basis_structure(self, rb87_model):
        basis = rb87_model.basis
        assert basis.n_sites == 2
        assert basis.local_dim == 7
        assert basis.total_dim == 49
        assert basis.site_labels == ("A", "B")
        assert basis.local_levels == ("0", "1", "e1", "e2", "e3", "r", "r_garb")

    def test_basis_level_index(self, rb87_model):
        assert rb87_model.basis.level_index("r") == 5
        assert rb87_model.basis.level_index("0") == 0
        assert rb87_model.basis.level_index("r_garb") == 6

    def test_all_expected_blocks_registered(self, rb87_model):
        expected = {"H_const", "H_1013", "H_1013_conj", "drive_420", "drive_420_dag", "lightshift_zero"}
        registered = set(rb87_model.blocks.list())
        assert expected == registered

    def test_all_expected_observables_registered(self, rb87_model):
        expected = {
            "pop_0", "pop_1", "pop_e1", "pop_e2", "pop_e3", "pop_r", "pop_r_garb",
            "pop_A_ground", "pop_A_qubit", "pop_B_ground", "pop_B_qubit",
            "pop_A_ryd", "pop_B_ryd",
        }
        registered = set(rb87_model.observables.list_names())
        assert expected == registered

    def test_drive_420_matches_system(self, rb87_model):
        block_op = rb87_model.blocks.get("drive_420")
        system_op = rb87_model.system.tq_ham_420
        np.testing.assert_array_equal(block_op, system_op)

    def test_drive_420_dag_matches_system(self, rb87_model):
        block_op = rb87_model.blocks.get("drive_420_dag")
        system_op = rb87_model.system.tq_ham_420_conj
        np.testing.assert_array_equal(block_op, system_op)

    def test_system_property_returns_atomic_system(self, rb87_model):
        assert isinstance(rb87_model.system, AtomicSystem)
        assert rb87_model.system.param_set == "our"

    def test_lukin_param_set(self):
        model = Rb87TwoAtomModel.from_param_set("lukin")
        assert model.param_set == "lukin"
        assert model.basis.local_dim == 7

    def test_invalid_param_set_raises(self):
        with pytest.raises(ValueError, match="Unknown param_set"):
            Rb87TwoAtomModel.from_param_set("analog")

    def test_rejects_3level_system(self):
        system = create_analog_system()
        with pytest.raises(ValueError, match="7-level"):
            Rb87TwoAtomModel(system)


# ---------------------------------------------------------------------------
# Lattice2LevelModel
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lattice_model():
    system = create_lattice_system(Lx=2, Ly=2, V_nn=24.0)
    return Lattice2LevelModel(system)


class TestLattice2LevelModel:
    def test_is_system_model(self, lattice_model):
        assert isinstance(lattice_model, SystemModel)

    def test_param_set(self, lattice_model):
        assert lattice_model.param_set == "lattice"

    def test_basis_structure(self, lattice_model):
        basis = lattice_model.basis
        assert basis.n_sites == 4
        assert basis.local_dim == 2
        assert basis.total_dim == 16
        assert basis.site_labels == ("0", "1", "2", "3")
        assert basis.local_levels == ("g", "r")

    def test_expected_blocks(self, lattice_model):
        expected = {"global_X", "global_n", "H_vdw", "n_0", "n_1", "n_2", "n_3"}
        registered = set(lattice_model.blocks.list())
        assert expected == registered

    def test_expected_observables(self, lattice_model):
        expected = {"n_r_0", "n_r_1", "n_r_2", "n_r_3"}
        registered = set(lattice_model.observables.list_names())
        assert expected == registered

    def test_site_block_matches_system(self, lattice_model):
        for i in range(4):
            block_op = lattice_model.blocks.get(f"n_{i}")
            system_op = lattice_model.system.n_list[i]
            # Sparse matrices: compare via toarray
            np.testing.assert_array_equal(block_op.toarray(), system_op.toarray())

    def test_system_property_returns_lattice_system(self, lattice_model):
        assert isinstance(lattice_model.system, LatticeSystem)
        assert lattice_model.system.N == 4

    def test_observable_per_site_flag(self, lattice_model):
        obs = lattice_model.observables.get("n_r_0")
        assert obs.per_site is True


# ---------------------------------------------------------------------------
# Analog3LevelModel
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def analog_model():
    return Analog3LevelModel.from_defaults()


class TestAnalog3LevelModel:
    def test_is_system_model(self, analog_model):
        assert isinstance(analog_model, SystemModel)

    def test_param_set(self, analog_model):
        assert analog_model.param_set == "analog"

    def test_basis_structure(self, analog_model):
        basis = analog_model.basis
        assert basis.n_sites == 2
        assert basis.local_dim == 3
        assert basis.total_dim == 9
        assert basis.site_labels == ("A", "B")
        assert basis.local_levels == ("g", "e", "r")

    def test_basis_level_index(self, analog_model):
        assert analog_model.basis.level_index("r") == 2
        assert analog_model.basis.level_index("g") == 0
        assert analog_model.basis.level_index("e") == 1

    def test_expected_blocks(self, analog_model):
        expected = {"H_const", "H_1013", "H_1013_conj", "drive_420", "drive_420_dag", "lightshift_zero"}
        registered = set(analog_model.blocks.list())
        assert expected == registered

    def test_expected_observables(self, analog_model):
        base = {"pop_g", "pop_e", "pop_r", "pop_A_g", "pop_A_r", "pop_B_g", "pop_B_r"}
        # Joint product-state projectors: pop_gg, pop_ge, pop_gr, ...
        levels = ("g", "e", "r")
        joint = {f"pop_{a}{b}" for a in levels for b in levels}
        expected = base | joint
        registered = set(analog_model.observables.list_names())
        assert expected == registered

    def test_measure_pop_g_ground_state(self, analog_model):
        """Both atoms in |g>: total ground population should be 2.0."""
        psi0 = np.zeros(9, dtype=complex)
        psi0[0] = 1.0  # |g,g> = |0,0> in tensor product
        result = analog_model.observables.measure("pop_g", psi0)
        np.testing.assert_allclose(result, 2.0, atol=1e-12)

    def test_measure_pop_r_ground_state(self, analog_model):
        """Both atoms in |g>: Rydberg population should be 0."""
        psi0 = np.zeros(9, dtype=complex)
        psi0[0] = 1.0
        result = analog_model.observables.measure("pop_r", psi0)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_measure_per_atom_g(self, analog_model):
        """Both atoms in |g>: each atom individually in |g> with probability 1."""
        psi0 = np.zeros(9, dtype=complex)
        psi0[0] = 1.0
        assert np.isclose(analog_model.observables.measure("pop_A_g", psi0), 1.0)
        assert np.isclose(analog_model.observables.measure("pop_B_g", psi0), 1.0)

    def test_system_property_returns_atomic_system(self, analog_model):
        assert isinstance(analog_model.system, AtomicSystem)
        assert analog_model.system.param_set == "analog"
        assert analog_model.system.n_levels == 3

    def test_rejects_7level_system(self):
        from ryd_gate.core.atomic_system import create_our_system
        system = create_our_system()
        with pytest.raises(ValueError, match="3-level"):
            Analog3LevelModel(system)
