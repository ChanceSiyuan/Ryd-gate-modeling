"""Tests for the Phase 0 core abstractions: BasisSpec, BlockRegistry, ObservableRegistry."""

import numpy as np
import pytest

from ryd_gate.core.basis import BasisSpec
from ryd_gate.core.blocks import BlockRegistry
from ryd_gate.core.observables import ObservableRegistry


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
    def test_creation(self):
        bs = _make_basis()
        assert bs.n_sites == 2
        assert bs.local_dim == 7
        assert bs.total_dim == 49

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

    def test_projector_shape_and_trace(self):
        bs = _make_basis()
        proj = bs.projector("A", "r")
        assert proj.shape == (49, 49)
        # |r><r| on site A tensored with I_7 on site B -> trace = 7
        assert np.isclose(np.trace(proj), 7.0)

    def test_projector_idempotent(self):
        bs = _make_basis()
        proj = bs.projector("B", "e1")
        assert np.allclose(proj @ proj, proj)

    def test_projector_hermitian(self):
        bs = _make_basis()
        proj = bs.projector("A", "1")
        assert np.allclose(proj, proj.conj().T)

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


# ---------------------------------------------------------------------------
# BlockRegistry
# ---------------------------------------------------------------------------


class TestBlockRegistry:
    def test_register_and_get(self):
        reg = BlockRegistry()
        op = np.eye(4)
        reg.register("H_const", op, description="constant term")
        retrieved = reg.get("H_const")
        assert np.array_equal(retrieved, op)

    def test_get_info(self):
        reg = BlockRegistry()
        op = np.eye(4)
        reg.register("drive_420", op, description="420nm drive", hermitian=True)
        info = reg.get_info("drive_420")
        assert info.name == "drive_420"
        assert info.description == "420nm drive"
        assert info.hermitian is True

    def test_list(self):
        reg = BlockRegistry()
        reg.register("a", np.eye(2))
        reg.register("b", np.eye(2))
        assert sorted(reg.list()) == ["a", "b"]

    def test_has_and_contains(self):
        reg = BlockRegistry()
        reg.register("H_vdw", np.eye(2))
        assert reg.has("H_vdw")
        assert "H_vdw" in reg
        assert not reg.has("missing")
        assert "missing" not in reg

    def test_len(self):
        reg = BlockRegistry()
        assert len(reg) == 0
        reg.register("x", np.eye(2))
        assert len(reg) == 1

    def test_get_missing_raises(self):
        reg = BlockRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")


# ---------------------------------------------------------------------------
# ObservableRegistry
# ---------------------------------------------------------------------------


class TestObservableRegistry:
    def test_register_and_get(self):
        reg = ObservableRegistry()
        op = np.diag([1.0, 0.0])
        reg.register("n0", op, description="level-0 occupation")
        obs = reg.get("n0")
        assert obs.name == "n0"
        assert np.array_equal(obs.operator, op)

    def test_list_names(self):
        reg = ObservableRegistry()
        reg.register("a", np.eye(2))
        reg.register("b", np.eye(2))
        assert sorted(reg.list_names()) == ["a", "b"]

    def test_contains_and_len(self):
        reg = ObservableRegistry()
        assert len(reg) == 0
        reg.register("obs", np.eye(2))
        assert "obs" in reg
        assert "missing" not in reg
        assert len(reg) == 1

    def test_measure_known_state(self):
        """Measure |0> against sigma_z: expect +1."""
        reg = ObservableRegistry()
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        reg.register("sigma_z", sigma_z)
        psi = np.array([1.0, 0.0], dtype=complex)
        assert np.isclose(reg.measure("sigma_z", psi), 1.0)

    def test_measure_all(self):
        reg = ObservableRegistry()
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        identity = np.eye(2, dtype=complex)
        reg.register("sigma_z", sigma_z)
        reg.register("I", identity)
        psi = np.array([1.0, 0.0], dtype=complex)
        results = reg.measure_all(psi)
        assert np.isclose(results["sigma_z"], 1.0)
        assert np.isclose(results["I"], 1.0)

    def test_measure_occupation_two_atom(self):
        """Measure occupation of level 0 on a two-atom system in |0,0> state.

        The occupation operator for level 0 on site A is |0><0|_A x I_B.
        For |0,0> we expect <psi|n0_A|psi> = 1.
        """
        bs = BasisSpec(
            site_labels=("A", "B"),
            local_levels=LEVELS_7,
            local_dim=7,
            total_dim=49,
        )
        n0_A = bs.projector("A", "0")

        reg = ObservableRegistry()
        reg.register("n0_A", n0_A, description="level-0 occupation on A", per_site=True)

        # |0,0> is the first basis state
        psi = np.zeros(49, dtype=complex)
        psi[0] = 1.0

        assert np.isclose(reg.measure("n0_A", psi), 1.0)

    def test_get_missing_raises(self):
        reg = ObservableRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")
