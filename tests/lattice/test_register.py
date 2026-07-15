"""Tests for the frozen Register geometry API (ryd_gate.lattice, S03/S09/S14)."""

import numpy as np
import pytest

from ryd_gate.lattice import Register


class TestConstructors:
    def test_chain_coords(self):
        reg = Register.chain(3, 4.0)
        assert reg.N == 3
        np.testing.assert_allclose(reg.coords, [(0.0, 0.0), (4.0, 0.0), (8.0, 0.0)])

    def test_chain_default_spacing(self):
        # spacing_um is an input, not a stored property; check via geometry.
        reg = Register.chain(2)
        np.testing.assert_allclose(reg.coords, [(0.0, 0.0), (4.0, 0.0)])

    def test_rectangle_row_major(self):
        reg = Register.rectangle(2, 3, 5.0)
        assert reg.N == 6
        np.testing.assert_allclose(
            reg.coords,
            [
                (0.0, 0.0), (0.0, 5.0), (0.0, 10.0),
                (5.0, 0.0), (5.0, 5.0), (5.0, 10.0),
            ],
        )

    def test_square_equals_rectangle(self):
        sq = Register.square(2, 5.0)
        rect = Register.rectangle(2, 2, 5.0)
        np.testing.assert_allclose(sq.coords, rect.coords)
        np.testing.assert_allclose(sq.coords, [(0.0, 0.0), (0.0, 5.0), (5.0, 0.0), (5.0, 5.0)])

    def test_triangular_conventions(self):
        reg = Register.triangular(2, 3, 4.0)
        assert reg.N == 6
        coords = reg.coords
        np.testing.assert_allclose(coords[0], [0.0, 0.0])
        np.testing.assert_allclose(coords[3], [2.0, 4.0 * np.sqrt(3) / 2])
        np.testing.assert_allclose(coords[4], [6.0, 4.0 * np.sqrt(3) / 2])

    def test_direct_coords_kept_verbatim(self):
        # No hidden centering (S04): the coordinates are used as given.
        reg = Register([(0.0, 0.0), (4.0, 0.0)])
        np.testing.assert_allclose(reg.coords, [(0.0, 0.0), (4.0, 0.0)])

    def test_invalid_sizes_raise(self):
        with pytest.raises(ValueError):
            Register.chain(0, 4.0)
        with pytest.raises(ValueError):
            Register.rectangle(2, 3, -1.0)
        with pytest.raises(ValueError):
            Register.triangular(0, 3)


class TestValidationRules:
    def test_n_derived_from_coords(self):
        reg = Register([[0.0, 0.0], [0.0, 2.0], [0.0, 5.0]])
        assert reg.N == 3
        assert reg.coords.shape == (3, 2)

    def test_3d_coords_rejected(self):
        with pytest.raises(ValueError):
            Register([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def test_bad_coords_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            Register([[0.0], [1.0]])

    def test_nonfinite_coords_raise(self):
        with pytest.raises(ValueError, match="finite"):
            Register([[np.inf, 0.0]])

    def test_coords_readonly(self):
        reg = Register.chain(3, 4.0)
        assert reg.coords.flags.writeable is False
        with pytest.raises(ValueError):
            reg.coords[0, 0] = 1.0
        # Cannot be re-enabled for writing (view of a frozen base).
        with pytest.raises(ValueError):
            reg.coords.setflags(write=True)


class TestRemovedNames:
    @pytest.mark.parametrize(
        "name",
        ["ids", "sublattice", "spacing_um"],
    )
    def test_removed_instance_attributes(self, name):
        reg = Register.chain(2, 4.0)
        assert not hasattr(reg, name)

    @pytest.mark.parametrize(
        "name",
        ["from_coordinates", "distances_um", "distance_pairs", "blockade_edges", "draw",
         "index", "id_at"],
    )
    def test_removed_methods(self, name):
        assert not hasattr(Register, name)

    def test_lattice_all_is_register_only(self):
        import ryd_gate.lattice as lattice

        assert lattice.__all__ == ["Register"]

    @pytest.mark.parametrize(
        "name",
        ["is_in_domain", "nn_nnn_relative_pairs", "cylinder_nn_nnn_pairs",
         "plot_spatial_rydberg", "RegisterLayout", "LatticeGeometry"],
    )
    def test_removed_module_symbols(self, name):
        import ryd_gate.lattice as lattice

        assert not hasattr(lattice, name)
