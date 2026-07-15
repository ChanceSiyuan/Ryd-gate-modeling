"""Tests for the Register product API (ryd_gate.lattice)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from ryd_gate.lattice import Register


class TestConstructors:
    def test_chain_ids_coords_sublattice(self):
        reg = Register.chain(3, 4.0)
        assert reg.N == 3
        assert reg.ids == ("q0", "q1", "q2")
        np.testing.assert_allclose(reg.coords, [(0.0, 0.0), (4.0, 0.0), (8.0, 0.0)])
        np.testing.assert_array_equal(reg.sublattice, [1, -1, 1])

    def test_chain_derives_n_and_spacing(self):
        reg = Register.chain(3, spacing_um=4)
        assert reg.N == 3
        assert reg.spacing_um == 4.0

    def test_chain_default_spacing(self):
        reg = Register.chain(2)
        assert reg.spacing_um == 4.0

    def test_rectangle_row_major_checkerboard(self):
        reg = Register.rectangle(2, 3, 5.0)
        assert reg.N == 6
        assert reg.ids == ("q0", "q1", "q2", "q3", "q4", "q5")
        np.testing.assert_allclose(
            reg.coords,
            [
                (0.0, 0.0), (0.0, 5.0), (0.0, 10.0),
                (5.0, 0.0), (5.0, 5.0), (5.0, 10.0),
            ],
        )
        np.testing.assert_array_equal(reg.sublattice, [1, -1, 1, -1, 1, -1])
        assert reg.spacing_um == 5.0

    def test_square_equals_rectangle(self):
        sq = Register.square(2, 5.0, prefix="a")
        rect = Register.rectangle(2, 2, 5.0, prefix="a")
        assert sq.ids == rect.ids == ("a0", "a1", "a2", "a3")
        np.testing.assert_allclose(sq.coords, rect.coords)
        np.testing.assert_allclose(
            sq.coords, [(0.0, 0.0), (0.0, 5.0), (5.0, 0.0), (5.0, 5.0)]
        )
        np.testing.assert_array_equal(sq.sublattice, [1, -1, -1, 1])

    def test_triangular_conventions(self):
        reg = Register.triangular(2, 3, 4.0)
        assert reg.N == 6
        assert reg.ids[0] == "q0" and reg.ids[-1] == "q5"
        coords = reg.coords
        # row 0: no offset; row 1: offset by spacing/2; row pitch sqrt(3)/2 * spacing
        np.testing.assert_allclose(coords[0], [0.0, 0.0])
        np.testing.assert_allclose(coords[3], [2.0, 4.0 * np.sqrt(3) / 2])
        np.testing.assert_allclose(coords[4], [6.0, 4.0 * np.sqrt(3) / 2])
        np.testing.assert_array_equal(reg.sublattice, np.zeros(6, dtype=int))

    def test_from_coordinates_ids_and_center(self):
        reg = Register.from_coordinates([(0.0, 0.0), (4.0, 0.0)], center=False)
        assert reg.ids == ("q0", "q1")
        np.testing.assert_allclose(reg.coords, [(0.0, 0.0), (4.0, 0.0)])
        assert reg.spacing_um == 4.0

        centered = Register.from_coordinates([(0.0, 0.0), (4.0, 0.0)], center=True)
        np.testing.assert_allclose(centered.coords, [(-2.0, 0.0), (2.0, 0.0)])

    def test_from_coordinates_l_shape_derives_spacing(self):
        # L-shape with pair distances 3, 4, 5: spacing is the smallest one.
        reg = Register.from_coordinates(
            [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)], center=False
        )
        assert reg.N == 3
        assert reg.spacing_um == 3.0

    def test_from_coordinates_explicit_ids_and_sublattice(self):
        reg = Register.from_coordinates(
            [(0.0, 0.0), (4.0, 0.0)],
            ids=("left", "right"),
            center=False,
            sublattice=[1, -1],
        )
        assert reg.ids == ("left", "right")
        np.testing.assert_array_equal(reg.sublattice, [1, -1])

    def test_from_coordinates_single_atom_spacing_zero(self):
        reg = Register.from_coordinates([(1.0, 2.0)], center=False)
        assert reg.N == 1
        assert reg.spacing_um == 0.0

    def test_from_coordinates_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Register.from_coordinates([])

    def test_invalid_sizes_raise(self):
        with pytest.raises(ValueError):
            Register.chain(0, 4.0)
        with pytest.raises(ValueError):
            Register.rectangle(2, 3, -1.0)
        with pytest.raises(ValueError):
            Register.square(2, 4.0, prefix="")
        with pytest.raises(ValueError):
            Register.triangular(0, 3)


class TestValidationRules:
    def test_duplicate_ids_raise(self):
        with pytest.raises(ValueError, match="unique"):
            Register([[0, 0], [1, 0]], ids=("a", "a"))

    def test_mixed_coordinate_dimensions_raise(self):
        with pytest.raises(ValueError):
            Register.from_coordinates([(0.0, 0.0), (1.0, 1.0, 1.0)])

    def test_bad_coords_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            Register([[0.0], [1.0]])

    def test_omitted_ids_autogenerate(self):
        reg = Register([[0, 0], [1, 0]])
        assert reg.ids == ("q0", "q1")

    def test_nonfinite_coords_raise(self):
        with pytest.raises(ValueError, match="finite"):
            Register([[np.inf, 0.0]])

    def test_mismatched_sublattice_raises(self):
        with pytest.raises(ValueError, match="sublattice"):
            Register([[0, 0], [1, 0]], sublattice=[0])


class TestDerivedGeometry:
    def test_n_and_dimensionality_derived_from_coords(self):
        reg = Register([[0.0, 0.0], [0.0, 2.0], [0.0, 5.0]])
        assert reg.N == 3
        assert reg.coords.shape == (3, 2)
        assert reg.spacing_um == 2.0

    def test_3d_coords_supported(self):
        reg = Register([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert reg.N == 2
        assert reg.coords.shape == (2, 3)
        assert reg.spacing_um == 1.0


class TestIndexing:
    def test_index_id_at_roundtrip(self):
        reg = Register.square(2, 5.0)
        for i, atom_id in enumerate(reg.ids):
            assert reg.index(atom_id) == i
            assert reg.id_at(i) == atom_id

    def test_unknown_id_raises_keyerror(self):
        reg = Register.chain(2, 4.0)
        with pytest.raises(KeyError):
            reg.index("nope")

    def test_out_of_range_index_raises_indexerror(self):
        reg = Register.chain(2, 4.0)
        with pytest.raises(IndexError):
            reg.id_at(2)
        with pytest.raises(IndexError):
            reg.id_at(-1)


class TestGeometryQueries:
    def test_distances_symmetric_zero_diagonal(self):
        reg = Register.square(2, 5.0)
        d = reg.distances_um()
        assert d.shape == (4, 4)
        np.testing.assert_allclose(d, d.T)
        np.testing.assert_allclose(np.diag(d), 0.0)

    def test_distance_pairs_cutoff(self):
        reg = Register.square(2, 5.0)
        all_pairs = reg.distance_pairs()
        assert len(all_pairs) == 6
        assert all(i < j for i, j, _ in all_pairs)
        nn = reg.distance_pairs(cutoff_um=5.1)
        assert len(nn) == 4
        with pytest.raises(ValueError):
            reg.distance_pairs(cutoff_um=-1.0)

    def test_blockade_edges(self):
        reg = Register.square(2, 5.0)
        edges = reg.blockade_edges(radius_um=5.1)
        assert set(edges) == {(0, 1), (0, 2), (1, 3), (2, 3)}
        assert reg.blockade_edges(radius_um=0.1) == ()
        with pytest.raises(ValueError):
            reg.blockade_edges(radius_um=-1.0)


class TestDraw:
    def test_draw_returns_figure(self):
        import matplotlib.pyplot as plt

        reg = Register.square(2, 5.0)
        fig = reg.draw(blockade_radius_um=6.0, show=False)
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_draw_3d_not_implemented(self):
        reg = Register([[0, 0, 0], [1, 0, 0]])
        with pytest.raises(NotImplementedError):
            reg.draw(show=False)


class TestRemovedNames:
    def test_lattice_geometry_not_importable(self):
        with pytest.raises(ImportError):
            from ryd_gate.lattice import LatticeGeometry  # noqa: F401

    def test_register_layout_removed(self):
        with pytest.raises(ImportError):
            from ryd_gate.lattice import RegisterLayout  # noqa: F401
        with pytest.raises(ImportError):
            from ryd_gate.lattice import define_register  # noqa: F401

    @pytest.mark.parametrize(
        "name",
        [
            "make_chain",
            "make_square_lattice",
            "make_triangular_lattice",
            "make_geometry_from_coords",
        ],
    )
    def test_make_factories_not_importable(self, name):
        import importlib

        lattice = importlib.import_module("ryd_gate.lattice")
        with pytest.raises(AttributeError):
            getattr(lattice, name)
