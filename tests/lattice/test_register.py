"""Tests for the Register / RegisterLayout product API (lattice/geometry.py)."""

import dataclasses

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from ryd_gate.lattice import Register, RegisterLayout


class TestConstructors:
    def test_chain_ids_coords_sublattice(self):
        reg = Register.chain(3, 4.0)
        assert reg.N == 3
        assert reg.ids == ("q0", "q1", "q2")
        assert reg.coords_um == ((0.0, 0.0), (4.0, 0.0), (8.0, 0.0))
        np.testing.assert_array_equal(reg.sublattice, [1, -1, 1])
        assert reg.layout is None

    def test_chain_default_spacing(self):
        reg = Register.chain(2)
        assert reg.spacing_um == 4.0

    def test_rectangle_row_major_checkerboard(self):
        reg = Register.rectangle(2, 3, 5.0)
        assert reg.N == 6
        assert reg.ids == ("q0", "q1", "q2", "q3", "q4", "q5")
        assert reg.coords_um == (
            (0.0, 0.0), (0.0, 5.0), (0.0, 10.0),
            (5.0, 0.0), (5.0, 5.0), (5.0, 10.0),
        )
        np.testing.assert_array_equal(reg.sublattice, [1, -1, 1, -1, 1, -1])
        assert reg.layout is None

    def test_square_equals_rectangle(self):
        sq = Register.square(2, 5.0, prefix="a")
        rect = Register.rectangle(2, 2, 5.0, prefix="a")
        assert sq.ids == rect.ids == ("a0", "a1", "a2", "a3")
        assert sq.coords_um == rect.coords_um == (
            (0.0, 0.0), (0.0, 5.0), (5.0, 0.0), (5.0, 5.0),
        )
        np.testing.assert_array_equal(sq.sublattice, [1, -1, -1, 1])
        assert sq.layout is None

    def test_triangular_conventions(self):
        reg = Register.triangular(2, 3, 4.0)
        assert reg.N == 6
        assert reg.ids[0] == "q0" and reg.ids[-1] == "q5"
        coords = reg.coords_array
        # row 0: no offset; row 1: offset by spacing/2; row pitch sqrt(3)/2 * spacing
        np.testing.assert_allclose(coords[0], [0.0, 0.0])
        np.testing.assert_allclose(coords[3], [2.0, 4.0 * np.sqrt(3) / 2])
        np.testing.assert_allclose(coords[4], [6.0, 4.0 * np.sqrt(3) / 2])
        np.testing.assert_array_equal(reg.sublattice, np.zeros(6, dtype=int))
        assert reg.layout is None

    def test_from_coordinates_ids_and_center(self):
        reg = Register.from_coordinates([(0.0, 0.0), (4.0, 0.0)], center=False)
        assert reg.ids == ("q0", "q1")
        assert reg.coords_um == ((0.0, 0.0), (4.0, 0.0))
        assert reg.spacing_um == 4.0
        assert reg.layout is None

        centered = Register.from_coordinates([(0.0, 0.0), (4.0, 0.0)], center=True)
        assert centered.coords_um == ((-2.0, 0.0), (2.0, 0.0))

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
            Register(N=2, coords=[[0, 0], [1, 0]], sublattice=[0, 0],
                     spacing_um=1.0, ids=("a", "a"))

    def test_mixed_coordinate_dimensions_raise(self):
        with pytest.raises(ValueError):
            Register.from_coordinates([(0.0, 0.0), (1.0, 1.0, 1.0)])

    def test_omitted_ids_autogenerate(self):
        reg = Register(N=2, coords=[[0, 0], [1, 0]], sublattice=[0, 0], spacing_um=1.0)
        assert reg.ids == ("q0", "q1")

    def test_nonfinite_coords_raise(self):
        with pytest.raises(ValueError, match="finite"):
            Register(N=1, coords=[[np.inf, 0.0]], sublattice=[0], spacing_um=1.0)

    def test_mismatched_sublattice_raises(self):
        with pytest.raises(ValueError, match="sublattice"):
            Register(N=2, coords=[[0, 0], [1, 0]], sublattice=[0], spacing_um=1.0)


class TestProperties:
    def test_coords_array_is_a_copy(self):
        reg = Register.chain(2, 4.0)
        arr = reg.coords_array
        arr[0, 0] = 99.0
        assert reg.coords[0, 0] == 0.0

    def test_coords_um_tuple_of_tuples(self):
        reg = Register.chain(2, 4.0)
        assert isinstance(reg.coords_um, tuple)
        assert all(isinstance(row, tuple) for row in reg.coords_um)

    def test_n_atoms_and_dimensions(self):
        reg = Register.chain(3, 4.0)
        assert reg.n_atoms == 3
        assert reg.dimensions == 2


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
        reg = Register(N=2, coords=[[0, 0, 0], [1, 0, 0]], sublattice=[0, 0], spacing_um=1.0)
        with pytest.raises(NotImplementedError):
            reg.draw(show=False)


class TestValidateDelegation:
    def test_validate_delegates_to_device(self):
        class _StubDevice:
            def validate_register(self, register):
                return ["sentinel", register]

        reg = Register.chain(2, 4.0)
        result = reg.validate(_StubDevice())
        assert result[0] == "sentinel"
        assert result[1] is reg


class TestLayout:
    def test_classmethods_never_attach_layouts(self):
        for reg in (
            Register.chain(2),
            Register.square(2),
            Register.rectangle(1, 2),
            Register.triangular(1, 2),
            Register.from_coordinates([(0.0, 0.0), (1.0, 0.0)]),
        ):
            assert reg.layout is None

    def test_replace_attaches_layout_and_revalidates(self):
        reg = Register.square(2, 5.0)
        layout = RegisterLayout(
            name="square_2x2",
            trap_coords_um=reg.coords_um,
            kind="square",
        )
        reg2 = dataclasses.replace(reg, layout=layout)
        assert reg2.layout is layout
        assert reg2.ids == reg.ids
        with pytest.raises(ValueError):
            dataclasses.replace(reg, spacing_um=-1.0)

    def test_layout_validation(self):
        with pytest.raises(ValueError, match="kind"):
            RegisterLayout(name="x", trap_coords_um=((0.0, 0.0),), kind="hexagonal")
        with pytest.raises(ValueError, match="empty"):
            RegisterLayout(name="x", trap_coords_um=(), kind="custom")
        with pytest.raises(ValueError, match="name"):
            RegisterLayout(name="", trap_coords_um=((0.0, 0.0),), kind="custom")


class TestRemovedNames:
    def test_lattice_geometry_not_importable(self):
        with pytest.raises(ImportError):
            from ryd_gate.lattice import LatticeGeometry  # noqa: F401

    @pytest.mark.parametrize(
        "name",
        ["make_chain", "make_square_lattice", "make_triangular_lattice", "make_geometry_from_coords"],
    )
    def test_make_factories_not_importable(self, name):
        import importlib

        lattice = importlib.import_module("ryd_gate.lattice")
        with pytest.raises(AttributeError):
            getattr(lattice, name)
