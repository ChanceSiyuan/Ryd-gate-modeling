"""Tests for TN lattice spec and snake-order mapping."""

import numpy as np
import pytest

from ryd_gate import DEFAULT_C6, RydbergSystem, SweepProtocol, compile_hamiltonian_ir
from ryd_gate.backends.tn_common.compiler import TNCompiler, tn_lattice_spec_from_system
from ryd_gate.backends.tn_common.lattice_spec import (
    create_tn_lattice_spec,
    snake_order_mapping,
)
from ryd_gate.core.level_structures import InteractionSpec, level_structure
from ryd_gate.lattice import Register


def _sweep(t_gate=1.0, omega=1.0, delta=0.0, n_steps=200):
    return SweepProtocol(
        t_gate=t_gate,
        omega_half_fn=lambda t: 0.5 * omega,
        delta_fn=lambda t: delta,
        n_steps=n_steps,
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
        assert spec.level_structure == "1r"
        assert spec.level_spec == level_structure("1r")
        assert spec.interaction_mode == "nnn"
        assert len(spec.snake_to_2d) == 9
        assert len(spec.inv_snake) == 9

    def test_three_level_spec(self):
        spec = create_tn_lattice_spec(Lx=2, Ly=2, level_structure="01r")
        assert spec.level_structure == "01r"
        assert spec.level_spec == level_structure("01r")

    def test_accepts_shared_level_structure_spec(self):
        shared_spec = level_structure("01r")
        spec = create_tn_lattice_spec(Lx=2, Ly=2, level_structure=shared_spec)
        assert spec.level_spec == shared_spec

    def test_invalid_level_structure_raises(self):
        with pytest.raises(ValueError, match="level_structure"):
            create_tn_lattice_spec(Lx=2, Ly=2, level_structure="bad")

    def test_registered_but_unsupported_level_structure_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            create_tn_lattice_spec(Lx=2, Ly=2, level_structure="analog_3")

    def test_nn_interaction_mode_filters_diagonals(self):
        spec = create_tn_lattice_spec(Lx=2, Ly=2, interaction_mode="nn")
        assert spec.interaction_mode == "nn"
        assert len(spec.vdw_pairs) == 4
        assert all(np.isclose(v_rel, 1.0) for _, _, v_rel in spec.vdw_pairs)

    def test_invalid_interaction_mode_raises(self):
        with pytest.raises(ValueError, match="interaction_mode"):
            create_tn_lattice_spec(Lx=2, Ly=2, interaction_mode="all")

    def test_sublattice_consistency(self):
        """Sublattice matches ryd_gate.lattice convention."""

        spec = create_tn_lattice_spec(Lx=4, Ly=4)
        sq = Register.rectangle(4, 4, spacing_um=1.0)
        np.testing.assert_array_equal(spec.sublattice, sq.sublattice)
        np.testing.assert_array_equal(spec.coords, sq.coords)

    def test_vdw_pairs_consistency(self):
        """VdW pairs match the shared NN/NNN lattice convention."""
        from ryd_gate.lattice import nn_nnn_relative_pairs

        spec = create_tn_lattice_spec(Lx=3, Ly=3)
        assert spec.vdw_pairs == nn_nnn_relative_pairs(3, 3)

    def test_frozen(self):
        """Spec is immutable."""
        spec = create_tn_lattice_spec(Lx=2, Ly=2)
        with pytest.raises(AttributeError):
            spec.Lx = 5


def test_tn_compiler_uses_system_level_spec_and_interactions():
    proto = _sweep(omega=2.0)
    system = RydbergSystem.from_lattice(
        Register.rectangle(2, 2, spacing_um=10.0),
        level_structure="1r",
        interaction=InteractionSpec(C6=DEFAULT_C6, mode="nn"),
        protocol=proto,
        Omega=2.0,
    )

    params = system.unpack_params([])
    ir = TNCompiler().compile(system, params)

    assert ir.spec.level_spec == system.meta("level_spec")
    assert ir.spec.vdw_pairs == system.meta("interaction_pairs")
    assert ir.spec.interaction_mode == "system"
    assert ir.spec.Omega == 2.0
    assert ir.hamiltonian is not None


def test_incompatible_protocol_level_structure_is_rejected():
    system = RydbergSystem.from_lattice(
        Register.rectangle(2, 2, spacing_um=10.0),
        level_structure="01r",
        interaction=InteractionSpec(C6=DEFAULT_C6, mode="nn"),
        protocol=_sweep(omega=2.0),
        Omega=2.0,
    )
    params = system.unpack_params([])

    with pytest.raises(ValueError, match="channel mismatch"):
        compile_hamiltonian_ir(system, params)


def test_unified_hamiltonian_ir_lowers_to_exact_and_tn():
    proto = _sweep(omega=2.0)
    system = RydbergSystem.from_lattice(
        Register.rectangle(2, 2, spacing_um=10.0),
        level_structure="1r",
        interaction=InteractionSpec(C6=DEFAULT_C6, mode="nn"),
        protocol=proto,
        Omega=2.0,
    )
    params = system.unpack_params([])

    hamiltonian = compile_hamiltonian_ir(system, params)
    tn_ir = TNCompiler().compile(hamiltonian)

    from ryd_gate.backends.exact.compiler import ExactSparseCompiler

    exact_ir = ExactSparseCompiler(max_dim=1000).compile(hamiltonian)

    assert tn_ir.hamiltonian is hamiltonian
    assert tn_ir.spec.vdw_pairs == hamiltonian.metadata["interaction_pairs"]
    assert exact_ir.metadata["source_compiler"] == "ryd_gate"
    assert exact_ir.static_terms
    assert exact_ir.drive_terms


def test_tn_lattice_spec_from_system_rejects_non_rectangular_geometry():

    system = RydbergSystem.from_lattice(
        Register.triangular(2, 2),
        level_structure="1r",
        interaction=InteractionSpec(mode="nn"),
        protocol=_sweep(),
    )
    with pytest.raises(ValueError, match="rectangular"):
        tn_lattice_spec_from_system(system)
