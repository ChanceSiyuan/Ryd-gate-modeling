"""Tests for the compiler layer: HamiltonianIR, DenseAtomicCompiler, SparseLatticeCompiler.

Verifies that the new compile + evolve_ir path produces the same results
as the legacy solve_gate() / solve_lattice() paths.
"""

import numpy as np
import pytest

from ryd_gate.compilers.dense_atomic import DenseAtomicCompiler
from ryd_gate.compilers.ir import HamiltonianIR, HamiltonianTerm
from ryd_gate.compilers.sparse_lattice import SparseLatticeCompiler
from ryd_gate.core.atomic_system import (
    create_analog_system,
    create_lattice_system,
    create_our_system,
)
from ryd_gate.protocols.gate_cz_to import TOProtocol
from ryd_gate.protocols.sweep import SweepProtocol
from ryd_gate.solvers.schrodinger import evolve_ir, solve_gate


class TestHamiltonianIR:
    """Test IR dataclasses."""

    def test_creation(self):
        op = np.eye(2, dtype=complex)
        term = HamiltonianTerm("test", op, 1.0)
        ir = HamiltonianIR(
            static_terms=[term],
            drive_terms=[],
            dim=2,
        )
        assert ir.dim == 2
        assert not ir.is_sparse
        assert len(ir.static_terms) == 1
        assert len(ir.drive_terms) == 0

    def test_metadata(self):
        ir = HamiltonianIR([], [], dim=4, metadata={"t_gate": 1e-6})
        assert ir.metadata["t_gate"] == 1e-6


class TestDenseAtomicCompiler:
    """Test DenseAtomicCompiler produces correct IR."""

    def test_compile_our_to(self):
        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]
        params = proto.unpack_params(x, system)
        compiler = DenseAtomicCompiler()
        ir = compiler.compile(system, proto, params)

        assert ir.dim == 49
        assert not ir.is_sparse
        assert len(ir.static_terms) == 3  # H_const, H_1013, H_1013_conj
        assert len(ir.drive_terms) == 3  # drive_420, drive_420_dag, lightshift_zero
        assert ir.metadata["t_gate"] == params["t_gate"]

    def test_compile_with_ham_additions(self):
        """Compiler should incorporate protocol's ham_const_additions."""
        system = create_analog_system()
        proto = SweepProtocol(addressing={0: 2 * np.pi * 12e6}, scatter_rate=35.0)
        x = [-5.0, 5.0, 1.5]
        params = proto.unpack_params(x, system)
        compiler = DenseAtomicCompiler()
        ir = compiler.compile(system, proto, params)

        # H_const should include pinning additions
        H_const_term = ir.static_terms[0]
        assert H_const_term.name == "H_const"
        # The compiled H_const should differ from system.tq_ham_const
        assert not np.allclose(H_const_term.operator, system.tq_ham_const)

    def test_drive_term_coefficients_are_callable(self):
        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]
        params = proto.unpack_params(x, system)
        compiler = DenseAtomicCompiler()
        ir = compiler.compile(system, proto, params)

        for term in ir.drive_terms:
            assert callable(term.coefficient)
            # Should return a complex value at t=0
            val = term.coefficient(0)
            assert isinstance(val, (complex, float, np.complexfloating, np.floating))

    def test_amplitude_scale(self):
        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]
        params = proto.unpack_params(x, system)

        compiler_1 = DenseAtomicCompiler(amplitude_scale=1.0)
        compiler_2 = DenseAtomicCompiler(amplitude_scale=0.5)
        ir1 = compiler_1.compile(system, proto, params)
        ir2 = compiler_2.compile(system, proto, params)

        t_mid = params["t_gate"] / 2
        c1_420 = ir1.drive_terms[0].coefficient(t_mid)
        c2_420 = ir2.drive_terms[0].coefficient(t_mid)
        # drive_420 scales linearly
        np.testing.assert_allclose(abs(c2_420), 0.5 * abs(c1_420), rtol=1e-10)


@pytest.mark.slow
class TestCompilerSolverConsistency:
    """Verify compile + evolve_ir matches legacy solve_gate()."""

    def test_to_protocol_final_state_matches(self):
        """TO protocol: new path vs old path produce same final state."""
        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]

        # Initial state: |01>
        psi0 = np.zeros(49, dtype=complex)
        psi0[1] = 1.0

        # Old path
        psi_old = solve_gate(system, proto, x, psi0)

        # New path
        params = proto.unpack_params(x, system)
        compiler = DenseAtomicCompiler()
        ir = compiler.compile(system, proto, params)
        psi_new = evolve_ir(ir, psi0, params["t_gate"])

        np.testing.assert_allclose(psi_new, psi_old, atol=1e-6)

    def test_sweep_analog_final_state_matches(self):
        """Sweep protocol (3-level analog): new path vs old path."""
        system = create_analog_system()
        proto = SweepProtocol()
        x = [-5.0, 5.0, 1.5]

        # Initial state: |00> in 3-level basis (9-dim)
        psi0 = np.zeros(9, dtype=complex)
        psi0[0] = 1.0

        # Old path
        psi_old = solve_gate(system, proto, x, psi0)

        # New path
        params = proto.unpack_params(x, system)
        compiler = DenseAtomicCompiler()
        ir = compiler.compile(system, proto, params)
        psi_new = evolve_ir(ir, psi0, params["t_gate"])

        np.testing.assert_allclose(psi_new, psi_old, atol=1e-6)


class TestSparseLatticeCompiler:
    """Test SparseLatticeCompiler produces correct IR."""

    def test_compile_lattice_sweep(self):
        system = create_lattice_system(Lx=2, Ly=2)
        proto = SweepProtocol()
        x = [-5.0, 5.0, 1.5]
        params = proto.unpack_params(x, system)
        compiler = SparseLatticeCompiler()
        ir = compiler.compile(system, proto, params)

        assert ir.dim == 2 ** system.N
        assert ir.is_sparse
        assert len(ir.static_terms) == 1  # H_static (VdW + pinning)
        assert len(ir.drive_terms) == 2  # global_X, global_n
        assert ir.metadata["N"] == system.N

    def test_compile_with_pinning(self):
        system = create_lattice_system(Lx=2, Ly=2)
        proto = SweepProtocol(addressing={0: 5.0, 1: 3.0})
        x = [-5.0, 5.0, 1.5]
        params = proto.unpack_params(x, system)
        compiler = SparseLatticeCompiler()
        ir = compiler.compile(system, proto, params)

        # H_static should include pinning (differ from bare H_vdw)
        H_static_diag = ir.static_terms[0].operator.diagonal()
        H_vdw_diag = system.H_vdw.diagonal()
        assert not np.allclose(H_static_diag, H_vdw_diag)

    def test_drive_coefficients_at_midpoint(self):
        system = create_lattice_system(Lx=2, Ly=2)
        proto = SweepProtocol()
        x = [-5.0, 5.0, 1.5]
        params = proto.unpack_params(x, system)
        compiler = SparseLatticeCompiler()
        ir = compiler.compile(system, proto, params)

        t_mid = params["t_gate"] / 2
        global_X_coeff = ir.drive_terms[0].coefficient(t_mid)
        global_n_coeff = ir.drive_terms[1].coefficient(t_mid)
        # At midpoint, Delta should be ~0 (midway between -5 and +5)
        assert abs(global_n_coeff) < 0.5  # close to zero
        # Omega should be at full value (past ramp)
        assert global_X_coeff > 0
