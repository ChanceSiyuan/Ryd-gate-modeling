"""Tests for the solver backend layer: SolverBackend ABC, DenseODEBackend,
SparseExpmBackend, simulate() dispatch, and EvolutionResult.

Verifies that the new backend classes produce correct results and that
the simulate() convenience function auto-selects appropriate backends.
"""

import numpy as np
import pytest

from ryd_gate.compilers.dense_atomic import DenseAtomicCompiler
from ryd_gate.compilers.sparse_lattice import SparseLatticeCompiler
from ryd_gate.core.atomic_system import (
    create_lattice_system,
    create_our_system,
)
from ryd_gate.protocols.gate_cz_to import TOProtocol
from ryd_gate.protocols.sweep import SweepProtocol
from ryd_gate.solvers.base import EvolutionResult, SolverBackend
from ryd_gate.solvers.dense_ode import DenseODEBackend
from ryd_gate.solvers.dispatch import simulate
from ryd_gate.solvers.sparse_expm import SparseExpmBackend


# ---------------------------------------------------------------------------
# EvolutionResult dataclass
# ---------------------------------------------------------------------------


class TestEvolutionResult:
    """Test EvolutionResult dataclass creation and fields."""

    def test_minimal_creation(self):
        psi = np.array([1.0, 0.0], dtype=complex)
        result = EvolutionResult(psi_final=psi)
        assert result.psi_final is psi
        assert result.times is None
        assert result.states is None
        assert result.metadata == {}

    def test_full_creation(self):
        psi = np.array([0.0, 1.0], dtype=complex)
        times = np.linspace(0, 1, 10)
        states = np.random.randn(2, 10)
        meta = {"t_gate": 1e-6, "param_set": "our"}
        result = EvolutionResult(
            psi_final=psi,
            times=times,
            states=states,
            metadata=meta,
        )
        assert result.psi_final is psi
        np.testing.assert_array_equal(result.times, times)
        assert result.states is states
        assert result.metadata["t_gate"] == 1e-6
        assert result.metadata["param_set"] == "our"

    def test_metadata_default_is_independent(self):
        """Each EvolutionResult should get its own metadata dict."""
        r1 = EvolutionResult(psi_final=np.zeros(2))
        r2 = EvolutionResult(psi_final=np.zeros(2))
        r1.metadata["key"] = "value"
        assert "key" not in r2.metadata


# ---------------------------------------------------------------------------
# SolverBackend ABC
# ---------------------------------------------------------------------------


class TestSolverBackendABC:
    """Test that SolverBackend is a proper ABC."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            SolverBackend()

    def test_subclass_must_implement_evolve(self):
        class IncompleteSolver(SolverBackend):
            pass

        with pytest.raises(TypeError):
            IncompleteSolver()

    def test_concrete_subclass_works(self):
        class TrivialSolver(SolverBackend):
            def evolve(self, ir, psi0, t_gate, t_eval=None):
                return EvolutionResult(psi_final=psi0)

        solver = TrivialSolver()
        result = solver.evolve(None, np.zeros(2), 1.0)
        assert isinstance(result, EvolutionResult)


# ---------------------------------------------------------------------------
# DenseODEBackend
# ---------------------------------------------------------------------------


class TestDenseODEBackend:
    """Test DenseODEBackend with create_our_system() + TOProtocol."""

    def test_evolve_01_norm_preserved(self):
        """Evolve |01> under TO protocol; norm should be preserved."""
        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]
        params = proto.unpack_params(x, system)

        compiler = DenseAtomicCompiler()
        ir = compiler.compile(system, proto, params)

        # Initial state |01>
        psi0 = np.zeros(49, dtype=complex)
        psi0[1] = 1.0

        backend = DenseODEBackend()
        result = backend.evolve(ir, psi0, params["t_gate"])

        assert isinstance(result, EvolutionResult)
        assert result.psi_final.shape == (49,)
        np.testing.assert_allclose(np.linalg.norm(result.psi_final), 1.0, atol=1e-6)
        assert result.times is None
        assert result.states is None
        assert "t_gate" in result.metadata

    def test_evolve_with_t_eval(self):
        """Evolve with t_eval returns times and states."""
        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]
        params = proto.unpack_params(x, system)

        compiler = DenseAtomicCompiler()
        ir = compiler.compile(system, proto, params)

        psi0 = np.zeros(49, dtype=complex)
        psi0[1] = 1.0

        t_eval = np.linspace(0, params["t_gate"], 50)
        backend = DenseODEBackend()
        result = backend.evolve(ir, psi0, params["t_gate"], t_eval=t_eval)

        assert result.times is not None
        assert result.states is not None
        assert len(result.times) == 50
        # states shape: (49, 50) -- columns are state vectors at each time
        assert result.states.shape == (49, 50)
        # Final state matches psi_final
        np.testing.assert_allclose(result.states[:, -1], result.psi_final, atol=1e-12)

    def test_identity_evolution(self):
        """Zero Hamiltonian should leave state unchanged."""
        from ryd_gate.compilers.ir import HamiltonianIR, HamiltonianTerm

        dim = 4
        ir = HamiltonianIR(
            static_terms=[HamiltonianTerm("zero", np.zeros((dim, dim), dtype=complex), 1.0)],
            drive_terms=[],
            dim=dim,
            metadata={"t_gate": 1.0},
        )
        psi0 = np.array([1, 0, 0, 0], dtype=complex)
        backend = DenseODEBackend()
        result = backend.evolve(ir, psi0, t_gate=1.0)

        np.testing.assert_allclose(result.psi_final, psi0, atol=1e-10)


@pytest.mark.slow
class TestDenseODEBackendConsistency:
    """Verify DenseODEBackend matches legacy solve_gate()."""

    def test_matches_solve_gate(self):
        """DenseODEBackend final state should match solve_gate() result."""
        from ryd_gate.solvers.schrodinger import solve_gate

        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]

        psi0 = np.zeros(49, dtype=complex)
        psi0[1] = 1.0

        # Legacy path
        psi_legacy = solve_gate(system, proto, x, psi0)

        # New backend path
        params = proto.unpack_params(x, system)
        compiler = DenseAtomicCompiler()
        ir = compiler.compile(system, proto, params)
        backend = DenseODEBackend()
        result = backend.evolve(ir, psi0, params["t_gate"])

        np.testing.assert_allclose(result.psi_final, psi_legacy, atol=1e-6)


# ---------------------------------------------------------------------------
# SparseExpmBackend
# ---------------------------------------------------------------------------


class TestSparseExpmBackend:
    """Test SparseExpmBackend with LatticeSystem + SweepProtocol."""

    def test_evolve_lattice_norm_preserved(self):
        """Evolve ground state on 2x2 lattice; norm should be preserved."""
        system = create_lattice_system(Lx=2, Ly=2)
        proto = SweepProtocol()
        x = [-5.0, 5.0, 1.5]
        params = proto.unpack_params(x, system)

        compiler = SparseLatticeCompiler()
        ir = compiler.compile(system, proto, params)

        # Initial state: all-ground |0000>
        dim = 2 ** system.N
        psi0 = np.zeros(dim, dtype=complex)
        psi0[0] = 1.0

        backend = SparseExpmBackend(n_steps=100)
        result = backend.evolve(ir, psi0, params["t_gate"])

        assert isinstance(result, EvolutionResult)
        assert result.psi_final.shape == (dim,)
        np.testing.assert_allclose(np.linalg.norm(result.psi_final), 1.0, atol=1e-10)
        assert result.times is None
        assert result.states is None

    def test_evolve_with_t_eval(self):
        """Evolve with t_eval stores all intermediate states."""
        system = create_lattice_system(Lx=2, Ly=2)
        proto = SweepProtocol()
        x = [-5.0, 5.0, 1.5]
        params = proto.unpack_params(x, system)

        compiler = SparseLatticeCompiler()
        ir = compiler.compile(system, proto, params)

        dim = 2 ** system.N
        psi0 = np.zeros(dim, dtype=complex)
        psi0[0] = 1.0

        n_steps = 50
        # t_eval just triggers storage; actual stored times are per-step
        t_eval = np.linspace(0, params["t_gate"], 10)
        backend = SparseExpmBackend(n_steps=n_steps)
        result = backend.evolve(ir, psi0, params["t_gate"], t_eval=t_eval)

        assert result.times is not None
        assert result.states is not None
        assert len(result.times) == n_steps
        assert result.states.shape == (n_steps, dim)
        # Each stored state should be normalised
        for i in range(result.states.shape[0]):
            np.testing.assert_allclose(
                np.linalg.norm(result.states[i]), 1.0, atol=1e-10
            )

    def test_state_evolves_nontrivially(self):
        """With nonzero drive, final state should differ from initial."""
        system = create_lattice_system(Lx=2, Ly=2)
        proto = SweepProtocol()
        x = [-5.0, 5.0, 1.5]
        params = proto.unpack_params(x, system)

        compiler = SparseLatticeCompiler()
        ir = compiler.compile(system, proto, params)

        dim = 2 ** system.N
        psi0 = np.zeros(dim, dtype=complex)
        psi0[0] = 1.0

        backend = SparseExpmBackend(n_steps=100)
        result = backend.evolve(ir, psi0, params["t_gate"])

        # State should have evolved away from |0000>
        overlap = abs(np.dot(psi0.conj(), result.psi_final)) ** 2
        assert overlap < 0.99


# ---------------------------------------------------------------------------
# simulate() dispatch
# ---------------------------------------------------------------------------


class TestSimulate:
    """Test the simulate() convenience function."""

    def test_simulate_dense_atomic(self):
        """simulate() with AtomicSystem should auto-select DenseODEBackend."""
        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]

        psi0 = np.zeros(49, dtype=complex)
        psi0[1] = 1.0

        result = simulate(system, proto, x, psi0)

        assert isinstance(result, EvolutionResult)
        assert result.psi_final.shape == (49,)
        np.testing.assert_allclose(np.linalg.norm(result.psi_final), 1.0, atol=1e-6)

    def test_simulate_sparse_lattice(self):
        """simulate() with LatticeSystem should auto-select SparseExpmBackend."""
        system = create_lattice_system(Lx=2, Ly=2)
        proto = SweepProtocol()
        x = [-5.0, 5.0, 1.5]

        dim = 2 ** system.N
        psi0 = np.zeros(dim, dtype=complex)
        psi0[0] = 1.0

        result = simulate(system, proto, x, psi0)

        assert isinstance(result, EvolutionResult)
        assert result.psi_final.shape == (dim,)
        np.testing.assert_allclose(np.linalg.norm(result.psi_final), 1.0, atol=1e-10)

    def test_simulate_with_explicit_backend(self):
        """simulate() should use the explicitly provided backend."""
        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]

        psi0 = np.zeros(49, dtype=complex)
        psi0[1] = 1.0

        backend = DenseODEBackend(rtol=1e-6, atol=1e-10)
        result = simulate(system, proto, x, psi0, backend=backend)

        assert isinstance(result, EvolutionResult)
        np.testing.assert_allclose(np.linalg.norm(result.psi_final), 1.0, atol=1e-5)

    def test_simulate_with_t_eval(self):
        """simulate() should pass t_eval through to the backend."""
        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]

        psi0 = np.zeros(49, dtype=complex)
        psi0[1] = 1.0

        params = proto.unpack_params(x, system)
        t_eval = np.linspace(0, params["t_gate"], 20)

        result = simulate(system, proto, x, psi0, t_eval=t_eval)

        assert result.times is not None
        assert result.states is not None

    def test_simulate_with_explicit_compiler(self):
        """simulate() should use the explicitly provided compiler."""
        system = create_our_system()
        proto = TOProtocol()
        x = [1.0, 2.0, 0.5, 0.1, 0.0, 2.0]

        psi0 = np.zeros(49, dtype=complex)
        psi0[1] = 1.0

        compiler = DenseAtomicCompiler(amplitude_scale=0.8)
        result = simulate(system, proto, x, psi0, compiler=compiler)

        assert isinstance(result, EvolutionResult)
        assert result.metadata.get("amplitude_scale") == 0.8
