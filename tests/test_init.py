"""Tests for ryd_gate package initialization."""

import pytest


class TestPackageImports:
    """Tests for package-level imports."""

    def test_version_defined(self):
        """Package should have __version__."""
        import ryd_gate
        assert hasattr(ryd_gate, "__version__")
        assert isinstance(ryd_gate.__version__, str)

    def test_blackman_exports(self):
        """Blackman functions should be exported."""
        from ryd_gate import blackman_pulse, blackman_pulse_sqrt, blackman_window

        assert callable(blackman_pulse)
        assert callable(blackman_pulse_sqrt)
        assert callable(blackman_window)

    def test_all_exports_match(self):
        """__all__ should list exactly the expected exports."""
        import ryd_gate

        expected = {
            # Systems
            "RydbergSystem",
            "LevelStructureSpec", "TransitionSpec",
            "InteractionSpec", "DEFAULT_C6", "level_structure",
            "compute_shift_scatter",
            # Protocols
            "Protocol", "TOProtocol", "ARProtocol", "SweepProtocol",
            "DigitalAnalogProtocol", "Segment",
            # Simulation
            "simulate", "EvolutionResult", "SolverBackend",
            "HamiltonianIR", "HamiltonianTerm",
            "ExactSparseCompiler", "compile_expm_ir",
            # Analysis
            "average_gate_infidelity", "error_budget", "AddressingEvaluator",
            # Pulse utilities
            "blackman_pulse", "blackman_pulse_sqrt", "blackman_window",
            # Advanced primitives
            "SystemModel", "BasisSpec", "BlockRegistry",
            "ObservableRegistry", "Observable",
        }
        assert set(ryd_gate.__all__) == expected

    def test_all_exports_defined(self):
        """All items in __all__ should be defined."""
        import ryd_gate

        for name in ryd_gate.__all__:
            assert hasattr(ryd_gate, name)

    def test_direct_import_ideal_cz(self):
        """Should be able to import CZGateSimulator directly (with deprecation warning)."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ryd_gate.legacy.ideal_cz import CZGateSimulator

        assert CZGateSimulator is not None

    def test_legacy_import(self):
        """Should be able to import from legacy module without deprecation."""
        from ryd_gate.legacy.ideal_cz import CZGateSimulator

        assert CZGateSimulator is not None

    def test_new_architecture_exports(self):
        """New architecture types should be importable from top level."""
        from ryd_gate import (
            SystemModel, BasisSpec, BlockRegistry, ObservableRegistry,
            HamiltonianIR, HamiltonianTerm, SolverBackend, EvolutionResult,
            ExactSparseCompiler, compile_expm_ir, simulate, RydbergSystem,
        )
        assert SystemModel is not None
        assert RydbergSystem is not None
        assert simulate is not None
        assert ExactSparseCompiler is not None
        assert compile_expm_ir is not None

    def test_new_package_namespaces(self):
        """Model, IR, backend, and simulate namespaces should be importable."""
        from ryd_gate.model import RydbergSystem, BasisSpec
        from ryd_gate.ir import HamiltonianIR, HamiltonianTerm
        from ryd_gate.backends import DenseODEBackend, SparseExpmBackend
        from ryd_gate.simulate import simulate

        assert RydbergSystem is not None
        assert BasisSpec is not None
        assert HamiltonianIR is not None
        assert HamiltonianTerm is not None
        assert DenseODEBackend is not None
        assert SparseExpmBackend is not None
        assert simulate is not None
