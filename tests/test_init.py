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

    def test_cz_gate_simulator_export(self):
        """CZGateSimulator should be exported."""
        from ryd_gate import CZGateSimulator

        assert CZGateSimulator is not None

    def test_monte_carlo_result_export(self):
        """MonteCarloResult should be exported."""
        from ryd_gate import MonteCarloResult

        assert MonteCarloResult is not None

    def test_all_exports_match(self):
        """__all__ should list exactly the expected exports."""
        import ryd_gate

        expected = {
            # Legacy (backward-compatible)
            "CZGateSimulator", "MonteCarloResult",
            "AtomicSystem", "Protocol",
            "create_our_system", "create_lukin_system",
            "create_analog_system", "compatible_protocols",
            "PROTOCOL_REGISTRY", "compute_shift_scatter",
            "TOProtocol", "ARProtocol",
            "SweepProtocol",
            "LatticeSystem", "create_lattice_system",
            "AddressingEvaluator",
            "blackman_pulse", "blackman_pulse_sqrt", "blackman_window",
            # New architecture
            "SystemModel", "BasisSpec", "BlockRegistry",
            "ObservableRegistry", "Observable",
            "HamiltonianIR", "HamiltonianTerm",
            "SolverBackend", "EvolutionResult",
            "simulate",
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
            from ryd_gate.ideal_cz import CZGateSimulator

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
            simulate,
        )
        assert SystemModel is not None
        assert simulate is not None
