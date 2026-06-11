"""Tests for ryd_gate package initialization."""


class TestPackageImports:
    """Tests for package-level imports."""

    def test_version_defined(self):
        """Package should have __version__."""
        import ryd_gate
        assert hasattr(ryd_gate, "__version__")
        assert isinstance(ryd_gate.__version__, str)

    def test_product_layer_exports(self):
        """Stage 1 product objects should be exported at top level."""
        from ryd_gate import (
            ChannelSpec,
            DeviceSpec,
            Pulse,
            Register,
            RegisterLayout,
            ValidationIssue,
            Waveform,
            raise_for_errors,
        )

        assert all(
            item is not None
            for item in (
                Register, RegisterLayout, DeviceSpec, ChannelSpec,
                ValidationIssue, raise_for_errors, Waveform, Pulse,
            )
        )

    def test_removed_top_level_exports(self):
        """Kernel blackman helpers are soft-closed: not top-level API."""
        import pytest

        with pytest.raises(ImportError):
            from ryd_gate import blackman_pulse  # noqa: F401
        with pytest.raises(ImportError):
            from ryd_gate import blackman_window  # noqa: F401
        with pytest.raises(ImportError):
            from ryd_gate import blackman_pulse_sqrt  # noqa: F401

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
            "Protocol", "TOProtocol", "ARProtocol", "DoubleARPProtocol", "SweepProtocol",
            "TFIMAnnealProtocol", "TFIMQuenchProtocol", "TFIMRydbergControls",
            "tfim_to_rydberg_controls", "interaction_longitudinal_shifts",
            "DigitalAnalogProtocol",
            # IR
            "EvolutionResult", "HamiltonianIR", "HamiltonianTerm", "compile_hamiltonian_ir",
            # Analysis
            "average_gate_infidelity", "error_budget", "AddressingEvaluator",
            # Simulation
            "simulate",
            # Product data layer (Stage 1)
            "Register", "RegisterLayout", "DeviceSpec", "ChannelSpec",
            "ValidationIssue", "raise_for_errors", "Waveform", "Pulse",
            # Sequence layer (Stage 2)
            "Sequence", "PulseOp", "DelayOp", "MeasureOp", "SequenceProtocol",
            "simulate_sequence", "SimulationResult", "ExactStateHandle",
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
            from ryd_gate.backends.exact.legacy.ideal_cz import CZGateSimulator

        assert CZGateSimulator is not None

    def test_legacy_import(self):
        """Should be able to import from legacy module without deprecation."""
        from ryd_gate.backends.exact.legacy.ideal_cz import CZGateSimulator

        assert CZGateSimulator is not None

    def test_new_architecture_exports(self):
        """New architecture types should be importable from top level."""
        from ryd_gate import (
            BasisSpec,
            BlockRegistry,
            EvolutionResult,
            HamiltonianIR,
            HamiltonianTerm,
            ObservableRegistry,
            RydbergSystem,
            SystemModel,
            compile_hamiltonian_ir,
        )
        from ryd_gate.backends.exact import ExactSparseCompiler, SolverBackend, compile_expm_ir, simulate

        assert all(
            item is not None
            for item in (
                BasisSpec,
                BlockRegistry,
                EvolutionResult,
                HamiltonianIR,
                HamiltonianTerm,
                ObservableRegistry,
                RydbergSystem,
                SystemModel,
                compile_hamiltonian_ir,
                ExactSparseCompiler,
                SolverBackend,
                compile_expm_ir,
                simulate,
            )
        )

    def test_new_package_namespaces(self):
        """Core, IR, backend, and simulate namespaces should be importable."""
        from ryd_gate.backends.exact import DenseODEBackend, SparseExpmBackend, simulate
        from ryd_gate.core import BasisSpec, RydbergSystem
        from ryd_gate.ir import HamiltonianIR, HamiltonianTerm, compile_hamiltonian_ir

        assert all(
            item is not None
            for item in (
                BasisSpec,
                DenseODEBackend,
                HamiltonianIR,
                HamiltonianTerm,
                compile_hamiltonian_ir,
                RydbergSystem,
                SparseExpmBackend,
                simulate,
            )
        )
