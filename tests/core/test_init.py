"""Tests for ryd_gate package initialization."""


class TestPackageImports:
    """Tests for package-level imports."""

    def test_version_defined(self):
        """Package should have __version__."""
        import ryd_gate
        assert hasattr(ryd_gate, "__version__")
        assert isinstance(ryd_gate.__version__, str)

    def test_product_layer_exports(self):
        """Lattice geometry stays top-level; validation lives in core.serialization."""
        from ryd_gate import Register, RegisterLayout
        from ryd_gate.core.serialization import ValidationIssue, raise_for_errors

        assert all(
            item is not None
            for item in (Register, RegisterLayout, ValidationIssue, raise_for_errors)
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

    def test_demoted_top_level_exports(self):
        """Symbols moved to submodules are no longer top-level."""
        import pytest

        import ryd_gate

        for name in (
            "TOProtocol",
            "ARProtocol",
            "CZProtocol",
            "CZGateReport",
            "cz_gate_report",
            "average_gate_infidelity",
            "error_budget",
            "Protocol",
            "DigitalAnalogProtocol",
            "HamiltonianIR",
            "HamiltonianTerm",
            "compile_hamiltonian_ir",
            "SystemModel",
            "BasisSpec",
            "BlockRegistry",
            "ObservableRegistry",
            "Observable",
            "TransitionSpec",
            "ValidationIssue",
            "raise_for_errors",
            "AddressingEvaluator",
            "compute_shift_scatter",
            "TFIMRydbergControls",
            "tfim_to_rydberg_controls",
            "interaction_longitudinal_shifts",
        ):
            assert name not in ryd_gate.__all__
            with pytest.raises(AttributeError):
                getattr(ryd_gate, name)

    def test_all_exports_match(self):
        """__all__ should list exactly the minimal top-level surface."""
        import ryd_gate

        expected = {
            # Systems & geometry
            "RydbergSystem",
            "Register",
            "RegisterLayout",
            "InteractionSpec",
            "LevelStructureSpec",
            "level_structure",
            "DEFAULT_C6",
            # Noise layer
            "NoiseModel",
            "configure_monte_carlo_runner",
            # Lattice-dynamics protocols
            "SweepProtocol",
            "TFIMQuenchProtocol",
            "TFIMAnnealProtocol",
            # Simulation
            "simulate",
            "EvolutionResult",
        }
        assert set(ryd_gate.__all__) == expected

    def test_all_exports_defined(self):
        """All items in __all__ should be defined."""
        import ryd_gate

        for name in ryd_gate.__all__:
            assert hasattr(ryd_gate, name)

    def test_architecture_types_importable_from_submodules(self):
        """Core/IR types are importable from their owning submodules."""
        from ryd_gate import EvolutionResult, RydbergSystem
        from ryd_gate.core import BasisSpec, BlockRegistry, ObservableRegistry
        from ryd_gate.core.model import SystemModel
        from ryd_gate.ir import HamiltonianIR, HamiltonianTerm, compile_hamiltonian_ir

        assert all(
            item is not None
            for item in (
                EvolutionResult,
                RydbergSystem,
                BasisSpec,
                BlockRegistry,
                ObservableRegistry,
                SystemModel,
                HamiltonianIR,
                HamiltonianTerm,
                compile_hamiltonian_ir,
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
