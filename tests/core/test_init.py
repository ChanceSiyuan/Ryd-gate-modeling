"""Tests for ryd_gate package initialization."""

TOP_LEVEL_API = [
    "Register",
    "RydbergSystem",
    "InteractionSpec",
    "level_structure",
    "NoiseModel",
    "EvolutionResult",
    "EnsembleResult",
    "simulate",
    "simulate_ensemble",
    "SweepProtocol",
    "TFIMQuenchProtocol",
    "TFIMAnnealProtocol",
]

PROTOCOLS_API = [
    "CZProtocol",
    "TOProtocol",
    "ARProtocol",
    "Direct297PiProtocol",
    "Direct297CZProtocol",
    "Direct297TOProtocol",
    "SweepProtocol",
    "TFIMQuenchProtocol",
    "TFIMAnnealProtocol",
    "DigitalAnalogProtocol",
]


class TestPackageImports:
    """Tests for package-level imports."""

    def test_version_defined(self):
        """Package should have __version__."""
        import ryd_gate
        assert hasattr(ryd_gate, "__version__")
        assert isinstance(ryd_gate.__version__, str)

    def test_product_layer_exports(self):
        """Geometry, interaction, and level-structure entry points are top-level."""
        from ryd_gate import InteractionSpec, Register, level_structure

        assert all(
            item is not None for item in (Register, InteractionSpec, level_structure)
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
        """Symbols moved to submodules or deleted are no longer top-level."""
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
            "ObservableRegistry",
            "Observable",
            "DEFAULT_C6",  # lives in ryd_gate.core.level_structures
            "phase_from_chirp",  # lives in ryd_gate.protocols.gate_cz
            "RegisterLayout",  # deleted with the layout/serialization layer
            "LevelStructureSpec",  # deleted: no public level-structure DSL
            "TransitionSpec",
            "ValidationIssue",
            "raise_for_errors",
            "AddressingEvaluator",
            "compute_shift_scatter",
            "TFIMRydbergControls",
            "tfim_to_rydberg_controls",
            "interaction_longitudinal_shifts",
            "configure_monte_carlo_runner",  # deleted with the Monte-Carlo runner layer
        ):
            assert name not in ryd_gate.__all__
            with pytest.raises(AttributeError):
                getattr(ryd_gate, name)

    def test_all_exports_match(self):
        """__all__ is exactly the minimal top-level surface, in API order."""
        import ryd_gate

        assert ryd_gate.__all__ == TOP_LEVEL_API

    def test_all_exports_defined(self):
        """All items in __all__ should be defined."""
        import ryd_gate

        for name in ryd_gate.__all__:
            assert hasattr(ryd_gate, name)

    def test_protocols_namespace_is_user_selectable_protocols(self):
        """ryd_gate.protocols exports exactly the user-selectable protocols."""
        import pytest

        import ryd_gate.protocols as protocols

        assert protocols.__all__ == PROTOCOLS_API
        for name in PROTOCOLS_API:
            assert hasattr(protocols, name)
        # The Protocol ABC and pulse helpers live only in their modules.
        for name in (
            "Protocol",
            "phase_from_chirp",
            "TFIMRydbergControls",
            "tfim_to_rydberg_controls",
            "interaction_longitudinal_shifts",
        ):
            assert name not in protocols.__all__
            with pytest.raises(AttributeError):
                getattr(protocols, name)

        from ryd_gate.protocols.base import Protocol
        from ryd_gate.protocols.gate_cz import phase_from_chirp
        from ryd_gate.protocols.lattice_dynamics import (
            TFIMRydbergControls,
            interaction_longitudinal_shifts,
            tfim_to_rydberg_controls,
        )

        assert all(
            item is not None
            for item in (
                Protocol,
                phase_from_chirp,
                TFIMRydbergControls,
                tfim_to_rydberg_controls,
                interaction_longitudinal_shifts,
            )
        )

    def test_architecture_types_importable_from_submodules(self):
        """Core/IR types are importable from their owning submodules."""
        from ryd_gate import EvolutionResult, RydbergSystem
        from ryd_gate.core import BasisSpec, ObservableExpr, ObservableFactory
        from ryd_gate.ir import HamiltonianIR, HamiltonianTerm, compile_hamiltonian_ir

        assert all(
            item is not None
            for item in (
                EvolutionResult,
                RydbergSystem,
                BasisSpec,
                ObservableExpr,
                ObservableFactory,
                HamiltonianIR,
                HamiltonianTerm,
                compile_hamiltonian_ir,
            )
        )

    def test_new_package_namespaces(self):
        """Core, IR, backend, and simulate namespaces should be importable."""
        from ryd_gate.backends.exact import ExactCompiler, ExactODEBackend, simulate
        from ryd_gate.core import BasisSpec, RydbergSystem
        from ryd_gate.ir import HamiltonianIR, HamiltonianTerm, compile_hamiltonian_ir

        assert all(
            item is not None
            for item in (
                BasisSpec,
                ExactCompiler,
                ExactODEBackend,
                HamiltonianIR,
                HamiltonianTerm,
                compile_hamiltonian_ir,
                RydbergSystem,
                simulate,
            )
        )
