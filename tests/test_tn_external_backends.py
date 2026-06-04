from ryd_gate.backends.base import EvolutionResult
from ryd_gate.protocols.lattice_dynamics import TFIMQuenchProtocol
from ryd_gate.tn.external_backends import (
    ExternalSolverDependencyError,
    available_external_solver_packages,
)
from ryd_gate.tn.lattice_spec import create_tn_lattice_spec
from ryd_gate.tn.simulate import simulate_tn


def test_external_ttn_backend_dispatches_to_engine():
    spec = create_tn_lattice_spec(1, 1)
    proto = TFIMQuenchProtocol(hx=1.0, hz=0.0, t_gate=0.25)

    class FakeEngine:
        def evolve(self, payload, initial_state, **kwargs):
            assert payload["method"] == "ttn_tdvp"
            assert payload["lattice"]["N"] == 1
            assert payload["protocol"]["class"] == "TFIMQuenchProtocol"
            assert payload["protocol"]["params"]["t_gate"] == 0.25
            assert payload["observables"] == ["sigma_z"]
            assert initial_state == "all_ground"
            assert kwargs["chi_max"] == 8
            return EvolutionResult(psi_final="external-state", metadata={})

    result = simulate_tn(
        spec,
        proto,
        [],
        backend="ttn",
        observables=["sigma_z"],
        backend_options={"engine": FakeEngine(), "chi_max": 8},
    )

    assert result.psi_final == "external-state"
    assert result.metadata["backend"] == "ttn"
    assert result.metadata["method"] == "ttn_tdvp"


def test_external_2dtn_backend_can_still_select_python_package(monkeypatch):
    spec = create_tn_lattice_spec(1, 1)
    proto = TFIMQuenchProtocol(hx=1.0, t_gate=0.25)

    monkeypatch.setattr(
        "ryd_gate.tn.external_backends.importlib.util.find_spec",
        lambda name: None,
    )

    try:
        simulate_tn(
            spec,
            proto,
            [],
            backend="2dtn",
            backend_options={"engine_package": "yastn"},
        )
    except ExternalSolverDependencyError as exc:
        message = str(exc)
        assert "engine_package='yastn'" in message
        assert "tn-2d" in message
    else:
        raise AssertionError("simulate_tn should report the missing default 2DTN package")


def test_external_backend_rejects_wrong_package():
    spec = create_tn_lattice_spec(1, 1)
    proto = TFIMQuenchProtocol(hx=1.0, t_gate=0.25)

    try:
        simulate_tn(
            spec,
            proto,
            [],
            backend="2dtn",
            backend_options={"engine_package": "pytreenet"},
        )
    except ValueError as exc:
        assert "does not support engine_package" in str(exc)
    else:
        raise AssertionError("2DTN should reject a TTN package")


def test_available_external_solver_packages_lists_expected_roles():
    packages = available_external_solver_packages("nqs")

    assert set(packages) == {"netket", "jvmc"}
    assert packages["netket"].extra == "nqs"
