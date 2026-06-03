"""Tests for the pluggable GPU TN backend adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ryd_gate.backends.base import EvolutionResult
from ryd_gate.protocols.sweep import SweepProtocol
from ryd_gate.tn import gpu_backends
from ryd_gate.tn.lattice_spec import create_tn_lattice_spec
from ryd_gate.tn.simulate import simulate_tn


@pytest.fixture
def spec_1x1():
    return create_tn_lattice_spec(Lx=1, Ly=1)


def test_invalid_tn_backend_raises(spec_1x1):
    with pytest.raises(ValueError, match="Unknown TN backend"):
        simulate_tn(spec_1x1, SweepProtocol(), [0.0, 0.0, 1.0], backend="bad")


def test_gputn_rejects_dmrg_before_dependency_check(spec_1x1):
    with pytest.raises(ValueError, match="method='tdvp'"):
        simulate_tn(
            spec_1x1,
            SweepProtocol(),
            [0.0, 0.0, 1.0],
            method="dmrg",
            backend="gputn",
        )


def test_gputn_dependency_error_names_missing_module(monkeypatch, spec_1x1):
    def fake_find_spec(module_name):
        if module_name == "cupy":
            return None
        return object()

    monkeypatch.setattr(gpu_backends.importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(gpu_backends.GPUTNDependencyError, match="cupy"):
        simulate_tn(
            spec_1x1,
            SweepProtocol(),
            [0.0, 0.0, 1.0],
            backend="gputn",
        )


def test_gputn_dispatches_to_configured_engine(monkeypatch, spec_1x1):
    monkeypatch.setattr(
        gpu_backends,
        "_require_gputn_dependencies",
        lambda **_: SimpleNamespace(cupy=object(), cuquantum=object(), n_devices=1),
    )

    class FakeEngine:
        def evolve(self, spec, protocol, params, psi0, **kwargs):
            assert spec.N == 1
            assert isinstance(protocol, SweepProtocol)
            assert params["t_gate"] == 1.0
            assert kwargs["chi_max"] == 7
            assert kwargs["dt"] == 0.25
            return EvolutionResult(
                psi_final=psi0,
                metadata={"method": "tdvp"},
            )

    result = simulate_tn(
        spec_1x1,
        SweepProtocol(),
        [0.0, 0.0, 1.0],
        initial_state="all_ground",
        backend="gputn",
        backend_options={"engine": FakeEngine(), "chi_max": 7, "dt": 0.25},
    )

    assert result.psi_final == "all_ground"
    assert result.metadata["backend"] == "gputn"
    assert result.metadata["accelerator"] == "cuda"
