"""Tests for the pluggable GPU TN backend adapter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ryd_gate.backends.gputn import backend as gpu_backends
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.simulate import simulate_tn
from ryd_gate.ir.evolution import EvolutionResult
from ryd_gate.protocols.sweep import SweepProtocol


@pytest.fixture
def spec_1x1():
    return create_tn_lattice_spec(Lx=1, Ly=1)


def _sweep(t_gate=1.0, omega=1.0, delta=0.0):
    return SweepProtocol(
        t_gate=t_gate,
        omega_half_fn=lambda t: 0.5 * omega,
        delta_fn=lambda t: delta,
    )


def test_invalid_tn_backend_raises(spec_1x1):
    with pytest.raises(ValueError, match="Unknown TN backend"):
        simulate_tn(spec_1x1, _sweep(), [], backend="bad")


def test_gputn_rejects_dmrg_before_dependency_check(spec_1x1):
    with pytest.raises(ValueError, match="method='tdvp'"):
        simulate_tn(
            spec_1x1,
            _sweep(),
            [],
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
            _sweep(),
            [],
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
        _sweep(),
        [],
        initial_state="all_ground",
        backend="gputn",
        backend_options={"engine": FakeEngine(), "chi_max": 7, "dt": 0.25},
    )

    assert result.psi_final == "all_ground"
    assert result.metadata["backend"] == "gputn"
    assert result.metadata["accelerator"] == "cuda"


def test_gputn_builtin_statevector_kernel_runs_with_dependency_injection(monkeypatch, spec_1x1):
    monkeypatch.setattr(
        gpu_backends,
        "_require_gputn_dependencies",
        lambda **_: SimpleNamespace(cupy=np, cuquantum=SimpleNamespace(), n_devices=None),
    )

    result = simulate_tn(
        spec_1x1,
        _sweep(t_gate=0.2),
        [],
        initial_state="all_ground",
        backend="gputn",
        t_eval=np.array([0.0, 0.2]),
        observables=["n_mean", "sigma_z"],
        backend_options={
            "kernel": "statevector",
            "require_gpu": False,
            "dt": 0.1,
        },
    )

    assert result.metadata["backend"] == "gputn"
    assert result.metadata["accelerator"] == "cuda"
    assert result.metadata["engine_package"] == "gputn"
    assert result.metadata["kernel"] == "statevector_trotter"
    assert result.metadata["obs"]["n_mean"].shape == (2,)
    assert result.metadata["obs"]["sigma_z"].shape == (2, 1)
    np.testing.assert_allclose(np.linalg.norm(result.psi_final.reshape(-1)), 1.0, atol=1e-12)


def test_gputn_builtin_cutensornet_kernel_uses_network_state_api(monkeypatch, spec_1x1):
    class FakeNetworkState:
        def __init__(self, extents, **kwargs):
            del kwargs
            self.psi = np.zeros(int(np.prod(extents)), dtype=complex)
            self.psi[0] = 1.0
            self.applied = False

        def apply_tensor_operator(self, modes, tensor, unitary=True):
            del unitary
            assert tuple(modes) == (0,)
            self.psi = np.asarray(tensor) @ self.psi
            self.applied = True

        def compute_reduced_density_matrix(self, modes):
            assert tuple(modes) == (0,)
            if not self.applied:
                raise RuntimeError("RDM cannot be computed before an operator has been applied.")
            return np.outer(self.psi, self.psi.conjugate())

        def compute_state_vector(self):
            return self.psi

    class FakeMPSConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_cuquantum = SimpleNamespace(
        tensornet=SimpleNamespace(
            experimental=SimpleNamespace(
                NetworkState=FakeNetworkState,
                MPSConfig=FakeMPSConfig,
            )
        )
    )
    monkeypatch.setattr(
        gpu_backends,
        "_require_gputn_dependencies",
        lambda **_: SimpleNamespace(cupy=np, cuquantum=fake_cuquantum, n_devices=None),
    )

    result = simulate_tn(
        spec_1x1,
        _sweep(t_gate=0.2),
        [],
        initial_state="all_ground",
        backend="gputn",
        t_eval=np.array([0.0, 0.2]),
        observables=["n_mean"],
        backend_options={
            "kernel": "cutensornet_mps",
            "require_gpu": False,
            "return_state_vector": True,
            "dt": 0.1,
        },
    )

    assert result.metadata["kernel"] == "cutensornet_mps_trotter"
    assert result.psi_final.shape == (2,)
    assert result.metadata["obs"]["n_mean"].shape == (2,)
    assert result.metadata["obs"]["n_mean"][0] == 0.0
