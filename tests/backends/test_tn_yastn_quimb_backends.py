import numpy as np
import pytest

from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.simulate import simulate_tn
from ryd_gate.protocols.lattice_dynamics import TFIMQuenchProtocol


def test_yastn_mps_gpu_alias_runs_cpu_smoke():
    pytest.importorskip("yastn")

    spec = create_tn_lattice_spec(1, 2, V_nn=4.0, interaction_mode="nn")
    proto = TFIMQuenchProtocol(hx=0.1, hz=0.0, t_gate=0.05)

    result = simulate_tn(
        spec,
        proto,
        [],
        backend="mps_gpu",
        t_eval=np.array([0.0, 0.05]),
        observables=["sigma_z", "czz_centerline"],
        backend_options={"chi_max": 2, "dt": 0.05, "use_cuda": False},
    )

    assert result.metadata["backend"] == "yastn_mps"
    assert result.metadata["method"] == "mps_tdvp"
    assert result.metadata["engine_package"] == "yastn"
    assert result.metadata["gpu"] is False
    assert result.metadata["obs"]["sigma_z"].shape == (2, 2)
    assert result.metadata["obs"]["czz_centerline"].shape == (2, 1)


def test_2dtn_yastn_package_runs_concrete_smoke():
    pytest.importorskip("yastn")

    spec = create_tn_lattice_spec(1, 2, V_nn=4.0, interaction_mode="nn")
    proto = TFIMQuenchProtocol(hx=0.1, hz=0.0, t_gate=0.05)

    result = simulate_tn(
        spec,
        proto,
        [],
        backend="2dtn",
        t_eval=np.array([0.0, 0.05]),
        observables=["sigma_z", "czz_centerline"],
        backend_options={
            "engine_package": "yastn",
            "chi_max": 2,
            "dt": 0.05,
            "use_cuda": False,
            "measurement_environment": "ctm",
        },
    )

    assert result.metadata["backend"] == "2dtn"
    assert result.metadata["method"] == "2dtn_yastn"
    assert result.metadata["engine_package"] == "yastn"
    assert result.metadata["obs"]["sigma_z"].shape == (2, 2)
    assert result.metadata["obs"]["czz_centerline"].shape == (2, 1)


def test_2dtn_quimb_package_runs_concrete_smoke():
    pytest.importorskip("quimb")
    pytest.importorskip("networkx")

    spec = create_tn_lattice_spec(1, 2, V_nn=4.0, interaction_mode="nn")
    proto = TFIMQuenchProtocol(hx=0.1, hz=0.0, t_gate=0.05)

    result = simulate_tn(
        spec,
        proto,
        [],
        backend="2dtn",
        t_eval=np.array([0.0, 0.05]),
        observables=["sigma_z", "czz_centerline"],
        backend_options={
            "engine_package": "quimb",
            "chi_max": 2,
            "dt": 0.05,
            "use_cuda": False,
            "algorithm": "simple_update",
        },
    )

    assert result.metadata["backend"] == "2dtn"
    assert result.metadata["method"] == "2dtn_quimb"
    assert result.metadata["engine_package"] == "quimb"
    assert result.metadata["obs"]["sigma_z"].shape == (2, 2)
    assert result.metadata["obs"]["czz_centerline"].shape == (2, 1)
