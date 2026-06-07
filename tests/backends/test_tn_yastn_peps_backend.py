import numpy as np
import pytest

from ryd_gate.backends.peps2d.yastn_backend import build_yastn_peps_payload
from ryd_gate.backends.tenpy_mps.backends import _TNProtocolContext
from ryd_gate.backends.tn_common.compiler import TNEvolutionIR
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.simulate import simulate_tn
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol
from ryd_gate.protocols.lattice_dynamics import TFIMQuenchProtocol


def test_peps_yastn_package_runs_concrete_smoke():
    pytest.importorskip("yastn")

    spec = create_tn_lattice_spec(1, 2, V_nn=4.0, interaction_mode="nn")
    proto = TFIMQuenchProtocol(hx=0.1, hz=0.0, t_gate=0.05)

    result = simulate_tn(
        spec,
        proto,
        [],
        backend="peps",
        t_eval=np.array([0.0, 0.05]),
        observables=["sigma_z", "czz_centerline"],
        backend_options={
            "chi_max": 2,
            "dt": 0.05,
            "use_cuda": False,
            "measurement_environment": "ctm",
        },
    )

    assert result.metadata["backend"] == "peps"
    assert result.metadata["method"] == "peps_yastn"
    assert result.metadata["engine_package"] == "yastn"
    assert result.metadata["obs"]["sigma_z"].shape == (2, 2)
    assert result.metadata["obs"]["czz_centerline"].shape == (2, 1)


def test_peps_yastn_payload_supports_01r_qutrit_profiles():
    spec = create_tn_lattice_spec(1, 2, V_nn=4.0, interaction_mode="nn", level_structure="01r")
    proto = DigitalAnalogProtocol.constant(
        omega_R=[0.2, 0.4],
        omega_hf=[0.6, 0.8],
        delta_R=[0.1, 0.2],
        delta_hf=[0.03, 0.04],
        t_gate=0.1,
    )
    params = proto.unpack_params([], _TNProtocolContext(spec))
    ir = TNEvolutionIR(spec=spec, protocol=proto, params=params, method="peps_yastn")

    payload = build_yastn_peps_payload(
        ir,
        initial_state=["0", "1"],
        t_eval=np.array([0.0, 0.1]),
        observables=["n_0", "n_1", "n_r", "n_mean"],
        dt=0.05,
        chi_max=4,
        svd_min=1e-9,
        use_cuda=False,
    )

    assert payload["method"] == "peps_yastn"
    assert payload["lattice"]["level_structure"] == "01r"
    assert payload["lattice"]["levels"] == ("0", "1", "r")
    assert payload["lattice"]["local_dim"] == 3
    assert payload["initial_labels_1d"] == ["0", "1"]
    assert payload["record_steps"] == [0, 2]
    np.testing.assert_allclose(payload["schedule"][0]["omega_R_1d"], [0.2, 0.4])
    np.testing.assert_allclose(payload["schedule"][0]["omega_hf_1d"], [0.6, 0.8])
    np.testing.assert_allclose(payload["schedule"][0]["delta_R_1d"], [0.1, 0.2])
    np.testing.assert_allclose(payload["schedule"][0]["delta_hf_1d"], [0.03, 0.04])


def test_peps_yastn_package_runs_01r_qutrit_smoke():
    pytest.importorskip("yastn")

    spec = create_tn_lattice_spec(1, 2, V_nn=0.1, interaction_mode="nn", level_structure="01r")
    proto = DigitalAnalogProtocol.constant(
        omega_R=0.2,
        omega_hf=0.1,
        delta_R=0.03,
        delta_hf=-0.02,
        t_gate=0.02,
    )

    result = simulate_tn(
        spec,
        proto,
        [],
        backend="peps",
        initial_state="all_1",
        t_eval=np.array([0.0, 0.02]),
        observables=["n_0", "n_1", "n_r", "n_mean", "sigma_z", "czz"],
        backend_options={
            "chi_max": 3,
            "dt": 0.02,
            "use_cuda": False,
            "measurement_environment": "bp",
        },
    )

    assert result.metadata["backend"] == "peps"
    assert result.metadata["method"] == "peps_yastn"
    assert result.metadata["level_structure"] == "01r"
    assert result.metadata["local_dim"] == 3
    assert result.metadata["obs"]["n_0"].shape == (2, 2)
    assert result.metadata["obs"]["n_1"].shape == (2, 2)
    assert result.metadata["obs"]["n_r"].shape == (2, 2)
    assert result.metadata["obs"]["n_mean"].shape == (2,)
    assert result.metadata["obs"]["czz"].shape == (2, 1)
    np.testing.assert_allclose(result.metadata["obs"]["n_1"][0], [1.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(result.metadata["obs"]["n_r"][0], [0.0, 0.0], atol=1e-10)


def test_peps_rejects_non_yastn_engine_package():
    spec = create_tn_lattice_spec(1, 2, V_nn=4.0, interaction_mode="nn")
    proto = TFIMQuenchProtocol(hx=0.1, hz=0.0, t_gate=0.05)

    with pytest.raises(ValueError, match="engine_package='yastn'"):
        simulate_tn(
            spec,
            proto,
            [],
            backend="peps",
            backend_options={"engine_package": "quimb"},
        )
