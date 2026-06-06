import numpy as np
import pytest

from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.simulate import simulate_tn
from ryd_gate.protocols.lattice_dynamics import TFIMQuenchProtocol


def test_nqs_netket_package_runs_concrete_smoke():
    pytest.importorskip("netket")

    spec = create_tn_lattice_spec(1, 2, V_nn=1.0, interaction_mode="nn")
    proto = TFIMQuenchProtocol(hx=0.1, hz=0.0, t_gate=0.002)

    result = simulate_tn(
        spec,
        proto,
        [],
        backend="nqs",
        t_eval=np.array([0.0, 0.002]),
        observables=["sigma_z", "n_mean"],
        backend_options={
            "engine_package": "netket",
            "dt": 0.001,
            "sampler": "exact",
            "n_samples": 64,
            "n_chains": 4,
            "integrator": "euler",
            "initial_bias_strength": 0.5,
            "seed": 2,
        },
    )

    assert result.metadata["backend"] == "nqs"
    assert result.metadata["method"] == "nqs_tvmc"
    assert result.metadata["engine_package"] == "netket"
    assert result.metadata["algorithm"] == "netket_tdvp_mcstate"
    assert result.metadata["obs"]["sigma_z"].shape == (2, 2)
    assert result.metadata["obs"]["n_mean"].shape == (2,)
