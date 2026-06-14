"""Stage D: RydTNPEPSBackend real-time smoke tests (mirror the YASTN smokes)."""

import numpy as np

from ryd_gate.backends.rydtn.backend import RydTNPEPSBackend
from ryd_gate.backends.tn_common.compiler import TNEvolutionIR
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.protocol_context import TNProtocolContext
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol
from ryd_gate.protocols.lattice_dynamics import TFIMQuenchProtocol


def _ir(spec, proto):
    params = proto.unpack_params([], TNProtocolContext(spec))
    return TNEvolutionIR(spec=spec, protocol=proto, params=params, method="peps_rydtn")


def test_rydtn_peps_runs_concrete_smoke():
    spec = create_tn_lattice_spec(1, 2, V_nn=4.0, interaction_mode="nn")
    proto = TFIMQuenchProtocol(hx=0.1, hz=0.0, t_gate=0.05)
    result = RydTNPEPSBackend(chi_max=2, dt=0.05).evolve_ir(
        _ir(spec, proto), t_eval=np.array([0.0, 0.05]),
        observables=["sigma_z", "czz_centerline"],
    )
    assert result.metadata["backend"] == "peps"
    assert result.metadata["method"] == "peps_rydtn"
    assert result.metadata["engine_package"] == "rydtn"
    assert result.metadata["obs"]["sigma_z"].shape == (2, 2)
    assert result.metadata["obs"]["czz_centerline"].shape == (2, 1)
    assert np.isfinite(result.metadata["max_truncation_error"])


def test_rydtn_peps_runs_01r_qutrit_smoke():
    spec = create_tn_lattice_spec(1, 2, V_nn=0.1, interaction_mode="nn", level_structure="01r")
    proto = DigitalAnalogProtocol(
        t_gate=0.02,
        omega_R_fn=lambda t: 0.2,
        omega_hf_fn=lambda t: 0.1,
        delta_R_fn=lambda t: 0.03,
        delta_hf_fn=lambda t: -0.02,
    )
    result = RydTNPEPSBackend(chi_max=3, dt=0.02).evolve_ir(
        _ir(spec, proto), initial_state="all_1", t_eval=np.array([0.0, 0.02]),
        observables=["n_0", "n_1", "n_r", "n_mean", "sigma_z", "czz"],
    )
    assert result.metadata["level_structure"] == "01r"
    assert result.metadata["local_dim"] == 3
    assert result.metadata["obs"]["n_0"].shape == (2, 2)
    assert result.metadata["obs"]["n_1"].shape == (2, 2)
    assert result.metadata["obs"]["n_mean"].shape == (2,)
    assert result.metadata["obs"]["czz"].shape == (2, 1)
    # initial product state |1,1>: n_1 = 1, n_r = 0 exactly
    np.testing.assert_allclose(result.metadata["obs"]["n_1"][0], [1.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(result.metadata["obs"]["n_r"][0], [0.0, 0.0], atol=1e-10)


def test_legacy_yastn_backend_option_aliases_to_rydtn():
    """YASTN-style backend_options (yastn_backend=...) keep working under the rydtn default."""
    from ryd_gate.backends.tn_common.simulate import simulate_tn

    spec = create_tn_lattice_spec(1, 2, V_nn=2.0, interaction_mode="nn")
    proto = TFIMQuenchProtocol(hx=0.5, hz=0.1, t_gate=0.05)
    res = simulate_tn(
        spec, proto, [], backend="peps", t_eval=np.array([0.05]), observables=["n_mean"],
        backend_options={"chi_max": 4, "dt": 0.05, "yastn_backend": "np"},  # legacy option name
    )
    assert res.metadata["engine_package"] == "rydtn"
    assert res.metadata["array_backend"] == "numpy"


def test_rydtn_peps_torch_cpu_smoke():
    import pytest

    pytest.importorskip("torch")
    spec = create_tn_lattice_spec(1, 2, V_nn=4.0, interaction_mode="nn")
    proto = TFIMQuenchProtocol(hx=0.1, hz=0.0, t_gate=0.05)
    result = RydTNPEPSBackend(chi_max=2, dt=0.05, backend_name="torch", device="cpu").evolve_ir(
        _ir(spec, proto), t_eval=np.array([0.0, 0.05]), observables=["n_mean"],
    )
    assert result.metadata["array_backend"] == "torch"
    assert result.metadata["obs"]["n_mean"].shape == (2,)
