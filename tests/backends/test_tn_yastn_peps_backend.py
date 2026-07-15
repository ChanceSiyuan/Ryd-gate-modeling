import numpy as np
import pytest

from ryd_gate.backends.peps2d import build_yastn_peps_payload
from ryd_gate.backends.tn_common._expressions import tn_basis
from ryd_gate.backends.tn_common.compiler import TNEvolutionIR
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.protocol_context import (
    TNProtocolContext,
    plan_tn_segments,
)
from ryd_gate.backends.tn_common.simulate import simulate_tn
from ryd_gate.core.observables import ObservableFactory
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol
from ryd_gate.protocols.lattice_dynamics import TFIMQuenchProtocol


def _factory(spec) -> ObservableFactory:
    return ObservableFactory(tn_basis(spec))


def _sigma_z(obs, site):
    return 2.0 * obs.n("r", site) - obs.identity()


def test_peps_yastn_package_runs_concrete_smoke():
    pytest.importorskip("yastn")

    spec = create_tn_lattice_spec(1, 2, V_nn=4.0, interaction_mode="nn")
    proto = TFIMQuenchProtocol(hx=0.1, hz=0.0, t_gate=0.05)
    obs = _factory(spec)
    t_eval = np.array([0.0, 0.05])

    result = simulate_tn(
        spec,
        proto,
        backend="peps",
        t_eval=t_eval,
        observables={
            "z_0": _sigma_z(obs, 0),
            "z_1": _sigma_z(obs, 1),
            "zz_01": _sigma_z(obs, 0) @ _sigma_z(obs, 1),
        },
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
    np.testing.assert_array_equal(result.times, t_eval)
    for name in ("z_0", "z_1", "zz_01"):
        values = result.expectation(name)
        assert values.shape == (2,)
        assert values.dtype == complex
    # connected C_zz is script post-processing of the stored expectations
    czz = result.expectation("zz_01") - result.expectation("z_0") * result.expectation("z_1")
    assert np.all(np.isfinite(czz))
    # environment convergence is recorded per sampling anchor
    infos = result.metadata["measurement_env_info"]
    assert len(infos) == 2
    assert all(info["environment"] == "ctm" for info in infos)
    assert all(info["chi"] == result.metadata["ctm_chi"] for info in infos)


def test_peps_yastn_payload_supports_01r_qutrit_profiles():
    spec = create_tn_lattice_spec(1, 2, V_nn=4.0, interaction_mode="nn", level_structure="01r")
    proto = DigitalAnalogProtocol(
        t_gate=0.1,
        coupling_r1=lambda t: [0.1, 0.2 + 0.1j],
        coupling_10=lambda t: [0.3, 0.4],
        coupling_r0=lambda t: [0.05j, 0.0],
        energy_r=lambda t: [-0.1, -0.2],
        energy_1=lambda t: [-0.03, -0.04],
    )
    params = proto._resolve(TNProtocolContext(spec))
    ir = TNEvolutionIR(spec=spec, protocol=proto, params=params, method="peps_yastn")

    record_at_start, segments = plan_tn_segments(0.1, np.array([0.0, 0.1]), 0.05)
    payload = build_yastn_peps_payload(
        ir,
        initial_state=["0", "1"],
        segments=segments,
        chi_max=4,
        svd_min=1e-9,
        use_cuda=False,
    )

    assert record_at_start is True
    assert payload["method"] == "peps_yastn"
    assert payload["lattice"]["level_structure"] == "01r"
    assert payload["lattice"]["levels"] == ("0", "1", "r")
    assert payload["lattice"]["local_dim"] == 3
    assert payload["initial_labels_1d"] == ["0", "1"]
    # per-segment schedule: two steps of dt=0.05, midpoints, record on the anchor step
    assert [entry["dt"] for entry in payload["schedule"]] == [0.05, 0.05]
    np.testing.assert_allclose([entry["t_mid"] for entry in payload["schedule"]], [0.025, 0.075])
    assert [entry["record"] for entry in payload["schedule"]] == [False, True]
    np.testing.assert_allclose(payload["schedule"][0]["coupling_r1_1d"], [0.1, 0.2 + 0.1j])
    np.testing.assert_allclose(payload["schedule"][0]["coupling_10_1d"], [0.3, 0.4])
    np.testing.assert_allclose(payload["schedule"][0]["coupling_r0_1d"], [0.05j, 0.0])
    np.testing.assert_allclose(payload["schedule"][0]["energy_r_1d"], [-0.1, -0.2])
    np.testing.assert_allclose(payload["schedule"][0]["energy_1_1d"], [-0.03, -0.04])


def test_peps_yastn_package_runs_01r_qutrit_smoke():
    pytest.importorskip("yastn")

    spec = create_tn_lattice_spec(1, 2, V_nn=0.1, interaction_mode="nn", level_structure="01r")
    proto = DigitalAnalogProtocol(
        t_gate=0.02,
        coupling_r1=lambda t: 0.1,
        coupling_10=lambda t: 0.05,
        energy_r=lambda t: -0.03,
        energy_1=lambda t: 0.02,
    )
    obs = _factory(spec)
    observables = {}
    for level in ("0", "1", "r"):
        observables.update(obs.site_populations(level))
    observables["n_mean"] = (1.0 / spec.N) * obs.level_sum("r")
    observables["nrnr_01"] = obs.n("r", 0) @ obs.n("r", 1)
    t_eval = np.array([0.0, 0.02])

    result = simulate_tn(
        spec,
        proto,
        backend="peps",
        initial_state="all_1",
        t_eval=t_eval,
        observables=observables,
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
    np.testing.assert_array_equal(result.times, t_eval)
    for name in observables:
        assert result.expectation(name).shape == (2,)
    # BP has no two-point support: the pair term uses the documented CTM fallback
    assert result.metadata["pair_measurement_environment"] == "ctm"
    assert "pair_fallback" in result.metadata["measurement_env_info"][0]
    # initial all-1 product state
    for i in range(2):
        np.testing.assert_allclose(result.expectation(f"n_1_{i}")[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(result.expectation(f"n_r_{i}")[0], 0.0, atol=1e-10)


def test_peps_three_site_product_is_capability_error():
    """PEPS floor is local/sums/two-point: 3-site products fail at preflight."""
    spec = create_tn_lattice_spec(1, 3, V_nn=4.0, interaction_mode="nn")
    proto = TFIMQuenchProtocol(hx=0.1, hz=0.0, t_gate=0.05)
    obs = _factory(spec)
    triple = obs.n("r", 0) @ obs.n("r", 1) @ obs.n("r", 2)
    with pytest.raises(ValueError, match="3 distinct sites"):
        simulate_tn(
            spec, proto, backend="peps",
            t_eval=np.array([0.05]),
            observables={"triple": triple},
            backend_options={"chi_max": 2, "dt": 0.05},
        )


def test_peps_bp_measurement_converges_to_exact():
    """Converged-BP measurement matches exact within the BP loop error.

    Guards the fix for the single-sweep EnvBP path: one ``update_()`` sweep is
    not the BP fixed point, and on this entangled 01r quench state it returned
    n_mean ~ 0.0435 against the exact 0.02127 (err 2.2e-2).  Converged BP
    carries only the plaquette loop error (~1e-3 here).
    """
    pytest.importorskip("yastn")
    import ryd_gate as rg
    from ryd_gate import InteractionSpec, RydbergSystem, level_structure
    from ryd_gate.lattice import Register

    omega_r = 2 * np.pi * 491e6 * 2 * np.pi * 185e6 / (2 * 2 * np.pi * 9.1e9)
    dt = 0.1 / omega_r
    t_gate = 40 * dt
    proto = DigitalAnalogProtocol(
        t_gate=t_gate,
        coupling_r1=lambda t: omega_r / 2,
        coupling_10=lambda t: np.pi * 2e6,
        energy_r=lambda t: -2 * np.pi * 10e6,
        energy_1=lambda t: 2 * np.pi * 5e6,
    )
    system = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.rectangle(2, 2, spacing_um=7.0),
        interaction=InteractionSpec(C6=2 * np.pi * 874e9, mode="nn"),
        protocol=proto,
    )
    observables = {"n_mean": system.observables.level_sum("r") * (1.0 / system.N)}

    exact = rg.simulate(
        system, None, backend="exact_ode",
        t_eval=np.array([t_gate]), observables=observables,
    ).expectation("n_mean")[0].real

    res = rg.simulate(
        system, None, backend="peps",
        t_eval=np.array([t_gate]), observables=observables,
        backend_options={
            "chi_max": 6, "dt": dt, "svd_min": 1e-10,
            "update_environment": "ntu", "measurement_environment": "bp",
            "initialization": "SVD", "max_iter": 30, "tol_iter": 1e-12,
            "use_cuda": False,
        },
    )
    bp = res.expectation("n_mean")[0].real
    assert abs(bp - exact) < 3e-3, f"BP n_mean {bp} vs exact {exact}"
    info = res.metadata["measurement_env_info"][0]
    assert info["environment"] == "bp"
    assert info["converged"] is True or bool(info["converged"])


def test_peps_yastn_ctm_measurement_is_physical():
    """Converged-CTM measurement must be physical (0<=n_r<=1), symmetric, ~ BP.

    Guards the fix for the previously unconverged CTM path, which returned
    out-of-range, symmetry-breaking occupations (e.g. n_r = -0.84, 1.88).
    """
    pytest.importorskip("yastn")
    spec = create_tn_lattice_spec(2, 2, V_nn=3.0, interaction_mode="nn")
    proto = TFIMQuenchProtocol(hx=1.2, hz=0.2, t_gate=0.4)
    obs = _factory(spec)

    def run(meas):
        result = simulate_tn(
            spec, proto, backend="peps",
            t_eval=np.array([0.4]),
            observables=obs.site_populations("r"),
            backend_options={
                "chi_max": 8, "dt": 0.05,
                "measurement_environment": meas, "ctm_chi": 24, "ctm_iters": 60,
            },
        )
        return np.array(
            [result.expectation(f"n_r_{i}")[-1].real for i in range(spec.N)]
        )

    nr_ctm, nr_bp = run("ctm"), run("bp")
    assert np.all(nr_ctm >= -1e-6) and np.all(nr_ctm <= 1.0 + 1e-6)  # physical bounds
    assert float(nr_ctm.max() - nr_ctm.min()) < 1e-3                  # 2x2 uniform-drive symmetry
    np.testing.assert_allclose(nr_ctm, nr_bp, atol=2e-2)             # agree up to BP loop error


def test_peps_ctm_requires_iterations():
    """Guards: CTM measurement rejects degenerate (chi<=1 / iters<=0) settings."""
    pytest.importorskip("yastn")
    import yastn.tn.fpeps as fpeps

    from ryd_gate.backends.peps2d import YASTNPEPSError, _measurement_env

    with pytest.raises(YASTNPEPSError, match="ctm_chi > 1"):
        _measurement_env(fpeps, None, "ctm", ctm_chi=1, ctm_iters=10, ctm_tol=1e-8)
    with pytest.raises(YASTNPEPSError, match="ctm_iters > 0"):
        _measurement_env(fpeps, None, "ctm", ctm_chi=16, ctm_iters=0, ctm_tol=1e-8)
