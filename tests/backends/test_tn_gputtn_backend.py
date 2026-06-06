import stat
import sys
import textwrap

import numpy as np

from ryd_gate.backends.itensor.gputtn_backend import build_gputtn_payload
from ryd_gate.backends.tn_common.compiler import TNEvolutionIR
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.simulate import simulate_tn
from ryd_gate.protocols.lattice_dynamics import TFIMQuenchProtocol


def test_build_gputtn_payload_contains_ttn_runtime_options():
    spec = create_tn_lattice_spec(1, 2, V_nn=4.0)
    proto = TFIMQuenchProtocol(hx=0.5, hz=0.0, t_gate=0.2)
    params = proto.unpack_params([], _Context(spec))
    ir = TNEvolutionIR(spec=spec, protocol=proto, params=params, method="gputtn_tdvp")

    payload = build_gputtn_payload(
        ir,
        initial_state="af1",
        t_eval=np.array([0.0, 0.2]),
        observables=["sigma_z", "n_mean"],
        dt=0.1,
        chi_max=8,
        svd_min=1e-9,
        use_cuda=True,
        require_gpu=True,
        rk_order=4,
        tdvp_nsites=2,
    )

    assert payload["method"] == "gputtn_tdvp"
    assert payload["runtime"]["use_cuda"] is True
    assert payload["runtime"]["require_gpu"] is True
    assert payload["runtime"]["tree"] == "balanced_physical"
    assert payload["runtime"]["rk_order"] == 4
    assert payload["runtime"]["tdvp_nsites"] == 2
    assert payload["record_steps"] == [0, 2]
    assert len(payload["schedule"]) == 2


def test_gputtn_payload_rejects_single_site_with_clear_message():
    spec = create_tn_lattice_spec(1, 1)
    proto = TFIMQuenchProtocol(hx=0.5, hz=0.0, t_gate=0.2)
    params = proto.unpack_params([], _Context(spec))
    ir = TNEvolutionIR(spec=spec, protocol=proto, params=params, method="gputtn_tdvp")

    try:
        build_gputtn_payload(
            ir,
            initial_state="all_ground",
            t_eval=np.array([0.0, 0.2]),
            observables=["sigma_z"],
            dt=0.1,
            chi_max=8,
            svd_min=1e-9,
            use_cuda=True,
            require_gpu=True,
            rk_order=4,
            tdvp_nsites=2,
        )
    except ValueError as exc:
        assert "requires at least two physical sites" in str(exc)
    else:
        raise AssertionError("gputtn should reject single-site payloads clearly")


def test_simulate_tn_gputtn_dispatches_subprocess(tmp_path):
    fake_julia = tmp_path / "fake_julia"
    fake_julia.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """\
            import json
            import sys
            import numpy as np

            args = [arg for arg in sys.argv[1:] if not arg.startswith("--project=")]
            _script, input_json, output_npz, output_json = args
            payload = json.load(open(input_json))
            n = payload["lattice"]["N"]
            times = np.array([0.0, payload["runtime"]["dt"] * payload["runtime"]["n_steps"]])
            np.savez(
                output_npz,
                times=times,
                obs_sigma_z=np.zeros((2, n)),
                obs_n_mean=np.zeros(2),
                final_sigma_z=np.ones(n),
            )
            json.dump(
                {
                    "backend": "gputtn",
                    "method": "gputtn_tdvp",
                    "engine_package": "ITensorNetworks.jl",
                    "gpu": payload["runtime"]["use_cuda"],
                    "n_sites": n,
                },
                open(output_json, "w"),
            )
            """
        )
    )
    fake_julia.chmod(fake_julia.stat().st_mode | stat.S_IXUSR)
    fake_kernel = tmp_path / "kernel.jl"
    fake_kernel.write_text("# fake gputtn kernel\n")

    spec = create_tn_lattice_spec(1, 2)
    proto = TFIMQuenchProtocol(hx=0.5, t_gate=0.2)
    result = simulate_tn(
        spec,
        proto,
        [],
        backend="gputtn",
        t_eval=np.array([0.0, 0.2]),
        observables=["sigma_z", "n_mean"],
        backend_options={
            "julia_cmd": str(fake_julia),
            "project_dir": str(tmp_path),
            "script_path": str(fake_kernel),
            "source_bashrc": False,
            "dt": 0.1,
            "chi_max": 8,
            "use_cuda": True,
            "require_gpu": True,
        },
    )

    assert result.psi_final == "gputtn_itensornetworks_external"
    assert result.metadata["backend"] == "gputtn"
    assert result.metadata["method"] == "gputtn_tdvp"
    assert result.metadata["engine_package"] == "ITensorNetworks.jl"
    assert result.metadata["gpu"] is True
    assert result.metadata["obs"]["sigma_z"].shape == (2, 2)
    assert result.metadata["obs"]["n_mean"].shape == (2,)
    assert result.metadata["final_sigma_z"].shape == (2,)


class _Context:
    def __init__(self, spec):
        self._spec = spec
        self.N = spec.N

    def meta(self, name, default=None):
        if name == "interaction_pairs":
            return tuple((int(i), int(j), float(self._spec.V_nn) * float(v)) for i, j, v in self._spec.vdw_pairs)
        if name == "Omega":
            return self._spec.Omega
        if name == "n_sites":
            return self._spec.N
        return default
