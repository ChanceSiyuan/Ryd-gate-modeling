import stat
import sys
import textwrap

import numpy as np

from ryd_gate.backends.itensor.backend import build_itensors_payload
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.simulate import simulate_tn
from ryd_gate.protocols.lattice_dynamics import TFIMQuenchProtocol


def test_build_itensors_payload_contains_schedule_and_1d_ordering():
    from ryd_gate.backends.tn_common.compiler import TNEvolutionIR

    spec = create_tn_lattice_spec(1, 2, V_nn=4.0)
    proto = TFIMQuenchProtocol(hx=0.5, hz=0.0, t_gate=0.2)
    params = proto.unpack_params([], _Context(spec))
    ir = TNEvolutionIR(spec=spec, protocol=proto, params=params, method="itensors_tebd")

    payload = build_itensors_payload(
        ir,
        initial_state="af1",
        t_eval=np.array([0.0, 0.2]),
        observables=["sigma_z"],
        dt=0.1,
        chi_max=8,
        svd_min=1e-9,
        use_cuda=False,
    )

    assert payload["method"] == "itensors_tebd"
    assert payload["runtime"]["n_steps"] == 2
    assert payload["record_steps"] == [0, 2]
    assert payload["initial_occupations_1d"] in ([1, 0], [0, 1])
    assert len(payload["schedule"]) == 2
    assert len(payload["schedule"][0]["omega_1d"]) == 2
    assert payload["lattice"]["vdw_pairs_1d"][0][0] >= 1


def test_simulate_tn_itensors_dispatches_subprocess(tmp_path):
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
                final_sigma_z=np.ones(n),
            )
            json.dump(
                {
                    "backend": "itensors",
                    "method": "itensors_tebd",
                    "n_sites": n,
                },
                open(output_json, "w"),
            )
            """
        )
    )
    fake_julia.chmod(fake_julia.stat().st_mode | stat.S_IXUSR)
    fake_kernel = tmp_path / "kernel.jl"
    fake_kernel.write_text("# fake kernel\n")

    spec = create_tn_lattice_spec(1, 2)
    proto = TFIMQuenchProtocol(hx=0.5, t_gate=0.2)
    result = simulate_tn(
        spec,
        proto,
        [],
        backend="itensors",
        t_eval=np.array([0.0, 0.2]),
        observables=["sigma_z"],
        backend_options={
            "julia_cmd": str(fake_julia),
            "project_dir": str(tmp_path),
            "script_path": str(fake_kernel),
            "source_bashrc": False,
            "dt": 0.1,
            "chi_max": 8,
        },
    )

    assert result.psi_final == "itensors_mps_external"
    assert result.metadata["backend"] == "itensors"
    assert result.metadata["method"] == "itensors_tebd"
    assert result.metadata["obs"]["sigma_z"].shape == (2, 2)
    assert result.metadata["final_sigma_z"].shape == (2,)
    assert np.allclose(result.times, [0.0, 0.2])


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
