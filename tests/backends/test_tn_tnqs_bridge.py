import os
import stat
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from ryd_gate.backends.itensor import tnqs_backend as tnqs_mod
from ryd_gate.backends.itensor.tnqs_backend import TNQSJulia2DTNBackend, build_tnqs_payload
from ryd_gate.backends.tn_common.compiler import TNEvolutionIR
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.simulate import simulate_tn
from ryd_gate.protocols.lattice_dynamics import TFIMQuenchProtocol


def test_build_tnqs_payload_contains_2d_runtime_options():
    spec = create_tn_lattice_spec(1, 2, V_nn=4.0)
    proto = TFIMQuenchProtocol(hx=0.5, hz=0.0, t_gate=0.2)
    params = proto.unpack_params([], _Context(spec))
    ir = TNEvolutionIR(spec=spec, protocol=proto, params=params, method="2dtn_bp")

    payload = build_tnqs_payload(
        ir,
        initial_state="af1",
        t_eval=np.array([0.0, 0.2]),
        observables=["sigma_z", "czz_centerline"],
        dt=0.1,
        chi_max=8,
        svd_min=1e-9,
        use_cuda=True,
        measurement_alg="boundary_mps",
        measurement_bond_dim=16,
        chi_2d_prime=4,
        normalize_tensors=True,
        eltype="ComplexF32",
    )

    assert payload["method"] == "2dtn_bp"
    assert payload["runtime"]["chi_max"] == 8
    assert payload["runtime"]["measurement_alg"] == "boundarymps"
    assert payload["runtime"]["measurement_bond_dim"] == 16
    assert payload["runtime"]["chi_2d_prime"] == 4
    assert payload["runtime"]["normalize_tensors"] is True
    assert payload["runtime"]["eltype"] == "ComplexF32"
    assert payload["lattice"]["vdw_pairs_1d"][0][0] >= 1


def test_simulate_tn_2dtn_dispatches_to_tnqs_subprocess(tmp_path):
    fake_julia = tmp_path / "fake_julia"
    fake_julia.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """\
            import json
            import sys
            import numpy as np

            args = [arg for arg in sys.argv[1:] if not arg.startswith(("--project=", "--sysimage="))]
            _script, input_json, output_npz, output_json = args
            payload = json.load(open(input_json))
            n = payload["lattice"]["N"]
            times = np.array([0.0, payload["runtime"]["dt"] * payload["runtime"]["n_steps"]])
            np.savez(
                output_npz,
                times=times,
                obs_sigma_z=np.zeros((2, n)),
                obs_czz_centerline=np.zeros((2, 1)),
                final_sigma_z=np.ones(n),
                truncation_error=np.array([0.0, 1.0e-8]),
            )
            json.dump(
                {
                    "backend": "2dtn",
                    "method": "2dtn_bp",
                    "engine_package": "TensorNetworkQuantumSimulator.jl",
                    "n_sites": n,
                    "measurement_alg": payload["runtime"]["measurement_alg"],
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
        backend="2dtn",
        t_eval=np.array([0.0, 0.2]),
        observables=["sigma_z", "czz_centerline"],
        backend_options={
            "julia_cmd": str(fake_julia),
            "project_dir": str(tmp_path),
            "script_path": str(fake_kernel),
            "source_bashrc": False,
            "dt": 0.1,
            "chi_max": 8,
            "chi_2d_prime": 4,
            "measurement_alg": "bp",
        },
    )

    assert result.psi_final == "tnqs_2dtn_external"
    assert result.metadata["backend"] == "2dtn"
    assert result.metadata["method"] == "2dtn_bp"
    assert result.metadata["engine_package"] == "TensorNetworkQuantumSimulator.jl"
    assert result.metadata["obs"]["sigma_z"].shape == (2, 2)
    assert result.metadata["obs"]["czz_centerline"].shape == (2, 1)
    assert result.metadata["final_sigma_z"].shape == (2,)
    assert result.metadata["truncation_error"].shape == (2,)
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


def _command(backend):
    return backend._subprocess_command(Path("in.json"), Path("o.npz"), Path("o.json"))


def _backend(tmp_path, **kwargs):
    kernel = tmp_path / "kernel.jl"
    kernel.write_text("# fake kernel\n")
    kwargs.setdefault("julia_cmd", "/bin/true")
    kwargs.setdefault("script_path", str(kernel))
    kwargs.setdefault("project_dir", str(tmp_path))
    return TNQSJulia2DTNBackend(**kwargs)


def test_subprocess_command_injects_explicit_sysimage(tmp_path):
    img = tmp_path / "ryd_tnqs.so"
    img.write_bytes(b"")
    cmd = _command(_backend(tmp_path, sysimage=str(img)))
    assert f"--sysimage={img}" in cmd
    # --sysimage must be a julia flag, i.e. precede --project and the script
    project_idx = next(i for i, c in enumerate(cmd) if c.startswith("--project="))
    assert cmd.index(f"--sysimage={img}") < project_idx


def test_subprocess_command_uses_env_sysimage(tmp_path, monkeypatch):
    img = tmp_path / "env.so"
    img.write_bytes(b"")
    monkeypatch.setenv("RYD_TNQS_SYSIMAGE", str(img))
    assert f"--sysimage={img}" in _command(_backend(tmp_path))


def test_subprocess_command_omits_sysimage_when_absent(tmp_path, monkeypatch):
    # No default sysimage built, no env, no explicit path -> zero-config fallback.
    monkeypatch.delenv("RYD_TNQS_SYSIMAGE", raising=False)
    monkeypatch.setattr(tnqs_mod, "_default_project_dir", lambda: tmp_path)
    assert not any(c.startswith("--sysimage=") for c in _command(_backend(tmp_path)))
    # Explicit but missing path is silently skipped, not an error.
    cmd = _command(_backend(tmp_path, sysimage=str(tmp_path / "missing.so")))
    assert not any(c.startswith("--sysimage=") for c in cmd)


# ---------------------------------------------------------------------------
# Opt-in real-Julia smoke tests. The TNQS kernel pays full cold JIT (tens of
# seconds), so these only run when RYD_RUN_JULIA_TESTS=1 (and RYD_TEST_GPU=1 for
# the CUDA path). Following the repo convention, the default suite never spawns a
# real Julia process.
# ---------------------------------------------------------------------------
def _quench_2x2():
    spec = create_tn_lattice_spec(2, 2, V_nn=4.0, interaction_mode="nn")
    proto = TFIMQuenchProtocol(hx=1.0, t_gate=0.2)
    return spec, proto


@pytest.mark.skipif(
    os.environ.get("RYD_RUN_JULIA_TESTS") != "1",
    reason="set RYD_RUN_JULIA_TESTS=1 to run the real TNQS Julia kernel (slow cold JIT)",
)
def test_tnqs_2dtn_kernel_runs_and_is_physical():
    spec, proto = _quench_2x2()
    t_eval = np.array([0.0, 0.1, 0.2])
    res = simulate_tn(
        spec, proto, [], backend="2dtn", t_eval=t_eval,
        observables=["sigma_z", "czz_centerline"],
        backend_options={"chi_max": 16, "dt": 0.02, "measurement_alg": "bp",
                         "measurement_bond_dim": 16, "chi_2d_prime": 16, "timeout": 1800},
    )
    sigma_z = np.asarray(res.metadata["obs"]["sigma_z"])
    assert sigma_z.shape == (3, 4)
    occ = 0.5 * (sigma_z + 1.0)
    assert np.all(occ >= -1e-6) and np.all(occ <= 1.0 + 1e-6)
    # all_ground (|Dn>) has zero Rydberg occupation at t=0
    assert abs(float(occ[0].mean())) < 1e-6


@pytest.mark.skipif(
    os.environ.get("RYD_RUN_JULIA_TESTS") != "1" or os.environ.get("RYD_TEST_GPU") != "1",
    reason="set RYD_RUN_JULIA_TESTS=1 and RYD_TEST_GPU=1 to run the CUDA TNQS path",
)
def test_tnqs_2dtn_gpu_smoke():
    spec, proto = _quench_2x2()
    t_eval = np.array([0.0, 0.2])
    res = simulate_tn(
        spec, proto, [], backend="2dtn", t_eval=t_eval, observables=["sigma_z"],
        backend_options={"chi_max": 8, "dt": 0.05, "use_cuda": True, "timeout": 1800},
    )
    assert np.asarray(res.metadata["obs"]["sigma_z"]).shape == (2, 4)
    assert res.metadata["use_cuda"] is True
