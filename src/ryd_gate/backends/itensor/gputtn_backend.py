"""Subprocess bridge to ITensorNetworks.jl TTN-TDVP kernels."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ryd_gate.ir.evolution import EvolutionResult

from .backend import (
    ITensorsJuliaError,
    _command_list,
    _default_project_dir,
    _executable_available,
    _jsonable,
    build_itensors_payload,
)


class GPUITensorNetworksTTNError(ITensorsJuliaError):
    """Raised when the ITensorNetworks.jl GPU TTN bridge fails."""


@dataclass
class GPUITensorNetworksTTNBackend:
    """Run TTN-TDVP through ITensorNetworks.jl.

    This backend is intentionally parallel to ``backend="ttn"``: it consumes
    the same compiled TN IR, but delegates the evolution to Julia
    ITensorNetworks.  By default it requires CUDA and moves both the TTN state
    and TTN Hamiltonian to GPU storage before TDVP sweeps.
    """

    chi_max: int = 64
    dt: float = 0.05
    svd_min: float = 1e-10
    rk_order: int = 4
    tdvp_nsites: int = 2
    use_cuda: bool = True
    require_gpu: bool = True
    julia_cmd: str | Sequence[str] = "julia"
    project_dir: str | os.PathLike | None = None
    script_path: str | os.PathLike | None = None
    timeout: float | None = None
    work_dir: str | os.PathLike | None = None
    keep_workdir: bool = False
    source_bashrc: bool = True
    threads: int | None = None

    def evolve_ir(
        self,
        ir,
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        payload = build_gputtn_payload(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
            dt=self.dt,
            chi_max=self.chi_max,
            svd_min=self.svd_min,
            use_cuda=self.use_cuda,
            require_gpu=self.require_gpu,
            rk_order=self.rk_order,
            tdvp_nsites=self.tdvp_nsites,
        )

        if self.keep_workdir:
            run_dir = Path(tempfile.mkdtemp(prefix="ryd-gputtn-", dir=self.work_dir))
            return self._run_in_dir(run_dir, payload, cleanup=False)

        with tempfile.TemporaryDirectory(prefix="ryd-gputtn-", dir=self.work_dir) as tmp:
            return self._run_in_dir(Path(tmp), payload, cleanup=True)

    def _run_in_dir(self, run_dir: Path, payload: dict, *, cleanup: bool) -> EvolutionResult:
        input_json = run_dir / "payload.json"
        output_npz = run_dir / "result.npz"
        output_json = run_dir / "result.json"
        input_json.write_text(json.dumps(_jsonable(payload), sort_keys=True), encoding="utf-8")

        cmd = self._subprocess_command(input_json, output_npz, output_json)
        env = os.environ.copy()
        if self.threads is not None:
            env["JULIA_NUM_THREADS"] = str(int(self.threads))

        try:
            proc = subprocess.run(
                cmd,
                cwd=run_dir,
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise GPUITensorNetworksTTNError(
                "ITensorNetworks.jl GPU TTN kernel timed out after "
                f"{self.timeout} seconds. Increase `backend_options['timeout']`, "
                "especially for first-time Julia/CUDA JIT compilation."
            ) from exc
        if proc.returncode != 0:
            raise GPUITensorNetworksTTNError(
                "ITensorNetworks.jl GPU TTN kernel failed with exit code "
                f"{proc.returncode}.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not output_npz.exists() or not output_json.exists():
            raise GPUITensorNetworksTTNError(
                "ITensorNetworks.jl GPU TTN kernel finished but did not produce "
                "result.npz/result.json."
            )
        return _load_gputtn_result(
            output_npz,
            output_json,
            stdout=proc.stdout,
            stderr=proc.stderr,
            workdir=None if cleanup else str(run_dir),
        )

    def _subprocess_command(
        self,
        input_json: Path,
        output_npz: Path,
        output_json: Path,
    ) -> list[str]:
        project_dir = Path(self.project_dir) if self.project_dir is not None else _default_project_dir()
        script_path = Path(self.script_path) if self.script_path is not None else _default_script_path()
        if not script_path.exists():
            raise GPUITensorNetworksTTNError(f"ITensorNetworks.jl GPU TTN script not found: {script_path}")

        julia_args = [
            f"--project={project_dir}",
            str(script_path),
            str(input_json),
            str(output_npz),
            str(output_json),
        ]
        cmd = _command_list(self.julia_cmd) + julia_args
        executable = cmd[0]
        if _executable_available(executable):
            return cmd
        if not self.source_bashrc:
            raise GPUITensorNetworksTTNError(
                f"Julia executable {executable!r} was not found on PATH. "
                "Pass backend_options={'julia_cmd': '/path/to/julia'} or enable source_bashrc."
            )
        return [
            "bash",
            "-lc",
            "source ~/.bashrc && exec " + " ".join(shlex.quote(part) for part in cmd),
        ]


def build_gputtn_payload(
    ir,
    *,
    initial_state: str | np.ndarray | object,
    t_eval: np.ndarray | None,
    observables: list[str] | None,
    dt: float,
    chi_max: int,
    svd_min: float,
    use_cuda: bool,
    require_gpu: bool,
    rk_order: int,
    tdvp_nsites: int,
) -> dict:
    """Build a JSON-serializable payload for the ITensorNetworks TTN kernel."""
    if ir.spec.N < 2:
        raise ValueError(
            "backend='gputtn' requires at least two physical sites because "
            "ITensorNetworks.jl OpSum-to-TTN conversion needs a leaf root. "
            "Use backend='ttn', 'exact', or 'mps_gpu' for single-site checks."
        )
    payload = build_itensors_payload(
        ir,
        initial_state=initial_state,
        t_eval=t_eval,
        observables=observables,
        dt=dt,
        chi_max=chi_max,
        svd_min=svd_min,
        use_cuda=use_cuda,
    )
    payload["method"] = "gputtn_tdvp"
    payload["runtime"].update(
        {
            "require_gpu": bool(require_gpu),
            "rk_order": int(rk_order),
            "tdvp_nsites": int(tdvp_nsites),
            "tree": "balanced_physical",
        }
    )
    return payload


def _load_gputtn_result(
    output_npz: Path,
    output_json: Path,
    *,
    stdout: str,
    stderr: str,
    workdir: str | None,
) -> EvolutionResult:
    arrays = np.load(output_npz)
    metadata = json.loads(output_json.read_text(encoding="utf-8"))
    obs = {}
    for key in arrays.files:
        if key.startswith("obs_"):
            obs[key[4:]] = arrays[key]
    if obs:
        metadata["obs"] = obs
    for key in ("final_sigma_z",):
        if key in arrays.files:
            metadata[key] = arrays[key]
    metadata.setdefault("backend", "gputtn")
    metadata.setdefault("method", "gputtn_tdvp")
    metadata["julia_stdout"] = stdout
    metadata["julia_stderr"] = stderr
    if workdir is not None:
        metadata["workdir"] = workdir
    times = arrays["times"] if "times" in arrays.files else None
    return EvolutionResult(
        psi_final="gputtn_itensornetworks_external",
        times=times,
        states=None,
        metadata=metadata,
    )


def _default_script_path() -> Path:
    return _default_project_dir() / "run_gputtn_tdvp.jl"
