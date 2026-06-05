"""Subprocess bridge to TensorNetworkQuantumSimulator.jl 2D-TN kernels."""

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
    _command_list,
    _default_project_dir,
    _executable_available,
    _jsonable,
    build_itensors_payload,
)


class TNQSJuliaError(RuntimeError):
    """Raised when the TensorNetworkQuantumSimulator.jl subprocess bridge fails."""


@dataclass
class TNQSJulia2DTNBackend:
    """Run a compiled TN IR through TensorNetworkQuantumSimulator.jl.

    This backend implements the 2D tensor-network simple-update/BP path used for
    effective two-level Rydberg/TFIM lattice dynamics. It consumes the same
    compiled ``1r`` TN IR as the ITensors MPS bridge, but keeps the 2D graph
    structure instead of flattening evolution to an MPS.
    """

    chi_max: int = 256
    dt: float = 0.1
    svd_min: float = 1e-10
    chi_2d: int | None = None
    chi_2d_prime: int | None = None
    measurement_alg: str = "bp"
    measurement_bond_dim: int = 32
    r_2d: int | None = None
    normalize_tensors: bool = False
    julia_cmd: str | Sequence[str] = "julia"
    project_dir: str | os.PathLike | None = None
    script_path: str | os.PathLike | None = None
    timeout: float | None = None
    work_dir: str | os.PathLike | None = None
    keep_workdir: bool = False
    source_bashrc: bool = True
    threads: int | None = None
    use_cuda: bool = False
    eltype: str | None = None

    def evolve_ir(
        self,
        ir,
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        """Serialize a TN IR, run the Julia 2D-TN kernel, and parse its result."""
        payload = build_tnqs_payload(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
            dt=self.dt,
            chi_max=self.chi_2d or self.chi_max,
            svd_min=self.svd_min,
            use_cuda=self.use_cuda,
            measurement_alg=self.measurement_alg,
            measurement_bond_dim=self.r_2d or self.measurement_bond_dim,
            chi_2d_prime=self.chi_2d_prime,
            normalize_tensors=self.normalize_tensors,
            eltype=self.eltype,
        )

        if self.keep_workdir:
            run_dir = Path(tempfile.mkdtemp(prefix="ryd-tnqs-", dir=self.work_dir))
            return self._run_in_dir(run_dir, payload, cleanup=False)

        with tempfile.TemporaryDirectory(prefix="ryd-tnqs-", dir=self.work_dir) as tmp:
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
            raise TNQSJuliaError(
                "TensorNetworkQuantumSimulator.jl kernel timed out after "
                f"{self.timeout} seconds. Increase `backend_options['timeout']`, "
                "especially for first-time Julia/CUDA JIT compilation."
            ) from exc
        if proc.returncode != 0:
            raise TNQSJuliaError(
                "TensorNetworkQuantumSimulator.jl kernel failed with exit code "
                f"{proc.returncode}.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not output_npz.exists() or not output_json.exists():
            raise TNQSJuliaError(
                "TensorNetworkQuantumSimulator.jl kernel finished but did not produce "
                "result.npz/result.json."
            )
        return _load_tnqs_result(
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
            raise TNQSJuliaError(f"TensorNetworkQuantumSimulator.jl kernel script not found: {script_path}")

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
            raise TNQSJuliaError(
                f"Julia executable {executable!r} was not found on PATH. "
                "Pass backend_options={'julia_cmd': '/path/to/julia'} or enable source_bashrc."
            )
        return [
            "bash",
            "-lc",
            "source ~/.bashrc && exec " + " ".join(shlex.quote(part) for part in cmd),
        ]


def build_tnqs_payload(
    ir,
    *,
    initial_state: str | np.ndarray | object,
    t_eval: np.ndarray | None,
    observables: list[str] | None,
    dt: float,
    chi_max: int,
    svd_min: float,
    use_cuda: bool,
    measurement_alg: str,
    measurement_bond_dim: int,
    chi_2d_prime: int | None,
    normalize_tensors: bool,
    eltype: str | None = None,
) -> dict:
    """Build a JSON-serializable payload for the Julia 2D-TN/BP kernel."""
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
    payload["method"] = "2dtn_bp"
    payload["runtime"].update(
        {
            "chi_max": int(chi_max),
            "measurement_alg": _normalize_measurement_alg(measurement_alg),
            "measurement_bond_dim": int(measurement_bond_dim),
            "chi_2d_prime": None if chi_2d_prime is None else int(chi_2d_prime),
            "normalize_tensors": bool(normalize_tensors),
        }
    )
    if eltype is not None:
        payload["runtime"]["eltype"] = str(eltype)
    return payload


def _normalize_measurement_alg(measurement_alg: str) -> str:
    key = str(measurement_alg).lower()
    if key in {"bp", "belief_propagation", "belief-propagation"}:
        return "bp"
    if key in {"boundarymps", "boundary_mps", "bmps"}:
        return "boundarymps"
    if key == "exact":
        return "exact"
    raise ValueError("measurement_alg must be 'bp', 'boundarymps', or 'exact'.")


def _load_tnqs_result(
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
    for key in ("final_sigma_z", "truncation_error"):
        if key in arrays.files:
            metadata[key] = arrays[key]
    metadata.setdefault("backend", "2dtn")
    metadata.setdefault("method", "2dtn_bp")
    metadata["julia_stdout"] = stdout
    metadata["julia_stderr"] = stderr
    if workdir is not None:
        metadata["workdir"] = workdir
    times = arrays["times"] if "times" in arrays.files else None
    return EvolutionResult(
        psi_final="tnqs_2dtn_external",
        times=times,
        states=None,
        metadata=metadata,
    )


def _default_script_path() -> Path:
    return _default_project_dir() / "run_tnqs_2d_bp.jl"
