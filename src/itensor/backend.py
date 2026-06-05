"""Subprocess bridge to ITensors.jl / ITensorMPS.jl kernels."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ryd_gate.ir.evolution import EvolutionResult
from ryd_gate.core.channel_lowering import two_level_drive_and_detuning_from_coeffs

from tenpy_mps.backends import _merge_pin_deltas, _pin_deltas_from_params
from tn_common.external_backends import build_external_solver_payload


class ITensorsJuliaError(RuntimeError):
    """Raised when the ITensors.jl subprocess bridge fails."""


@dataclass
class ITensorsJuliaBackend:
    """Run a compiled TN IR through a Julia ITensors/ITensorMPS kernel.

    The current Julia kernel is an MPS TEBD implementation for the effective
    ``1r`` lattice Hamiltonian. It is intentionally exposed as
    ``backend='itensors'`` rather than ``backend='2dtn'`` because the 2D-TN/BP
    algorithm in ``main.tex`` needs TensorNetworkQuantumSimulator.jl on top of
    ITensors, not just a plain ITensors MPS path.
    """

    chi_max: int = 256
    dt: float = 0.1
    svd_min: float = 1e-10
    julia_cmd: str | Sequence[str] = "julia"
    project_dir: str | os.PathLike | None = None
    script_path: str | os.PathLike | None = None
    timeout: float | None = None
    work_dir: str | os.PathLike | None = None
    keep_workdir: bool = False
    source_bashrc: bool = True
    threads: int | None = None
    use_cuda: bool = False

    def evolve_ir(
        self,
        ir,
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        """Serialize a TN IR, run the Julia kernel, and parse its result."""
        payload = build_itensors_payload(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
            dt=self.dt,
            chi_max=self.chi_max,
            svd_min=self.svd_min,
            use_cuda=self.use_cuda,
        )

        if self.keep_workdir:
            run_dir = Path(tempfile.mkdtemp(prefix="ryd-itensors-", dir=self.work_dir))
            return self._run_in_dir(run_dir, payload, cleanup=False)

        with tempfile.TemporaryDirectory(prefix="ryd-itensors-", dir=self.work_dir) as tmp:
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
            raise ITensorsJuliaError(
                "ITensors.jl kernel timed out after "
                f"{self.timeout} seconds. Increase `backend_options['timeout']`, "
                "especially for first-time Julia/CUDA JIT compilation."
            ) from exc
        if proc.returncode != 0:
            raise ITensorsJuliaError(
                "ITensors.jl kernel failed with exit code "
                f"{proc.returncode}.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not output_npz.exists() or not output_json.exists():
            raise ITensorsJuliaError(
                "ITensors.jl kernel finished but did not produce result.npz/result.json."
            )
        return _load_itensors_result(
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
            raise ITensorsJuliaError(f"ITensors.jl kernel script not found: {script_path}")

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
            raise ITensorsJuliaError(
                f"Julia executable {executable!r} was not found on PATH. "
                "Pass backend_options={'julia_cmd': '/path/to/julia'} or enable source_bashrc."
            )
        return [
            "bash",
            "-lc",
            "source ~/.bashrc && exec " + " ".join(shlex.quote(part) for part in cmd),
        ]


def build_itensors_payload(
    ir,
    *,
    initial_state: str | np.ndarray | object,
    t_eval: np.ndarray | None,
    observables: list[str] | None,
    dt: float,
    chi_max: int,
    svd_min: float,
    use_cuda: bool,
) -> dict:
    """Build a JSON-serializable payload for the Julia ITensors kernel."""
    spec = ir.spec
    if spec.level_structure != "1r":
        raise ValueError("ITensors.jl bridge currently supports only level_structure='1r'.")

    t_gate = float(ir.params["t_gate"])
    if t_gate <= 0:
        raise ValueError("ITensors.jl bridge requires a positive t_gate.")
    n_steps = max(1, int(np.ceil(t_gate / float(dt))))
    dt_actual = t_gate / n_steps

    record_steps = _record_steps(t_eval, dt_actual, n_steps)
    schedule = _drive_schedule(ir, dt_actual=dt_actual, n_steps=n_steps)
    base_payload = build_external_solver_payload(
        spec,
        ir.protocol,
        ir.params,
        t_eval=t_eval,
        observables=observables,
    )
    base_payload["method"] = "itensors_tebd"
    base_payload["metadata"] = ir.metadata or {}
    base_payload["initial_occupations_1d"] = _initial_occupations_1d(spec, initial_state)
    base_payload["lattice"]["snake_to_2d"] = np.asarray(spec.snake_to_2d, dtype=int)
    base_payload["lattice"]["inv_snake"] = np.asarray(spec.inv_snake, dtype=int)
    base_payload["lattice"]["vdw_pairs_1d"] = [
        [
            int(spec.inv_snake[int(i)]) + 1,
            int(spec.inv_snake[int(j)]) + 1,
            float(spec.V_nn) * float(v_rel),
        ]
        for i, j, v_rel in spec.vdw_pairs
    ]
    base_payload["schedule"] = schedule
    base_payload["record_steps"] = record_steps
    base_payload["runtime"] = {
        "dt": dt_actual,
        "requested_dt": float(dt),
        "n_steps": n_steps,
        "chi_max": int(chi_max),
        "svd_min": float(svd_min),
        "use_cuda": bool(use_cuda),
    }
    return base_payload


def _drive_schedule(ir, *, dt_actual: float, n_steps: int) -> list[dict]:
    spec = ir.spec
    static_pin = _pin_deltas_from_params(ir.params, spec.N)
    schedule = []
    for step in range(n_steps):
        t_mid = (step + 0.5) * dt_actual
        coeffs = ir.protocol.get_drive_coefficients(t_mid, ir.params)
        omega_t, delta_t, channel_pin = two_level_drive_and_detuning_from_coeffs(coeffs, spec)
        omega_profile = _profile(omega_t, spec.N)
        pin = _merge_pin_deltas(static_pin, channel_pin, n_sites=spec.N)
        delta_profile = np.full(spec.N, float(delta_t), dtype=float)
        if pin is not None:
            delta_profile = delta_profile + pin
        schedule.append(
            {
                "step": step + 1,
                "t_mid": float(t_mid),
                "omega_1d": omega_profile[np.asarray(spec.snake_to_2d, dtype=int)],
                "delta_1d": delta_profile[np.asarray(spec.snake_to_2d, dtype=int)],
            }
        )
    return schedule


def _profile(value, n_sites: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_sites, float(arr), dtype=float)
    if arr.shape != (n_sites,):
        raise ValueError(f"Expected scalar or length-{n_sites} profile, got shape {arr.shape}.")
    return arr


def _record_steps(t_eval: np.ndarray | None, dt_actual: float, n_steps: int) -> list[int]:
    if t_eval is None:
        return []
    steps = set()
    for t_req in np.asarray(t_eval, dtype=float):
        step = int(round(float(t_req) / dt_actual))
        steps.add(max(0, min(step, n_steps)))
    return sorted(steps)


def _initial_occupations_1d(spec, initial_state: str | np.ndarray | object) -> list[int]:
    if not isinstance(initial_state, (str, list, tuple, np.ndarray)):
        raise TypeError("ITensors.jl bridge requires a string, label list, or 0/1 array initial_state.")
    occ_2d = _initial_occupations_2d(spec, initial_state)
    return [int(x) for x in occ_2d[np.asarray(spec.snake_to_2d, dtype=int)]]


def _initial_occupations_2d(spec, initial_state: str | np.ndarray | Sequence) -> np.ndarray:
    if isinstance(initial_state, str):
        if initial_state in {"all_ground", "all_1"}:
            return np.zeros(spec.N, dtype=int)
        if initial_state == "all_r":
            return np.ones(spec.N, dtype=int)
        if initial_state == "af1":
            return (np.asarray(spec.sublattice) > 0).astype(int)
        if initial_state == "af2":
            return (np.asarray(spec.sublattice) < 0).astype(int)
        raise ValueError(f"Unknown ITensors.jl initial_state string: {initial_state!r}")

    arr = np.asarray(initial_state)
    if arr.shape != (spec.N,):
        raise ValueError(f"initial_state must have shape ({spec.N},), got {arr.shape}.")
    if arr.dtype.kind in {"U", "S", "O"}:
        return np.array([1 if str(label) == "r" else 0 for label in arr], dtype=int)
    return arr.astype(int)


def _load_itensors_result(
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
    if "final_sigma_z" in arrays.files:
        metadata["final_sigma_z"] = arrays["final_sigma_z"]
    metadata.setdefault("backend", "itensors")
    metadata.setdefault("method", "itensors_tebd")
    metadata["julia_stdout"] = stdout
    metadata["julia_stderr"] = stderr
    if workdir is not None:
        metadata["workdir"] = workdir
    times = arrays["times"] if "times" in arrays.files else None
    return EvolutionResult(
        psi_final="itensors_mps_external",
        times=times,
        states=None,
        metadata=metadata,
    )


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__") and value.__class__.__module__.startswith(("ryd_gate", "tn_common")):
        return repr(value)
    return value


def _command_list(cmd: str | Sequence[str]) -> list[str]:
    if isinstance(cmd, str):
        return shlex.split(cmd)
    return [str(part) for part in cmd]


def _executable_available(executable: str) -> bool:
    if os.path.sep in executable:
        return Path(executable).exists()
    return shutil.which(executable) is not None


def _default_project_dir() -> Path:
    return Path(__file__).resolve().parent / "julia"


def _default_script_path() -> Path:
    return _default_project_dir() / "run_mps_tebd.jl"
