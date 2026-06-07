"""Subprocess bridge to a PEPSKit.jl iPEPS real-time simple-update kernel.

This backend models the **infinite-lattice (bulk) limit** of a 2D Rydberg quench
with PEPSKit.jl: a translation-invariant ``2x2`` iPEPS evolved by real-time simple
update (``exp(-iH dt)``) with CTMRG measurement. It supports the ``1r`` (2-level)
and ``01r`` (3-level) local structures. Site-dependent driving is restricted to
either uniform fields (``unit_cell="uniform"``) or an A/B sublattice pattern
(``unit_cell="sublattice"``); finer site dependence is rejected.
"""

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

from ryd_gate.backends.tenpy_mps.backends import _merge_pin_deltas, _pin_deltas_from_params
from ryd_gate.core.channel_lowering import (
    three_level_profiles_from_coeffs,
    two_level_drive_and_detuning_from_coeffs,
)
from ryd_gate.ir.evolution import EvolutionResult


class PEPSKitJuliaError(RuntimeError):
    """Raised when the PEPSKit.jl iPEPS subprocess bridge fails."""


class PEPSKitDriveError(ValueError):
    """Raised when a protocol requests driving an iPEPS cannot represent."""


@dataclass
class PEPSKitIPEPSBackend:
    """Run a compiled TN IR through a PEPSKit.jl iPEPS real-time simple update."""

    unit_cell: str = "uniform"  # "uniform" (homogeneous) | "sublattice" (A/B)
    bond_dim: int = 4  # iPEPS virtual bond dimension D
    env_dim: int = 16  # CTMRG environment/boundary dimension chi
    dt: float = 0.05
    trotter_order: int = 2  # 1 or 2 (2 => symmetrized Trotter gates)
    ctmrg_tol: float = 1e-8
    ctmrg_maxiter: int = 200
    su_trunc_atol: float = 1e-10
    init_noise: float = 1e-3
    julia_cmd: str | Sequence[str] = "julia"
    project_dir: str | os.PathLike | None = None
    script_path: str | os.PathLike | None = None
    timeout: float | None = None
    work_dir: str | os.PathLike | None = None
    keep_workdir: bool = False
    source_bashrc: bool = True
    threads: int | None = None
    sysimage: str | os.PathLike | None = None

    def evolve_ir(
        self,
        ir,
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        """Serialize a TN IR, run the PEPSKit iPEPS kernel, and parse its result."""
        if self.unit_cell not in {"uniform", "sublattice"}:
            raise PEPSKitDriveError("unit_cell must be 'uniform' or 'sublattice'.")
        payload = build_pepskit_payload(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
            dt=self.dt,
            unit_cell=self.unit_cell,
            bond_dim=self.bond_dim,
            env_dim=self.env_dim,
            trotter_order=self.trotter_order,
            ctmrg_tol=self.ctmrg_tol,
            ctmrg_maxiter=self.ctmrg_maxiter,
            su_trunc_atol=self.su_trunc_atol,
            init_noise=self.init_noise,
        )

        if self.keep_workdir:
            run_dir = Path(tempfile.mkdtemp(prefix="ryd-pepskit-", dir=self.work_dir))
            return self._run_in_dir(run_dir, payload, cleanup=False)

        with tempfile.TemporaryDirectory(prefix="ryd-pepskit-", dir=self.work_dir) as tmp:
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
            raise PEPSKitJuliaError(
                "PEPSKit.jl iPEPS kernel timed out after "
                f"{self.timeout} seconds. Increase `backend_options['timeout']`, "
                "especially for first-time Julia JIT compilation."
            ) from exc
        if proc.returncode != 0:
            raise PEPSKitJuliaError(
                "PEPSKit.jl iPEPS kernel failed with exit code "
                f"{proc.returncode}.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not output_npz.exists() or not output_json.exists():
            raise PEPSKitJuliaError(
                "PEPSKit.jl iPEPS kernel finished but did not produce result.npz/result.json."
            )
        return _load_pepskit_result(
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
            raise PEPSKitJuliaError(f"PEPSKit.jl kernel script not found: {script_path}")

        julia_args: list[str] = []
        sysimage = self._resolve_sysimage()
        if sysimage is not None:
            julia_args.append(f"--sysimage={sysimage}")
        julia_args += [
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
            raise PEPSKitJuliaError(
                f"Julia executable {executable!r} was not found on PATH. "
                "Pass backend_options={'julia_cmd': '/path/to/julia'} or enable source_bashrc."
            )
        return [
            "bash",
            "-lc",
            "source ~/.bashrc && exec " + " ".join(shlex.quote(part) for part in cmd),
        ]

    def _resolve_sysimage(self) -> Path | None:
        if self.sysimage is not None:
            candidate = Path(self.sysimage)
        elif os.environ.get("RYD_PEPSKIT_SYSIMAGE"):
            candidate = Path(os.environ["RYD_PEPSKIT_SYSIMAGE"])
        else:
            candidate = _default_project_dir() / "sysimages" / "ryd_pepskit.so"
        return candidate if candidate.exists() else None


_DRIVE_FIELDS = ("omega_R", "omega_hf", "delta_R", "delta_hf")


def build_pepskit_payload(
    ir,
    *,
    initial_state: str | np.ndarray | object,
    t_eval: np.ndarray | None,
    observables: list[str] | None,
    dt: float,
    unit_cell: str,
    bond_dim: int,
    env_dim: int,
    trotter_order: int,
    ctmrg_tol: float,
    ctmrg_maxiter: int,
    su_trunc_atol: float,
    init_noise: float,
) -> dict:
    """Build a JSON-serializable payload for the PEPSKit iPEPS kernel."""
    spec = ir.spec
    level_spec = spec.level_spec
    level_structure = spec.level_structure

    t_gate = float(ir.params["t_gate"])
    n_steps = max(1, int(np.ceil(t_gate / dt)))
    dt_actual = t_gate / n_steps
    record_steps = _record_steps(t_eval, dt_actual, n_steps) or [n_steps]

    static_pin = _pin_deltas_from_params(ir.params, spec.N)
    schedule = [
        _schedule_entry(ir, spec, level_structure, step, dt_actual, static_pin, unit_cell)
        for step in range(n_steps)
    ]

    return {
        "method": "pepskit_ipeps_su",
        "lattice": {
            "unit_cell": unit_cell,
            "Nr": 2,
            "Nc": 2,
            "physical_dim": int(level_spec.local_dim),
            "level_structure": level_structure,
            "levels": list(level_spec.levels),
            "V_nn": float(spec.V_nn),
        },
        "initial_state": _initial_state_payload(spec, initial_state, unit_cell),
        "schedule": schedule,
        "record_steps": record_steps,
        "observables": list(observables) if observables else ["n_r"],
        "runtime": {
            "dt": float(dt_actual),
            "requested_dt": float(dt),
            "n_steps": int(n_steps),
            "bond_dim": int(bond_dim),
            "env_dim": int(env_dim),
            "trotter_order": int(trotter_order),
            "ctmrg_tol": float(ctmrg_tol),
            "ctmrg_maxiter": int(ctmrg_maxiter),
            "su_trunc_atol": float(su_trunc_atol),
            "init_noise": float(init_noise),
        },
        "metadata": ir.metadata or {},
    }


def _schedule_entry(ir, spec, level_structure, step, dt_actual, static_pin, unit_cell):
    t_mid = (step + 0.5) * dt_actual
    coeffs = ir.protocol.get_drive_coefficients(float(t_mid), ir.params)
    if level_structure == "01r":
        profiles = three_level_profiles_from_coeffs(coeffs, spec)
        omega_R = _profile(profiles["omega_R"], spec.N)
        omega_hf = _profile(profiles["omega_hf"], spec.N)
        delta_R = _profile(profiles["delta_R"], spec.N)
        delta_hf = _profile(profiles["delta_hf"], spec.N)
        if static_pin is not None:
            delta_R = delta_R + static_pin
    else:
        omega_t, delta_t, channel_pin = two_level_drive_and_detuning_from_coeffs(coeffs, spec)
        omega_R = _profile(omega_t, spec.N)
        omega_hf = np.zeros(spec.N)
        pin = _merge_pin_deltas(static_pin, channel_pin, n_sites=spec.N)
        delta_R = np.full(spec.N, float(delta_t), dtype=float)
        if pin is not None:
            delta_R = delta_R + pin
        delta_hf = np.zeros(spec.N)

    fields = {"omega_R": omega_R, "omega_hf": omega_hf, "delta_R": delta_R, "delta_hf": delta_hf}
    entry: dict = {"step": step + 1, "t_mid": float(t_mid)}
    if unit_cell == "uniform":
        for name in _DRIVE_FIELDS:
            entry[name] = _reduce_uniform(fields[name], name)
    else:
        sub_a, sub_b = {}, {}
        for name in _DRIVE_FIELDS:
            a, b = _reduce_sublattice(fields[name], spec, name)
            sub_a[name] = a
            sub_b[name] = b
        entry["A"] = sub_a
        entry["B"] = sub_b
    return entry


def _reduce_uniform(profile, name: str) -> float:
    arr = np.asarray(profile, dtype=float)
    if not np.allclose(arr, arr.flat[0]):
        raise PEPSKitDriveError(
            f"unit_cell='uniform' but drive field {name!r} is site-dependent; "
            "use unit_cell='sublattice' or a homogeneous protocol."
        )
    return float(arr.flat[0])


def _reduce_sublattice(profile, spec, name: str) -> tuple[float, float]:
    arr = np.asarray(profile, dtype=float)
    sub = np.asarray(spec.sublattice)
    fallback = float(arr.flat[0])

    def constant(vals: np.ndarray) -> float:
        if vals.size == 0:
            return fallback
        if not np.allclose(vals, vals[0]):
            raise PEPSKitDriveError(
                f"drive field {name!r} varies within a sublattice; iPEPS supports "
                "only uniform or A/B sublattice patterns."
            )
        return float(vals[0])

    return constant(arr[sub > 0]), constant(arr[sub < 0])


def _initial_state_payload(spec, initial_state, unit_cell: str) -> dict:
    labels = _initial_labels(spec, initial_state)
    sub = np.asarray(spec.sublattice)
    if unit_cell == "uniform":
        if len(set(labels)) != 1:
            raise PEPSKitDriveError(
                "unit_cell='uniform' requires a homogeneous initial state; use unit_cell='sublattice'."
            )
        return {"pattern": "uniform", "label": labels[0]}

    def single(mask, fallback):
        present = {labels[i] for i in range(spec.N) if mask[i]}
        if len(present) > 1:
            raise PEPSKitDriveError("initial_state varies within a sublattice; not representable as A/B.")
        return present.pop() if present else fallback
    return {
        "pattern": "sublattice",
        "A": single(sub > 0, labels[0]),
        "B": single(sub < 0, labels[0]),
    }


def _initial_labels(spec, initial_state) -> list[str]:
    levels = set(spec.level_spec.levels)
    ground = "1" if "1" in levels else spec.level_spec.levels[0]
    if isinstance(initial_state, str):
        if initial_state in {"all_ground", "all_1"}:
            return [ground] * spec.N
        if initial_state == "all_r":
            return ["r"] * spec.N
        if initial_state in {"all_0", "all_zero"}:
            return ["0"] * spec.N
        if initial_state == "af1":
            return ["r" if s > 0 else ground for s in spec.sublattice]
        if initial_state == "af2":
            return ["r" if s < 0 else ground for s in spec.sublattice]
        raise PEPSKitDriveError(f"Unknown PEPSKit initial_state string: {initial_state!r}")
    arr = np.asarray(initial_state)
    if arr.shape != (spec.N,):
        raise PEPSKitDriveError(f"initial_state must have shape ({spec.N},), got {arr.shape}.")
    if arr.dtype.kind in {"U", "S", "O"}:
        return [str(x) for x in arr]
    return ["r" if int(c) == 1 else ground for c in arr]


def _load_pepskit_result(
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
    metadata.setdefault("backend", "pepskit")
    metadata.setdefault("method", "pepskit_ipeps_su")
    metadata["julia_stdout"] = stdout
    metadata["julia_stderr"] = stderr
    if workdir is not None:
        metadata["workdir"] = workdir
    times = arrays["times"] if "times" in arrays.files else None
    return EvolutionResult(
        psi_final="pepskit_ipeps_external",
        times=times,
        states=None,
        metadata=metadata,
    )


def _default_project_dir() -> Path:
    return Path(__file__).resolve().parent / "julia"


def _default_script_path() -> Path:
    return _default_project_dir() / "run_pepskit_ipeps.jl"


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
