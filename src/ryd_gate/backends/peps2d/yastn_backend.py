"""YASTN fPEPS real-time adapter for 2D Rydberg lattice dynamics."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np

from ryd_gate.analysis.spin_observables import line_pairs_from_reference
from ryd_gate.backends.tn_common.protocol_context import merge_pin_deltas, pin_deltas_from_params
from ryd_gate.core.channel_lowering import (
    three_level_profiles_from_coeffs,
    two_level_drive_and_detuning_from_coeffs,
)
from ryd_gate.ir.evolution import EvolutionResult


class YASTNPEPSError(RuntimeError):
    """Raised when the YASTN PEPS adapter cannot run."""


@dataclass
class YASTNPEPSBackend:
    """2D fPEPS real-time evolution through YASTN.

    The evolution kernel uses YASTN's finite PEPS ``evolution_step_`` with an
    ``EnvNTU`` update by default.  GPU support is inherited from YASTN's
    PyTorch backend by setting ``yastn_backend="torch", device="cuda"``.
    """

    chi_max: int = 64
    dt: float = 0.05
    svd_min: float = 1e-10
    use_cuda: bool = False
    yastn_backend: str | None = None
    device: str | None = None
    dtype: str = "complex128"
    update_environment: str = "ntu"
    measurement_environment: str = "bp"
    initialization: str = "SVD"
    max_iter: int = 20
    tol_iter: float = 1e-12
    require_gpu: bool = False

    def evolve_ir(
        self,
        ir,
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        yastn, fpeps, gates, cfg = self._load_yastn()
        payload = build_yastn_peps_payload(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
            dt=self.dt,
            chi_max=self.chi_max,
            svd_min=self.svd_min,
            use_cuda=self.use_cuda,
        )

        ops = _YASTNPEPSOps(yastn, cfg, payload["lattice"]["levels"])
        geom = fpeps.SquareLattice(
            dims=(int(payload["lattice"]["Lx"]), int(payload["lattice"]["Ly"])),
            boundary="obc",
        )
        psi = _yastn_product_peps(fpeps, geom, ops, payload)
        if observables is None and t_eval is not None:
            observables = ["n_mean", "n_r"]
        obs_data = {name: [] for name in observables or []}
        recorded_times: list[float] = []
        record_steps = set(int(step) for step in payload["record_steps"])
        truncation_error = []
        infoss = []

        def record(t_value: float) -> None:
            if not observables:
                return
            recorded_times.append(float(t_value))
            measured = _measure_yastn(fpeps, psi, ops, payload, observables, self.measurement_environment)
            for name, value in measured.items():
                obs_data[name].append(value)

        if 0 in record_steps:
            record(0.0)

        opts_svd = {"D_total": int(self.chi_max), "tol": float(self.svd_min)}
        env = _make_update_env(fpeps, psi, self.update_environment)
        opts_post_truncation = None
        if self.update_environment.lower() in {"ctm", "envctm"}:
            env.update_(dict(opts_svd))
            opts_post_truncation = {"opts_svd": dict(opts_svd)}
        for step_data in payload["schedule"]:
            step_gates = _yastn_gates(gates, ops, payload, step_data, float(payload["runtime"]["dt"]))
            infos = fpeps.evolution_step_(
                env,
                step_gates,
                opts_svd,
                initialization=self.initialization,
                max_iter=int(self.max_iter),
                tol_iter=float(self.tol_iter),
                opts_post_truncation=opts_post_truncation,
            )
            infoss.append(infos)
            truncation_error.append(max((float(getattr(info, "truncation_error", 0.0)) for info in infos), default=0.0))
            psi = env.psi
            step = int(step_data["step"])
            if step in record_steps:
                record(step * float(payload["runtime"]["dt"]))

        for name in obs_data:
            obs_data[name] = np.asarray(obs_data[name])

        accumulated_truncation_error = None
        if infoss and hasattr(fpeps, "accumulated_truncation_error"):
            try:
                accumulated_truncation_error = float(fpeps.accumulated_truncation_error(infoss))
            except Exception:
                accumulated_truncation_error = None

        result = EvolutionResult(
            psi_final=psi,
            metadata={
                **(ir.metadata or {}),
                "backend": "peps",
                "method": "peps_yastn",
                "engine_package": "yastn",
                "algorithm": f"fpeps_{self.update_environment}",
                "level_structure": payload["lattice"]["level_structure"],
                "local_dim": int(payload["lattice"]["local_dim"]),
                "measurement_environment": self.measurement_environment,
                "accelerator": "cuda" if self._uses_cuda else "cpu",
                "gpu": bool(self._uses_cuda),
                "yastn_backend": self._selected_backend,
                "device": self._selected_device,
                "chi_max": int(self.chi_max),
                "dt": float(payload["runtime"]["dt"]),
                "n_steps": int(payload["runtime"]["n_steps"]),
                "svd_min": float(self.svd_min),
                "truncation_error": np.asarray(truncation_error, dtype=float),
                "accumulated_truncation_error": accumulated_truncation_error,
                "max_truncation_error": max(truncation_error, default=0.0),
                "obs": obs_data,
            },
        )
        if recorded_times:
            result.times = np.asarray(recorded_times, dtype=float)
        return result

    def _load_yastn(self):
        if importlib.util.find_spec("yastn") is None:
            raise YASTNPEPSError(
                "engine_package='yastn' requires yastn. Install it from GitHub with "
                "`pip install git+https://github.com/yastn/yastn.git`."
            )
        import yastn
        import yastn.tn.fpeps as fpeps
        import yastn.tn.fpeps.gates as gates

        backend = self.yastn_backend or ("torch" if self.use_cuda else "np")
        device = self.device or ("cuda" if self.use_cuda else "cpu")
        if self.require_gpu or self.use_cuda:
            if backend != "torch":
                raise YASTNPEPSError("YASTN CUDA runs require yastn_backend='torch'.")
            if importlib.util.find_spec("torch") is None:
                raise YASTNPEPSError(
                    "YASTN CUDA runs require PyTorch with CUDA support. Install torch, "
                    "or set use_cuda=False for CPU smoke tests."
                )
            import torch

            if not torch.cuda.is_available():
                raise YASTNPEPSError("use_cuda=True but torch.cuda.is_available() is false.")

        cfg = yastn.make_config(
            backend=backend,
            sym="none",
            default_device=device,
            default_dtype=self.dtype,
        )
        self._selected_backend = backend
        self._selected_device = device
        self._uses_cuda = backend == "torch" and str(device).startswith("cuda")
        return yastn, fpeps, gates, cfg


def build_yastn_peps_payload(
    ir,
    *,
    initial_state: str | np.ndarray | object,
    t_eval: np.ndarray | None,
    observables: list[str] | None,
    dt: float,
    chi_max: int,
    svd_min: float,
    use_cuda: bool,
) -> dict[str, Any]:
    """Build a YASTN finite-PEPS payload directly from the TN IR.

    Unlike the Julia ITensors bridges, this lowering keeps the local physical
    dimension from the central level spec, so ``01r`` becomes a genuine qutrit
    PEPS rather than an effective two-level model.
    """
    spec = ir.spec
    if spec.level_structure not in {"1r", "01r"}:
        raise ValueError("YASTN PEPS supports TN level_structure '1r' and '01r' only.")

    t_gate = float(ir.params["t_gate"])
    if t_gate <= 0:
        raise ValueError("YASTN PEPS requires a positive t_gate.")
    n_steps = max(1, int(np.ceil(t_gate / float(dt))))
    dt_actual = t_gate / n_steps

    schedule = _drive_schedule(ir, dt_actual=dt_actual, n_steps=n_steps)
    initial_labels_1d, initial_superposition = _initial_state_payload_entries(spec, initial_state)
    return {
        "method": "peps_yastn",
        "metadata": ir.metadata or {},
        "lattice": {
            "Lx": int(spec.Lx),
            "Ly": int(spec.Ly),
            "N": int(spec.N),
            "sublattice": np.asarray(spec.sublattice, dtype=float),
            "level_structure": spec.level_structure,
            "levels": tuple(spec.level_spec.levels),
            "local_dim": int(spec.level_spec.local_dim),
            "snake_to_2d": np.asarray(spec.snake_to_2d, dtype=int),
            "inv_snake": np.asarray(spec.inv_snake, dtype=int),
            "vdw_pairs_1d": [
                [
                    int(spec.inv_snake[int(i)]) + 1,
                    int(spec.inv_snake[int(j)]) + 1,
                    float(spec.V_nn) * float(v_rel),
                ]
                for i, j, v_rel in spec.vdw_pairs
            ],
        },
        "initial_labels_1d": initial_labels_1d,
        "initial_superposition": initial_superposition,
        "record_steps": _record_steps(t_eval, dt_actual, n_steps),
        "observables": list(observables or []),
        "schedule": schedule,
        "runtime": {
            "dt": dt_actual,
            "requested_dt": float(dt),
            "n_steps": n_steps,
            "chi_max": int(chi_max),
            "svd_min": float(svd_min),
            "use_cuda": bool(use_cuda),
        },
    }


def _drive_schedule(ir, *, dt_actual: float, n_steps: int) -> list[dict[str, Any]]:
    spec = ir.spec
    static_pin = pin_deltas_from_params(ir.params, spec.N)
    schedule = []
    for step in range(n_steps):
        t_mid = (step + 0.5) * dt_actual
        coeffs = ir.protocol.get_drive_coefficients(t_mid, ir.params)
        if spec.level_structure == "01r":
            profiles = three_level_profiles_from_coeffs(coeffs, spec)
            if static_pin is not None:
                profiles["delta_R"] = profiles["delta_R"] + static_pin
        else:
            omega_t, delta_t, channel_pin = two_level_drive_and_detuning_from_coeffs(coeffs, spec)
            pin = merge_pin_deltas(static_pin, channel_pin, n_sites=spec.N)
            profiles = {
                "omega_R": _profile(omega_t, spec.N),
                "omega_hf": np.zeros(spec.N, dtype=float),
                "delta_R": np.full(spec.N, float(delta_t), dtype=float),
                "delta_hf": np.zeros(spec.N, dtype=float),
            }
            if pin is not None:
                profiles["delta_R"] = profiles["delta_R"] + pin

        order = np.asarray(spec.snake_to_2d, dtype=int)
        schedule.append(
            {
                "step": step + 1,
                "t_mid": float(t_mid),
                "omega_R_1d": np.asarray(profiles["omega_R"], dtype=float)[order],
                "omega_hf_1d": np.asarray(profiles["omega_hf"], dtype=float)[order],
                "delta_R_1d": np.asarray(profiles["delta_R"], dtype=float)[order],
                "delta_hf_1d": np.asarray(profiles["delta_hf"], dtype=float)[order],
            }
        )
    return schedule


def _profile(value: float | np.ndarray, n_sites: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_sites, float(arr), dtype=float)
    if arr.shape != (n_sites,):
        raise ValueError(f"Profile must be scalar or length-{n_sites}; got {arr.shape}.")
    return arr.astype(float, copy=False)


def _record_steps(t_eval: np.ndarray | None, dt_actual: float, n_steps: int) -> list[int]:
    if t_eval is None:
        return []
    steps = set()
    for t_req in np.asarray(t_eval, dtype=float):
        step = int(round(float(t_req) / dt_actual))
        steps.add(max(0, min(step, n_steps)))
    return sorted(steps)


def _initial_labels_1d(spec, initial_state: str | np.ndarray | object) -> list[str]:
    labels_2d = _initial_labels_2d(spec, initial_state)
    return [labels_2d[int(i)] for i in np.asarray(spec.snake_to_2d, dtype=int)]


def _initial_labels_2d(spec, initial_state: str | np.ndarray | object) -> list[str]:
    if isinstance(initial_state, str):
        return _named_initial_labels(spec, initial_state)
    if isinstance(initial_state, (list, tuple)):
        arr = np.asarray(initial_state)
    else:
        arr = np.asarray(initial_state)
    if arr.shape != (spec.N,):
        raise ValueError(f"initial_state must have shape ({spec.N},), got {arr.shape}.")
    if arr.dtype.kind in {"U", "S", "O"}:
        labels = [str(label) for label in arr]
        _validate_labels(spec, labels)
        return labels
    occ = arr.astype(int)
    labels = ["r" if int(value) == 1 else "1" for value in occ]
    _validate_labels(spec, labels)
    return labels


def _named_initial_labels(spec, name: str) -> list[str]:
    if name in {"all_ground", "all_1"}:
        labels = ["1"] * spec.N
    elif name in {"all_0", "all_zero"}:
        labels = ["0"] * spec.N
    elif name == "all_r":
        labels = ["r"] * spec.N
    elif name == "af1":
        labels = ["r" if s > 0 else "1" for s in spec.sublattice]
    elif name == "af2":
        labels = ["r" if s < 0 else "1" for s in spec.sublattice]
    else:
        raise ValueError(f"Unknown YASTN PEPS initial_state string: {name!r}.")
    _validate_labels(spec, labels)
    return labels


def _validate_labels(spec, labels: list[str]) -> None:
    allowed = set(spec.level_spec.levels)
    unknown = sorted(set(labels) - allowed)
    if unknown:
        raise ValueError(f"Unknown level label(s) for {spec.level_structure}: {unknown}.")


def _initial_state_payload_entries(spec, initial_state):
    """Return ``(initial_labels_1d, initial_superposition)`` for the payload.

    For ``initial_state="plus"`` the PEPS is built from a uniform per-site
    superposition (|0>+|1>)/√2 instead of basis labels.
    """
    if isinstance(initial_state, str) and initial_state == "plus":
        from ryd_gate.core.states import plus_local_amplitudes

        amps = [complex(a) for a in plus_local_amplitudes(spec.level_spec.levels)]
        return None, amps
    return _initial_labels_1d(spec, initial_state), None


class _YASTNPEPSOps:
    def __init__(self, yastn, config, levels) -> None:
        self.yastn = yastn
        self.config = config
        self.levels = tuple(str(level) for level in levels)
        self.dim = len(self.levels)

    def matrix(self, values: np.ndarray):
        tensor = self.yastn.Tensor(config=self.config, s=(1, -1))
        tensor.set_block(ts=(), Ds=values.shape, val=np.asarray(values, dtype=complex))
        return tensor

    def vector(self, label: str):
        values = np.zeros(self.dim, dtype=complex)
        values[self.index(label)] = 1.0
        tensor = self.yastn.Tensor(config=self.config, s=(1,))
        tensor.set_block(ts=(), Ds=(self.dim,), val=values)
        return tensor

    def superposition_vector(self, amps):
        values = np.asarray(amps, dtype=complex)
        if values.shape != (self.dim,):
            raise ValueError(f"superposition amps must have shape ({self.dim},); got {values.shape}.")
        tensor = self.yastn.Tensor(config=self.config, s=(1,))
        tensor.set_block(ts=(), Ds=(self.dim,), val=values)
        return tensor

    def index(self, label: str) -> int:
        try:
            return self.levels.index(label)
        except ValueError:
            raise ValueError(f"Unknown PEPS level label {label!r}; available levels are {self.levels}.") from None

    def projector(self, label: str):
        mat = np.zeros((self.dim, self.dim), dtype=complex)
        if label in self.levels:
            idx = self.index(label)
            mat[idx, idx] = 1.0
        return self.matrix(mat)

    def x_between(self, lower: str, upper: str):
        mat = np.zeros((self.dim, self.dim), dtype=complex)
        if lower in self.levels and upper in self.levels:
            lo = self.index(lower)
            up = self.index(upper)
            mat[lo, up] = 1.0
            mat[up, lo] = 1.0
        return self.matrix(mat)

    @property
    def I(self):  # noqa: E743 - conventional identity-operator symbol.
        return self.matrix(np.eye(self.dim, dtype=complex))

    @property
    def X_01(self):
        return self.x_between("0", "1")

    @property
    def X_1r(self):
        return self.x_between("1", "r")

    @property
    def n_0(self):
        return self.projector("0")

    @property
    def n_1(self):
        return self.projector("1")

    @property
    def n_r(self):
        return self.projector("r")

    @property
    def Z(self):
        return 2.0 * self.n_r - self.I


def _yastn_product_peps(fpeps, geom, ops: _YASTNPEPSOps, payload: dict):
    lattice = payload["lattice"]
    snake_to_2d = np.asarray(lattice["snake_to_2d"], dtype=int)
    superposition = payload.get("initial_superposition")
    Ly = int(lattice["Ly"])
    vectors = {}
    for pos in range(len(snake_to_2d)):
        site_2d = int(snake_to_2d[pos])
        coord = (site_2d // Ly, site_2d % Ly)
        if superposition is not None:
            vectors[coord] = ops.superposition_vector(superposition)
        else:
            vectors[coord] = ops.vector(str(payload["initial_labels_1d"][pos]))
    return fpeps.product_peps(geom, vectors)


def _make_update_env(fpeps, psi, name: str):
    key = name.lower()
    if key in {"ntu", "envntu"}:
        return fpeps.EnvNTU(psi)
    if key in {"approx", "approximate"}:
        return fpeps.EnvApproximate(psi)
    if key in {"ctm", "envctm"}:
        return fpeps.EnvCTM(psi)
    raise ValueError("update_environment must be 'ntu', 'approximate', or 'ctm'.")


def _yastn_gates(gates, ops: _YASTNPEPSOps, payload: dict, step_data: dict, dt: float):
    lattice = payload["lattice"]
    Ly = int(lattice["Ly"])
    inv_snake = np.asarray(lattice["inv_snake"], dtype=int)
    omega_R = np.asarray(step_data["omega_R_1d"], dtype=float)
    omega_hf = np.asarray(step_data["omega_hf_1d"], dtype=float)
    delta_R = np.asarray(step_data["delta_R_1d"], dtype=float)
    delta_hf = np.asarray(step_data["delta_hf_1d"], dtype=float)
    out = []

    for site_2d in range(int(lattice["N"])):
        pos = int(inv_snake[site_2d])
        coord = (site_2d // Ly, site_2d % Ly)
        H = _local_hamiltonian(ops, omega_R[pos], omega_hf[pos], delta_R[pos], delta_hf[pos])
        out.append(gates.gate_local_exp(0.5j * dt, ops.I, H, site=coord))

    Hnn = gates.fkron(ops.n_r, ops.n_r)
    for i_pos, j_pos, strength in lattice["vdw_pairs_1d"]:
        i_2d = int(lattice["snake_to_2d"][int(i_pos) - 1])
        j_2d = int(lattice["snake_to_2d"][int(j_pos) - 1])
        if i_2d == j_2d or abs(float(strength)) == 0:
            continue
        ci = (i_2d // Ly, i_2d % Ly)
        cj = (j_2d // Ly, j_2d % Ly)
        out.append(gates.gate_nn_exp(1j * dt, ops.I, float(strength) * Hnn, bond=(ci, cj)))

    for site_2d in range(int(lattice["N"])):
        pos = int(inv_snake[site_2d])
        coord = (site_2d // Ly, site_2d % Ly)
        H = _local_hamiltonian(ops, omega_R[pos], omega_hf[pos], delta_R[pos], delta_hf[pos])
        out.append(gates.gate_local_exp(0.5j * dt, ops.I, H, site=coord))
    return out


def _local_hamiltonian(
    ops: _YASTNPEPSOps,
    omega_R: float,
    omega_hf: float,
    delta_R: float,
    delta_hf: float,
):
    return (
        0.5 * float(omega_R) * ops.X_1r
        + 0.5 * float(omega_hf) * ops.X_01
        - float(delta_R) * ops.n_r
        - float(delta_hf) * ops.n_1
    )


def _measure_yastn(fpeps, psi, ops: _YASTNPEPSOps, payload: dict, observables: list[str], env_name: str):
    lattice = payload["lattice"]
    Lx = int(lattice["Lx"])
    Ly = int(lattice["Ly"])
    n_sites = int(lattice["N"])
    out: dict[str, Any] = {}
    env = _measurement_env(fpeps, psi, env_name)
    z_profile: np.ndarray | None = None
    level_profiles: dict[str, np.ndarray] = {}

    def level_occ(level: str) -> np.ndarray:
        if level not in level_profiles:
            values = np.empty(n_sites, dtype=float)
            measured = env.measure_1site(ops.projector(level))
            for site in range(n_sites):
                coord = (site // Ly, site % Ly)
                values[site] = float(np.real(measured[coord]))
            level_profiles[level] = values
        return level_profiles[level]

    def sigma_z() -> np.ndarray:
        nonlocal z_profile
        if z_profile is None:
            values = np.empty(n_sites, dtype=float)
            measured = env.measure_1site(ops.Z)
            for site in range(n_sites):
                coord = (site // Ly, site % Ly)
                values[site] = float(np.real(measured[coord]))
            z_profile = values
        return z_profile

    for name in observables:
        if name in {"sigma_z", "z_i"}:
            out[name] = sigma_z().copy()
        elif name == "n_mean":
            out[name] = float(np.mean(level_occ("r")))
        elif name in {"n_i", "n_r"}:
            out[name] = level_occ("r").copy()
        elif name in {"n_0", "n_1"}:
            out[name] = level_occ(name[-1]).copy()
        elif name == "m_s":
            sublattice = np.asarray(lattice["sublattice"], dtype=float)
            out[name] = float(np.sum(sublattice * sigma_z()) / n_sites)
        elif name in {"czz", "czz_centerline"}:
            z = sigma_z()
            pair_env = env if hasattr(env, "measure_nsite") else fpeps.EnvCTM(psi)
            values = []
            for i, j in line_pairs_from_reference(Lx, Ly, axis="horizontal"):
                ci = (int(i) // Ly, int(i) % Ly)
                cj = (int(j) // Ly, int(j) % Ly)
                zz = pair_env.measure_nsite(ops.Z, ops.Z, sites=(ci, cj))
                values.append(float(np.real(zz)) - z[int(i)] * z[int(j)])
            out[name] = np.asarray(values, dtype=float)
    return out


def _measurement_env(fpeps, psi, name: str):
    key = name.lower()
    if key in {"ctm", "envctm"}:
        return fpeps.EnvCTM(psi)
    if key in {"bp", "envbp"}:
        env = fpeps.EnvBP(psi)
        env.update_()
        return env
    raise ValueError("measurement_environment must be 'ctm' or 'bp'.")
