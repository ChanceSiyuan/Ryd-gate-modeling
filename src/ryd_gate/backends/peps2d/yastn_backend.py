"""YASTN fPEPS real-time adapter for 2D Rydberg lattice dynamics."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np

from ryd_gate.analysis.spin_observables import line_pairs_from_reference
from ryd_gate.backends.itensor.backend import build_itensors_payload
from ryd_gate.ir.evolution import EvolutionResult


class YASTN2DTNError(RuntimeError):
    """Raised when the YASTN 2D-TN adapter cannot run."""


@dataclass
class YASTN2DTNBackend:
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
    measurement_environment: str = "ctm"
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
        payload["method"] = "2dtn_yastn"

        ops = _YASTNPEPSOps(yastn, cfg)
        geom = fpeps.SquareLattice(
            dims=(int(payload["lattice"]["Lx"]), int(payload["lattice"]["Ly"])),
            boundary="obc",
        )
        psi = _yastn_product_peps(fpeps, geom, ops, payload)
        if observables is None and t_eval is not None:
            observables = ["sigma_z"]
        obs_data = {name: [] for name in observables or []}
        recorded_times: list[float] = []
        record_steps = set(int(step) for step in payload["record_steps"])
        truncation_error = []

        def record(t_value: float) -> None:
            if not observables:
                return
            recorded_times.append(float(t_value))
            measured = _measure_yastn(fpeps, psi, ops, payload, observables, self.measurement_environment)
            for name, value in measured.items():
                obs_data[name].append(value)

        if 0 in record_steps:
            record(0.0)

        env = _make_update_env(fpeps, psi, self.update_environment)
        for step_data in payload["schedule"]:
            step_gates = _yastn_gates(gates, ops, payload, step_data, float(payload["runtime"]["dt"]))
            infos = fpeps.evolution_step_(
                env,
                step_gates,
                {"D_total": int(self.chi_max), "tol": float(self.svd_min)},
                initialization=self.initialization,
                max_iter=int(self.max_iter),
                tol_iter=float(self.tol_iter),
            )
            truncation_error.append(max((float(getattr(info, "truncation_error", 0.0)) for info in infos), default=0.0))
            psi = env.psi
            step = int(step_data["step"])
            if step in record_steps:
                record(step * float(payload["runtime"]["dt"]))

        for name in obs_data:
            obs_data[name] = np.asarray(obs_data[name])

        result = EvolutionResult(
            psi_final=psi,
            metadata={
                **(ir.metadata or {}),
                "backend": "2dtn",
                "method": "2dtn_yastn",
                "engine_package": "yastn",
                "algorithm": f"fpeps_{self.update_environment}",
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
                "max_truncation_error": max(truncation_error, default=0.0),
                "obs": obs_data,
            },
        )
        if recorded_times:
            result.times = np.asarray(recorded_times, dtype=float)
        return result

    def _load_yastn(self):
        if importlib.util.find_spec("yastn") is None:
            raise YASTN2DTNError(
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
                raise YASTN2DTNError("YASTN CUDA runs require yastn_backend='torch'.")
            if importlib.util.find_spec("torch") is None:
                raise YASTN2DTNError(
                    "YASTN CUDA runs require PyTorch with CUDA support. Install torch, "
                    "or set use_cuda=False for CPU smoke tests."
                )
            import torch

            if not torch.cuda.is_available():
                raise YASTN2DTNError("use_cuda=True but torch.cuda.is_available() is false.")

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


class _YASTNPEPSOps:
    def __init__(self, yastn, config) -> None:
        self.yastn = yastn
        self.config = config

    def matrix(self, values: np.ndarray):
        tensor = self.yastn.Tensor(config=self.config, s=(1, -1))
        tensor.set_block(ts=(), Ds=values.shape, val=np.asarray(values, dtype=complex))
        return tensor

    def vector(self, occ: int):
        values = np.array([1.0, 0.0], dtype=complex) if int(occ) == 0 else np.array([0.0, 1.0], dtype=complex)
        tensor = self.yastn.Tensor(config=self.config, s=(1,))
        tensor.set_block(ts=(), Ds=(2,), val=values)
        return tensor

    @property
    def I(self):  # noqa: E743 - conventional identity-operator symbol.
        return self.matrix(np.eye(2, dtype=complex))

    @property
    def Sx(self):
        return 0.5 * self.matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))

    @property
    def n(self):
        return self.matrix(np.diag([0.0, 1.0]).astype(complex))

    @property
    def Z(self):
        return self.matrix(np.diag([-1.0, 1.0]).astype(complex))


def _yastn_product_peps(fpeps, geom, ops: _YASTNPEPSOps, payload: dict):
    lattice = payload["lattice"]
    occ_1d = np.asarray(payload["initial_occupations_1d"], dtype=int)
    snake_to_2d = np.asarray(lattice["snake_to_2d"], dtype=int)
    vectors = {}
    for pos, occ in enumerate(occ_1d):
        site_2d = int(snake_to_2d[pos])
        vectors[(site_2d // int(lattice["Ly"]), site_2d % int(lattice["Ly"]))] = ops.vector(int(occ))
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
    omega = np.asarray(step_data["omega_1d"], dtype=float)
    delta = np.asarray(step_data["delta_1d"], dtype=float)
    out = []

    for site_2d in range(int(lattice["N"])):
        pos = int(inv_snake[site_2d])
        coord = (site_2d // Ly, site_2d % Ly)
        H = float(omega[pos]) * ops.Sx - float(delta[pos]) * ops.n
        out.append(gates.gate_local_exp(0.5j * dt, ops.I, H, site=coord))

    Hnn = gates.fkron(ops.n, ops.n)
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
        H = float(omega[pos]) * ops.Sx - float(delta[pos]) * ops.n
        out.append(gates.gate_local_exp(0.5j * dt, ops.I, H, site=coord))
    return out


def _measure_yastn(fpeps, psi, ops: _YASTNPEPSOps, payload: dict, observables: list[str], env_name: str):
    lattice = payload["lattice"]
    Lx = int(lattice["Lx"])
    Ly = int(lattice["Ly"])
    n_sites = int(lattice["N"])
    out: dict[str, Any] = {}
    env = _measurement_env(fpeps, psi, env_name)
    z_profile: np.ndarray | None = None

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
            out[name] = float(np.mean(0.5 * (sigma_z() + 1.0)))
        elif name in {"n_i", "n_r"}:
            out[name] = 0.5 * (sigma_z() + 1.0)
        elif name == "m_s":
            sublattice = np.asarray(lattice["sublattice"], dtype=float)
            out[name] = float(np.sum(sublattice * sigma_z()) / n_sites)
        elif name == "czz_centerline":
            z = sigma_z()
            values = []
            for i, j in line_pairs_from_reference(Lx, Ly, axis="horizontal"):
                ci = (int(i) // Ly, int(i) % Ly)
                cj = (int(j) // Ly, int(j) % Ly)
                zz = env.measure_nsite(ops.Z, ops.Z, sites=(ci, cj))
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
