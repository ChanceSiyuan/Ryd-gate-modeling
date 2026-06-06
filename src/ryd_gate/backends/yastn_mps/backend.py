"""GPU-capable MPS TDVP backend implemented with YASTN."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np

from ryd_gate.analysis.spin_observables import line_pairs_from_reference
from ryd_gate.backends.itensor.backend import _initial_occupations_1d, _record_steps
from ryd_gate.backends.tenpy_mps.backends import _merge_pin_deltas, _pin_deltas_from_params
from ryd_gate.core.channel_lowering import two_level_drive_and_detuning_from_coeffs
from ryd_gate.ir.evolution import EvolutionResult


class YASTNBackendError(RuntimeError):
    """Raised when the YASTN backend cannot run the requested problem."""


@dataclass
class YASTNMPSBackend:
    """Run MPS TDVP with YASTN.

    Parameters
    ----------
    chi_max
        Maximum MPS bond dimension used by two-site TDVP truncation.
    dt
        Requested TDVP time step in the same dimensionless units as the TN IR.
    svd_min
        Singular-value truncation tolerance.
    tdvp_method
        YASTN TDVP method: ``"1site"``, ``"2site"``, or ``"12site"``.
    yastn_backend
        ``"np"`` for CPU NumPy or ``"torch"`` for PyTorch. If ``use_cuda`` is
        true and this is omitted, ``"torch"`` is selected.
    device
        YASTN default device. If ``use_cuda`` is true and this is omitted,
        ``"cuda"`` is selected.
    require_gpu
        If true, fail when the selected backend/device is not CUDA-capable.
    """

    chi_max: int = 256
    dt: float = 0.05
    svd_min: float = 1e-10
    tdvp_method: str = "2site"
    order: str = "2nd"
    use_cuda: bool = False
    yastn_backend: str | None = None
    device: str | None = None
    dtype: str = "complex128"
    require_gpu: bool = False
    progress: bool = False

    def evolve_ir(
        self,
        ir,
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        if ir.spec.level_structure != "1r":
            raise ValueError("YASTN MPS TDVP currently supports level_structure='1r' only.")
        if ir.spec.bc != "open":
            raise ValueError("YASTN MPS TDVP currently supports open boundary conditions only.")
        if self.tdvp_method not in {"1site", "2site", "12site"}:
            raise ValueError("tdvp_method must be '1site', '2site', or '12site'.")

        yastn, ymps, cfg = self._load_yastn()
        ops = _YASTNLocalOps(yastn, cfg)
        psi = ymps.product_mps([
            ops.basis_vector(int(occ))
            for occ in _initial_occupations_1d(ir.spec, initial_state)
        ])

        t_gate = float(ir.params["t_gate"])
        if t_gate <= 0:
            raise ValueError("YASTN MPS TDVP requires a positive t_gate.")
        n_steps = max(1, int(np.ceil(t_gate / float(self.dt))))
        dt_actual = t_gate / n_steps
        record_at = set(_record_steps(t_eval, dt_actual, n_steps))

        if observables is None and t_eval is not None:
            observables = ["m_s", "n_mean"]
        obs_data = {name: [] for name in observables or []}
        recorded_times: list[float] = []

        def record(t_value: float) -> None:
            if not observables:
                return
            recorded_times.append(float(t_value))
            measured = _measure_observables(ymps, psi, ops, ir.spec, observables)
            for name, value in measured.items():
                obs_data[name].append(value)

        if 0 in record_at:
            record(0.0)

        generator = ymps.Generator(ir.spec.N, ops)
        for step in range(n_steps):
            t_mid = (step + 0.5) * dt_actual
            H = _build_yastn_mpo(ymps, generator, ops, ir, t_mid)
            list(ymps.tdvp_(
                psi,
                H,
                times=(0.0, dt_actual),
                dt=dt_actual,
                u=1j,
                method=self.tdvp_method,
                order=self.order,
                opts_svd={"D_total": int(self.chi_max), "tol": float(self.svd_min)},
                progressbar=bool(self.progress),
            ))

            step_num = step + 1
            if step_num in record_at:
                record(step_num * dt_actual)

        for name in obs_data:
            obs_data[name] = np.asarray(obs_data[name])

        result = EvolutionResult(
            psi_final=psi,
            metadata={
                **(ir.metadata or {}),
                "backend": "yastn_mps",
                "method": "mps_tdvp",
                "engine_package": "yastn",
                "accelerator": "cuda" if self._uses_cuda else "cpu",
                "gpu": bool(self._uses_cuda),
                "yastn_backend": self._selected_backend,
                "device": self._selected_device,
                "chi_max": int(self.chi_max),
                "dt": dt_actual,
                "n_steps": n_steps,
                "tdvp_method": self.tdvp_method,
                "order": self.order,
                "obs": obs_data,
            },
        )
        if recorded_times:
            result.times = np.asarray(recorded_times, dtype=float)
        return result

    def _load_yastn(self):
        if importlib.util.find_spec("yastn") is None:
            raise YASTNBackendError(
                "YASTN MPS TDVP requires yastn. Install it from GitHub with "
                "`pip install git+https://github.com/yastn/yastn.git`."
            )
        import yastn
        import yastn.tn.mps as ymps

        backend = self.yastn_backend or ("torch" if self.use_cuda else "np")
        device = self.device or ("cuda" if self.use_cuda else "cpu")
        if self.require_gpu or self.use_cuda:
            if backend != "torch":
                raise YASTNBackendError("YASTN CUDA runs require yastn_backend='torch'.")
            if importlib.util.find_spec("torch") is None:
                raise YASTNBackendError(
                    "YASTN CUDA runs require PyTorch with CUDA support. Install torch, "
                    "or set use_cuda=False for CPU smoke tests."
                )
            import torch

            if not torch.cuda.is_available():
                raise YASTNBackendError("use_cuda=True but torch.cuda.is_available() is false.")

        cfg = yastn.make_config(
            backend=backend,
            sym="none",
            default_device=device,
            default_dtype=self.dtype,
        )
        self._selected_backend = backend
        self._selected_device = device
        self._uses_cuda = backend == "torch" and str(device).startswith("cuda")
        return yastn, ymps, cfg


class _YASTNLocalOps:
    def __init__(self, yastn, config) -> None:
        self.yastn = yastn
        self.config = config

    def _matrix(self, values: np.ndarray):
        tensor = self.yastn.Tensor(config=self.config, s=(1, -1))
        tensor.set_block(ts=(), Ds=values.shape, val=np.asarray(values, dtype=complex))
        return tensor

    def basis_vector(self, occ: int):
        values = np.array([1.0, 0.0], dtype=complex) if int(occ) == 0 else np.array([0.0, 1.0], dtype=complex)
        tensor = self.yastn.Tensor(config=self.config, s=(1,))
        tensor.set_block(ts=(), Ds=(2,), val=values)
        return tensor

    def I(self, site: int | None = None):  # noqa: E743 - YASTN expects an identity operator named I.
        del site
        return self._matrix(np.eye(2, dtype=complex))

    def X(self, site: int | None = None):
        del site
        return self._matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))

    def Sx(self, site: int | None = None):
        return 0.5 * self.X(site)

    def n(self, site: int | None = None):
        del site
        return self._matrix(np.diag([0.0, 1.0]).astype(complex))

    def Z(self, site: int | None = None):
        del site
        return self._matrix(np.diag([-1.0, 1.0]).astype(complex))


def _build_yastn_mpo(ymps, generator, ops: _YASTNLocalOps, ir, t_mid: float):
    spec = ir.spec
    coeffs = ir.protocol.get_drive_coefficients(float(t_mid), ir.params)
    omega_t, delta_t, channel_pin = two_level_drive_and_detuning_from_coeffs(coeffs, spec)
    omega = _profile_in_1d(omega_t, spec)
    delta = _profile_in_1d(delta_t, spec)
    pin = _merge_pin_deltas(
        _pin_deltas_from_params(ir.params, spec.N),
        channel_pin,
        n_sites=spec.N,
    )
    if pin is not None:
        delta = delta + np.asarray(pin, dtype=float)[np.asarray(spec.snake_to_2d, dtype=int)]

    terms = []
    for site in range(spec.N):
        if abs(omega[site]) > 0:
            terms.append(ymps.Hterm(float(omega[site]), (site,), (ops.Sx(),)))
        if abs(delta[site]) > 0:
            terms.append(ymps.Hterm(float(-delta[site]), (site,), (ops.n(),)))
    for i_2d, j_2d, v_rel in spec.vdw_pairs:
        i = int(spec.inv_snake[int(i_2d)])
        j = int(spec.inv_snake[int(j_2d)])
        strength = float(spec.V_nn) * float(v_rel)
        if i != j and abs(strength) > 0:
            terms.append(ymps.Hterm(strength, (i, j), (ops.n(), ops.n())))
    return ymps.generate_mpo(generator.I(), terms, opts_svd={"tol": 5e-15})


def _profile_in_1d(value: Any, spec) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        profile_2d = np.full(spec.N, float(arr), dtype=float)
    else:
        if arr.shape != (spec.N,):
            raise ValueError(f"Expected scalar or length-{spec.N} profile, got {arr.shape}.")
        profile_2d = arr
    return profile_2d[np.asarray(spec.snake_to_2d, dtype=int)]


def _measure_observables(ymps, psi, ops: _YASTNLocalOps, spec, observables: list[str]) -> dict[str, Any]:
    z_1d = None
    n_1d = None
    out: dict[str, Any] = {}

    def sigma_z_1d() -> np.ndarray:
        nonlocal z_1d
        if z_1d is None:
            values = ymps.measure_1site(psi, ops.Z(), psi)
            z_1d = np.asarray([float(np.real(values[i])) for i in range(spec.N)], dtype=float)
        return z_1d

    def rydberg_occ_1d() -> np.ndarray:
        nonlocal n_1d
        if n_1d is None:
            n_1d = 0.5 * (sigma_z_1d() + 1.0)
        return n_1d

    for name in observables:
        if name in {"sigma_z", "z_i"}:
            z_2d = np.empty(spec.N, dtype=float)
            z_2d[np.asarray(spec.snake_to_2d, dtype=int)] = sigma_z_1d()
            out[name] = z_2d
        elif name == "n_mean":
            out[name] = float(np.mean(rydberg_occ_1d()))
        elif name in {"n_i", "n_r"}:
            n_2d = np.empty(spec.N, dtype=float)
            n_2d[np.asarray(spec.snake_to_2d, dtype=int)] = rydberg_occ_1d()
            out[name] = n_2d
        elif name == "m_s":
            z_2d = np.empty(spec.N, dtype=float)
            z_2d[np.asarray(spec.snake_to_2d, dtype=int)] = sigma_z_1d()
            out[name] = float(np.sum(spec.sublattice * z_2d) / spec.N)
        elif name == "czz_centerline":
            z = sigma_z_1d()
            values = []
            for i_2d, j_2d in line_pairs_from_reference(spec.Lx, spec.Ly, axis="horizontal"):
                i = int(spec.inv_snake[int(i_2d)])
                j = int(spec.inv_snake[int(j_2d)])
                zz = ymps.measure_nsite(psi, ops.Z(), ops.Z(), ket=psi, sites=(i, j))
                values.append(float(np.real(zz)) - z[i] * z[j])
            out[name] = np.asarray(values, dtype=float)
    return out
