"""YASTN fPEPS real-time adapter for 2D Rydberg lattice dynamics."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np

from ryd_gate.analysis.observables import line_pairs_from_reference
from ryd_gate.backends.tn_common.protocol_context import merge_pin_deltas, pin_deltas_from_params
from ryd_gate.core.level_structures import (
    three_level_profiles_from_coeffs,
    two_level_drive_and_detuning_from_coeffs,
)
from ryd_gate.ir import EvolutionResult


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
    ctm_chi: int | None = None
    ctm_iters: int = 50
    ctm_tol: float = 1e-8

    def _ctm_chi(self) -> int:
        """Environment bond dimension for converged CTM measurement."""
        return int(self.ctm_chi) if self.ctm_chi else max(2 * int(self.chi_max), 16)

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
            measured = _measure_yastn(
                fpeps, psi, ops, payload, observables, self.measurement_environment,
                ctm_chi=self._ctm_chi(), ctm_iters=int(self.ctm_iters), ctm_tol=float(self.ctm_tol),
            )
            for name, value in measured.items():
                obs_data[name].append(value)

        if 0 in record_steps:
            record(0.0)

        opts_svd = {"D_total": int(self.chi_max), "tol": float(self.svd_min)}
        env = _make_update_env(fpeps, psi, self.update_environment, opts_svd=opts_svd)
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

    def find_ground_state(
        self,
        ir,
        *,
        dtau_schedule: tuple[tuple[float, int], ...] = ((0.1, 30), (0.03, 30), (0.01, 40)),
        warmup_env: str | None = None,
        step_max_iter: int = 8,
        step_tol_iter: float = 1e-8,
        energy_convergence: bool = True,
        energy_tol: float = 1e-5,
        ctm_chi: int = 16,
        ctm_iters: int = 3,
        observables: list[str] | None = None,
        initial_state: str | np.ndarray | object = "af1",
    ) -> EvolutionResult:
        """Imaginary-time ground state of a *static* Rydberg/TFIM Hamiltonian.

        ``ir`` must carry a constant-in-time protocol (e.g.
        :class:`~ryd_gate.protocols.lattice_dynamics.TFIMQuenchProtocol`). The
        Hamiltonian is read from the first schedule step and applied as
        imaginary-time gates ``exp(-dtau H)`` over the ``(dtau, n_steps)`` schedule.

        Accelerations over a plain fixed NTU sweep:

        * **optional Simple-Update warmup** -- ``warmup_env="approximate"`` cools the
          first ``dtau`` stage with ``EnvApproximate`` before refining with
          ``self.update_environment`` (default NTU). Off by default: its boundary-MPS
          compression costs more than NTU on small clusters and only pays off on
          larger lattices.
        * **looser per-step bond optimisation** -- ``step_max_iter``/``step_tol_iter``
          (the Trotter+truncation error dwarfs a 1e-12 bond tolerance).
        * **energy-based early stop** -- a light CTM energy after each stage halts
          cooling once ``|dE| <= energy_tol * max(1, |E|)``.

        ``metadata["energy"]`` is the O(N) ground-state energy (1-site fields plus
        nearest-neighbour ``<n_r n_r>``) -- the cheap, robust cross-check quantity.
        Use a checkerboard seed (``"af1"``) to select the antiferromagnetic sector.
        """
        yastn, fpeps, gates, cfg = self._load_yastn()
        if observables is None:
            observables = ["m_s", "n_mean"]
        payload = build_yastn_peps_payload(
            ir, initial_state=initial_state, t_eval=None, observables=observables,
            dt=self.dt, chi_max=self.chi_max, svd_min=self.svd_min, use_cuda=self.use_cuda,
        )
        if not payload["schedule"]:
            raise YASTNPEPSError("find_ground_state requires a non-empty protocol schedule.")
        step0 = payload["schedule"][0]

        ops = _YASTNPEPSOps(yastn, cfg, payload["lattice"]["levels"])
        geom = fpeps.SquareLattice(
            dims=(int(payload["lattice"]["Lx"]), int(payload["lattice"]["Ly"])),
            boundary="obc",
        )
        psi = _yastn_product_peps(fpeps, geom, ops, payload)
        opts_svd = {"D_total": int(self.chi_max), "tol": float(self.svd_min)}

        truncation_error: list[float] = []
        n_steps = 0
        energy = float("nan")
        prev_E = None
        stopped_early = False
        for stage, (dtau, nsteps) in enumerate(dtau_schedule):
            env_name = warmup_env if (warmup_env and stage == 0) else self.update_environment
            env = _make_update_env(fpeps, psi, env_name, opts_svd=opts_svd)
            opts_post_truncation = None
            if env_name.lower() in {"ctm", "envctm"}:
                env.update_(dict(opts_svd))
                opts_post_truncation = {"opts_svd": dict(opts_svd)}
            step_gates = _yastn_imag_gates(gates, ops, payload, step0, float(dtau))
            for _ in range(int(nsteps)):
                infos = fpeps.evolution_step_(
                    env, step_gates, opts_svd,
                    initialization=self.initialization,
                    max_iter=int(step_max_iter),
                    tol_iter=float(step_tol_iter),
                    opts_post_truncation=opts_post_truncation,
                )
                psi = env.psi
                truncation_error.append(
                    max((float(getattr(info, "truncation_error", 0.0)) for info in infos), default=0.0))
                n_steps += 1
            if energy_convergence:
                energy, _ = _peps_energy_ctm(fpeps, psi, ops, payload, step0, ctm_chi, ctm_iters)
                if prev_E is not None and abs(energy - prev_E) <= energy_tol * max(1.0, abs(energy)):
                    stopped_early = True
                    break
                prev_E = energy

        measured = _measure_yastn(
            fpeps, psi, ops, payload, observables, self.measurement_environment,
            ctm_chi=self._ctm_chi(), ctm_iters=int(self.ctm_iters), ctm_tol=float(self.ctm_tol),
        )
        if not np.isfinite(energy):
            energy, _ = _peps_energy_ctm(fpeps, psi, ops, payload, step0, ctm_chi, ctm_iters)
        return EvolutionResult(
            psi_final=psi,
            metadata={
                **(ir.metadata or {}),
                "backend": "peps",
                "method": "peps_yastn_imag",
                "engine_package": "yastn",
                "algorithm": f"fpeps_imag_warmup-{warmup_env}_refine-{self.update_environment}",
                "level_structure": payload["lattice"]["level_structure"],
                "local_dim": int(payload["lattice"]["local_dim"]),
                "chi_max": int(self.chi_max),
                "dtau_schedule": [list(s) for s in dtau_schedule],
                "n_steps": int(n_steps),
                "stopped_early": stopped_early,
                "energy": float(energy),
                "max_truncation_error": max(truncation_error, default=0.0),
                "obs": measured,
            },
        )

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


def _make_update_env(fpeps, psi, name: str, opts_svd=None):
    key = name.lower()
    if key in {"ntu", "envntu"}:
        return fpeps.EnvNTU(psi)
    if key in {"approx", "approximate"}:
        # EnvApproximate compresses a boundary MPS, so it needs the SVD options.
        return fpeps.EnvApproximate(psi, opts_svd=opts_svd)
    if key in {"ctm", "envctm"}:
        return fpeps.EnvCTM(psi)
    raise ValueError("update_environment must be 'ntu', 'approximate', or 'ctm'.")


def _yastn_gates(gates, ops: _YASTNPEPSOps, payload: dict, step_data: dict, dt: float):
    """Second-order Trotter gate list ``[local(½dt), nn(dt), local(½dt)]`` for one step.

    Each ``gate_*_exp`` runs an ``eigh``/SVD on the GPU, and the drive is usually
    spatially uniform (global ``Omega``/``Delta``), so the per-site and per-bond
    exponentials repeat. We build each *distinct* gate once -- keyed by its field
    tuple / bond strength -- and place copies with ``Gate._replace(sites=...)``,
    the same mechanism YASTN's own ``gates.distribute`` uses. This collapses the
    ~56 gate builds per step down to the handful that are physically distinct
    (one local + one nn when the drive is uniform). The two outer local
    half-steps are identical, so the same list is reused for both.
    """
    local_gates = _local_gate_list(gates, ops, payload, step_data, 0.5j * dt)
    nn_gates = _nn_gate_list(gates, ops, payload, 1j * dt)
    return local_gates + nn_gates + local_gates


def _local_gate_list(gates, ops: _YASTNPEPSOps, payload: dict, step_data: dict, coeff: complex):
    """One ``exp(-coeff·H_local)`` gate per site, built once per distinct field tuple."""
    lattice = payload["lattice"]
    Ly = int(lattice["Ly"])
    inv_snake = np.asarray(lattice["inv_snake"], dtype=int)
    omega_R = np.asarray(step_data["omega_R_1d"], dtype=float)
    omega_hf = np.asarray(step_data["omega_hf_1d"], dtype=float)
    delta_R = np.asarray(step_data["delta_R_1d"], dtype=float)
    delta_hf = np.asarray(step_data["delta_hf_1d"], dtype=float)
    I = ops.I
    cache: dict = {}
    out = []
    for site_2d in range(int(lattice["N"])):
        pos = int(inv_snake[site_2d])
        coord = (site_2d // Ly, site_2d % Ly)
        key = (omega_R[pos], omega_hf[pos], delta_R[pos], delta_hf[pos])
        proto = cache.get(key)
        if proto is None:
            H = _local_hamiltonian(ops, *key)
            proto = gates.gate_local_exp(coeff, I, H, site=coord)
            cache[key] = proto
        out.append(proto._replace(sites=(coord,)))
    return out


def _nn_gate_list(gates, ops: _YASTNPEPSOps, payload: dict, coeff: complex):
    """One ``exp(-coeff·strength·n_r⊗n_r)`` gate per bond, built once per distinct strength.

    ``n_r⊗n_r`` is symmetric under site swap, so a single decomposed gate is valid
    on every bond (horizontal or vertical) -- ``apply_gate_`` routes the auxiliary
    leg from the bond geometry. This is exactly the reuse pattern of ``distribute``.
    """
    lattice = payload["lattice"]
    Ly = int(lattice["Ly"])
    Hnn = gates.fkron(ops.n_r, ops.n_r)
    I = ops.I
    cache: dict = {}
    out = []
    for i_pos, j_pos, strength in lattice["vdw_pairs_1d"]:
        i_2d = int(lattice["snake_to_2d"][int(i_pos) - 1])
        j_2d = int(lattice["snake_to_2d"][int(j_pos) - 1])
        if i_2d == j_2d or abs(float(strength)) == 0:
            continue
        ci = (i_2d // Ly, i_2d % Ly)
        cj = (j_2d // Ly, j_2d % Ly)
        proto = cache.get(float(strength))
        if proto is None:
            proto = gates.gate_nn_exp(coeff, I, float(strength) * Hnn, bond=(ci, cj))
            cache[float(strength)] = proto
        out.append(proto._replace(sites=(ci, cj)))
    return out


def _yastn_imag_gates(gates, ops: _YASTNPEPSOps, payload: dict, step_data: dict, dtau: float):
    """Imaginary-time Trotter gates ``exp(-dtau H)`` for ground-state NTU evolution.

    Mirrors :func:`_yastn_gates` but uses a *real* coefficient, so ``gate_*_exp``
    produces ``exp(-dtau H)`` instead of the real-time ``exp(-i dt H)``. The local
    field is split into two half-steps around the nearest-neighbour interaction
    (second-order Trotter).
    """
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
        out.append(gates.gate_local_exp(0.5 * dtau, ops.I, H, site=coord))

    Hnn = gates.fkron(ops.n_r, ops.n_r)
    for i_pos, j_pos, strength in lattice["vdw_pairs_1d"]:
        i_2d = int(lattice["snake_to_2d"][int(i_pos) - 1])
        j_2d = int(lattice["snake_to_2d"][int(j_pos) - 1])
        if i_2d == j_2d or abs(float(strength)) == 0:
            continue
        ci = (i_2d // Ly, i_2d % Ly)
        cj = (j_2d // Ly, j_2d % Ly)
        out.append(gates.gate_nn_exp(dtau, ops.I, float(strength) * Hnn, bond=(ci, cj)))

    for site_2d in range(int(lattice["N"])):
        pos = int(inv_snake[site_2d])
        coord = (site_2d // Ly, site_2d % Ly)
        H = _local_hamiltonian(ops, omega_R[pos], omega_hf[pos], delta_R[pos], delta_hf[pos])
        out.append(gates.gate_local_exp(0.5 * dtau, ops.I, H, site=coord))
    return out


def _peps_energy_ctm(fpeps, psi, ops, payload, step0, ctm_chi, ctm_iters):
    """Total ground-state energy <H> from a light CTM environment.

    O(N): one CTM convergence, four 1-site measurements (X_1r, X_01, n_r, n_1)
    combined with the per-site fields, plus nearest-neighbour ``<n_r n_r>`` over the
    ~2N interaction bonds. Far cheaper than the O(N^2) staggered structure factor and
    a robust, symmetry-agnostic cross-check quantity. Returns ``(energy, env)``.
    """
    lat = payload["lattice"]
    Ly = int(lat["Ly"])
    N = int(lat["N"])
    inv = np.asarray(lat["inv_snake"], dtype=int)
    oR = np.asarray(step0["omega_R_1d"], dtype=float)
    oh = np.asarray(step0["omega_hf_1d"], dtype=float)
    dR = np.asarray(step0["delta_R_1d"], dtype=float)
    dh = np.asarray(step0["delta_hf_1d"], dtype=float)

    env = fpeps.EnvCTM(psi)
    for _ in range(int(ctm_iters)):
        env.update_({"D_total": int(ctm_chi), "tol": 1e-8})

    xr = env.measure_1site(ops.X_1r)
    nr = env.measure_1site(ops.n_r)
    x01 = env.measure_1site(ops.X_01) if np.any(oh) else None
    n1 = env.measure_1site(ops.n_1) if np.any(dh) else None

    E = 0.0
    for s2 in range(N):
        pos = int(inv[s2])
        coord = (s2 // Ly, s2 % Ly)
        E += 0.5 * oR[pos] * float(np.real(xr[coord])) - dR[pos] * float(np.real(nr[coord]))
        if x01 is not None:
            E += 0.5 * oh[pos] * float(np.real(x01[coord]))
        if n1 is not None:
            E += -dh[pos] * float(np.real(n1[coord]))

    for i_pos, j_pos, strength in lat["vdw_pairs_1d"]:
        i2 = int(lat["snake_to_2d"][int(i_pos) - 1])
        j2 = int(lat["snake_to_2d"][int(j_pos) - 1])
        if i2 == j2 or abs(float(strength)) == 0:
            continue
        ci = (i2 // Ly, i2 % Ly)
        cj = (j2 // Ly, j2 % Ly)
        E += float(strength) * float(np.real(env.measure_nsite(ops.n_r, ops.n_r, sites=(ci, cj))))
    return E, env


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


def _measure_yastn(
    fpeps, psi, ops: _YASTNPEPSOps, payload: dict, observables: list[str], env_name: str,
    *, ctm_chi: int, ctm_iters: int, ctm_tol: float,
):
    lattice = payload["lattice"]
    Lx = int(lattice["Lx"])
    Ly = int(lattice["Ly"])
    n_sites = int(lattice["N"])
    out: dict[str, Any] = {}
    env = _measurement_env(fpeps, psi, env_name, ctm_chi=ctm_chi, ctm_iters=ctm_iters, ctm_tol=ctm_tol)
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
            # BP has no 2-site measurement; fall back to a *converged* CTM for the pair.
            pair_env = env if hasattr(env, "measure_nsite") else _measurement_env(
                fpeps, psi, "ctm", ctm_chi=ctm_chi, ctm_iters=ctm_iters, ctm_tol=ctm_tol,
            )
            values = []
            for i, j in line_pairs_from_reference(Lx, Ly, axis="horizontal"):
                ci = (int(i) // Ly, int(i) % Ly)
                cj = (int(j) // Ly, int(j) % Ly)
                zz = pair_env.measure_nsite(ops.Z, ops.Z, sites=(ci, cj))
                values.append(float(np.real(zz)) - z[int(i)] * z[int(j)])
            out[name] = np.asarray(values, dtype=float)
    return out


def _measurement_env(fpeps, psi, name: str, *, ctm_chi: int, ctm_iters: int, ctm_tol: float):
    key = name.lower()
    if key in {"ctm", "envctm"}:
        # A freshly constructed EnvCTM is a random, chi=1 *seed*; the CTM
        # environment is only physical at the CTMRG fixed point.  Iterate the
        # directional moves to convergence before measuring -- otherwise the
        # corner/edge tensors are meaningless and <n_r> can leave [0, 1].
        if not ctm_chi or int(ctm_chi) <= 1:
            raise YASTNPEPSError(f"CTM measurement requires ctm_chi > 1; got {ctm_chi!r}.")
        if int(ctm_iters) <= 0:
            raise YASTNPEPSError(f"CTM measurement requires ctm_iters > 0; got {ctm_iters!r}.")
        env = fpeps.EnvCTM(psi, init="eye")
        env.iterate_(
            {"D_total": int(ctm_chi), "tol": 1e-10},
            max_sweeps=int(ctm_iters),
            corner_tol=float(ctm_tol),
        )
        return env
    if key in {"bp", "envbp"}:
        env = fpeps.EnvBP(psi)
        env.update_()
        return env
    raise ValueError("measurement_environment must be 'ctm' or 'bp'.")
