"""YASTN fPEPS real-time adapter for 2D Rydberg lattice dynamics."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np

from ryd_gate.backends.tn_common._expressions import (
    PEPS_MAX_TERM_SITES,
    preflight_tn_expressions,
    tn_basis,
)
from ryd_gate.backends.tn_common.initial_state import level_labels
from ryd_gate.backends.tn_common.protocol_context import (
    TNSegment,
    analog3_dt_guard,
    merge_pin_deltas,
    pin_deltas_from_params,
    plan_tn_segments,
    require_tn_dt,
)
from ryd_gate.core.level_structures import (
    three_level_profiles_from_coeffs,
    two_level_drive_and_detuning_from_coeffs,
)
from ryd_gate.ir import EvolutionResult, validate_evolution_request


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
    dt: float | None = None
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
        observables=None,
    ) -> EvolutionResult:
        """Evolve a compiled TN IR with YASTN finite-PEPS Trotter steps.

        ``t_eval`` selects measurement times only (never the evolution
        endpoint) and is returned exactly as ``result.times``; ``observables``
        is a ``dict[str, ObservableExpr]``.  PEPS expectations are evaluated at
        the BP/CTM environment fixed point, i.e. environment-normalized; PEPS
        evolution is Hermitian-only, so the survival norm is identically 1 and
        normalized values coincide with the raw ``<psi|O|psi>`` contract.
        """
        spec = ir.spec
        t_gate = float(ir.params["t_gate"])
        times, exprs = validate_evolution_request(t_gate, t_eval, observables)
        preflight_tn_expressions(exprs, spec, backend="peps", max_term_sites=PEPS_MAX_TERM_SITES)
        dt = require_tn_dt(self.dt)
        analog3_dt_guard(spec, dt)
        out_times = times if times is not None else np.array([t_gate])
        record_at_start, segments = plan_tn_segments(t_gate, out_times, dt)

        yastn, fpeps, gates, cfg = self._load_yastn()
        payload = build_yastn_peps_payload(
            ir,
            initial_state=initial_state,
            segments=segments,
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

        lowered, has_pair_terms = _lower_peps_expressions(exprs, ops, int(spec.Ly))
        records: dict[str, list[complex]] = {label: [] for label in exprs}
        measurement_env_info: list[dict] = []
        truncation_error = []
        infoss = []

        def record() -> None:
            if not exprs:
                return
            measured, env_info = _measure_lowered_peps(
                fpeps, psi, lowered, self.measurement_environment,
                ctm_chi=self._ctm_chi(), ctm_iters=int(self.ctm_iters), ctm_tol=float(self.ctm_tol),
            )
            measurement_env_info.append(env_info)
            for label, value in measured.items():
                records[label].append(value)

        if record_at_start:
            record()

        opts_svd = {"D_total": int(self.chi_max), "tol": float(self.svd_min)}
        env = _make_update_env(fpeps, psi, self.update_environment, opts_svd=opts_svd)
        opts_post_truncation = None
        if self.update_environment.lower() in {"ctm", "envctm"}:
            env.update_(dict(opts_svd))
            opts_post_truncation = {"opts_svd": dict(opts_svd)}
        for step_data in payload["schedule"]:
            step_gates = _yastn_gates(gates, ops, payload, step_data, float(step_data["dt"]))
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
            if step_data["record"]:
                record()

        expectations = {
            label: np.asarray(values, dtype=complex) for label, values in records.items()
        }

        accumulated_truncation_error = None
        if infoss and hasattr(fpeps, "accumulated_truncation_error"):
            try:
                accumulated_truncation_error = float(fpeps.accumulated_truncation_error(infoss))
            except Exception:
                accumulated_truncation_error = None

        metadata = {
            **(ir.metadata or {}),
            "backend": "peps",
            "method": "peps_yastn",
            "engine_package": "yastn",
            "algorithm": f"fpeps_{self.update_environment}",
            "level_structure": payload["lattice"]["level_structure"],
            "local_dim": int(payload["lattice"]["local_dim"]),
            "measurement_environment": self.measurement_environment,
            "ctm_chi": self._ctm_chi(),
            "ctm_iters": int(self.ctm_iters),
            "ctm_tol": float(self.ctm_tol),
            "measurement_env_info": measurement_env_info,
            "accelerator": "cuda" if self._uses_cuda else "cpu",
            "gpu": bool(self._uses_cuda),
            "yastn_backend": self._selected_backend,
            "device": self._selected_device,
            "chi_max": int(self.chi_max),
            "dt": dt,
            "n_steps": len(payload["schedule"]),
            "svd_min": float(self.svd_min),
            "truncation_error": np.asarray(truncation_error, dtype=float),
            "accumulated_truncation_error": accumulated_truncation_error,
            "max_truncation_error": max(truncation_error, default=0.0),
        }
        if has_pair_terms and self.measurement_environment.lower() in {"bp", "envbp"}:
            # BP has no two-point measurement; the documented fallback builds one
            # converged CTM environment per sampling anchor for the pair terms.
            metadata["pair_measurement_environment"] = "ctm"
        return EvolutionResult(
            final_state=psi,
            times=out_times,
            expectations=expectations,
            metadata=metadata,
            basis=tn_basis(spec),
        )

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
        observables=None,
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

        ``observables`` is an optional ``dict[str, ObservableExpr]`` measured
        once on the converged state; a ground-state search has no evolution
        clock, so the result carries the minimal time axis ``times == [0.0]``
        and each expectation is a shape-``(1,)`` complex array.
        """
        spec = ir.spec
        if spec.level_structure == "analog_3":
            raise YASTNPEPSError(
                "find_ground_state is not defined for analog_3 "
                "(time-dependent physical ladder protocol)."
            )
        exprs = dict(observables or {})
        preflight_tn_expressions(exprs, spec, backend="peps", max_term_sites=PEPS_MAX_TERM_SITES)
        yastn, fpeps, gates, cfg = self._load_yastn()
        # The protocol must be constant in time; the Hamiltonian is read from a
        # single full-duration schedule step (coefficients at t_gate / 2).
        t_gate = float(ir.params["t_gate"])
        payload = build_yastn_peps_payload(
            ir, initial_state=initial_state,
            segments=[TNSegment(0.0, t_gate, 1, t_gate, False)],
            chi_max=self.chi_max, svd_min=self.svd_min, use_cuda=self.use_cuda,
        )
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

        expectations: dict[str, np.ndarray] = {}
        measurement_env_info: list[dict] = []
        if exprs:
            lowered, _ = _lower_peps_expressions(exprs, ops, int(spec.Ly))
            measured, env_info = _measure_lowered_peps(
                fpeps, psi, lowered, self.measurement_environment,
                ctm_chi=self._ctm_chi(), ctm_iters=int(self.ctm_iters), ctm_tol=float(self.ctm_tol),
            )
            measurement_env_info.append(env_info)
            expectations = {
                label: np.asarray([value], dtype=complex) for label, value in measured.items()
            }
        if not np.isfinite(energy):
            energy, _ = _peps_energy_ctm(fpeps, psi, ops, payload, step0, ctm_chi, ctm_iters)
        return EvolutionResult(
            final_state=psi,
            times=np.array([0.0]),
            expectations=expectations,
            metadata={
                **(ir.metadata or {}),
                "backend": "peps",
                "method": "peps_yastn_imag",
                "engine_package": "yastn",
                "algorithm": f"fpeps_imag_warmup-{warmup_env}_refine-{self.update_environment}",
                "level_structure": payload["lattice"]["level_structure"],
                "local_dim": int(payload["lattice"]["local_dim"]),
                "measurement_environment": self.measurement_environment,
                "ctm_chi": self._ctm_chi(),
                "ctm_iters": int(self.ctm_iters),
                "ctm_tol": float(self.ctm_tol),
                "measurement_env_info": measurement_env_info,
                "chi_max": int(self.chi_max),
                "dtau_schedule": [list(s) for s in dtau_schedule],
                "n_steps": int(n_steps),
                "stopped_early": stopped_early,
                "energy": float(energy),
                "max_truncation_error": max(truncation_error, default=0.0),
            },
            basis=tn_basis(spec),
        )

    def _load_yastn(self):
        if importlib.util.find_spec("yastn") is None:
            raise YASTNPEPSError(
                "backend='peps' requires yastn. Install it with the tn-2d extra, e.g. "
                "`pip install -e '.[tn-2d]'` (or `pip install git+https://github.com/yastn/yastn.git`)."
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
    segments: "list[TNSegment]",
    chi_max: int,
    svd_min: float,
    use_cuda: bool,
) -> dict[str, Any]:
    """Build a YASTN finite-PEPS payload directly from the TN IR.

    Unlike the Julia ITensors bridges, this lowering keeps the local physical
    dimension from the central level spec, so ``01r`` becomes a genuine qutrit
    PEPS rather than an effective two-level model.

    ``segments`` is the anchor-exact stepping plan
    (:func:`~ryd_gate.backends.tn_common.protocol_context.plan_tn_segments`):
    each schedule entry carries its own ``dt`` and ``record`` flag, and each
    step's drive coefficients are evaluated at that step's own midpoint.
    """
    spec = ir.spec
    if spec.level_structure not in {"1r", "01r", "analog_3"}:
        raise ValueError(
            "TN PEPS payload supports level_structure '1r', '01r', and 'analog_3' only."
        )
    t_gate = float(ir.params["t_gate"])
    if t_gate <= 0:
        raise ValueError("YASTN PEPS requires a positive t_gate.")

    schedule = _drive_schedule(ir, segments)
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
            "local_blocks": spec.local_blocks,
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
        "schedule": schedule,
        "runtime": {
            "n_steps": len(schedule),
            "chi_max": int(chi_max),
            "svd_min": float(svd_min),
            "use_cuda": bool(use_cuda),
        },
    }


def _drive_schedule(ir, segments: "list[TNSegment]") -> list[dict[str, Any]]:
    spec = ir.spec
    static_pin = pin_deltas_from_params(ir.params, spec.N)
    schedule: list[dict[str, Any]] = []
    for segment in segments:
        for k in range(segment.n_sub):
            t_mid = segment.t0 + (k + 0.5) * segment.dt_sub
            record = segment.record and k == segment.n_sub - 1
            schedule.append(
                _drive_schedule_entry(
                    ir, spec, static_pin,
                    step=len(schedule) + 1, t_mid=float(t_mid),
                    dt=float(segment.dt_sub), record=record,
                )
            )
    return schedule


def _drive_schedule_entry(
    ir, spec, static_pin, *, step: int, t_mid: float, dt: float, record: bool
) -> dict[str, Any]:
    coeffs = ir.protocol.get_drive_coefficients(t_mid, ir.params)
    entry: dict[str, Any] = {"step": step, "t_mid": t_mid, "dt": dt, "record": record}
    if spec.level_structure == "analog_3":
        # Physical g/e/r ladder: the local 3x3 blocks live on spec.local_blocks;
        # the protocol modulates the g-e drive by a (complex) scalar. Drive is
        # spatially uniform, so no per-site profile/reorder is needed.  Under the
        # TN context the protocol emits the unitless g-e envelope on E[e,g]; the
        # payload re-applies the full Rabi via local_blocks.drive_420.
        entry["drive_coeffs"] = {"drive_420": complex(coeffs.get("E[e,g]", 0.0))}
        return entry
    if spec.level_structure == "01r":
        profiles = three_level_profiles_from_coeffs(coeffs, spec)
        if static_pin is not None:
            # A pin adds to the Rydberg detuning, i.e. subtracts energy.
            profiles["energy_r"] = profiles["energy_r"] - static_pin
    else:
        omega_t, delta_t, channel_pin = two_level_drive_and_detuning_from_coeffs(coeffs, spec)
        pin = merge_pin_deltas(static_pin, channel_pin, n_sites=spec.N)
        delta_profile = np.full(spec.N, float(delta_t), dtype=float)
        if pin is not None:
            delta_profile = delta_profile + pin
        profiles = {
            "coupling_r1": 0.5 * _profile(omega_t, spec.N).astype(complex),
            "coupling_10": np.zeros(spec.N, dtype=complex),
            "coupling_r0": np.zeros(spec.N, dtype=complex),
            "energy_1": np.zeros(spec.N, dtype=float),
            "energy_r": -delta_profile,
        }

    order = np.asarray(spec.snake_to_2d, dtype=int)
    entry.update(
        {
            "coupling_r1_1d": np.asarray(profiles["coupling_r1"], dtype=complex)[order],
            "coupling_10_1d": np.asarray(profiles["coupling_10"], dtype=complex)[order],
            "coupling_r0_1d": np.asarray(profiles["coupling_r0"], dtype=complex)[order],
            "energy_1_1d": np.asarray(profiles["energy_1"], dtype=float)[order],
            "energy_r_1d": np.asarray(profiles["energy_r"], dtype=float)[order],
        }
    )
    return entry


def _profile(value: float | np.ndarray, n_sites: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_sites, float(arr), dtype=float)
    if arr.shape != (n_sites,):
        raise ValueError(f"Profile must be scalar or length-{n_sites}; got {arr.shape}.")
    return arr.astype(float, copy=False)


def _initial_labels_1d(spec, initial_state: str | np.ndarray | object) -> list[str]:
    labels_2d = level_labels(spec, initial_state)
    return [labels_2d[int(i)] for i in np.asarray(spec.snake_to_2d, dtype=int)]


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

    def matrix_hamiltonian(self, static, drive_mat, coeff):
        return self.matrix(_matrix_hamiltonian(static, drive_mat, coeff))

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

    def sp_between(self, lower: str, upper: str):
        """The raising operator ``|upper><lower|`` (zero when a level is absent)."""
        mat = np.zeros((self.dim, self.dim), dtype=complex)
        if lower in self.levels and upper in self.levels:
            mat[self.index(upper), self.index(lower)] = 1.0
        return self.matrix(mat)

    @property
    def I(self):  # noqa: E743 - conventional identity-operator symbol.
        return self.matrix(np.eye(self.dim, dtype=complex))

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
    local_blocks = lattice.get("local_blocks")
    if local_blocks is not None:
        if not local_blocks.hermitian:
            raise YASTNPEPSError(
                "YASTN PEPS analog_3 requires Hermitian local blocks; "
                "decay/noise-enabled analog_3 is not supported by PEPS."
            )
        proto = gates.gate_local_exp(
            coeff,
            ops.I,
            ops.matrix_hamiltonian(
                local_blocks.static,
                local_blocks.drive_420,
                step_data["drive_coeffs"]["drive_420"],
            ),
            site=(0, 0),
        )
        return [
            proto._replace(sites=((site_2d // Ly, site_2d % Ly),))
            for site_2d in range(int(lattice["N"]))
        ]

    inv_snake = np.asarray(lattice["inv_snake"], dtype=int)
    c_r1 = np.asarray(step_data["coupling_r1_1d"], dtype=complex)
    c_10 = np.asarray(step_data["coupling_10_1d"], dtype=complex)
    c_r0 = np.asarray(step_data["coupling_r0_1d"], dtype=complex)
    e_1 = np.asarray(step_data["energy_1_1d"], dtype=float)
    e_r = np.asarray(step_data["energy_r_1d"], dtype=float)
    I = ops.I
    cache: dict = {}
    out = []
    for site_2d in range(int(lattice["N"])):
        pos = int(inv_snake[site_2d])
        coord = (site_2d // Ly, site_2d % Ly)
        key = (c_r1[pos], c_10[pos], c_r0[pos], e_1[pos], e_r[pos])
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
    if lattice.get("local_blocks") is not None:
        raise YASTNPEPSError("Imaginary-time PEPS is not defined for analog_3 local blocks.")
    Ly = int(lattice["Ly"])
    inv_snake = np.asarray(lattice["inv_snake"], dtype=int)
    c_r1 = np.asarray(step_data["coupling_r1_1d"], dtype=complex)
    c_10 = np.asarray(step_data["coupling_10_1d"], dtype=complex)
    c_r0 = np.asarray(step_data["coupling_r0_1d"], dtype=complex)
    e_1 = np.asarray(step_data["energy_1_1d"], dtype=float)
    e_r = np.asarray(step_data["energy_r_1d"], dtype=float)
    out = []

    for site_2d in range(int(lattice["N"])):
        pos = int(inv_snake[site_2d])
        coord = (site_2d // Ly, site_2d % Ly)
        H = _local_hamiltonian(ops, c_r1[pos], c_10[pos], c_r0[pos], e_1[pos], e_r[pos])
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
        H = _local_hamiltonian(ops, c_r1[pos], c_10[pos], c_r0[pos], e_1[pos], e_r[pos])
        out.append(gates.gate_local_exp(0.5 * dtau, ops.I, H, site=coord))
    return out


def _peps_energy_ctm(fpeps, psi, ops, payload, step0, ctm_chi, ctm_iters):
    """Total ground-state energy <H> from a light CTM environment.

    O(N): one CTM convergence, a handful of 1-site measurements (one raising
    operator per active coupling channel, n_r, n_1) combined with the per-site
    fields, plus nearest-neighbour ``<n_r n_r>`` over the ~2N interaction bonds.
    Far cheaper than the O(N^2) staggered structure factor and a robust,
    symmetry-agnostic cross-check quantity. Returns ``(energy, env)``.
    """
    lat = payload["lattice"]
    Ly = int(lat["Ly"])
    N = int(lat["N"])
    inv = np.asarray(lat["inv_snake"], dtype=int)
    c_r1 = np.asarray(step0["coupling_r1_1d"], dtype=complex)
    c_10 = np.asarray(step0["coupling_10_1d"], dtype=complex)
    c_r0 = np.asarray(step0["coupling_r0_1d"], dtype=complex)
    e_1 = np.asarray(step0["energy_1_1d"], dtype=float)
    e_r = np.asarray(step0["energy_r_1d"], dtype=float)

    env = fpeps.EnvCTM(psi)
    for _ in range(int(ctm_iters)):
        env.update_({"D_total": int(ctm_chi), "tol": 1e-8})

    # <c |u><l| + h.c.> = 2 Re(c <|u><l|>) per active coupling channel.
    coupling_meas = [
        (c, env.measure_1site(ops.sp_between(lower, upper)))
        for lower, upper, c in (("1", "r", c_r1), ("0", "1", c_10), ("0", "r", c_r0))
        if np.any(c != 0)
    ]
    nr = env.measure_1site(ops.n_r) if np.any(e_r) else None
    n1 = env.measure_1site(ops.n_1) if np.any(e_1) else None

    E = 0.0
    for s2 in range(N):
        pos = int(inv[s2])
        coord = (s2 // Ly, s2 % Ly)
        for c, meas in coupling_meas:
            E += 2.0 * float(np.real(c[pos] * meas[coord]))
        if nr is not None:
            E += e_r[pos] * float(np.real(nr[coord]))
        if n1 is not None:
            E += e_1[pos] * float(np.real(n1[coord]))

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
    coupling_r1: complex,
    coupling_10: complex,
    coupling_r0: complex,
    energy_1: float,
    energy_r: float,
):
    """Local 1r/01r H in matrix-element convention:
    ``c |u><l| + conj(c) |l><u|`` per coupling plus ``e_1 n_1 + e_r n_r``."""
    mat = np.zeros((ops.dim, ops.dim), dtype=complex)
    for lower, upper, c in (
        ("1", "r", coupling_r1), ("0", "1", coupling_10), ("0", "r", coupling_r0),
    ):
        c = complex(c)
        if lower not in ops.levels or upper not in ops.levels:
            if c != 0:
                raise YASTNPEPSError(
                    f"Nonzero |{lower}>-|{upper}> coupling on levels {ops.levels}."
                )
            continue
        lo, up = ops.index(lower), ops.index(upper)
        mat[up, lo] += c
        mat[lo, up] += np.conj(c)
    for level, energy in (("1", energy_1), ("r", energy_r)):
        if level in ops.levels:
            idx = ops.index(level)
            mat[idx, idx] += float(energy)
    return ops.matrix(mat)


def _matrix_hamiltonian(static, drive_mat, coeff) -> np.ndarray:
    """analog_3 local H(t) = static + c*drive + c.conj()*drive^dag."""
    static = np.asarray(static, dtype=np.complex128)
    drive_mat = np.asarray(drive_mat, dtype=np.complex128)
    coeff = complex(coeff)
    return static + coeff * drive_mat + np.conj(coeff) * drive_mat.conj().T


def _lower_peps_expressions(exprs: dict, ops: _YASTNPEPSOps, Ly: int):
    """Lower validated observable expressions onto the PEPS geometry.

    Register site ``i`` lives at PEPS coordinate ``(i // Ly, i % Ly)`` (the
    row-major 2D layout; no snake permutation on the PEPS path).  Distinct
    local matrices are wrapped into yastn tensors once and shared, so each
    sampling anchor needs one ``measure_1site`` sweep per distinct matrix.
    Returns ``(lowered, has_pair_terms)``.
    """
    tensor_cache: dict[bytes, Any] = {}

    def wrap(matrix) -> tuple[bytes, Any]:
        arr = np.asarray(matrix, dtype=complex)
        key = arr.tobytes()
        tensor = tensor_cache.get(key)
        if tensor is None:
            tensor = ops.matrix(arr)
            tensor_cache[key] = tensor
        return key, tensor

    lowered: dict[str, tuple] = {}
    has_pair_terms = False
    for label, expr in exprs.items():
        identity_coeff = 0.0 + 0.0j
        singles: list[tuple] = []
        pairs: list[tuple] = []
        for term in expr._terms:
            if not term.factors:
                identity_coeff += complex(term.coefficient)
            elif len(term.factors) == 1:
                site, matrix = term.factors[0]
                key, tensor = wrap(matrix)
                singles.append(
                    (complex(term.coefficient), key, tensor, (site // Ly, site % Ly))
                )
            else:
                # preflight_tn_expressions(max_term_sites=PEPS_MAX_TERM_SITES) guarantees <= 2.
                (site_i, mat_i), (site_j, mat_j) = term.factors
                _, tensor_i = wrap(mat_i)
                _, tensor_j = wrap(mat_j)
                pairs.append(
                    (
                        complex(term.coefficient),
                        tensor_i, (site_i // Ly, site_i % Ly),
                        tensor_j, (site_j // Ly, site_j % Ly),
                    )
                )
                has_pair_terms = True
        lowered[label] = (identity_coeff, singles, pairs)
    return lowered, has_pair_terms


def _measure_lowered_peps(
    fpeps, psi, lowered: dict, env_name: str,
    *, ctm_chi: int, ctm_iters: int, ctm_tol: float,
) -> tuple[dict[str, complex], dict]:
    """Evaluate lowered expressions against ONE converged environment.

    One environment convergence per sampling anchor; single-site profiles are
    measured once per distinct local matrix and shared across labels.  When
    the anchor environment is BP (no two-point support), pair terms use the
    documented fallback: one additional *converged* CTM environment for this
    anchor, recorded in the returned env info.  Values stay complex (raw
    ``<psi|O|psi>`` contract; PEPS evolution is Hermitian, norm 1).
    """
    env, env_info = _measurement_env(
        fpeps, psi, env_name, ctm_chi=ctm_chi, ctm_iters=ctm_iters, ctm_tol=ctm_tol
    )
    profiles: dict[bytes, dict] = {}
    pair_env = None

    def profile(key: bytes, tensor) -> dict:
        if key not in profiles:
            profiles[key] = env.measure_1site(tensor)
        return profiles[key]

    def get_pair_env():
        nonlocal pair_env, env_info
        if pair_env is None:
            if hasattr(env, "measure_nsite"):
                pair_env = env
            else:
                pair_env, pair_info = _measurement_env(
                    fpeps, psi, "ctm", ctm_chi=ctm_chi, ctm_iters=ctm_iters, ctm_tol=ctm_tol
                )
                env_info = {**env_info, "pair_fallback": pair_info}
        return pair_env

    out: dict[str, complex] = {}
    for label, (identity_coeff, singles, pairs) in lowered.items():
        value = identity_coeff
        for coeff, key, tensor, coord in singles:
            value += coeff * complex(profile(key, tensor)[coord])
        for coeff, tensor_i, coord_i, tensor_j, coord_j in pairs:
            value += coeff * complex(
                get_pair_env().measure_nsite(tensor_i, tensor_j, sites=(coord_i, coord_j))
            )
        out[label] = value
    return out, env_info


def _measurement_env(fpeps, psi, name: str, *, ctm_chi: int, ctm_iters: int, ctm_tol: float):
    """Build one converged measurement environment; returns ``(env, info)``.

    ``info`` records the method and its convergence settings/diagnostics for
    ``metadata["measurement_env_info"]``.
    """
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
        ctm_out = env.iterate_(
            {"D_total": int(ctm_chi), "tol": 1e-10},
            max_sweeps=int(ctm_iters),
            corner_tol=float(ctm_tol),
        )
        info = {
            "environment": "ctm",
            "chi": int(ctm_chi),
            "max_sweeps": int(ctm_iters),
            "corner_tol": float(ctm_tol),
            "sweeps": getattr(ctm_out, "sweeps", None),
            "max_dsv": getattr(ctm_out, "max_dsv", None),
            "converged": getattr(ctm_out, "converged", None),
        }
        return env, info
    if key in {"bp", "envbp"}:
        # A single BP update_() is one sweep, NOT the BP fixed point; on an
        # entangled state one sweep can be off by ~x2 (converged BP carries
        # only the loop error).  Iterate to convergence, reusing the
        # measurement-environment budget/tolerance knobs (ctm_iters/ctm_tol).
        if int(ctm_iters) <= 0:
            raise YASTNPEPSError(f"BP measurement requires ctm_iters > 0; got {ctm_iters!r}.")
        env = fpeps.EnvBP(psi)
        bp_out = env.iterate_(max_sweeps=int(ctm_iters), diff_tol=float(ctm_tol))
        info = {
            "environment": "bp",
            "max_sweeps": int(ctm_iters),
            "diff_tol": float(ctm_tol),
            "sweeps": getattr(bp_out, "sweeps", None),
            "max_diff": getattr(bp_out, "max_diff", None),
            "converged": getattr(bp_out, "converged", None),
        }
        return env, info
    raise ValueError("measurement_environment must be 'ctm' or 'bp'.")
