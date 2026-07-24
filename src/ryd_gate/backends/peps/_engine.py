"""YASTN/device loading, local operators, product PEPS, Trotter gates, orchestration.

This is the only module that imports YASTN — and only lazily, inside ``_load_yastn``
(no module-scope import), so the dependency-free preflight in the dispatcher runs
and every geometry/option/topology error surfaces even on a machine without YASTN.

Real time keeps the physical second-order Strang decomposition and never normalizes
the Hamiltonian; imaginary time normalizes by the physical scale ``Lambda`` (PEPS09)
and runs the full fixed schedule with no early stop and no per-stage environment.
NTU truncation error is report-only (PEPS02); it is validated finite/nonnegative and
recorded, never compared with a tolerance.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

from ryd_gate.backends.peps._boundary import _ContractionControls
from ryd_gate.backends.peps._environment import lower_observables, measure, measurement_environment
from ryd_gate.backends.peps._numerics import PEPSError, finite_nonneg, real_scalar
from ryd_gate.backends.peps._readout import _GroundLedger, _GroundStateReader, _RealTimeLedger, _RealTimeReader

# Fixed private NTU algorithm (PEPS §7); not a public option.
_NTU_METHOD = "mpo"
_NTU_INIT = "EAT"  # EAT_SVD's parallel SVD path wins <10% of bonds at ~20% step cost (2026-07 profile)
_NTU_FIX_METRIC = 0
_NTU_PINV_CUTOFFS = (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
_NTU_POST_TRUNCATION = None

# fpeps rank-5 site-tensor leg order: (top, left, bottom, right, phys[fused]).
_PHYS_AX = 4


@dataclass(frozen=True, slots=True)
class _Handles:
    yastn: object
    fpeps: object
    gates: object
    mps: object
    cfg: object
    device: str


def _load_yastn(device: str) -> _Handles:
    if importlib.util.find_spec("yastn") is None:
        raise PEPSError("backend='peps' requires yastn (install the tn-2d extra: pip install -e '.[tn-2d]').")
    import yastn
    import yastn.tn.fpeps as fpeps
    import yastn.tn.fpeps.gates as gates
    import yastn.tn.mps as mps

    if device == "cuda":
        if importlib.util.find_spec("torch") is None:
            raise PEPSError("device='cuda' requires PyTorch with CUDA; none available.")
        import torch

        if not torch.cuda.is_available():
            raise PEPSError("device='cuda' requested but torch.cuda.is_available() is false.")
        cfg = yastn.make_config(backend="torch", sym="none", default_device="cuda", default_dtype="complex128")
    else:
        cfg = yastn.make_config(backend="np", sym="none", default_dtype="complex128")
    return _Handles(yastn=yastn, fpeps=fpeps, gates=gates, mps=mps, cfg=cfg, device=device)


class _PEPSOps:
    """Local YASTN tensors for a TN level structure (device-resident)."""

    __slots__ = ("_h", "terms", "d", "levels")

    def __init__(self, h: _Handles, terms) -> None:
        self._h = h
        self.terms = terms
        self.d = terms.local_dim
        self.levels = terms.levels

    def matrix(self, values):
        t = self._h.yastn.Tensor(config=self._h.cfg, s=(1, -1))
        t.set_block(ts=(), Ds=(self.d, self.d), val=np.asarray(values, dtype=complex))
        return t

    def vector(self, values):
        t = self._h.yastn.Tensor(config=self._h.cfg, s=(1,))
        t.set_block(ts=(), Ds=(self.d,), val=np.asarray(values, dtype=complex))
        return t

    @property
    def identity(self):
        return self.matrix(np.eye(self.d))

    def projector(self, level: str):
        m = np.zeros((self.d, self.d), dtype=complex)
        m[self.levels.index(level), self.levels.index(level)] = 1.0
        return self.matrix(m)

    def rydberg(self):
        return self.matrix(self.terms.rydberg_projector())

    def bra_leg(self, level_idx: int):
        """Fused-ancilla conjugated bra leg for physical-leg projection (PEPS §12.2)."""
        v = np.zeros(self.d, dtype=complex)
        v[level_idx] = 1.0
        return self.vector(v).add_leg(s=-1).fuse_legs(axes=((0, 1),)).conj()


def _product_peps(h: _Handles, geom, ops: _PEPSOps, amps, site_to_coord):
    vectors = {coord: ops.vector(amps[i]) for i, coord in enumerate(site_to_coord)}
    return h.fpeps.product_peps(geom, vectors)


def _finite_local_hamiltonians(terms, t: float):
    h_local = np.asarray(terms.local_hamiltonians(float(t)), dtype=complex)
    if not np.all(np.isfinite(h_local)):
        raise PEPSError(f"local Hamiltonian at t={t} has a non-finite matrix element.")
    return h_local


def _ntu_step_errors(infos) -> tuple[float, float]:
    """(max individual truncation error, one-step error) for a single evolution step (PEPS §8)."""
    max_individual = 0.0
    per_bond: dict = {}
    for info in infos:
        e = finite_nonneg(info.truncation_error, "NTU truncation_error")
        max_individual = max(max_individual, e)
        per_bond[info.bond] = per_bond.get(info.bond, 0.0) + e
    one_step = max(per_bond.values(), default=0.0)
    return max_individual, one_step


def _scale_scan(psi, site_to_coord) -> dict:
    """O(N) final local tensor validity scan → positive per-site Frobenius scales (PEPS §8)."""
    scales = {}
    for coord in site_to_coord:
        s = float(psi[coord].norm())
        if not np.isfinite(s) or s <= 0.0:
            raise PEPSError(f"final PEPS site tensor at {coord} has non-finite/zero Frobenius norm {s!r}.")
        scales[coord] = s
    return scales


# ── real-time evolution (PEPS §8) ────────────────────────────────────────────


def evolve_peps(terms, lattice_spec, pair_bonds, amps, out_times, obs_exprs, options):
    """NTU real-time PEPS evolution; return ``(out_times, real_expectations, reader)``."""
    from ryd_gate.backends.tn_common.compiler import plan_segments

    h = _load_yastn(options.device)
    ops = _PEPSOps(h, terms)
    site_to_coord = lattice_spec.site_to_coord
    geom = h.fpeps.SquareLattice(dims=lattice_spec.shape, boundary="obc")
    psi = _product_peps(h, geom, ops, amps, site_to_coord)

    lowered = lower_observables(obs_exprs, ops, site_to_coord, options.measurement_method) if obs_exprs else None
    records: dict[str, list[float]] = {label: [] for label in obs_exprs}
    env_residuals: list[float | None] = []
    env_iterations: list[int] = []

    def measure_now():
        if not obs_exprs:
            return
        env, summary = measurement_environment(h, psi, options)
        env_residuals.append(summary.residual)
        env_iterations.append(summary.iterations)
        for label, value in measure(env, lowered).items():
            records[label].append(value)

    opts_svd = {"D_total": options.bond_dimension, "tol": options.svd_tolerance}
    record_at_start, segments = plan_segments(terms.t_gate, out_times, options.time_step_s)
    env_update = h.fpeps.EnvNTU(psi, which="NN")
    max_ntu = 0.0
    cumulative = 0.0
    if record_at_start:
        measure_now()
    for segment in segments:
        for k in range(segment.n_sub):
            t_mid = segment.t0 + (k + 0.5) * segment.dt_sub
            h_local = _finite_local_hamiltonians(terms, t_mid)
            step_gates = _real_time_gates(h, ops, h_local, pair_bonds, site_to_coord, segment.dt_sub)
            infos = h.fpeps.evolution_step_(
                env_update, step_gates, opts_svd,
                method=_NTU_METHOD, fix_metric=_NTU_FIX_METRIC, pinv_cutoffs=_NTU_PINV_CUTOFFS,
                max_iter=options.ntu_max_iterations, tol_iter=options.ntu_iteration_tolerance,
                initialization=_NTU_INIT, opts_post_truncation=_NTU_POST_TRUNCATION,
            )
            psi = env_update.psi
            step_max, one_step = _ntu_step_errors(infos)
            max_ntu = max(max_ntu, step_max)
            cumulative += one_step
        if segment.record:
            measure_now()

    scales = _scale_scan(psi, site_to_coord)
    ledger = _RealTimeLedger(
        parameters=options.parameters(),
        max_ntu_truncation_error=max_ntu,
        cumulative_ntu_truncation_error=cumulative,
        env_residuals=tuple(env_residuals),
        env_iterations=tuple(env_iterations),
    )
    controls = _ContractionControls(options.environment_bond_dimension, options.environment_tolerance,
                                    options.environment_max_iterations)
    reader = _RealTimeReader(h, ops, psi, terms, site_to_coord, scales, controls, options, ledger)
    expectations = {label: np.asarray(vals, dtype=float) for label, vals in records.items()}
    return out_times, expectations, reader


def _real_time_gates(h, ops, h_local, pair_bonds, site_to_coord, dt):
    """Second-order Strang gates ``[local(½dt), nn(dt), local(½dt)]`` (no normalization)."""
    gates = h.gates
    I = ops.identity
    local = [
        gates.gate_local_exp(0.5j * dt, I, ops.matrix(h_local[i]), site=coord)
        for i, coord in enumerate(site_to_coord)
    ]
    Hnn = gates.fkron(ops.rydberg(), ops.rydberg())
    nn = [gates.gate_nn_exp(1j * dt, I, V * Hnn, bond=(ci, cj)) for ci, cj, V in pair_bonds]
    return local + nn + local


# ── normalized imaginary-time ground state (PEPS §11) ────────────────────────


def solve_peps_ground_state(terms, lattice_spec, pair_bonds, at, amps, obs_exprs, options):
    """Normalized imaginary-time PEPS ground state; return ``(expectations, reader)``."""
    from ryd_gate.backends.peps._environment import build_ctm_environment, ground_energy

    h = _load_yastn(options.device)
    ops = _PEPSOps(h, terms)
    site_to_coord = lattice_spec.site_to_coord
    geom = h.fpeps.SquareLattice(dims=lattice_spec.shape, boundary="obc")
    psi = _product_peps(h, geom, ops, amps, site_to_coord)

    h_local = _finite_local_hamiltonians(terms, float(at))
    Lambda = _hamiltonian_scale(h_local, pair_bonds, site_to_coord)
    h_tilde = h_local / Lambda
    pair_tilde = tuple((ci, cj, V / Lambda) for ci, cj, V in pair_bonds)

    opts_svd = {"D_total": options.bond_dimension, "tol": options.svd_tolerance}
    env_update = h.fpeps.EnvNTU(psi, which="NN")
    max_ntu = 0.0
    for dtau, steps in options.imaginary_time_schedule:
        stage_gates = _imag_time_gates(h, ops, h_tilde, pair_tilde, site_to_coord, float(dtau))
        for _ in range(int(steps)):
            infos = h.fpeps.evolution_step_(
                env_update, stage_gates, opts_svd,
                method=_NTU_METHOD, fix_metric=_NTU_FIX_METRIC, pinv_cutoffs=_NTU_PINV_CUTOFFS,
                max_iter=options.ntu_max_iterations, tol_iter=options.ntu_iteration_tolerance,
                initialization=_NTU_INIT, opts_post_truncation=_NTU_POST_TRUNCATION,
            )
            psi = env_update.psi
            step_max, _ = _ntu_step_errors(infos)
            max_ntu = max(max_ntu, step_max)

    scales = _scale_scan(psi, site_to_coord)
    env, summary = build_ctm_environment(h, psi, options)
    energy = real_scalar(ground_energy(env, ops, site_to_coord, h_local, pair_bonds), "ground energy")
    expectations = {"energy": energy}
    if obs_exprs:
        lowered = lower_observables(obs_exprs, ops, site_to_coord, "ctm")
        expectations.update(measure(env, lowered))

    ledger = _GroundLedger(
        parameters=options.parameters(),
        hamiltonian_scale_rad_s=Lambda,
        max_ntu_truncation_error=max_ntu,
        env_residual=summary.residual,
        env_iterations=summary.iterations,
    )
    controls = _ContractionControls(options.environment_bond_dimension, options.environment_tolerance,
                                    options.environment_max_iterations)
    reader = _GroundStateReader(h, ops, psi, terms, site_to_coord, scales, controls, options, env, ledger)
    return expectations, reader


def _hamiltonian_scale(h_local, pair_bonds, site_to_coord) -> float:
    local_norm = max((float(np.linalg.norm(h_local[i], ord=2)) for i in range(len(site_to_coord))), default=0.0)
    pair_norm = max((abs(float(V)) for _, _, V in pair_bonds), default=0.0)
    for value in (local_norm, pair_norm):
        if not np.isfinite(value) or value < 0.0:
            raise PEPSError(f"Hamiltonian spectral scale component is not finite/nonnegative: {value!r}.")
    Lambda = max(local_norm, pair_norm)
    if not np.isfinite(Lambda) or Lambda <= 0.0:
        raise PEPSError(
            "imaginary-time ground state requires a nonzero Hamiltonian; the physical scale Lambda "
            f"is {Lambda!r}, so no ground-state representative is selected."
        )
    return Lambda


def _imag_time_gates(h, ops, h_tilde, pair_tilde, site_to_coord, dtau):
    """Second-order imaginary-time gates ``exp(-dtau H_tilde)`` from the normalized Hamiltonian."""
    gates = h.gates
    I = ops.identity
    local = [
        gates.gate_local_exp(0.5 * dtau, I, ops.matrix(h_tilde[i]), site=coord)
        for i, coord in enumerate(site_to_coord)
    ]
    Hnn = gates.fkron(ops.rydberg(), ops.rydberg())
    nn = [gates.gate_nn_exp(dtau, I, V * Hnn, bond=(ci, cj)) for ci, cj, V in pair_tilde]
    return local + nn + local
