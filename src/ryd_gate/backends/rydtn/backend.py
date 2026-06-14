"""``RydTNPEPSBackend`` — self-written finite-PEPS backend for ``backend="peps"``.

Drop-in replacement for ``YASTNPEPSBackend``: consumes the same engine-agnostic
payload (``build_yastn_peps_payload``), exposes ``evolve_ir`` / ``find_ground_state``
with the same signatures, and returns ``EvolutionResult`` with the same metadata
keys and ``obs`` names.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ryd_gate.analysis.observables import line_pairs_from_reference
from ryd_gate.backends.peps2d import build_yastn_peps_payload
from ryd_gate.ir import EvolutionResult

from .boundary import BoundaryMPS
from .evolve import trotter_step
from .operators import PEPSOps
from .peps import product_peps
from .tensors import RydTNError, resolve_backend


@dataclass
class RydTNPEPSBackend:
    """2D finite-PEPS real-time / imaginary-time evolution (NTU + boundary-MPS)."""

    chi_max: int = 64
    dt: float = 0.05
    svd_min: float = 1e-10
    use_cuda: bool = False
    backend_name: str | None = None
    device: str | None = None
    dtype: str = "complex128"
    update_environment: str = "ntu"
    measurement_environment: str = "bp"
    initialization: str = "SVD"
    max_iter: int = 20
    tol_iter: float = 1e-12
    require_gpu: bool = False
    measure_chi: int | None = None

    def _backend(self):
        return resolve_backend(
            use_cuda=self.use_cuda,
            backend=self.backend_name,
            device=self.device,
            dtype=self.dtype,
            require_gpu=self.require_gpu,
        )

    def _measure_chi(self) -> int:
        if self.measure_chi is not None:
            return int(self.measure_chi)
        # Boundary-MPS bond for measurement: a few x the PEPS bond is plenty for
        # accuracy; capped to keep the O(chi^2 D^4) zip-up affordable at large D.
        return int(min(max(4 * self.chi_max, 32), 128))

    def evolve_ir(self, ir, initial_state="all_ground", t_eval=None, observables=None) -> EvolutionResult:
        if self.update_environment.lower() not in {"ntu", "ctm", "approximate", "approx"}:
            raise RydTNError("update_environment must be 'ntu' (others map to NTU in rydtn).")
        ab = self._backend()
        payload = build_yastn_peps_payload(
            ir, initial_state=initial_state, t_eval=t_eval, observables=observables,
            dt=self.dt, chi_max=self.chi_max, svd_min=self.svd_min, use_cuda=self.use_cuda,
        )
        ops = PEPSOps(payload["lattice"]["levels"])
        psi = product_peps(payload, ops, ab)

        if observables is None and t_eval is not None:
            observables = ["n_mean", "n_r"]
        obs_data = {name: [] for name in observables or []}
        recorded_times: list[float] = []
        record_steps = {int(s) for s in payload["record_steps"]}
        per_step_errs: list[list[float]] = []

        def record(t_value: float) -> None:
            if not observables:
                return
            recorded_times.append(float(t_value))
            measured = _measure(ab, psi, ops, payload, observables, self._measure_chi())
            for name, value in measured.items():
                obs_data[name].append(value)

        if 0 in record_steps:
            record(0.0)

        dt_actual = float(payload["runtime"]["dt"])
        for step_data in payload["schedule"]:
            errs = trotter_step(
                ab, psi, ops, payload, step_data,
                dt=dt_actual, chi_max=int(self.chi_max), svd_min=float(self.svd_min),
                max_iter=int(self.max_iter), tol_iter=float(self.tol_iter), imag=False,
            )
            per_step_errs.append(errs if isinstance(errs, list) else [errs])
            step = int(step_data["step"])
            if step in record_steps:
                record(step * dt_actual)

        for name in obs_data:
            obs_data[name] = np.asarray(obs_data[name])

        truncation_error = np.asarray([max(e, default=0.0) for e in per_step_errs], dtype=float)
        accumulated = float(sum(sum(e) / len(e) for e in per_step_errs if e))

        result = EvolutionResult(
            psi_final=psi,
            metadata={
                **(ir.metadata or {}),
                "backend": "peps",
                "method": "peps_rydtn",
                "engine_package": "rydtn",
                "algorithm": "fpeps_ntu",
                "level_structure": payload["lattice"]["level_structure"],
                "local_dim": int(payload["lattice"]["local_dim"]),
                "measurement_environment": self.measurement_environment,
                "accelerator": "cuda" if ab.kind == "torch" and str(ab.device).startswith("cuda") else "cpu",
                "gpu": ab.kind == "torch" and str(ab.device).startswith("cuda"),
                "array_backend": ab.kind,
                "device": ab.device,
                "chi_max": int(self.chi_max),
                "dt": dt_actual,
                "n_steps": int(payload["runtime"]["n_steps"]),
                "svd_min": float(self.svd_min),
                "truncation_error": truncation_error,
                "accumulated_truncation_error": accumulated,
                "max_truncation_error": float(truncation_error.max()) if truncation_error.size else 0.0,
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
        dtau_schedule=((0.1, 30), (0.03, 30), (0.01, 40)),
        warmup_env=None,
        step_max_iter=8,
        step_tol_iter=1e-8,
        energy_convergence=True,
        energy_tol=1e-5,
        ctm_chi=16,
        ctm_iters=3,
        observables=None,
        initial_state="af1",
    ) -> EvolutionResult:
        """Imaginary-time ground state of a static Hamiltonian (mirror YASTN path).

        ``warmup_env``/``ctm_iters`` are accepted for signature compatibility;
        ``ctm_chi`` sets the boundary-MPS bond used for the energy cross-check.
        """
        ab = self._backend()
        if observables is None:
            observables = ["m_s", "n_mean"]
        payload = build_yastn_peps_payload(
            ir, initial_state=initial_state, t_eval=None, observables=observables,
            dt=self.dt, chi_max=self.chi_max, svd_min=self.svd_min, use_cuda=self.use_cuda,
        )
        if not payload["schedule"]:
            raise RydTNError("find_ground_state requires a non-empty protocol schedule.")
        step0 = payload["schedule"][0]
        ops = PEPSOps(payload["lattice"]["levels"])
        psi = product_peps(payload, ops, ab)
        energy_chi = int(max(self._measure_chi(), min(int(self.chi_max) ** 2, 128)))

        max_err = 0.0
        n_steps = 0
        energy = float("nan")
        prev_E = None
        stopped_early = False
        for dtau, nsteps in dtau_schedule:
            for _ in range(int(nsteps)):
                err = trotter_step(
                    ab, psi, ops, payload, step0,
                    dt=float(dtau), chi_max=int(self.chi_max), svd_min=float(self.svd_min),
                    max_iter=int(step_max_iter), tol_iter=float(step_tol_iter), imag=True,
                )
                max_err = max(max_err, err)
                n_steps += 1
            if energy_convergence:
                energy = _energy(ab, psi, ops, payload, step0, energy_chi)
                if prev_E is not None and abs(energy - prev_E) <= energy_tol * max(1.0, abs(energy)):
                    stopped_early = True
                    break
                prev_E = energy

        measured = _measure(ab, psi, ops, payload, observables, self._measure_chi())
        if not np.isfinite(energy):
            energy = _energy(ab, psi, ops, payload, step0, energy_chi)
        return EvolutionResult(
            psi_final=psi,
            metadata={
                **(ir.metadata or {}),
                "backend": "peps",
                "method": "peps_rydtn_imag",
                "engine_package": "rydtn",
                "algorithm": "fpeps_imag_ntu",
                "level_structure": payload["lattice"]["level_structure"],
                "local_dim": int(payload["lattice"]["local_dim"]),
                "chi_max": int(self.chi_max),
                "dtau_schedule": [list(s) for s in dtau_schedule],
                "n_steps": int(n_steps),
                "stopped_early": stopped_early,
                "energy": float(energy),
                "max_truncation_error": float(max_err),
                "obs": measured,
            },
        )


def _energy(ab, psi, ops: PEPSOps, payload, step0, measure_chi) -> float:
    """O(N) ground-state energy: 1-site fields + nearest-neighbour <n_r n_r>.

    Mirrors ``peps2d._peps_energy_ctm`` but uses the boundary-MPS environment.
    """
    lat = payload["lattice"]
    Ly, N = int(lat["Ly"]), int(lat["N"])
    inv = np.asarray(lat["inv_snake"], dtype=int)
    s2d = np.asarray(lat["snake_to_2d"], dtype=int)
    oR = np.asarray(step0["omega_R_1d"], dtype=float)
    ohf = np.asarray(step0["omega_hf_1d"], dtype=float)
    dR = np.asarray(step0["delta_R_1d"], dtype=float)
    dhf = np.asarray(step0["delta_hf_1d"], dtype=float)

    env = BoundaryMPS(psi, ab, chi=int(measure_chi))
    xr = env.measure_1site(ops.X_1r)
    nr = env.measure_1site(ops.n_r)
    x01 = env.measure_1site(ops.X_01) if np.any(ohf) else None
    n1 = env.measure_1site(ops.n_1) if np.any(dhf) else None

    E = 0.0
    for s2 in range(N):
        pos = int(inv[s2])
        coord = (s2 // Ly, s2 % Ly)
        E += 0.5 * oR[pos] * float(np.real(xr[coord])) - dR[pos] * float(np.real(nr[coord]))
        if x01 is not None:
            E += 0.5 * ohf[pos] * float(np.real(x01[coord]))
        if n1 is not None:
            E += -dhf[pos] * float(np.real(n1[coord]))
    for i_pos, j_pos, strength in lat["vdw_pairs_1d"]:
        i2 = int(s2d[int(i_pos) - 1])
        j2 = int(s2d[int(j_pos) - 1])
        if i2 == j2 or abs(float(strength)) == 0:
            continue
        ci = (i2 // Ly, i2 % Ly)
        cj = (j2 // Ly, j2 % Ly)
        E += float(strength) * float(np.real(env.measure_2site(ops.n_r, ops.n_r, ci, cj)))
    return E


def _measure(ab, psi, ops: PEPSOps, payload, observables, measure_chi) -> dict:
    """Assemble requested observables from a boundary-MPS environment.

    Mirrors ``peps2d._measure_yastn``: per-site arrays are in 2D site order.
    """
    lat = payload["lattice"]
    Lx, Ly, N = int(lat["Lx"]), int(lat["Ly"]), int(lat["N"])
    env = BoundaryMPS(psi, ab, chi=measure_chi)
    occ: dict[str, np.ndarray] = {}
    z_profile = {"v": None}

    def level_occ(level: str) -> np.ndarray:
        if level not in occ:
            m = env.measure_1site(ops.projector(level))
            occ[level] = np.array([float(np.real(m[(s // Ly, s % Ly)])) for s in range(N)])
        return occ[level]

    def sigma_z() -> np.ndarray:
        if z_profile["v"] is None:
            m = env.measure_1site(ops.Z)
            z_profile["v"] = np.array([float(np.real(m[(s // Ly, s % Ly)])) for s in range(N)])
        return z_profile["v"]

    out: dict = {}
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
            sub = np.asarray(lat["sublattice"], dtype=float)
            out[name] = float(np.sum(sub * sigma_z()) / N)
        elif name in {"czz", "czz_centerline"}:
            z = sigma_z()
            vals = []
            for i, j in line_pairs_from_reference(Lx, Ly, axis="horizontal"):
                ci = (int(i) // Ly, int(i) % Ly)
                cj = (int(j) // Ly, int(j) % Ly)
                zz = env.measure_2site(ops.Z, ops.Z, ci, cj)
                vals.append(float(np.real(zz)) - z[int(i)] * z[int(j)])
            out[name] = np.asarray(vals, dtype=float)
        elif name == "czz_full":
            # Full <Z_i Z_j> matrix in 2D site order (i = x*Ly + y); diagonal = 1.
            # O(N^2) boundary contractions -- intended for structure-factor / R(g).
            C = np.eye(N, dtype=float)
            for i in range(N):
                ci = (i // Ly, i % Ly)
                for j in range(i + 1, N):
                    zz = float(np.real(env.measure_2site(ops.Z, ops.Z, ci, (j // Ly, j % Ly))))
                    C[i, j] = C[j, i] = zz
            out[name] = C
    return out
