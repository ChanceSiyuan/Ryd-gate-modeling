"""quimb PEPS real-time adapter for 2D Rydberg lattice dynamics."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np

from ryd_gate.analysis.spin_observables import line_pairs_from_reference
from ryd_gate.backends.itensor.backend import build_itensors_payload
from ryd_gate.ir.evolution import EvolutionResult


class Quimb2DTNError(RuntimeError):
    """Raised when the quimb 2D-TN adapter cannot run."""


@dataclass
class Quimb2DTNBackend:
    """2D PEPS time evolution through quimb.

    The implementation uses quimb's finite-PEPS ``SimpleUpdate`` machinery.  For
    real time, quimb 1.11 disallows ``imag=False`` but accepts complex imaginary
    time steps in the lower-level sweep API, so this adapter directly calls
    ``presweep/sweep/postsweep`` with ``tau = 1j * dt``.
    """

    chi_max: int = 64
    dt: float = 0.05
    svd_min: float = 1e-10
    use_cuda: bool = False
    array_backend: str | None = None
    algorithm: str = "simple_update"
    second_order_reflect: bool = False
    compute_energy_final: bool = False
    progbar: bool = False
    contract_optimize: str = "auto-hq"

    def evolve_ir(
        self,
        ir,
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        qtn, xp, backend_name = self._load_quimb()
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
        payload["method"] = "2dtn_quimb"

        psi = _quimb_product_peps(qtn, xp, payload)
        if observables is None and t_eval is not None:
            observables = ["sigma_z"]
        obs_data = {name: [] for name in observables or []}
        recorded_times: list[float] = []

        record_steps = set(int(step) for step in payload["record_steps"])

        def record(t_value: float) -> None:
            if not observables:
                return
            recorded_times.append(float(t_value))
            measured = _measure_quimb(psi, xp, payload, observables, self.contract_optimize)
            for name, value in measured.items():
                obs_data[name].append(value)

        if 0 in record_steps:
            record(0.0)

        for step_data in payload["schedule"]:
            ham = _quimb_local_ham(qtn, xp, payload, step_data)
            evolver_cls = _quimb_evolver_class(qtn, self.algorithm)
            evolver = evolver_cls(
                psi,
                ham,
                tau=float(payload["runtime"]["dt"]),
                D=int(self.chi_max),
                chi=int(self.chi_max),
                imag=True,
                gate_opts={"cutoff": float(self.svd_min), "max_bond": int(self.chi_max)},
                second_order_reflect=bool(self.second_order_reflect),
                compute_energy_every=None,
                compute_energy_final=bool(self.compute_energy_final),
                progbar=bool(self.progbar),
            )
            evolver.presweep(0)
            evolver.sweep(1j * float(payload["runtime"]["dt"]))
            evolver.postsweep(0)
            evolver._n += 1
            psi = evolver.state

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
                "method": "2dtn_quimb",
                "engine_package": "quimb",
                "algorithm": self.algorithm,
                "accelerator": "cuda" if self.use_cuda else "cpu",
                "array_backend": backend_name,
                "gpu": bool(self.use_cuda),
                "chi_max": int(self.chi_max),
                "dt": float(payload["runtime"]["dt"]),
                "n_steps": int(payload["runtime"]["n_steps"]),
                "svd_min": float(self.svd_min),
                "obs": obs_data,
            },
        )
        if recorded_times:
            result.times = np.asarray(recorded_times, dtype=float)
        return result

    def _load_quimb(self):
        if importlib.util.find_spec("quimb") is None:
            raise Quimb2DTNError(
                "engine_package='quimb' requires quimb. Install with "
                "`pip install quimb cotengra autoray networkx`."
            )
        if importlib.util.find_spec("networkx") is None:
            raise Quimb2DTNError("quimb 2D TEBD ordering requires networkx.")
        import quimb.tensor as qtn

        backend = self.array_backend or ("cupy" if self.use_cuda else "numpy")
        if backend == "cupy":
            if importlib.util.find_spec("cupy") is None:
                raise Quimb2DTNError(
                    "use_cuda=True for quimb requires CuPy. Install a CUDA wheel, "
                    "for example `pip install cupy-cuda12x`."
                )
            import cupy as xp
        elif backend == "numpy":
            xp = np
        else:
            raise Quimb2DTNError("quimb array_backend currently supports 'numpy' or 'cupy'.")
        return qtn, xp, backend


def _quimb_evolver_class(qtn, algorithm: str):
    key = algorithm.lower().replace("-", "_")
    if key in {"simple", "simple_update", "su"}:
        return qtn.SimpleUpdate
    if key in {"full", "full_update", "fu"}:
        return qtn.FullUpdate
    if key in {"tebd", "tebd2d"}:
        return qtn.TEBD2D
    raise ValueError("algorithm must be 'simple_update', 'full_update', or 'tebd2d'.")


def _quimb_product_peps(qtn, xp, payload: dict):
    lattice = payload["lattice"]
    occ_1d = np.asarray(payload["initial_occupations_1d"], dtype=int)
    snake_to_2d = np.asarray(lattice["snake_to_2d"], dtype=int)
    site_map = {}
    for pos, occ in enumerate(occ_1d):
        site_2d = int(snake_to_2d[pos])
        coord = (site_2d // int(lattice["Ly"]), site_2d % int(lattice["Ly"]))
        values = np.array([1.0, 0.0], dtype=complex) if int(occ) == 0 else np.array([0.0, 1.0], dtype=complex)
        site_map[coord] = xp.asarray(values)
    return qtn.PEPS.product_state(site_map)


def _quimb_local_ham(qtn, xp, payload: dict, step_data: dict):
    lattice = payload["lattice"]
    Ly = int(lattice["Ly"])
    inv_snake = np.asarray(lattice["inv_snake"], dtype=int)
    sx = xp.asarray(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex) / 2.0)
    n_op = xp.asarray(np.diag([0.0, 1.0]).astype(complex))

    h1 = {}
    omega = np.asarray(step_data["omega_1d"], dtype=float)
    delta = np.asarray(step_data["delta_1d"], dtype=float)
    for site_2d in range(int(lattice["N"])):
        pos = int(inv_snake[site_2d])
        coord = (site_2d // Ly, site_2d % Ly)
        h1[coord] = float(omega[pos]) * sx - float(delta[pos]) * n_op

    h2 = {}
    for i_pos, j_pos, strength in lattice["vdw_pairs_1d"]:
        i_2d = int(lattice["snake_to_2d"][int(i_pos) - 1])
        j_2d = int(lattice["snake_to_2d"][int(j_pos) - 1])
        if i_2d == j_2d or abs(float(strength)) == 0:
            continue
        ci = (i_2d // Ly, i_2d % Ly)
        cj = (j_2d // Ly, j_2d % Ly)
        h2[(ci, cj)] = float(strength) * xp.kron(n_op, n_op)
    if not h2:
        raise Quimb2DTNError("quimb 2D PEPS evolution requires at least one interaction bond.")
    return qtn.LocalHamGen(h2, h1)


def _measure_quimb(psi, xp, payload: dict, observables: list[str], optimize: str) -> dict[str, Any]:
    lattice = payload["lattice"]
    Lx = int(lattice["Lx"])
    Ly = int(lattice["Ly"])
    n_sites = int(lattice["N"])
    z_op = xp.asarray(np.diag([-1.0, 1.0]).astype(complex))
    zz_op = xp.kron(z_op, z_op)
    z_profile: np.ndarray | None = None
    out: dict[str, Any] = {}

    def sigma_z() -> np.ndarray:
        nonlocal z_profile
        if z_profile is None:
            values = np.empty(n_sites, dtype=float)
            for site in range(n_sites):
                coord = (site // Ly, site % Ly)
                val = psi.local_expectation_exact(z_op, (coord,), optimize=optimize)
                values[site] = float(np.real(_to_numpy(val, xp)))
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
                zz = psi.local_expectation_exact(zz_op, (ci, cj), optimize=optimize)
                values.append(float(np.real(_to_numpy(zz, xp))) - z[int(i)] * z[int(j)])
            out[name] = np.asarray(values, dtype=float)
    return out


def _to_numpy(value, xp):
    if xp is np:
        return np.asarray(value)
    return xp.asnumpy(value)
