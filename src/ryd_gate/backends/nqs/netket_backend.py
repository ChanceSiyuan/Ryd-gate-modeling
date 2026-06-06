"""NetKet NQS-tVMC adapter for Rydberg lattice dynamics."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from ryd_gate.analysis.spin_observables import line_pairs_from_reference
from ryd_gate.backends.itensor.backend import build_itensors_payload
from ryd_gate.ir.evolution import EvolutionResult


class NetKetNQSError(RuntimeError):
    """Raised when the NetKet NQS-tVMC adapter cannot run."""


@dataclass
class NetKetNQSTVMCBackend:
    """Run NQS real-time tVMC with NetKet.

    The adapter builds the same two-level Rydberg Hamiltonian used by the TN
    backends, then advances a NetKet variational state with
    ``netket.experimental.TDVP``.  GPU execution is controlled by the installed
    JAX build; set ``use_cuda=True`` to require a visible GPU device.
    """

    dt: float = 0.01
    n_samples: int = 1024
    n_chains: int = 16
    n_discard_per_chain: int | None = None
    sampler: str = "metropolis"
    ansatz: str = "rbm"
    alpha: float = 1.0
    seed: int = 1234
    sampler_seed: int | None = None
    integrator: str = "euler"
    qgt: str = "dense"
    holomorphic: bool = True
    diag_shift: float = 1e-6
    pinv_rtol: float = 1e-10
    pinv_rtol_smooth: float = 1e-10
    initial_bias_strength: float = 1.0
    use_cuda: bool = False
    require_gpu: bool = False
    progress: bool = False

    def evolve_ir(
        self,
        ir,
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        nk, nkx, dynamics, jax, jnp = self._load_netket()
        payload = build_itensors_payload(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
            dt=self.dt,
            chi_max=0,
            svd_min=0.0,
            use_cuda=self.use_cuda,
        )
        payload["method"] = "nqs_tvmc_netket"

        if observables is None and t_eval is not None:
            observables = ["sigma_z"]
        obs_data = {name: [] for name in observables or []}
        recorded_times: list[float] = []

        hilbert = nk.hilbert.Qubit(int(payload["lattice"]["N"]))
        vstate = self._make_variational_state(nk, jnp, hilbert, payload)
        record_steps = set(int(step) for step in payload["record_steps"])

        def record(t_value: float) -> None:
            if not observables:
                return
            recorded_times.append(float(t_value))
            measured = _measure_netket(nk, hilbert, vstate, payload, observables)
            for name, value in measured.items():
                obs_data[name].append(value)

        if 0 in record_steps:
            record(0.0)

        ode_solver = _netket_integrator(dynamics, self.integrator, float(payload["runtime"]["dt"]))
        qgt_spec = _netket_qgt(nk, self.qgt, self.holomorphic, self.diag_shift)
        linear_solver = partial(
            nk.optimizer.solver.pinv_smooth,
            rtol=float(self.pinv_rtol),
            rtol_smooth=float(self.pinv_rtol_smooth),
        )

        for step_data in payload["schedule"]:
            hamiltonian = _netket_hamiltonian(nk, hilbert, payload, step_data)
            driver = nkx.TDVP(
                hamiltonian,
                vstate,
                ode_solver,
                propagation_type="real",
                qgt=qgt_spec,
                linear_solver=linear_solver,
            )
            driver.run(
                float(payload["runtime"]["dt"]),
                show_progress=bool(self.progress),
            )

            step = int(step_data["step"])
            if step in record_steps:
                record(step * float(payload["runtime"]["dt"]))

        for name in obs_data:
            obs_data[name] = np.asarray(obs_data[name])

        result = EvolutionResult(
            psi_final=vstate,
            metadata={
                **(ir.metadata or {}),
                "backend": "nqs",
                "method": "nqs_tvmc",
                "engine_package": "netket",
                "algorithm": "netket_tdvp_mcstate",
                "ansatz": self.ansatz,
                "sampler": self.sampler,
                "integrator": self.integrator,
                "qgt": self.qgt,
                "accelerator": _jax_accelerator(jax),
                "gpu": _jax_accelerator(jax) == "cuda",
                "dt": float(payload["runtime"]["dt"]),
                "n_steps": int(payload["runtime"]["n_steps"]),
                "n_samples": int(self.n_samples),
                "n_chains": int(self.n_chains),
                "initial_bias_strength": float(self.initial_bias_strength),
                "obs": obs_data,
            },
        )
        if recorded_times:
            result.times = np.asarray(recorded_times, dtype=float)
        return result

    def _load_netket(self):
        if importlib.util.find_spec("netket") is None:
            raise NetKetNQSError(
                "engine_package='netket' requires NetKet. Install with "
                "`pip install netket`, or `pip install -e '.[nqs]'`."
            )
        cache_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/ryd_gate_matplotlib"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

        import jax
        import jax.numpy as jnp
        import netket as nk
        import netket.experimental as nkx
        import netket.experimental.dynamics as dynamics

        if self.use_cuda or self.require_gpu:
            try:
                gpu_devices = jax.devices("gpu")
            except RuntimeError as exc:  # pragma: no cover - depends on local JAX build
                raise NetKetNQSError(
                    "NetKet NQS-tVMC requested CUDA, but this JAX build has no GPU backend. "
                    "Install a CUDA-enabled jax/jaxlib build or set use_cuda=False."
                ) from exc
            if not gpu_devices:
                raise NetKetNQSError(
                    "NetKet NQS-tVMC requested CUDA, but jax.devices('gpu') returned no devices."
                )
        return nk, nkx, dynamics, jax, jnp

    def _make_variational_state(self, nk, jnp, hilbert, payload: dict):
        model = _netket_model(nk, self.ansatz, self.alpha)
        sampler = _netket_sampler(nk, hilbert, self.sampler, self.n_chains)
        vstate = nk.vqs.MCState(
            sampler,
            model,
            n_samples=int(self.n_samples),
            n_discard_per_chain=self.n_discard_per_chain,
            seed=int(self.seed),
            sampler_seed=self.sampler_seed,
        )
        _initialize_product_bias(vstate, jnp, payload, float(self.initial_bias_strength))
        return vstate


def _netket_model(nk, ansatz: str, alpha: float):
    key = ansatz.lower()
    if key == "rbm":
        return nk.models.RBM(alpha=float(alpha), param_dtype=complex)
    if key == "jastrow":
        return nk.models.Jastrow(param_dtype=complex)
    raise ValueError("ansatz must be 'rbm' or 'jastrow'.")


def _netket_sampler(nk, hilbert, sampler: str, n_chains: int):
    key = sampler.lower()
    if key in {"metropolis", "metropolis_local", "local"}:
        return nk.sampler.MetropolisLocal(hilbert, n_chains=int(n_chains))
    if key in {"exact", "exact_sampler"}:
        return nk.sampler.ExactSampler(hilbert)
    raise ValueError("sampler must be 'metropolis' or 'exact'.")


def _initialize_product_bias(vstate, jnp, payload: dict, strength: float) -> None:
    if strength == 0:
        return
    occ = np.asarray(payload["initial_occupations_1d"], dtype=int)
    bias = np.where(occ > 0, strength, -strength).astype(complex)
    params = vstate.parameters
    if "visible_bias" in params and np.shape(params["visible_bias"]) == np.shape(bias):
        params["visible_bias"] = jnp.asarray(bias, dtype=jnp.complex128)
        vstate.parameters = params


def _netket_integrator(dynamics, name: str, dt: float):
    key = name.lower()
    if key == "euler":
        return dynamics.Euler(dt=dt)
    if key == "heun":
        return dynamics.Heun(dt=dt)
    if key == "rk4":
        return dynamics.RK4(dt=dt)
    if key == "rk12":
        return dynamics.RK12(dt=dt)
    if key == "rk23":
        return dynamics.RK23(dt=dt)
    if key == "rk45":
        return dynamics.RK45(dt=dt)
    raise ValueError("integrator must be 'euler', 'heun', 'rk4', 'rk12', 'rk23', or 'rk45'.")


def _netket_qgt(nk, name: str, holomorphic: bool, diag_shift: float):
    key = name.lower()
    kwargs = {"holomorphic": bool(holomorphic), "diag_shift": float(diag_shift)}
    if key in {"dense", "jacobian_dense"}:
        return nk.optimizer.qgt.QGTJacobianDense(**kwargs)
    if key in {"pytree", "jacobian_pytree"}:
        return nk.optimizer.qgt.QGTJacobianPyTree(**kwargs)
    if key in {"auto", "qgt_auto"}:
        return nk.optimizer.qgt.QGTAuto(**kwargs)
    if key in {"onthefly", "on_the_fly"}:
        return nk.optimizer.qgt.QGTOnTheFly(holomorphic=bool(holomorphic))
    raise ValueError("qgt must be 'dense', 'pytree', 'auto', or 'onthefly'.")


def _netket_hamiltonian(nk, hilbert, payload: dict, step_data: dict):
    lattice = payload["lattice"]
    sx = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=complex)
    n_op = np.diag([0.0, 1.0]).astype(complex)
    nn_op = np.kron(n_op, n_op)

    hamiltonian = nk.operator.LocalOperator(hilbert, dtype=complex)
    omega = np.asarray(step_data["omega_1d"], dtype=float)
    delta = np.asarray(step_data["delta_1d"], dtype=float)
    for site in range(int(lattice["N"])):
        if abs(float(omega[site])) > 0:
            hamiltonian += nk.operator.LocalOperator(hilbert, float(omega[site]) * sx, [site])
        if abs(float(delta[site])) > 0:
            hamiltonian += nk.operator.LocalOperator(hilbert, -float(delta[site]) * n_op, [site])
    for i_pos, j_pos, strength in lattice["vdw_pairs_1d"]:
        i = int(i_pos) - 1
        j = int(j_pos) - 1
        if i != j and abs(float(strength)) > 0:
            hamiltonian += nk.operator.LocalOperator(hilbert, float(strength) * nn_op, [i, j])
    return hamiltonian


def _measure_netket(nk, hilbert, vstate, payload: dict, observables: list[str]) -> dict[str, Any]:
    lattice = payload["lattice"]
    n_sites = int(lattice["N"])
    z_op = np.diag([-1.0, 1.0]).astype(complex)
    n_op = np.diag([0.0, 1.0]).astype(complex)
    zz_op = np.kron(z_op, z_op)
    snake_to_2d = np.asarray(lattice["snake_to_2d"], dtype=int)
    z_1d: np.ndarray | None = None
    out: dict[str, Any] = {}

    def expect(op) -> float:
        return float(np.real(vstate.expect(op).mean))

    def sigma_z_1d() -> np.ndarray:
        nonlocal z_1d
        if z_1d is None:
            z_1d = np.asarray([
                expect(nk.operator.LocalOperator(hilbert, z_op, [site]))
                for site in range(n_sites)
            ], dtype=float)
        return z_1d

    def rydberg_occ_1d() -> np.ndarray:
        z = sigma_z_1d()
        if np.all(np.isfinite(z)):
            return 0.5 * (z + 1.0)
        return np.asarray([
            expect(nk.operator.LocalOperator(hilbert, n_op, [site]))
            for site in range(n_sites)
        ], dtype=float)

    for name in observables:
        if name in {"sigma_z", "z_i"}:
            z_2d = np.empty(n_sites, dtype=float)
            z_2d[snake_to_2d] = sigma_z_1d()
            out[name] = z_2d
        elif name == "n_mean":
            out[name] = float(np.mean(rydberg_occ_1d()))
        elif name in {"n_i", "n_r"}:
            n_2d = np.empty(n_sites, dtype=float)
            n_2d[snake_to_2d] = rydberg_occ_1d()
            out[name] = n_2d
        elif name == "m_s":
            z_2d = np.empty(n_sites, dtype=float)
            z_2d[snake_to_2d] = sigma_z_1d()
            out[name] = float(np.sum(np.asarray(lattice["sublattice"], dtype=float) * z_2d) / n_sites)
        elif name == "czz_centerline":
            z = sigma_z_1d()
            values = []
            Lx = int(lattice["Lx"])
            Ly = int(lattice["Ly"])
            inv_snake = np.asarray(lattice["inv_snake"], dtype=int)
            for i_2d, j_2d in line_pairs_from_reference(Lx, Ly, axis="horizontal"):
                i = int(inv_snake[int(i_2d)])
                j = int(inv_snake[int(j_2d)])
                zz = expect(nk.operator.LocalOperator(hilbert, zz_op, [i, j]))
                values.append(zz - z[i] * z[j])
            out[name] = np.asarray(values, dtype=float)
    return out


def _jax_accelerator(jax) -> str:
    try:
        devices = jax.devices()
    except Exception:  # pragma: no cover - depends on JAX runtime
        return "unknown"
    if any(getattr(device, "platform", "") == "gpu" for device in devices):
        return "cuda"
    if any(getattr(device, "platform", "") == "tpu" for device in devices):
        return "tpu"
    return "cpu"
