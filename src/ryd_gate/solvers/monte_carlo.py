"""Monte Carlo noise simulation engine.

Provides MonteCarloResult dataclass, MonteCarloEngine for scipy-based MC,
run_monte_carlo_jax for JAX-accelerated MC, and AddressingMCEngine for
local addressing simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.analysis.gate_metrics import average_gate_infidelity, residuals_to_branching
from ryd_gate.core.atomic_system import build_atom_a_projector, build_vdw_unit_operator, get_nominal_distance

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ryd_gate.core.atomic_system import AtomicSystem
    from ryd_gate.protocols.base import Protocol, SweepProtocol


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation.

    Attributes
    ----------
    mean_fidelity : float
        Average gate fidelity over all shots.
    std_fidelity : float
        Standard deviation of gate fidelity.
    mean_infidelity : float
        Average gate infidelity (1 - fidelity) over all shots.
    std_infidelity : float
        Standard deviation of gate infidelity.
    n_shots : int
        Number of Monte Carlo shots performed.
    fidelities : NDArray[np.floating]
        Array of individual fidelities for each shot.
    detuning_samples : NDArray[np.floating] | None
        Sampled detuning errors (rad/s) if T2* dephasing was enabled.
    distance_samples : NDArray[np.floating] | None
        Sampled interatomic distances (μm) if position fluctuations were enabled.
    branch_XYZ : NDArray[np.floating] | None
        Per-shot XYZ (Pauli) error from residual populations.
    branch_AL : NDArray[np.floating] | None
        Per-shot atom loss error from residual Rydberg populations.
    branch_LG : NDArray[np.floating] | None
        Per-shot leakage error from residual populations.
    branch_phase : NDArray[np.floating] | None
        Per-shot phase error (infidelity minus population-based errors).
    mean_branch_XYZ : float | None
        Mean XYZ error across all shots.
    std_branch_XYZ : float | None
        Std of XYZ error across all shots.
    mean_branch_AL : float | None
        Mean atom loss error across all shots.
    std_branch_AL : float | None
        Std of atom loss error across all shots.
    mean_branch_LG : float | None
        Mean leakage error across all shots.
    std_branch_LG : float | None
        Std of leakage error across all shots.
    mean_branch_phase : float | None
        Mean phase error across all shots.
    std_branch_phase : float | None
        Std of phase error across all shots.
    """

    mean_fidelity: float
    std_fidelity: float
    mean_infidelity: float
    std_infidelity: float
    n_shots: int
    fidelities: "NDArray[np.floating]"
    detuning_samples: "NDArray[np.floating] | None" = None
    distance_samples: "NDArray[np.floating] | None" = None
    branch_XYZ: "NDArray[np.floating] | None" = None
    branch_AL: "NDArray[np.floating] | None" = None
    branch_LG: "NDArray[np.floating] | None" = None
    branch_phase: "NDArray[np.floating] | None" = None
    mean_branch_XYZ: "float | None" = None
    std_branch_XYZ: "float | None" = None
    mean_branch_AL: "float | None" = None
    std_branch_AL: "float | None" = None
    mean_branch_LG: "float | None" = None
    std_branch_LG: "float | None" = None
    mean_branch_phase: "float | None" = None
    std_branch_phase: "float | None" = None

    def save_to_file(self, filepath: str) -> None:
        """Save Monte Carlo results to a text file."""
        import datetime

        with open(filepath, "w") as f:
            f.write(f"# MonteCarloResult saved {datetime.datetime.now().isoformat()}\n")
            f.write(f"# n_shots = {self.n_shots}\n")
            f.write(f"# mean_fidelity = {self.mean_fidelity:.12e}\n")
            f.write(f"# std_fidelity = {self.std_fidelity:.12e}\n")
            f.write(f"# mean_infidelity = {self.mean_infidelity:.12e}\n")
            f.write(f"# std_infidelity = {self.std_infidelity:.12e}\n")

            if self.mean_branch_XYZ is not None:
                f.write(f"# mean_branch_XYZ = {self.mean_branch_XYZ:.12e}\n")
                f.write(f"# std_branch_XYZ = {self.std_branch_XYZ:.12e}\n")
                f.write(f"# mean_branch_AL = {self.mean_branch_AL:.12e}\n")
                f.write(f"# std_branch_AL = {self.std_branch_AL:.12e}\n")
                f.write(f"# mean_branch_LG = {self.mean_branch_LG:.12e}\n")
                f.write(f"# std_branch_LG = {self.std_branch_LG:.12e}\n")
                f.write(f"# mean_branch_phase = {self.mean_branch_phase:.12e}\n")
                f.write(f"# std_branch_phase = {self.std_branch_phase:.12e}\n")

            columns = ["shot", "fidelity", "infidelity"]
            if self.detuning_samples is not None:
                columns.append("detuning_sample")
            if self.distance_samples is not None:
                columns.append("distance_sample")
            if self.branch_XYZ is not None:
                columns.extend(["branch_XYZ", "branch_AL", "branch_LG", "branch_phase"])

            f.write(f"# columns: {' '.join(columns)}\n")

            for i in range(self.n_shots):
                row = [
                    str(i),
                    f"{self.fidelities[i]:.12e}",
                    f"{1 - self.fidelities[i]:.12e}",
                ]
                if self.detuning_samples is not None:
                    row.append(f"{self.detuning_samples[i]:.12e}")
                if self.distance_samples is not None:
                    row.append(f"{self.distance_samples[i]:.12e}")
                if self.branch_XYZ is not None:
                    row.append(f"{self.branch_XYZ[i]:.12e}")
                    row.append(f"{self.branch_AL[i]:.12e}")
                    row.append(f"{self.branch_LG[i]:.12e}")
                    row.append(f"{self.branch_phase[i]:.12e}")
                f.write("\t".join(row) + "\n")

    @classmethod
    def load_from_file(cls, filepath: str) -> MonteCarloResult:
        """Load Monte Carlo results from a text file."""
        column_names: list[str] = []
        data_rows: list[list[str]] = []

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line.startswith("# columns:"):
                    column_names = line.split(":", 1)[1].strip().split()
                elif line.startswith("#"):
                    continue
                elif line:
                    data_rows.append(line.split("\t"))

        n_shots = len(data_rows)
        data = np.array([[float(v) for v in row] for row in data_rows])
        col_idx = {name: i for i, name in enumerate(column_names)}

        fidelities = data[:, col_idx["fidelity"]]

        kwargs: dict = dict(
            mean_fidelity=float(np.mean(fidelities)),
            std_fidelity=float(np.std(fidelities)),
            mean_infidelity=float(np.mean(1 - fidelities)),
            std_infidelity=float(np.std(1 - fidelities)),
            n_shots=n_shots,
            fidelities=fidelities,
        )

        if "detuning_sample" in col_idx:
            kwargs["detuning_samples"] = data[:, col_idx["detuning_sample"]]
        if "distance_sample" in col_idx:
            kwargs["distance_samples"] = data[:, col_idx["distance_sample"]]
        if "branch_XYZ" in col_idx:
            bx = data[:, col_idx["branch_XYZ"]]
            ba = data[:, col_idx["branch_AL"]]
            bl = data[:, col_idx["branch_LG"]]
            bp = data[:, col_idx["branch_phase"]]
            kwargs.update(
                branch_XYZ=bx, branch_AL=ba, branch_LG=bl, branch_phase=bp,
                mean_branch_XYZ=float(np.mean(bx)),
                std_branch_XYZ=float(np.std(bx)),
                mean_branch_AL=float(np.mean(ba)),
                std_branch_AL=float(np.std(ba)),
                mean_branch_LG=float(np.mean(bl)),
                std_branch_LG=float(np.std(bl)),
                mean_branch_phase=float(np.mean(bp)),
                std_branch_phase=float(np.std(bp)),
            )

        return cls(**kwargs)


class MonteCarloEngine:
    """Quasi-static Monte Carlo noise sampler.

    Unlike the old design which mutated self.tq_ham_const per-shot,
    this computes a perturbed ham_const and passes it via ham_const_override.
    """

    def __init__(
        self,
        system: AtomicSystem,
        protocol: Protocol,
        *,
        enable_rydberg_dephasing: bool = False,
        enable_position_error: bool = False,
        sigma_detuning: float | None = None,
        sigma_pos_xyz: tuple[float, float, float] | None = None,
    ):
        self.system = system
        self.protocol = protocol
        self.enable_rydberg_dephasing = enable_rydberg_dephasing
        self.enable_position_error = enable_position_error
        self.sigma_detuning = sigma_detuning
        self.sigma_pos_xyz = sigma_pos_xyz

    def run(
        self,
        x: list[float],
        n_shots: int = 1000,
        sigma_detuning: float | None = None,
        sigma_pos_xyz: tuple[float, float, float] | None = None,
        seed: int | None = None,
        compute_branching: bool = False,
    ) -> MonteCarloResult:
        """Run MC simulation with per-shot Hamiltonian perturbations."""
        # Use passed-in values or fall back to constructor values
        sigma_det = sigma_detuning if sigma_detuning is not None else self.sigma_detuning
        sigma_pos = sigma_pos_xyz if sigma_pos_xyz is not None else self.sigma_pos_xyz

        rng = np.random.default_rng(seed)

        sigma_delta_rad = (
            2 * np.pi * sigma_det
            if self.enable_rydberg_dephasing and sigma_det is not None
            else None
        )
        use_position = (
            self.enable_position_error and sigma_pos is not None
        )

        ham_vdw_unit = build_vdw_unit_operator()
        original_ham_const = self.system.tq_ham_const
        original_v_ryd = self.system.v_ryd
        d_nominal = get_nominal_distance(self.system.param_set)
        C_6 = original_v_ryd * d_nominal**6

        if use_position:
            sx_um, sy_um, sz_um = [s * 1e6 for s in sigma_pos]

        fidelities = np.zeros(n_shots)
        detuning_samples = np.zeros(n_shots) if sigma_delta_rad else None
        distance_samples = np.zeros(n_shots) if use_position else None

        if compute_branching:
            branch_xyz_arr = np.zeros(n_shots)
            branch_al_arr = np.zeros(n_shots)
            branch_lg_arr = np.zeros(n_shots)
            branch_phase_arr = np.zeros(n_shots)

        for shot in range(n_shots):
            if n_shots >= 5:
                pct = (shot + 1) * 100 // n_shots
                prev_pct = shot * 100 // n_shots
                if shot == 0 or pct != prev_pct:
                    print(
                        f"\r  MC shot {shot + 1}/{n_shots} ({pct}%)",
                        end="", flush=True,
                    )

            # Build perturbed Hamiltonian without mutating system
            ham_perturbed = original_ham_const.copy()

            if sigma_delta_rad is not None:
                delta_err = rng.normal(0, sigma_delta_rad)
                detuning_samples[shot] = delta_err
                perturbation_sq = np.zeros((7, 7), dtype=np.complex128)
                perturbation_sq[5, 5] = delta_err
                perturbation_sq[6, 6] = delta_err
                perturbation_tq = np.kron(np.eye(7), perturbation_sq) + np.kron(
                    perturbation_sq, np.eye(7)
                )
                ham_perturbed = ham_perturbed + perturbation_tq

            if use_position:
                dx1 = rng.normal(0, sx_um)
                dy1 = rng.normal(0, sy_um)
                dz1 = rng.normal(0, sz_um)
                dx2 = rng.normal(0, sx_um)
                dy2 = rng.normal(0, sy_um)
                dz2 = rng.normal(0, sz_um)
                R0 = d_nominal
                d_new = np.sqrt(
                    (R0 + dx1 - dx2) ** 2
                    + (dy1 - dy2) ** 2
                    + (dz1 - dz2) ** 2
                )
                d_new = max(d_new, 0.1)
                distance_samples[shot] = d_new
                v_ryd_new = C_6 / d_new**6
                delta_v = v_ryd_new - original_v_ryd
                ham_perturbed = ham_perturbed + delta_v * ham_vdw_unit

            if compute_branching:
                infidelity, residuals = average_gate_infidelity(
                    self.system, self.protocol, x,
                    return_residuals=True,
                    ham_const_override=ham_perturbed,
                )
                fidelities[shot] = 1.0 - infidelity
                branching = residuals_to_branching(self.system, residuals)
                branch_xyz_arr[shot] = branching["XYZ"]
                branch_al_arr[shot] = branching["AL"]
                branch_lg_arr[shot] = branching["LG"]
                branch_phase_arr[shot] = max(
                    infidelity - (branching["XYZ"] + branching["AL"] + branching["LG"]),
                    0.0,
                )
            else:
                infidelity = average_gate_infidelity(
                    self.system, self.protocol, x,
                    ham_const_override=ham_perturbed,
                )
                fidelities[shot] = 1.0 - infidelity

        if n_shots >= 5:
            print(f"\r  MC done: {n_shots}/{n_shots} (100%)    ")

        mean_fid = float(np.mean(fidelities))
        std_fid = float(np.std(fidelities))
        infidelities = 1.0 - fidelities

        kwargs = dict(
            mean_fidelity=mean_fid,
            std_fidelity=std_fid,
            mean_infidelity=float(np.mean(infidelities)),
            std_infidelity=float(np.std(infidelities)),
            n_shots=n_shots,
            fidelities=fidelities,
            detuning_samples=detuning_samples,
            distance_samples=distance_samples,
        )

        if compute_branching:
            kwargs.update(
                branch_XYZ=branch_xyz_arr,
                branch_AL=branch_al_arr,
                branch_LG=branch_lg_arr,
                branch_phase=branch_phase_arr,
                mean_branch_XYZ=float(np.mean(branch_xyz_arr)),
                std_branch_XYZ=float(np.std(branch_xyz_arr)),
                mean_branch_AL=float(np.mean(branch_al_arr)),
                std_branch_AL=float(np.std(branch_al_arr)),
                mean_branch_LG=float(np.mean(branch_lg_arr)),
                std_branch_LG=float(np.std(branch_lg_arr)),
                mean_branch_phase=float(np.mean(branch_phase_arr)),
                std_branch_phase=float(np.std(branch_phase_arr)),
            )

        return MonteCarloResult(**kwargs)


def run_monte_carlo_jax(
    system: AtomicSystem,
    protocol: Protocol,
    x: list[float],
    n_shots: int = 1000,
    sigma_detuning: float | None = None,
    sigma_pos_xyz: tuple[float, float, float] | None = None,
    seed: int = 0,
    *,
    enable_rydberg_dephasing: bool = False,
    enable_position_error: bool = False,
) -> MonteCarloResult:
    """GPU-accelerated Monte Carlo simulation using JAX.

    TO strategy only. Uses JAX odeint + vmap + jit.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental.ode import odeint

    from ryd_gate.blackman import blackman_pulse as _  # noqa: F401
    from ryd_gate.protocols.gate_cz_to import TOProtocol

    jax.config.update("jax_enable_x64", True)

    if not isinstance(protocol, TOProtocol):
        raise NotImplementedError("run_monte_carlo_jax only supports TO strategy.")

    params = protocol.unpack_params(x, system)
    phase_amp = params["phase_amp"]
    omega = params["omega"]
    phase_init = params["phase_init"]
    delta = params["delta"]
    theta = params["theta"]
    t_gate = params["t_gate"]

    H_const_base = jnp.array(system.tq_ham_const)
    H_420 = jnp.array(system.tq_ham_420)
    H_420_conj = jnp.array(system.tq_ham_420_conj)
    H_1013 = jnp.array(system.tq_ham_1013)
    H_1013_conj = jnp.array(system.tq_ham_1013_conj)
    H_lightshift = jnp.array(system.tq_ham_lightshift_zero)
    H_1013_sum = H_1013 + H_1013_conj

    pert_sq = np.zeros((7, 7), dtype=np.complex128)
    pert_sq[5, 5] = 1.0
    pert_sq[6, 6] = 1.0
    detuning_diag = jnp.array(
        np.diag(np.kron(np.eye(7), pert_sq) + np.kron(pert_sq, np.eye(7)))
    )

    ham_vdw_unit = jnp.array(build_vdw_unit_operator())
    d_nominal = get_nominal_distance(system.param_set)
    C_6 = system.v_ryd * d_nominal**6

    sigma_delta_rad = (
        2 * np.pi * sigma_detuning
        if enable_rydberg_dephasing and sigma_detuning is not None
        else None
    )
    use_position = enable_position_error and sigma_pos_xyz is not None

    key = jax.random.PRNGKey(seed)

    if sigma_delta_rad is not None:
        key, key1 = jax.random.split(key)
        delta_errs = jax.random.normal(key1, (n_shots,)) * sigma_delta_rad
    else:
        delta_errs = jnp.zeros(n_shots)

    if use_position:
        sx_um, sy_um, sz_um = [s * 1e6 for s in sigma_pos_xyz]
        keys = jax.random.split(key, 7)
        dx1 = jax.random.normal(keys[0], (n_shots,)) * sx_um
        dy1 = jax.random.normal(keys[1], (n_shots,)) * sy_um
        dz1 = jax.random.normal(keys[2], (n_shots,)) * sz_um
        dx2 = jax.random.normal(keys[3], (n_shots,)) * sx_um
        dy2 = jax.random.normal(keys[4], (n_shots,)) * sy_um
        dz2 = jax.random.normal(keys[5], (n_shots,)) * sz_um
        d_new = jnp.sqrt(
            (d_nominal + dx1 - dx2) ** 2
            + (dy1 - dy2) ** 2
            + (dz1 - dz2) ** 2
        )
        d_new = jnp.maximum(d_new, 0.1)
        delta_v = C_6 / d_new**6 - system.v_ryd
    else:
        d_new = jnp.full(n_shots, d_nominal)
        delta_v = jnp.zeros(n_shots)

    H_const_batch = (
        H_const_base[None, :, :]
        + delta_errs[:, None, None] * jnp.diag(detuning_diag)[None, :, :]
        + delta_v[:, None, None] * ham_vdw_unit[None, :, :]
    )

    y0_01 = jnp.array(
        np.kron([1 + 0j, 0, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0])
    )
    y0_11 = jnp.array(
        np.kron([0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0])
    )

    t_eval = jnp.linspace(0.0, t_gate, 1000)

    use_blackman = system.blackmanflag
    t_rise = system.t_rise

    def rhs(y, t, H_c):
        phase_420_val = jnp.exp(
            -1j * (phase_amp * jnp.cos(omega * t + phase_init) + delta * t)
        )
        if use_blackman:
            bw = lambda tt, tr: (
                0.42 - 0.5 * jnp.cos(2 * jnp.pi * tt / (2 * tr))
                + 0.08 * jnp.cos(4 * jnp.pi * tt / (2 * tr))
            )
            amp_rise = bw(t, t_rise) * (t < t_rise)
            amp_fall = bw(t_gate - t, t_rise) * ((t_gate - t) < t_rise)
            amp_flat = ((t >= t_rise) & (t <= t_gate - t_rise)).astype(y.dtype)
            amplitude = amp_rise + amp_flat + amp_fall
        else:
            amplitude = 1.0
        H = (
            H_c
            + amplitude * phase_420_val * H_420
            + amplitude * jnp.conj(phase_420_val) * H_420_conj
            + H_1013_sum
            + amplitude * amplitude * H_lightshift
        )
        return -1j * (H @ y)

    def single_shot_infidelity(H_c):
        sol_01 = odeint(rhs, y0_01, t_eval, H_c, rtol=1e-8, atol=1e-12)
        psi_01 = sol_01[-1]
        sol_11 = odeint(rhs, y0_11, t_eval, H_c, rtol=1e-8, atol=1e-12)
        psi_11 = sol_11[-1]

        a01 = jnp.exp(-1.0j * theta) * jnp.vdot(y0_01, psi_01)
        a11 = jnp.exp(-2.0j * theta - 1.0j * jnp.pi) * jnp.vdot(y0_11, psi_11)

        avg_F = (1.0 / 20.0) * (
            jnp.abs(1.0 + 2.0 * a01 + a11) ** 2
            + 1.0
            + 2.0 * jnp.abs(a01) ** 2
            + jnp.abs(a11) ** 2
        )
        return 1.0 - avg_F.real

    batched_infidelity = jax.jit(jax.vmap(single_shot_infidelity))
    infidelities_jax = batched_infidelity(H_const_batch)

    infidelities = np.asarray(infidelities_jax)
    fidelities = 1.0 - infidelities

    mean_fid = float(np.mean(fidelities))
    std_fid = float(np.std(fidelities))

    return MonteCarloResult(
        mean_fidelity=mean_fid,
        std_fidelity=std_fid,
        mean_infidelity=float(np.mean(infidelities)),
        std_infidelity=float(np.std(infidelities)),
        n_shots=n_shots,
        fidelities=fidelities,
        detuning_samples=np.asarray(delta_errs) if sigma_delta_rad is not None else None,
        distance_samples=np.asarray(d_new) if use_position else None,
    )


class AddressingMCEngine:
    """Monte Carlo engine for local addressing experiments.

    Applies quasi-static noise per-shot:
    - Global detuning error on Rydberg states of both atoms
    - Local RIN error on Atom A's Rydberg states only

    Parameters
    ----------
    system : AtomicSystem
        Atomic system with precomputed Hamiltonians.
    protocol : SweepProtocol
        Sweep addressing protocol.
    sigma_detuning : float
        Standard deviation of global detuning noise (Hz).
    sigma_local_rin : float
        Relative intensity noise (fractional, e.g. 0.01 = 1%).
    """

    def __init__(
        self,
        system: AtomicSystem,
        protocol: SweepProtocol,
        *,
        sigma_detuning: float = 0.0,
        sigma_local_rin: float = 0.0,
    ) -> None:
        self.system = system
        self.protocol = protocol
        self.sigma_detuning = sigma_detuning
        self.sigma_local_rin = sigma_local_rin

        # Precompute noise operators
        # Global detuning: shift Rydberg states of both atoms
        detuning_sq = np.zeros((7, 7), dtype=np.complex128)
        detuning_sq[5, 5] = 1.0
        detuning_sq[6, 6] = 1.0
        self._detuning_op = (
            np.kron(np.eye(7), detuning_sq) + np.kron(detuning_sq, np.eye(7))
        )

        # Local RIN: shift Atom A's Rydberg states only
        self._rin_op = build_atom_a_projector(5) + build_atom_a_projector(6)

    def run(
        self,
        initial_state: "NDArray[np.complexfloating]",
        n_shots: int = 50,
        seed: int | None = None,
    ) -> "list[NDArray[np.complexfloating]]":
        """Run MC simulation, returning list of final state vectors.

        Parameters
        ----------
        initial_state : ndarray, shape (49,)
            Two-atom initial state vector.
        n_shots : int
            Number of Monte Carlo shots.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        list of ndarray
            Final state vectors (one per shot).
        """
        from ryd_gate.protocols.local_sweep import SweepAddressingProtocol
        from ryd_gate.solvers.schrodinger import evolve

        rng = np.random.default_rng(seed)
        sigma_delta_rad = 2 * np.pi * self.sigma_detuning
        local_detuning_mag = abs(self.protocol.local_detuning_A) if isinstance(
            self.protocol, SweepAddressingProtocol
        ) else 0.0

        final_states = []

        for shot in range(n_shots):
            if n_shots >= 5:
                pct = (shot + 1) * 100 // n_shots
                prev_pct = shot * 100 // n_shots
                if shot == 0 or pct != prev_pct:
                    print(
                        f"\r  Addressing MC {shot + 1}/{n_shots} ({pct}%)",
                        end="", flush=True,
                    )

            # Sample noise
            delta_err = rng.normal(0, sigma_delta_rad) if self.sigma_detuning > 0 else 0.0
            rin_err = (
                rng.normal(0, self.sigma_local_rin * local_detuning_mag)
                if self.sigma_local_rin > 0
                else 0.0
            )

            # Build per-shot noise Hamiltonian (added to protocol's H(t))
            H_noise = delta_err * self._detuning_op + rin_err * self._rin_op

            def hamiltonian_fn(t, _H_noise=H_noise):
                return self.protocol.get_hamiltonian(t, self.system) + _H_noise

            psi_final = evolve(
                hamiltonian_fn,
                self.protocol.t_gate,
                initial_state,
            )
            final_states.append(psi_final)

        if n_shots >= 5:
            print(f"\r  Addressing MC done: {n_shots}/{n_shots} (100%)    ")

        return final_states
