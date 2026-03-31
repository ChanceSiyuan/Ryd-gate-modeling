"""Monte Carlo noise simulation engine.

Provides MonteCarloResult dataclass, MonteCarloEngine for scipy-based MC,
and run_monte_carlo_jax for JAX-accelerated MC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.analysis.gate_metrics import average_gate_infidelity, residuals_to_branching
from ryd_gate.core.atomic_system import build_atom_a_projector, build_occ_operator, build_vdw_unit_operator, get_nominal_distance

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ryd_gate.core.atomic_system import AtomicSystem
    from ryd_gate.protocols.base import Protocol


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
    """Unified quasi-static Monte Carlo noise sampler.

    Noise sources are configured independently via ``setup_*`` methods,
    each corresponding to a distinct physical mechanism.  Only enabled
    noise sources participate in the per-shot perturbation.

    Parameters
    ----------
    system : AtomicSystem
        Atomic system with precomputed Hamiltonians.
    protocol : Protocol
        Pulse protocol providing ``phase_420``.
    x : list of float
        Parameter vector for the protocol.
    """

    def __init__(
        self,
        system: AtomicSystem,
        protocol: Protocol,
        x: list[float],
    ) -> None:
        self.system = system
        self.protocol = protocol
        self.x = x

        self._ham_const_base: NDArray[np.complexfloating] = system.tq_ham_const.copy()
        ham_additions = protocol.get_ham_const_additions()
        if ham_additions is not None:
            self._ham_const_base = self._ham_const_base + ham_additions

        # Noise sources (None until setup_* is called)
        self._sigma_delta_rad: float | None = None
        self._detuning_op: NDArray[np.complexfloating] | None = None

        self._sigma_pos_um: tuple[float, float, float] | None = None
        self._vdw_unit: NDArray[np.complexfloating] | None = None
        self._d_nominal: float | None = None
        self._C_6: float | None = None
        self._v_ryd_nominal: float | None = None

        self._sigma_local_rin: float | None = None
        self._local_detuning_mag: float = 0.0
        self._rin_op: NDArray[np.complexfloating] | None = None

        self._sigma_amplitude: float | None = None
        self._amp_op_1013: NDArray[np.complexfloating] | None = None

    # ── Noise source setup ───────────────────────────────────────────

    def setup_detuning_noise(self, sigma_detuning: float) -> None:
        """Enable quasi-static Rydberg detuning noise.

        Models laser-phase noise that shifts the Rydberg state energy
        of **both** atoms by a random amount each shot.

        Parameters
        ----------
        sigma_detuning : float
            Standard deviation in **Hz**.
        """
        self._sigma_delta_rad = 2 * np.pi * sigma_detuning
        n = self.system.n_levels
        self._detuning_op = sum(
            build_occ_operator(i, n) for i in self.system.rydberg_indices
        )

    def setup_position_noise(
        self, sigma_pos_xyz: tuple[float, float, float],
    ) -> None:
        """Enable interatomic-distance fluctuation noise.

        Each atom is displaced independently in 3-D; the resulting
        change in distance modifies the van-der-Waals interaction.

        Parameters
        ----------
        sigma_pos_xyz : tuple of 3 floats
            Position standard deviations ``(σ_x, σ_y, σ_z)`` in **metres**.
        """
        self._sigma_pos_um = tuple(s * 1e6 for s in sigma_pos_xyz)
        self._vdw_unit = build_vdw_unit_operator(
            self.system.rydberg_indices, self.system.n_levels,
        )
        self._d_nominal = get_nominal_distance(self.system.param_set)
        self._v_ryd_nominal = self.system.v_ryd
        self._C_6 = self._v_ryd_nominal * self._d_nominal ** 6

    def setup_rin_noise(self, sigma_local_rin: float) -> None:
        """Enable 784-nm laser relative-intensity noise on Atom A.

        RIN changes the AC Stark shift on Atom A's Rydberg states,
        proportional to the local detuning magnitude.

        Parameters
        ----------
        sigma_local_rin : float
            Fractional RIN (e.g. 0.01 = 1 %).
        """
        self._sigma_local_rin = sigma_local_rin
        n = self.system.n_levels
        self._rin_op = sum(
            build_atom_a_projector(i, n) for i in self.system.rydberg_indices
        )
        addr = getattr(self.protocol, "addressing", {})
        self._local_detuning_mag = abs(addr.get(0, 0.0))

    def setup_amplitude_noise(self, sigma_amplitude: float) -> None:
        """Enable global Rabi-frequency (amplitude) fluctuation.

        The 1013-nm coupling is perturbed via ``ham_const``;
        the 420-nm coupling is scaled via ``amplitude_scale`` in
        :func:`solve_gate`.

        Parameters
        ----------
        sigma_amplitude : float
            Fractional amplitude noise (e.g. 0.01 = 1 %).
        """
        self._sigma_amplitude = sigma_amplitude
        self._amp_op_1013 = (
            self.system.tq_ham_1013 + self.system.tq_ham_1013_conj
        )

    # ── Per-shot sampling ────────────────────────────────────────────

    def _sample_perturbation(
        self, rng: np.random.Generator,
    ) -> "tuple[NDArray[np.complexfloating], float, dict]":
        """Sample all enabled noise sources for one MC shot.

        Returns
        -------
        ham_perturbed : ndarray
            Perturbed constant Hamiltonian.
        amplitude_scale : float
            Multiplicative scale for the 420-nm laser amplitude.
        samples : dict
            Recorded noise samples (keys depend on enabled sources).
        """
        ham = self._ham_const_base.copy()
        amplitude_scale = 1.0
        samples: dict = {}

        if self._sigma_delta_rad is not None:
            delta_err = rng.normal(0, self._sigma_delta_rad)
            ham += delta_err * self._detuning_op
            samples["detuning"] = delta_err

        if self._sigma_pos_um is not None:
            sigmas = np.array(self._sigma_pos_um * 2)  # (sx,sy,sz,sx,sy,sz)
            displacements = rng.normal(0, sigmas)
            dx1, dy1, dz1, dx2, dy2, dz2 = displacements
            d_new = np.sqrt(
                (self._d_nominal + dx1 - dx2) ** 2
                + (dy1 - dy2) ** 2
                + (dz1 - dz2) ** 2
            )
            d_new = max(d_new, 0.1)
            delta_v = self._C_6 / d_new ** 6 - self._v_ryd_nominal
            ham += delta_v * self._vdw_unit
            samples["distance"] = d_new

        if (
            self._sigma_local_rin is not None
            and self._local_detuning_mag > 0
        ):
            rin_err = rng.normal(
                0, self._sigma_local_rin * self._local_detuning_mag,
            )
            ham += rin_err * self._rin_op
            samples["rin"] = rin_err

        if self._sigma_amplitude is not None:
            amp_err = rng.normal(0, self._sigma_amplitude)
            ham += amp_err * self._amp_op_1013
            amplitude_scale = 1.0 + amp_err
            samples["amplitude"] = amp_err

        return ham, amplitude_scale, samples

    # ── Run modes ────────────────────────────────────────────────────

    @staticmethod
    def _print_progress(label: str, shot: int, n_shots: int) -> None:
        if n_shots < 5:
            return
        pct = (shot + 1) * 100 // n_shots
        prev_pct = shot * 100 // n_shots
        if shot == 0 or pct != prev_pct:
            print(
                f"\r  {label} {shot + 1}/{n_shots} ({pct}%)",
                end="", flush=True,
            )

    def run_gate_fidelity(
        self,
        n_shots: int = 1000,
        seed: int | None = None,
        compute_branching: bool = False,
    ) -> MonteCarloResult:
        """CZ gate mode: compute average gate infidelity per shot.

        Returns
        -------
        MonteCarloResult
        """
        rng = np.random.default_rng(seed)

        fidelities = np.zeros(n_shots)
        detuning_samples = (
            np.zeros(n_shots) if self._sigma_delta_rad is not None else None
        )
        distance_samples = (
            np.zeros(n_shots) if self._sigma_pos_um is not None else None
        )

        if compute_branching:
            branch_xyz_arr = np.zeros(n_shots)
            branch_al_arr = np.zeros(n_shots)
            branch_lg_arr = np.zeros(n_shots)
            branch_phase_arr = np.zeros(n_shots)

        for shot in range(n_shots):
            self._print_progress("MC shot", shot, n_shots)

            ham_perturbed, amplitude_scale, samples = self._sample_perturbation(rng)

            if detuning_samples is not None and "detuning" in samples:
                detuning_samples[shot] = samples["detuning"]
            if distance_samples is not None and "distance" in samples:
                distance_samples[shot] = samples["distance"]

            if compute_branching:
                infidelity, residuals = average_gate_infidelity(
                    self.system, self.protocol, self.x,
                    return_residuals=True,
                    ham_const_override=ham_perturbed,
                    amplitude_scale=amplitude_scale,
                )
                fidelities[shot] = 1.0 - infidelity
                branching = residuals_to_branching(self.system, residuals)
                branch_xyz_arr[shot] = branching["XYZ"]
                branch_al_arr[shot] = branching["AL"]
                branch_lg_arr[shot] = branching["LG"]
                branch_phase_arr[shot] = max(
                    infidelity
                    - (branching["XYZ"] + branching["AL"] + branching["LG"]),
                    0.0,
                )
            else:
                infidelity = average_gate_infidelity(
                    self.system, self.protocol, self.x,
                    ham_const_override=ham_perturbed,
                    amplitude_scale=amplitude_scale,
                )
                fidelities[shot] = 1.0 - infidelity

        if n_shots >= 5:
            print(f"\r  MC done: {n_shots}/{n_shots} (100%)    ")

        mean_fid = float(np.mean(fidelities))
        std_fid = float(np.std(fidelities))
        infidelities = 1.0 - fidelities

        kwargs: dict = dict(
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

    def run_states(
        self,
        initial_state: "NDArray[np.complexfloating]",
        n_shots: int = 50,
        seed: int | None = None,
    ) -> "list[NDArray[np.complexfloating]]":
        """State evolution mode: evolve a single initial state per shot.

        Parameters
        ----------
        initial_state : ndarray
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
        from ryd_gate.solvers.schrodinger import solve_gate

        rng = np.random.default_rng(seed)
        final_states = []

        for shot in range(n_shots):
            self._print_progress("MC shot", shot, n_shots)

            ham_perturbed, amplitude_scale, _ = self._sample_perturbation(rng)

            psi_final = solve_gate(
                self.system,
                self.protocol,
                self.x,
                initial_state,
                ham_const_override=ham_perturbed,
                amplitude_scale=amplitude_scale,
            )
            final_states.append(psi_final)

        if n_shots >= 5:
            print(f"\r  MC done: {n_shots}/{n_shots} (100%)    ")

        return final_states


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


