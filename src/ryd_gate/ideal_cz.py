"""Backward-compatible facade for CZGateSimulator.

All new development should import from submodules directly::

    from ryd_gate.core.atomic_system import create_our_system, AtomicSystem
    from ryd_gate.protocols.gate_cz_to import TOProtocol
    from ryd_gate.solvers.schrodinger import solve_gate
    from ryd_gate.analysis.gate_metrics import average_gate_infidelity
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from qutip import Bloch
from scipy.optimize import minimize

from ryd_gate.core.atomic_system import (
    AtomicSystem,
    build_occ_operator,
    build_sss_state_map,
    build_vdw_unit_operator,
    create_lukin_system,
    create_our_system,
    get_nominal_distance,
)
from ryd_gate.protocols.base import Protocol
from ryd_gate.protocols.gate_cz_ar import ARProtocol
from ryd_gate.protocols.gate_cz_to import TOProtocol
from ryd_gate.solvers.schrodinger import solve_gate
from ryd_gate.solvers.monte_carlo import (
    MonteCarloEngine,
    MonteCarloResult,
    run_monte_carlo_jax,
)
from ryd_gate.analysis.gate_metrics import (
    average_gate_infidelity,
    bell_infidelity,
    error_budget as _error_budget,
    population_evolution as _population_evolution,
    residuals_to_branching as _residuals_to_branching,
    sss_infidelity,
    state_infidelity as _state_infidelity,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Supported initial state labels for diagnostic methods
InitialStateLabel = Literal[
    "00", "01", "10", "11",
    "SSS-0", "SSS-1", "SSS-2", "SSS-3", "SSS-4", "SSS-5",
    "SSS-6", "SSS-7", "SSS-8", "SSS-9", "SSS-10", "SSS-11",
]


class CZGateSimulator:
    """Backward-compatible facade delegating to modular subpackages.

    See module docstring for recommended direct imports.
    """

    def __init__(
        self,
        param_set: Literal["our", "lukin"] = "our",
        strategy: Literal["TO", "AR"] = "AR",
        blackmanflag: bool = True,
        detuning_sign: Literal[1, -1] = 1,
        *,
        enable_rydberg_decay: bool = False,
        enable_intermediate_decay: bool = False,
        enable_0_scattering: bool = True,
        use_jax_mc: bool = False,
        enable_rydberg_dephasing: bool = False,
        enable_position_error: bool = False,
        enable_polarization_leakage: bool = False,
        sigma_detuning: float | None = None,
        sigma_pos_xyz: tuple[float, float, float] | None = None,
        n_mc_shots: int = 100,
        mc_seed: int | None = None,
    ) -> None:
        self.strategy = strategy
        self.enable_rydberg_dephasing = enable_rydberg_dephasing
        self.enable_position_error = enable_position_error
        self.sigma_detuning = sigma_detuning
        self.sigma_pos_xyz = sigma_pos_xyz
        self.n_mc_shots = n_mc_shots
        self.mc_seed = mc_seed
        self.x_initial: list[float] | None = None

        if enable_rydberg_dephasing and sigma_detuning is None:
            raise ValueError(
                "sigma_detuning must be provided when enable_rydberg_dephasing=True. "
                "Typical value: 130e3 (Hz)."
            )
        if enable_position_error and sigma_pos_xyz is None:
            raise ValueError(
                "sigma_pos_xyz must be provided when enable_position_error=True. "
                "Typical value: (70e-9, 70e-9, 130e-9) (meters)."
            )

        # Create immutable atomic system
        _factory_kwargs = dict(
            detuning_sign=detuning_sign,
            blackmanflag=blackmanflag,
            enable_rydberg_decay=enable_rydberg_decay,
            enable_intermediate_decay=enable_intermediate_decay,
            enable_0_scattering=enable_0_scattering,
            enable_polarization_leakage=enable_polarization_leakage,
        )
        if param_set == "our":
            self._system: AtomicSystem = create_our_system(**_factory_kwargs)
        elif param_set == "lukin":
            self._system = create_lukin_system(**_factory_kwargs)
        else:
            raise ValueError(
                f"CZGateSimulator only supports 'our' or 'lukin' systems, "
                f"got '{param_set}'."
            )

        # Create protocol object
        if strategy == "TO":
            self._protocol: Protocol = TOProtocol()
        elif strategy == "AR":
            self._protocol = ARProtocol()
        else:
            raise ValueError(
                f"Unknown strategy: '{strategy}'. Choose 'TO' or 'AR'."
            )

    # ==================================================================
    # ATTRIBUTE COMPATIBILITY LAYER
    # ==================================================================

    @property
    def param_set(self):
        return self._system.param_set

    @property
    def blackmanflag(self):
        return self._system.blackmanflag

    @property
    def detuning_sign(self):
        return self._system.detuning_sign

    @property
    def enable_rydberg_decay(self):
        return self._system.enable_rydberg_decay

    @property
    def enable_intermediate_decay(self):
        return self._system.enable_intermediate_decay

    @property
    def enable_0_scattering(self):
        return self._system.enable_0_scattering

    @property
    def enable_polarization_leakage(self):
        return self._system.enable_polarization_leakage

    @property
    def rabi_eff(self):
        return self._system.rabi_eff

    @property
    def time_scale(self):
        return self._system.time_scale

    @property
    def rabi_420(self):
        return self._system.rabi_420

    @property
    def rabi_1013(self):
        return self._system.rabi_1013

    @property
    def rabi_420_garbage(self):
        return self._system.rabi_420_garbage

    @property
    def rabi_1013_garbage(self):
        return self._system.rabi_1013_garbage

    @property
    def Delta(self):
        return self._system.Delta

    @property
    def v_ryd(self):
        return self._system.v_ryd

    @property
    def ryd_level(self):
        return self._system.ryd_level

    @property
    def ryd_zeeman_shift(self):
        return self._system.ryd_zeeman_shift

    @property
    def atom(self):
        return self._system.atom

    @property
    def d_mid_ratio(self):
        return self._system.d_mid_ratio

    @property
    def d_ryd_ratio(self):
        return self._system.d_ryd_ratio

    @property
    def tq_ham_const(self):
        return self._system.tq_ham_const

    @property
    def tq_ham_420(self):
        return self._system.tq_ham_420

    @property
    def tq_ham_1013(self):
        return self._system.tq_ham_1013

    @property
    def tq_ham_420_conj(self):
        return self._system.tq_ham_420_conj

    @property
    def tq_ham_1013_conj(self):
        return self._system.tq_ham_1013_conj

    @property
    def tq_ham_lightshift_zero(self):
        return self._system.tq_ham_lightshift_zero

    @property
    def mid_state_decay_rate(self):
        return self._system.mid_state_decay_rate

    @property
    def ryd_state_decay_rate(self):
        return self._system.ryd_state_decay_rate

    @property
    def ryd_RD_rate(self):
        return self._system.ryd_RD_rate

    @property
    def ryd_BBR_rate(self):
        return self._system.ryd_BBR_rate

    @property
    def t_rise(self):
        return self._system.t_rise

    @property
    def _ryd_branch(self):
        return self._system.ryd_branch

    @property
    def _mid_branch(self):
        return self._system.mid_branch

    # ==================================================================
    # PROTOCOL SETUP
    # ==================================================================

    def _setup_protocol_TO(self, x: list[float]) -> None:
        if len(x) != 6:
            raise ValueError(
                f"TO parameters must be a list of 6 elements. Got {len(x)} elements."
            )
        self.x_initial = x
        print(f"TO parameters is set to: [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale] = {x}")

    def _setup_protocol_AR(self, x: list[float]) -> None:
        if len(x) != 8:
            raise ValueError(
                f"AR parameters must be a list of 8 elements. Got {len(x)} elements."
            )
        self.x_initial = x
        print(f"AR parameters is set to: [ω/Ω_eff, A₁, φ₁, A₂, φ₂, δ/Ω_eff, T/T_scale, θ] = {x}")

    def _setup_protocol(self, x: list[float]) -> None:
        if self.strategy == "TO":
            self._setup_protocol_TO(x)
        elif self.strategy == "AR":
            self._setup_protocol_AR(x)
        else:
            raise ValueError(
                f"Unknown strategy: '{self.strategy}'. Choose 'TO' or 'AR'."
            )

    def setup_protocol(self, x: list[float]) -> None:
        """Store pulse parameters for subsequent method calls."""
        self._setup_protocol(x)

    def _resolve_params(self, x: list[float] | None, caller: str = "") -> list[float]:
        if x is not None:
            return x
        if self.x_initial is not None:
            return self.x_initial
        raise ValueError(
            f"No pulse parameters available{' in ' + caller if caller else ''}. "
            "Call setup_protocol(x) first or pass x explicitly."
        )

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def optimize(
        self,
        x_initial=None,
        fid_type: Literal["average", "sss", "bell"] = "average",
    ) -> object:
        """Run pulse parameter optimization."""
        if self.strategy == "TO":
            return self._optimization_TO(fid_type, x=x_initial)
        elif self.strategy == "AR":
            return self._optimization_AR(fid_type, x=x_initial)
        else:
            raise ValueError(
                f"Unknown strategy: '{self.strategy}'. Choose 'TO' or 'AR'."
            )

    def _gate_infidelity_single(
        self,
        x: list[float],
        fid_type: Literal["average", "sss", "bell"] = "average",
        return_residuals: bool = False,
    ) -> "float | tuple[float, dict[str, float]]":
        """Compute single-shot gate infidelity (no MC averaging)."""
        if return_residuals and fid_type != "average":
            raise ValueError(
                "return_residuals is only supported for fid_type='average'."
            )
        if fid_type == "average":
            return average_gate_infidelity(
                self._system, self._protocol, x,
                return_residuals=return_residuals,
            )
        elif fid_type == "sss":
            return sss_infidelity(self._system, self._protocol, x)
        elif fid_type == "bell":
            return bell_infidelity(self._system, self._protocol, x)
        raise ValueError(
            f"Unknown fid_type: '{fid_type}'. Choose 'average', 'sss', or 'bell'."
        )

    def gate_fidelity(
        self,
        x: list[float] | None = None,
        fid_type: Literal["average", "sss", "bell"] = "average",
        use_jax_mc: bool = False,
    ) -> "float | tuple[float, float]":
        """Calculate average gate infidelity."""
        x = self._resolve_params(x, "gate_fidelity")
        if self.enable_rydberg_dephasing or self.enable_position_error:
            if use_jax_mc:
                result = run_monte_carlo_jax(
                    self._system, self._protocol, x,
                    n_shots=self.n_mc_shots,
                    sigma_detuning=self.sigma_detuning,
                    sigma_pos_xyz=self.sigma_pos_xyz,
                    seed=self.mc_seed if self.mc_seed is not None else 0,
                    enable_rydberg_dephasing=self.enable_rydberg_dephasing,
                    enable_position_error=self.enable_position_error,
                )
            else:
                engine = MonteCarloEngine(
                    self._system, self._protocol,
                    enable_rydberg_dephasing=self.enable_rydberg_dephasing,
                    enable_position_error=self.enable_position_error,
                    sigma_detuning=self.sigma_detuning,
                    sigma_pos_xyz=self.sigma_pos_xyz,
                )
                result = engine.run(
                    x,
                    n_shots=self.n_mc_shots,
                    sigma_detuning=self.sigma_detuning,
                    sigma_pos_xyz=self.sigma_pos_xyz,
                    seed=self.mc_seed,
                )
            return (result.mean_infidelity, result.std_infidelity)
        return self._gate_infidelity_single(x, fid_type)

    def error_budget(
        self,
        x: list[float] | None = None,
        initial_states: list[str] | None = None,
    ) -> dict:
        """Compute error budget with XYZ/AL/LG breakdown."""
        x = self._resolve_params(x, "error_budget")
        return _error_budget(self._system, self._protocol, x, initial_states)

    def _population_evolution(
        self, x: list[float], initial_state: str,
    ) -> "dict[str, NDArray[np.floating]]":
        return _population_evolution(self._system, self._protocol, x, initial_state)

    def diagnose_plot(
        self,
        x: list[float] | None = None,
        initial_state: InitialStateLabel | None = None,
    ) -> None:
        """Generate population evolution plot."""
        if initial_state is None:
            raise ValueError("initial_state is required.")
        x = self._resolve_params(x, "diagnose_plot")
        population_evo = self.diagnose_run(x, initial_state)

        params = self._protocol.unpack_params(x, self._system)
        t_gate = params["t_gate"]
        time_axis_ns = np.linspace(0, t_gate * 1e9, len(population_evo[0]))

        plt.style.use("seaborn-v0_8-whitegrid")
        _fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_axis_ns, population_evo[0],
                label=f"Intermediate states population for |{initial_state}⟩ state", lw=2)
        ax.plot(time_axis_ns, population_evo[1],
                label="Rydberg state |r⟩ population", linestyle="--", lw=2)
        ax.plot(time_axis_ns, population_evo[2],
                label="Unwanted Rydberg |r'⟩ population", linestyle=":", lw=2)
        ax.set_title(f"Population Evolution During CPHASE Gate ({self.strategy})", fontsize=16)
        ax.set_xlabel("Time (ns)", fontsize=12)
        ax.set_ylabel("Population Probability", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"population_{self.strategy}_{initial_state}.png")
        plt.show()

    def diagnose_run(
        self,
        x: list[float] | None = None,
        initial_state: InitialStateLabel | None = None,
    ) -> "list[NDArray[np.floating]]":
        """Run diagnostic simulation and return population arrays."""
        if initial_state is None:
            raise ValueError("initial_state is required.")
        x = self._resolve_params(x, "diagnose_run")
        pops = _population_evolution(self._system, self._protocol, x, initial_state)
        mid = pops["e1"] + pops["e2"] + pops["e3"]
        return [mid, pops["ryd"], pops["ryd_garb"]]

    def plot_bloch(self, x: list[float] | None = None, save: bool = True) -> None:
        """Bloch sphere visualization (TO only)."""
        x = self._resolve_params(x, "plot_bloch")
        if self.strategy == "TO":
            return self._plotBloch_TO(x, save)
        else:
            print("Bloch sphere plot is only implemented for the 'TO' strategy.")
            return

    def state_infidelity(
        self,
        initial_state: "InitialStateLabel | NDArray[np.complexfloating]",
        x: list[float] | None = None,
    ) -> float:
        """Compute state infidelity for a specific initial state."""
        x = self._resolve_params(x, "state_infidelity")
        return _state_infidelity(self._system, self._protocol, x, initial_state)

    def get_gate_result(
        self,
        state_mat: "NDArray[np.complexfloating]",
        x: list[float] | None = None,
        t_eval: "NDArray[np.floating] | None" = None,
    ) -> "NDArray[np.complexfloating]":
        """Evolve a quantum state under the configured pulse protocol."""
        x = self._resolve_params(x, "get_gate_result")
        return solve_gate(self._system, self._protocol, x, state_mat, t_eval)

    # ==================================================================
    # MONTE CARLO
    # ==================================================================

    def run_monte_carlo_simulation(
        self,
        x: list[float],
        n_shots: int = 1000,
        sigma_detuning: float | None = None,
        sigma_pos_xyz: tuple[float, float, float] | None = None,
        seed: int | None = None,
        compute_branching: bool = False,
    ) -> MonteCarloResult:
        """Run quasi-static Monte Carlo simulation."""
        engine = MonteCarloEngine(self._system, self._protocol, x)
        if self.enable_rydberg_dephasing:
            sigma_det = sigma_detuning if sigma_detuning is not None else self.sigma_detuning
            if sigma_det is not None:
                engine.setup_detuning_noise(sigma_det)
        if self.enable_position_error:
            sigma_pos = sigma_pos_xyz if sigma_pos_xyz is not None else self.sigma_pos_xyz
            if sigma_pos is not None:
                engine.setup_position_noise(sigma_pos)
        return engine.run_gate_fidelity(
            n_shots=n_shots,
            seed=seed,
            compute_branching=compute_branching,
        )

    def run_monte_carlo_jax(
        self,
        x: list[float],
        n_shots: int = 1000,
        sigma_detuning: float | None = None,
        sigma_pos_xyz: tuple[float, float, float] | None = None,
        seed: int = 0,
    ) -> MonteCarloResult:
        """GPU-accelerated Monte Carlo (TO only)."""
        return run_monte_carlo_jax(
            self._system, self._protocol, x,
            n_shots=n_shots,
            sigma_detuning=sigma_detuning,
            sigma_pos_xyz=sigma_pos_xyz,
            seed=seed,
            enable_rydberg_dephasing=self.enable_rydberg_dephasing,
            enable_position_error=self.enable_position_error,
        )

    # ==================================================================
    # BACKWARD-COMPATIBLE PRIVATE METHODS
    # ==================================================================

    def _get_gate_result_TO(
        self, phase_amp, omega, phase_init, delta, t_gate,
        state_mat, t_eval=None,
    ):
        """Backward-compatible wrapper for TO solver."""
        x = [
            phase_amp,
            omega / self.rabi_eff,
            phase_init,
            delta / self.rabi_eff,
            0.0,  # theta placeholder
            t_gate / self.time_scale,
        ]
        return solve_gate(self._system, TOProtocol(), x, state_mat, t_eval)

    def _get_gate_result_AR(
        self, omega, phase_amp1, phase_init1, phase_amp2, phase_init2,
        delta, t_gate, state_mat, t_eval=None,
    ):
        """Backward-compatible wrapper for AR solver."""
        x = [
            omega / self.rabi_eff,
            phase_amp1, phase_init1,
            phase_amp2, phase_init2,
            delta / self.rabi_eff,
            t_gate / self.time_scale,
            0.0,  # theta placeholder
        ]
        return solve_gate(self._system, ARProtocol(), x, state_mat, t_eval)

    def _fidelity_sss(self, x):
        return sss_infidelity(self._system, self._protocol, x)

    def _fidelity_bell(self, x):
        return bell_infidelity(self._system, self._protocol, x)

    def _fidelity_avg(self, x, return_residuals=False):
        return average_gate_infidelity(
            self._system, self._protocol, x, return_residuals=return_residuals,
        )

    def _avg_fidelity_AR(self, x, return_residuals=False):
        return average_gate_infidelity(
            self._system, self._protocol, x, return_residuals=return_residuals,
        )

    def _residuals_to_branching(self, residuals):
        return _residuals_to_branching(self._system, residuals)

    @staticmethod
    def _build_sss_state_map():
        return build_sss_state_map()

    def _occ_operator(self, index):
        return build_occ_operator(index)

    def _build_vdw_unit_operator(self):
        return build_vdw_unit_operator()

    def _get_nominal_distance(self):
        return get_nominal_distance(self._system.param_set)

    def _decay_integrate(self, t_list, occ_list, decay_rate):
        from ryd_gate.analysis.gate_metrics import decay_integrate
        return decay_integrate(t_list, occ_list, decay_rate)

    # ==================================================================
    # OPTIMIZATION
    # ==================================================================

    def _optimization_TO(self, fid_type, x=None):
        if x is None:
            x = self.x_initial
        if fid_type == "average":
            raw_objective = self._fidelity_avg
        if fid_type == "bell":
            raw_objective = self._fidelity_bell
        if fid_type == "sss":
            raw_objective = self._fidelity_sss

        cache = {}

        def objective(x):
            val = raw_objective(x)
            cache['last_val'] = val
            return val

        def callback_func(x, saveflag=False):
            if saveflag:
                with open("opt_hf_new.txt", "a") as f:
                    for var in x:
                        f.write("{:.9f},".format(var))
                    f.write("\n")
            print("Current iteration parameters:", x, "Infidelity:", cache.get('last_val', '?'))

        bounds = TOProtocol().get_optimization_bounds()
        optimres = minimize(
            fun=objective, x0=x, method="Nelder-Mead",
            options={"disp": True, "fatol": 1e-9},
            bounds=bounds, callback=callback_func,
        )
        print(f"The final optimized protocol is: {optimres.x.tolist()}, with infidelity: {optimres.fun}")
        return optimres

    def _optimization_AR(self, fid_type="average", x=None):
        if x is None:
            x = self.x_initial
        else:
            print(f"AR parameters is overwritten to: [ω/Ω_eff, A₁, φ₁, A₂, φ₂, δ/Ω_eff, T/T_scale, θ] = {x}")

        cache = {}

        def objective(x):
            val = self._gate_infidelity_single(x, fid_type=fid_type)
            cache['last_val'] = val
            return val

        def callback_func(x, saveflag=False):
            if saveflag:
                with open("opt_hf_new.txt", "a") as f:
                    for var in x:
                        f.write("{:.9f},".format(var))
                    f.write("\n")
            print("Current iteration parameters:", x, "Infidelity:", cache.get('last_val', '?'))
            print(f"overwrite protocol from {self.x_initial} to {x}")
            self._setup_protocol_AR(x)

        bounds = ARProtocol().get_optimization_bounds()
        optimres = minimize(
            fun=objective, x0=x, method="Nelder-Mead",
            options={"disp": True, "fatol": 1e-9},
            bounds=bounds, callback=callback_func,
        )
        print(f"The final optimized protocol is: {optimres.x.tolist()}, with infidelity: {optimres.fun}")
        return optimres

    # ==================================================================
    # BLOCH SPHERE (TO only)
    # ==================================================================

    def _plotBloch_TO(self, x, saveflag=True):
        basis_01 = np.kron([1 + 0j, 0, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0])
        basis_0r = np.kron([1 + 0j, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1 + 0j, 0])
        basis_01_mat = np.reshape(basis_01, (-1, 1))
        basis_0r_mat = np.reshape(basis_0r, (-1, 1))

        sigmaz = basis_01_mat.dot(basis_01_mat.T) - basis_0r_mat.dot(basis_0r_mat.T)
        sigmax = basis_0r_mat.dot(basis_01_mat.T) + basis_01_mat.dot(basis_0r_mat.T)
        sigmay = -1j * basis_01_mat.dot(basis_0r_mat.T) + 1j * basis_0r_mat.dot(basis_01_mat.T)

        params = self._protocol.unpack_params(x, self._system)
        t_gate = params["t_gate"]
        t_eval = np.linspace(0, t_gate, 1000)

        # First Bloch sphere: |01⟩ ↔ |0r⟩
        res = solve_gate(self._system, self._protocol, x, basis_01, t_eval=t_eval)
        zlist, xlist, ylist = [], [], []
        for t in range(len(res[0, :])):
            state = res[:, t]
            zlist.append(np.matmul(state.reshape(-1).conj(), sigmaz).dot(state.reshape(-1, 1))[0])
            xlist.append(np.matmul(state.reshape(-1).conj(), sigmax).dot(state.reshape(-1, 1))[0])
            ylist.append(np.matmul(state.reshape(-1).conj(), sigmay).dot(state.reshape(-1, 1))[0])

        b = Bloch()
        b.zlabel = [r"$ |01 \rangle$ ", r"$|0r\rangle$"]
        b.point_color = ["#CC6600"]
        b.point_size = [5]
        b.point_marker = ["^"]
        b.add_points([np.array(xlist), np.array(ylist), np.array(zlist)])
        b.make_sphere()

        # Second Bloch sphere: |11⟩ ↔ |W⟩
        basis_11 = np.kron([0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0])
        basis_W = np.sqrt(1 / 2) * (
            np.kron([0, 1 + 0j, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1 + 0j, 0])
            + np.kron([0, 0, 0, 0, 0, 1 + 0j, 0], [0, 1 + 0j, 0, 0, 0, 0, 0])
        )
        basis_11_mat = np.reshape(basis_11, (-1, 1))
        basis_W_mat = np.reshape(basis_W, (-1, 1))

        sigmaz = basis_11_mat.dot(basis_11_mat.T) - basis_W_mat.dot(basis_W_mat.T)
        sigmax = basis_W_mat.dot(basis_11_mat.T) + basis_11_mat.dot(basis_W_mat.T)
        sigmay = -1j * basis_11_mat.dot(basis_W_mat.T) + 1j * basis_W_mat.dot(basis_11_mat.T)

        res = solve_gate(self._system, self._protocol, x, basis_11, t_eval=t_eval)
        zlist, xlist, ylist = [], [], []
        for t in range(len(res[0, :])):
            state = res[:, t]
            zlist.append(np.matmul(state.reshape(-1).conj(), sigmaz).dot(state.reshape(-1, 1))[0])
            xlist.append(np.matmul(state.reshape(-1).conj(), sigmax).dot(state.reshape(-1, 1))[0])
            ylist.append(np.matmul(state.reshape(-1).conj(), sigmay).dot(state.reshape(-1, 1))[0])

        b2 = Bloch()
        b2.zlabel = [r"$ |11 \rangle$ ", r"$|W\rangle$"]
        b2.point_color = ["r"]
        b2.point_size = [5]
        b2.point_marker = ["^"]
        b2.add_points([np.array(xlist), np.array(ylist), np.array(zlist)])
        b2.make_sphere()

        if saveflag:
            b.save("10-r0_Bloch")
            b2.save("11-W_Bloch")
        plt.show()
