"""DMRG and TDVP backend wrappers for TeNPy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.solvers.base import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.protocols.sweep import SweepProtocol
    from .lattice_spec import TNLatticeSpec


def _require_tenpy():
    try:
        import tenpy
        return tenpy
    except ImportError as exc:
        raise ImportError(
            "TeNPy is required for tensor network simulations. "
            "Install via: pip install physics-tenpy  "
            "or: pip install ryd-gate[tn]"
        ) from exc


class TenpyDMRGBackend:
    """DMRG ground-state solver wrapping TeNPy finite DMRG.

    Parameters
    ----------
    chi_max : int
        Maximum bond dimension.
    svd_min : float
        Truncation cutoff.
    n_sweeps : int
        Maximum number of DMRG sweeps.
    mixer : bool
        Whether to use a density-matrix mixer for convergence.
    """

    def __init__(
        self,
        chi_max: int = 256,
        svd_min: float = 1e-10,
        n_sweeps: int = 20,
        mixer: bool = True,
    ) -> None:
        self.chi_max = chi_max
        self.svd_min = svd_min
        self.n_sweeps = n_sweeps
        self.mixer = mixer

    def find_ground_state(
        self,
        spec: TNLatticeSpec,
        Delta: float,
        pin_deltas: np.ndarray | None = None,
        initial_state: str | np.ndarray = "all_ground",
    ) -> EvolutionResult:
        """Find ground state via finite DMRG.

        Parameters
        ----------
        spec : TNLatticeSpec
        Delta : float
            Global detuning.
        pin_deltas : ndarray or None
            Per-site local detunings.
        initial_state : str or ndarray
            Initial MPS configuration (see :func:`product_state_mps`).

        Returns
        -------
        EvolutionResult
            ``psi_final`` is the TeNPy MPS ground state.
            ``metadata`` contains ``energy``, ``chi``, ``n_sweeps``.
        """
        tenpy = _require_tenpy()
        from tenpy.algorithms.dmrg import TwoSiteDMRGEngine

        from .model import build_tenpy_model
        from .state import product_state_mps

        model = build_tenpy_model(spec, Delta, pin_deltas=pin_deltas)

        if hasattr(initial_state, "expectation_value"):
            psi = initial_state  # already an MPS
        else:
            psi = product_state_mps(spec, initial_state)

        dmrg_params = {
            "trunc_params": {
                "chi_max": self.chi_max,
                "svd_min": self.svd_min,
            },
            "mixer": self.mixer,
            "max_sweeps": self.n_sweeps,
        }

        eng = TwoSiteDMRGEngine(psi, model, dmrg_params)
        E0, _ = eng.run()

        return EvolutionResult(
            psi_final=psi,
            metadata={
                "energy": E0,
                "chi": max(psi.chi),
                "n_sweeps": eng.sweeps,
                "method": "dmrg",
            },
        )


class TenpyTDVPBackend:
    """Two-site TDVP time evolution using TeNPy.

    Parameters
    ----------
    chi_max : int
        Maximum bond dimension (paper uses 1200 for production).
    dt : float
        Time step in natural units (paper uses 0.2 Omega^-1).
    svd_min : float
        Truncation cutoff.
    """

    def __init__(
        self,
        chi_max: int = 256,
        dt: float = 0.2,
        svd_min: float = 1e-10,
    ) -> None:
        self.chi_max = chi_max
        self.dt = dt
        self.svd_min = svd_min

    def evolve(
        self,
        spec: TNLatticeSpec,
        protocol: SweepProtocol,
        x: list[float],
        psi0: object,
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        """Evolve MPS under the sweep/hold protocol.

        Uses piecewise-constant Hamiltonian updates: at each TDVP step
        the protocol coefficients are evaluated at the midpoint and the
        MPO is rebuilt.

        Parameters
        ----------
        spec : TNLatticeSpec
        protocol : SweepProtocol
        x : [delta_start, delta_end, t_sweep]
        psi0 : tenpy.networks.mps.MPS
            Initial state.
        t_eval : ndarray or None
            Times at which to record observables. If None, only
            the final state is returned.
        observables : list of str or None
            Observable names to stream: ``"m_s"``, ``"n_mean"``,
            ``"n_i"`` (per-site occupations). If None, stores
            ``"m_s"`` and ``"n_mean"`` when ``t_eval`` is given.

        Returns
        -------
        EvolutionResult
            ``psi_final`` is the final MPS. ``metadata["obs"]`` contains
            time-series of requested observables.
        """
        tenpy = _require_tenpy()
        from tenpy.algorithms.tdvp import TwoSiteTDVPEngine

        from .model import build_tenpy_model
        from .observables import (
            measure_mean_rydberg,
            measure_site_occupations,
            measure_staggered_magnetization,
        )

        if observables is None and t_eval is not None:
            observables = ["m_s", "n_mean"]

        params = protocol.unpack_params(x, type("_Sys", (), {"Omega": spec.Omega})())
        t_gate = params["t_gate"]
        n_steps = int(np.ceil(t_gate / self.dt))
        dt_actual = t_gate / n_steps

        # Determine which t_eval indices to record
        if t_eval is not None:
            record_at = set()
            for t_req in t_eval:
                step = int(round(t_req / dt_actual))
                step = max(0, min(step, n_steps))
                record_at.add(step)
        else:
            record_at = set()

        # Observable storage
        obs_data: dict[str, list] = {}
        if observables:
            for name in observables:
                obs_data[name] = []
        recorded_times: list[float] = []

        psi = psi0.copy()

        for k in range(n_steps):
            t_mid = (k + 0.5) * dt_actual
            coeffs = protocol.get_drive_coefficients(t_mid, params)
            # Extract Omega(t) and Delta(t) from coefficients
            Omega_t = float(np.real(coeffs.get("global_X", spec.Omega / 2))) * 2
            Delta_t = -float(np.real(coeffs.get("global_n", 0)))

            pin_deltas = protocol.get_pin_deltas(spec.N) if hasattr(protocol, "get_pin_deltas") else None

            model = build_tenpy_model(spec, Delta_t, Omega=Omega_t,
                                      pin_deltas=pin_deltas)

            tdvp_params = {
                "trunc_params": {
                    "chi_max": self.chi_max,
                    "svd_min": self.svd_min,
                },
            }

            eng = TwoSiteTDVPEngine(psi, model, tdvp_params)
            eng.evolve(1, dt_actual)

            # Record observables if this step is requested
            step_num = k + 1
            if step_num in record_at and observables:
                recorded_times.append(step_num * dt_actual)
                for name in observables:
                    if name == "m_s":
                        obs_data[name].append(
                            measure_staggered_magnetization(psi, spec))
                    elif name == "n_mean":
                        obs_data[name].append(
                            measure_mean_rydberg(psi, spec))
                    elif name == "n_i":
                        obs_data[name].append(
                            measure_site_occupations(psi, spec).copy())

        # Convert to arrays
        for name in obs_data:
            obs_data[name] = np.array(obs_data[name])

        result = EvolutionResult(
            psi_final=psi,
            metadata={
                "method": "tdvp",
                "chi_max": self.chi_max,
                "dt": dt_actual,
                "n_steps": n_steps,
                "obs": obs_data,
            },
        )
        if recorded_times:
            result.times = np.array(recorded_times)

        return result
