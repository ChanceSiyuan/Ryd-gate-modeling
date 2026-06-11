"""DMRG and TDVP backend wrappers for TeNPy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.backends.tn_common.lattice_spec import TNLatticeSpec
from ryd_gate.backends.tn_common.protocol_context import (
    merge_pin_deltas,
    pin_deltas_from_params,
)
from ryd_gate.core.channel_lowering import (
    three_level_profiles_from_coeffs,
    two_level_drive_and_detuning_from_coeffs,
)
from ryd_gate.ir.evolution import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.protocols.base import Protocol


def measure_mps_observable(psi_mps: object, spec: TNLatticeSpec, name: str):
    """Measure a TeNPy MPS observable in register/2D site order."""
    from .observables import (
        measure_centerline_connected_zz,
        measure_level_occupations,
        measure_mean_rydberg,
        measure_sigma_z,
        measure_site_occupations,
        measure_staggered_magnetization,
    )

    if name == "m_s":
        return measure_staggered_magnetization(psi_mps, spec)
    if name == "n_mean":
        return measure_mean_rydberg(psi_mps, spec)
    if name in {"n_i", "n_r"}:
        return measure_site_occupations(psi_mps, spec)
    if name in {"sigma_z", "z_i"}:
        return measure_sigma_z(psi_mps, spec)
    if name == "czz_centerline":
        return measure_centerline_connected_zz(psi_mps, spec)
    if name in {"n_0", "n_1"}:
        return measure_level_occupations(psi_mps, spec, name[-1])
    if name == "sum_nr":
        return float(sum(
            float(np.sum(measure_level_occupations(psi_mps, spec, level)))
            for level in spec.level_spec.rydberg_levels
        ))
    if name.startswith("sum_n_"):
        level = name[len("sum_n_"):]
        return float(np.sum(measure_level_occupations(psi_mps, spec, level)))
    if name.startswith("n_"):
        try:
            level, site = name[2:].rsplit("_", 1)
        except ValueError:
            raise ValueError(f"mps.observable_unsupported: {name}") from None
        if site.isdecimal():
            index = int(site)
            if not 0 <= index < spec.N:
                raise ValueError(f"mps.observable_unsupported: {name}")
            return float(measure_level_occupations(psi_mps, spec, level)[index])
    raise ValueError(f"mps.observable_unsupported: {name}")


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
        keep_state: bool = True,
    ) -> None:
        self.chi_max = chi_max
        self.svd_min = svd_min
        self.n_sweeps = n_sweeps
        self.mixer = mixer
        self.keep_state = keep_state

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
        _require_tenpy()
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

        metadata = {
            "backend": "mps",
            "engine_package": "tenpy",
            "energy": E0,
            "chi": max(psi.chi),
            "n_sweeps": eng.sweeps,
            "method": "dmrg",
            "state_handle_kind": "mps" if self.keep_state else "unsupported",
        }
        if self.keep_state:
            metadata["native_state"] = psi
        return EvolutionResult(
            psi_final=psi,
            metadata=metadata,
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
        keep_state: bool = True,
    ) -> None:
        self.chi_max = chi_max
        self.dt = dt
        self.svd_min = svd_min
        self.keep_state = keep_state

    def evolve_ir(
        self,
        ir,
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        """Evolve a compiled TN IR with TeNPy two-site TDVP.

        Uses piecewise-constant Hamiltonian updates: at each TDVP step the protocol
        coefficients are evaluated at the interval midpoint and the MPO is rebuilt.

        Parameters
        ----------
        ir : TNEvolutionIR
            Carries the lattice spec, bound protocol, and unpacked params. The
            protocol supports ``SweepProtocol`` plus ``DigitalAnalogProtocol`` on
            either the effective ``1r`` TN subspace or explicit ``01r`` levels.
        initial_state : str, ndarray, or tenpy MPS
            Initial state; a string/array is built into a product-state MPS via
            :func:`product_state_mps`, while an existing MPS is used as-is.
        t_eval : ndarray or None
            Times at which to record observables. If None, only the final state is
            returned.
        observables : list of str or None
            Observable names to stream: ``"m_s"``, ``"n_mean"``, ``"n_i"``/``"n_r"``
            (per-site Rydberg occupations), ``"sigma_z"``/``"z_i"`` (per-site TFIM
            magnetization), ``"czz_centerline"``, ``"n_0"``, and ``"n_1"``. If None,
            stores ``"m_s"`` and ``"n_mean"`` when ``t_eval`` is given.

        Returns
        -------
        EvolutionResult
            ``psi_final`` is the final MPS. ``metadata["obs"]`` contains time-series
            of requested observables.
        """
        from .state import product_state_mps

        psi0 = (
            initial_state
            if hasattr(initial_state, "expectation_value")
            else product_state_mps(ir.spec, initial_state)
        )
        return self.evolve_compiled(
            ir.spec,
            ir.protocol,
            ir.params,
            psi0,
            t_eval=t_eval,
            observables=observables,
            metadata=ir.metadata,
        )

    def evolve_compiled(
        self,
        spec: TNLatticeSpec,
        protocol: Protocol,
        params: dict,
        psi0: object,
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
        metadata: dict | None = None,
    ) -> EvolutionResult:
        """Evolve MPS with already-unpacked protocol parameters."""
        _require_tenpy()
        from tenpy.algorithms.tdvp import TwoSiteTDVPEngine

        from .model import build_tenpy_model
        if observables is None and t_eval is not None:
            observables = ["m_s", "n_mean"]

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

        def record_observables(t_value: float) -> None:
            recorded_times.append(t_value)
            for name in observables or []:
                try:
                    value = measure_mps_observable(psi, spec, name)
                except ValueError:
                    continue
                if isinstance(value, np.ndarray):
                    value = value.copy()
                obs_data[name].append(value)

        if 0 in record_at and observables:
            record_observables(0.0)

        for k in range(n_steps):
            t_mid = (k + 0.5) * dt_actual
            coeffs = protocol.get_drive_coefficients(t_mid, params)
            if spec.level_structure == "01r":
                profiles = three_level_profiles_from_coeffs(coeffs, spec)
                pin = pin_deltas_from_params(params, spec.N)
                if pin is not None:
                    profiles["delta_R"] = profiles["delta_R"] + pin
                model = build_tenpy_model(spec, Delta=0.0, **profiles)
            else:
                Omega_t, Delta_t, channel_pin_deltas = (
                    two_level_drive_and_detuning_from_coeffs(coeffs, spec)
                )

                pin_deltas = merge_pin_deltas(
                    pin_deltas_from_params(params, spec.N),
                    channel_pin_deltas,
                    n_sites=spec.N,
                )

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
                record_observables(step_num * dt_actual)

        # Convert to arrays
        for name in obs_data:
            obs_data[name] = np.array(obs_data[name])

        metadata_out = {
            **(metadata or {}),
            "method": "mps_tdvp",
            "backend": "mps",
            "engine_package": "tenpy",
            "chi_max": self.chi_max,
            "dt": dt_actual,
            "n_steps": n_steps,
            "obs": obs_data,
            "state_handle_kind": "mps" if self.keep_state else "unsupported",
        }
        if self.keep_state:
            metadata_out["native_state"] = psi
        result = EvolutionResult(
            psi_final=psi,
            metadata=metadata_out,
        )
        if recorded_times:
            result.times = np.array(recorded_times)

        return result
