"""DMRG and TDVP backend wrappers for TeNPy."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.backends.base import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.protocols.base import Protocol

    from .lattice_spec import TNLatticeSpec


class _TNProtocolContext:
    def __init__(self, spec: TNLatticeSpec) -> None:
        self._spec = spec
        self.basis = SimpleNamespace(n_sites=spec.N)

    @property
    def N(self) -> int:
        return self._spec.N

    def meta(self, name: str, default=None):
        if name == "Omega":
            return self._spec.Omega
        if name == "n_sites":
            return self._spec.N
        return default


def _pin_deltas_from_params(params: dict, n_sites: int) -> np.ndarray | None:
    pin_map = params.get("pin_deltas") or {}
    if not pin_map:
        return None
    pin = np.zeros(n_sites)
    for idx, value in pin_map.items():
        if idx < n_sites:
            pin[idx] = value
    return pin


def _has_channel(coeffs: dict[str, complex], channel: str | None, n_sites: int) -> bool:
    if channel is None:
        return False
    if channel in coeffs:
        return True
    return any(f"{channel}_{i}" in coeffs for i in range(n_sites))


def _declared_channels(spec: TNLatticeSpec) -> set[str]:
    channels = {transition.channel for transition in spec.level_spec.transitions}
    channels.update(spec.level_spec.detuning_levels)
    return channels


def _validate_coeff_channels(coeffs: dict[str, complex], spec: TNLatticeSpec) -> None:
    declared = _declared_channels(spec)
    unknown = []
    for channel in coeffs:
        base = channel.rsplit("_", 1)[0] if channel.rsplit("_", 1)[-1].isdigit() else channel
        if base not in declared:
            unknown.append(channel)
    if unknown:
        raise ValueError(
            f"Protocol emitted channel(s) not declared by level_structure "
            f"{spec.level_structure!r}: {sorted(unknown)}."
        )


def _site_profile_from_coeffs(
    coeffs: dict[str, complex],
    channel: str,
    n_sites: int,
    scale: float,
) -> np.ndarray | None:
    keys = [f"{channel}_{i}" for i in range(n_sites)]
    if not any(key in coeffs for key in keys):
        return None
    return np.array([scale * float(np.real(coeffs.get(key, 0.0))) for key in keys])


def _channel_profile_from_coeffs(
    coeffs: dict[str, complex],
    channel: str,
    n_sites: int,
    scale: float,
) -> np.ndarray:
    site_profile = _site_profile_from_coeffs(coeffs, channel, n_sites, scale=scale)
    if site_profile is not None:
        return site_profile
    return np.full(n_sites, scale * float(np.real(coeffs.get(channel, 0.0))))


def _merge_pin_deltas(*profiles: np.ndarray | None, n_sites: int) -> np.ndarray | None:
    merged = np.zeros(n_sites)
    any_profile = False
    for profile in profiles:
        if profile is None:
            continue
        merged += profile
        any_profile = True
    return merged if any_profile else None


def _transition_channel(spec: TNLatticeSpec, lower: str, upper: str) -> str | None:
    for transition in spec.level_spec.transitions:
        if transition.lower == lower and transition.upper == upper:
            return transition.channel
    return None


def _detuning_channel(spec: TNLatticeSpec, level: str) -> str | None:
    for channel, detuned_level in spec.level_spec.detuning_levels.items():
        if detuned_level == level:
            return channel
    return None


def _profile_for_optional_channel(
    coeffs: dict[str, complex],
    channel: str | None,
    n_sites: int,
    scale: float,
) -> np.ndarray:
    if channel is None:
        return np.zeros(n_sites)
    return _channel_profile_from_coeffs(coeffs, channel, n_sites, scale=scale)


def _three_level_profiles_from_coeffs(
    coeffs: dict[str, complex],
    spec: TNLatticeSpec,
) -> dict[str, np.ndarray]:
    """Map protocol coefficients to explicit 01r per-site profiles."""
    _validate_coeff_channels(coeffs, spec)
    drive_r = _transition_channel(spec, "1", "r")
    drive_hf = _transition_channel(spec, "0", "1")
    delta_r = _detuning_channel(spec, "r")
    delta_hf = _detuning_channel(spec, "1")

    return {
        "omega_R": _profile_for_optional_channel(coeffs, drive_r, spec.N, scale=2.0),
        "omega_hf": _profile_for_optional_channel(coeffs, drive_hf, spec.N, scale=2.0),
        "delta_R": _profile_for_optional_channel(coeffs, delta_r, spec.N, scale=-1.0),
        "delta_hf": _profile_for_optional_channel(coeffs, delta_hf, spec.N, scale=-1.0),
    }


def _split_uniform_profile(profile: np.ndarray) -> tuple[float, np.ndarray | None]:
    if np.allclose(profile, profile[0]):
        return float(profile[0]), None
    return 0.0, profile


def _two_level_drive_and_detuning_from_coeffs(
    coeffs: dict[str, complex],
    spec: TNLatticeSpec,
) -> tuple[float | np.ndarray, float, np.ndarray | None]:
    """Map protocol channel coefficients onto the TN 1/r Hamiltonian.

    ``SweepProtocol`` already emits 2-level channels ``global_X`` and
    ``global_n``.  ``DigitalAnalogProtocol`` emits 01r channels; for the TN
    2-level subspace we identify ``|g>`` with ``|1>`` and keep only the
    ``|1><->|r>`` drive.  The hyperfine drive is outside that subspace, while
    hyperfine detuning contributes an effective Rydberg detuning because
    ``n_1 = I - n_r``.
    """
    profiles = _three_level_profiles_from_coeffs(coeffs, spec)

    if np.any(np.abs(profiles["omega_hf"]) > 0):
        raise ValueError(
            "TN TDVP supports the |1>-|r> two-level subspace only; "
            "DigitalAnalogProtocol segments with omega_hf != 0 are not supported."
        )

    omega_profile = profiles["omega_R"]
    if np.allclose(omega_profile, omega_profile[0]):
        Omega_t: float | np.ndarray = float(omega_profile[0])
    else:
        Omega_t = omega_profile

    # 01r -> effective 1r mapping:
    # -Delta_R n_r - Delta_hf n_1 = const - (Delta_R - Delta_hf) n_r.
    Delta_t, delta_profile = _split_uniform_profile(
        profiles["delta_R"] - profiles["delta_hf"]
    )
    return Omega_t, Delta_t, delta_profile


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
        protocol: Protocol,
        x: list[float],
        psi0: object,
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        """Evolve MPS under a TN lattice protocol.

        Uses piecewise-constant Hamiltonian updates: at each TDVP step
        the protocol coefficients are evaluated at the midpoint and the
        MPO is rebuilt.

        Parameters
        ----------
        spec : TNLatticeSpec
        protocol : Protocol
            Supports ``SweepProtocol`` plus ``DigitalAnalogProtocol`` on
            either the effective ``1r`` TN subspace or explicit ``01r``
            TN local levels.
        x : list
            Protocol parameter vector. ``DigitalAnalogProtocol`` expects
            ``[]`` because its schedule is stored on the protocol.
        psi0 : tenpy.networks.mps.MPS
            Initial state.
        t_eval : ndarray or None
            Times at which to record observables. If None, only
            the final state is returned.
        observables : list of str or None
            Observable names to stream: ``"m_s"``, ``"n_mean"``,
            ``"n_i"``/``"n_r"`` (per-site Rydberg occupations),
            ``"n_0"``, and ``"n_1"``. If None, stores ``"m_s"`` and
            ``"n_mean"`` when ``t_eval`` is given.

        Returns
        -------
        EvolutionResult
            ``psi_final`` is the final MPS. ``metadata["obs"]`` contains
            time-series of requested observables.
        """
        params = protocol.unpack_params(x, _TNProtocolContext(spec))
        return self.evolve_compiled(
            spec,
            protocol,
            params,
            psi0,
            t_eval=t_eval,
            observables=observables,
        )

    def evolve_ir(
        self,
        ir,
        psi0: object,
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        """Evolve a compiled TN IR."""
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
        from .observables import (
            measure_level_occupations,
            measure_mean_rydberg,
            measure_site_occupations,
            measure_staggered_magnetization,
        )

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
                if name == "m_s":
                    obs_data[name].append(
                        measure_staggered_magnetization(psi, spec))
                elif name == "n_mean":
                    obs_data[name].append(
                        measure_mean_rydberg(psi, spec))
                elif name == "n_i":
                    obs_data[name].append(
                        measure_site_occupations(psi, spec).copy())
                elif name in {"n_0", "n_1", "n_r"}:
                    obs_data[name].append(
                        measure_level_occupations(psi, spec, name[-1]).copy())

        if 0 in record_at and observables:
            record_observables(0.0)

        for k in range(n_steps):
            t_mid = (k + 0.5) * dt_actual
            coeffs = protocol.get_drive_coefficients(t_mid, params)
            if spec.level_structure == "01r":
                profiles = _three_level_profiles_from_coeffs(coeffs, spec)
                pin = _pin_deltas_from_params(params, spec.N)
                if pin is not None:
                    profiles["delta_R"] = profiles["delta_R"] + pin
                model = build_tenpy_model(spec, Delta=0.0, **profiles)
            else:
                Omega_t, Delta_t, channel_pin_deltas = (
                    _two_level_drive_and_detuning_from_coeffs(coeffs, spec)
                )

                pin_deltas = _merge_pin_deltas(
                    _pin_deltas_from_params(params, spec.N),
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

        result = EvolutionResult(
            psi_final=psi,
            metadata={
                **(metadata or {}),
                "method": "tdvp",
                "backend": "tenpy",
                "chi_max": self.chi_max,
                "dt": dt_actual,
                "n_steps": n_steps,
                "obs": obs_data,
            },
        )
        if recorded_times:
            result.times = np.array(recorded_times)

        return result
