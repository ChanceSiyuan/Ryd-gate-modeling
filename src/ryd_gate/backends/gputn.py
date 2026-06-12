"""GPU tensor-network backend: CuPy/cuTensorNet kernels for ``backend="gputn"``.

The default production path uses cuQuantum's experimental ``NetworkState``
MPS interface when available.  A CuPy state-vector fallback is kept for small
lattices and for dependency-injected tests; it is exact but scales as
``local_dim ** n_sites``.

CUDA libraries are never imported at module import time: the CPU TeNPy path
remains usable on machines without a GPU, while ``backend="gputn"`` fails
early with a clear dependency/device error (:class:`GPUTNDependencyError`).
The :class:`GPUTNTDVPBackend` adapter is wired into
:func:`tn_common.simulate_tn`, and :class:`GPUTNOptions` is the typed options
dataclass for that dispatch path.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import expm

from ryd_gate.backends.tn_common.protocol_context import pin_deltas_from_params
from ryd_gate.core.level_structures import (
    three_level_profiles_from_coeffs,
    two_level_drive_and_detuning_from_coeffs,
)
from ryd_gate.ir import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.backends.tn_common.lattice_spec import TNLatticeSpec
    from ryd_gate.protocols.base import Protocol


class GPUTNKernelError(RuntimeError):
    """Raised when the local GPU TN kernel cannot run the requested problem."""


@dataclass
class CuTensorNetRydbergEngine:
    """Trotter gate-evolution engine for Rydberg lattice TN IR.

    Parameters
    ----------
    kernel
        ``"auto"`` selects cuTensorNet MPS when the installed cuQuantum Python
        package exposes ``NetworkState``; otherwise it falls back to the exact
        state-vector path.  ``"cutensornet_mps"`` and ``"statevector"`` force a
        specific path.
    trotter_order
        First- or second-order onsite/interaction splitting.
    statevector_max_sites
        Safety cap for exact state-vector fallback.  Set to ``None`` to
        disable the cap.
    return_state_vector
        For cuTensorNet MPS runs, try to materialize the final dense state only
        when explicitly requested.  Large 2D lattices should keep this false.
    """

    kernel: str = "auto"
    trotter_order: int = 2
    statevector_max_sites: int | None = 24
    return_state_vector: bool = False

    def evolve(
        self,
        spec,
        protocol,
        params: dict,
        psi0: object,
        *,
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
        chi_max: int = 256,
        dt: float = 0.2,
        svd_min: float = 1e-10,
        device_id: int | None = 0,
        xp=None,
        cuquantum=None,
        **_: Any,
    ) -> EvolutionResult:
        if xp is None:
            raise GPUTNKernelError("CuTensorNetRydbergEngine requires an array module via xp=.")
        if cuquantum is None:
            raise GPUTNKernelError("CuTensorNetRydbergEngine requires a cuQuantum module.")
        if spec.level_structure not in {"1r", "01r"}:
            raise ValueError("GPUTN kernel supports TN level_structure '1r' and '01r' only.")
        if self.trotter_order not in {1, 2}:
            raise ValueError("trotter_order must be 1 or 2.")

        selected_kernel = self._select_kernel(cuquantum)
        auto_kernel = self.kernel.strip().lower() == "auto"
        if selected_kernel == "cutensornet_mps":
            try:
                runner = _CuTensorNetMPSRunner(
                    spec=spec,
                    protocol=protocol,
                    params=params,
                    psi0=psi0,
                    xp=xp,
                    cuquantum=cuquantum,
                    chi_max=chi_max,
                    dt=dt,
                    svd_min=svd_min,
                    device_id=device_id,
                    trotter_order=self.trotter_order,
                    return_state_vector=self.return_state_vector,
                )
            except Exception as exc:
                if not auto_kernel or not _is_cutensornet_not_supported(exc):
                    raise
                runner = _StateVectorRunner(
                    spec=spec,
                    protocol=protocol,
                    params=params,
                    psi0=psi0,
                    xp=xp,
                    chi_max=chi_max,
                    dt=dt,
                    svd_min=svd_min,
                    trotter_order=self.trotter_order,
                    statevector_max_sites=self.statevector_max_sites,
                )
        else:
            runner = _StateVectorRunner(
                spec=spec,
                protocol=protocol,
                params=params,
                psi0=psi0,
                xp=xp,
                chi_max=chi_max,
                dt=dt,
                svd_min=svd_min,
                trotter_order=self.trotter_order,
                statevector_max_sites=self.statevector_max_sites,
            )

        result = runner.run(t_eval=t_eval, observables=observables)
        result.metadata.setdefault("backend", "gputn")
        result.metadata.setdefault("accelerator", "cuda")
        result.metadata.setdefault("kernel", runner.kernel_name)
        result.metadata.setdefault("method", "gputn_tebd")
        result.metadata.setdefault("chi_max", chi_max)
        result.metadata.setdefault("svd_min", svd_min)
        return result

    def _select_kernel(self, cuquantum) -> str:
        key = self.kernel.strip().lower()
        aliases = {
            "mps": "cutensornet_mps",
            "cutensornet": "cutensornet_mps",
            "network_state": "cutensornet_mps",
            "networkstate": "cutensornet_mps",
            "sv": "statevector",
            "exact": "statevector",
        }
        key = aliases.get(key, key)
        if key == "auto":
            return "cutensornet_mps" if _network_state_api_available(cuquantum) else "statevector"
        if key not in {"cutensornet_mps", "statevector"}:
            raise ValueError("kernel must be 'auto', 'cutensornet_mps', or 'statevector'.")
        if key == "cutensornet_mps" and not _network_state_api_available(cuquantum):
            raise GPUTNKernelError(
                "kernel='cutensornet_mps' requires cuQuantum Python with "
                "cuquantum.tensornet.experimental.NetworkState."
            )
        return key


class _StateVectorRunner:
    kernel_name = "statevector_trotter"

    def __init__(
        self,
        *,
        spec,
        protocol,
        params: dict,
        psi0: object,
        xp,
        chi_max: int,
        dt: float,
        svd_min: float,
        trotter_order: int,
        statevector_max_sites: int | None,
    ) -> None:
        del chi_max, svd_min
        self.spec = spec
        self.protocol = protocol
        self.params = params
        self.xp = xp
        self.dt = float(dt)
        self.trotter_order = int(trotter_order)
        self.levels = tuple(spec.level_spec.levels)
        self.local_dim = int(spec.level_spec.local_dim)
        self.r_index = self.levels.index("r")
        if statevector_max_sites is not None and spec.N > int(statevector_max_sites):
            raise GPUTNKernelError(
                f"statevector kernel is capped at {statevector_max_sites} sites; "
                "use kernel='cutensornet_mps' for larger lattices."
            )
        self.psi = _product_or_dense_state_tensor(spec, psi0, xp, dtype=np.complex128)
        self._gate_cache: dict[float, list] = {}

    def run(self, *, t_eval: np.ndarray | None, observables: list[str] | None) -> EvolutionResult:
        if observables is None and t_eval is not None:
            observables = ["m_s", "n_mean"]
        t_gate = float(self.params["t_gate"])
        n_steps = max(1, int(np.ceil(t_gate / self.dt)))
        dt_actual = t_gate / n_steps
        record_at = _record_steps(t_eval, dt_actual, n_steps)
        obs_data = _init_obs_data(observables)
        recorded_times: list[float] = []

        if 0 in record_at and observables:
            self._record_observables(observables, obs_data, recorded_times, 0.0)

        for step in range(n_steps):
            t_mid = (step + 0.5) * dt_actual
            onsite_h = _onsite_hamiltonians(self.spec, self.protocol, self.params, t_mid)
            onsite_gates = [_to_xp(expm(-1j * dt_actual * h_i), self.xp) for h_i in onsite_h]
            if self.trotter_order == 2:
                self._apply_interactions(0.5 * dt_actual)
                self._apply_onsite(onsite_gates)
                self._apply_interactions(0.5 * dt_actual)
            else:
                self._apply_onsite(onsite_gates)
                self._apply_interactions(dt_actual)

            step_num = step + 1
            if step_num in record_at and observables:
                self._record_observables(
                    observables,
                    obs_data,
                    recorded_times,
                    step_num * dt_actual,
                )

        result = EvolutionResult(
            psi_final=self.psi,
            metadata={
                "method": "gputn_tebd",
                "backend": "gputn",
                "kernel": self.kernel_name,
                "dt": dt_actual,
                "n_steps": n_steps,
                "obs": _finalize_obs_data(obs_data),
            },
        )
        if recorded_times:
            result.times = np.asarray(recorded_times, dtype=float)
        return result

    def _apply_onsite(self, gates: list[Any]) -> None:
        for site, gate in enumerate(gates):
            self.psi = _apply_single_site_gate(self.psi, gate, site, self.xp)

    def _apply_interactions(self, dt_piece: float) -> None:
        for i, j, gate in self._interaction_gates(dt_piece):
            self.psi = _apply_two_site_gate(self.psi, gate, i, j, self.xp)

    def _interaction_gates(self, dt_piece: float) -> list:
        """Constant two-body phase gates for a Trotter slice, built once per dt_piece."""
        gates = self._gate_cache.get(dt_piece)
        if gates is None:
            gates = [
                (
                    int(i),
                    int(j),
                    _two_body_rydberg_phase_gate(
                        self.local_dim, self.r_index,
                        float(self.spec.V_nn) * float(v_rel), dt_piece, self.xp,
                    ),
                )
                for i, j, v_rel in self.spec.vdw_pairs
            ]
            self._gate_cache[dt_piece] = gates
        return gates

    def _record_observables(
        self,
        observables: list[str],
        obs_data: dict[str, list],
        recorded_times: list[float],
        t_value: float,
    ) -> None:
        recorded_times.append(float(t_value))
        occ_r = None
        for name in observables:
            if name in {"n_i", "n_r", "n_mean", "sigma_z", "z_i", "m_s", "czz_centerline"}:
                occ_r = self._rydberg_occupations() if occ_r is None else occ_r
            if name == "m_s":
                obs_data[name].append(float(np.sum(self.spec.sublattice * (2.0 * occ_r - 1.0)) / self.spec.N))
            elif name == "n_mean":
                obs_data[name].append(float(np.mean(occ_r)))
            elif name in {"n_i", "n_r"}:
                obs_data[name].append(occ_r.copy())
            elif name in {"sigma_z", "z_i"}:
                obs_data[name].append(2.0 * occ_r - 1.0)
            elif name == "czz_centerline":
                obs_data[name].append(self._centerline_connected_zz(occ_r))
            elif name in {"n_0", "n_1"}:
                obs_data[name].append(self._level_occupations(name[-1]))

    def _probabilities(self):
        return self.xp.abs(self.psi) ** 2

    def _rydberg_occupations(self) -> np.ndarray:
        return self._level_occupations("r")

    def _level_occupations(self, level: str) -> np.ndarray:
        if level not in self.levels:
            return np.zeros(self.spec.N, dtype=float)
        level_index = self.levels.index(level)
        probs = self._probabilities()
        occ = np.empty(self.spec.N, dtype=float)
        for site in range(self.spec.N):
            idx = [slice(None)] * self.spec.N
            idx[site] = level_index
            occ[site] = float(_as_numpy(probs[tuple(idx)].sum(), self.xp).real)
        return occ

    def _pair_rydberg(self, i: int, j: int) -> float:
        probs = self._probabilities()
        idx = [slice(None)] * self.spec.N
        idx[int(i)] = self.r_index
        idx[int(j)] = self.r_index
        return float(_as_numpy(probs[tuple(idx)].sum(), self.xp).real)

    def _centerline_connected_zz(self, occ_r: np.ndarray) -> np.ndarray:
        from ryd_gate.analysis.observables import line_pairs_from_reference

        values = []
        for i, j in line_pairs_from_reference(self.spec.Lx, self.spec.Ly, axis="horizontal"):
            nn = self._pair_rydberg(i, j)
            values.append(4.0 * (nn - occ_r[int(i)] * occ_r[int(j)]))
        return np.asarray(values, dtype=float)


class _CuTensorNetMPSRunner:
    kernel_name = "cutensornet_mps_trotter"

    def __init__(
        self,
        *,
        spec,
        protocol,
        params: dict,
        psi0: object,
        xp,
        cuquantum,
        chi_max: int,
        dt: float,
        svd_min: float,
        device_id: int | None,
        trotter_order: int,
        return_state_vector: bool,
    ) -> None:
        self.spec = spec
        self.protocol = protocol
        self.params = params
        self.psi0 = psi0
        self.xp = xp
        self.cuquantum = cuquantum
        self.chi_max = int(chi_max)
        self.dt = float(dt)
        self.svd_min = float(svd_min)
        self.device_id = device_id
        self.trotter_order = int(trotter_order)
        self.return_state_vector = bool(return_state_vector)
        self.levels = tuple(spec.level_spec.levels)
        self.local_dim = int(spec.level_spec.local_dim)
        self.r_index = self.levels.index("r")
        self._initial_labels = _state_labels_2d(self.spec, self.psi0)
        self._operators_applied = False
        self._api = _network_state_api(cuquantum)
        self.state = self._build_state()
        self._gate_cache: dict[float, list] = {}

    def run(self, *, t_eval: np.ndarray | None, observables: list[str] | None) -> EvolutionResult:
        if observables is None and t_eval is not None:
            observables = ["m_s", "n_mean"]
        t_gate = float(self.params["t_gate"])
        n_steps = max(1, int(np.ceil(t_gate / self.dt)))
        dt_actual = t_gate / n_steps
        record_at = _record_steps(t_eval, dt_actual, n_steps)
        obs_data = _init_obs_data(observables)
        recorded_times: list[float] = []

        if 0 in record_at and observables:
            self._record_observables(observables, obs_data, recorded_times, 0.0)

        for step in range(n_steps):
            t_mid = (step + 0.5) * dt_actual
            onsite_h = _onsite_hamiltonians(self.spec, self.protocol, self.params, t_mid)
            onsite_gates = [_to_xp(expm(-1j * dt_actual * h_i), self.xp) for h_i in onsite_h]
            if self.trotter_order == 2:
                self._apply_interactions(0.5 * dt_actual)
                self._apply_onsite(onsite_gates)
                self._apply_interactions(0.5 * dt_actual)
            else:
                self._apply_onsite(onsite_gates)
                self._apply_interactions(dt_actual)

            step_num = step + 1
            if step_num in record_at and observables:
                self._record_observables(
                    observables,
                    obs_data,
                    recorded_times,
                    step_num * dt_actual,
                )

        result = EvolutionResult(
            psi_final=self._final_state(),
            metadata={
                "method": "gputn_tebd",
                "backend": "gputn",
                "kernel": self.kernel_name,
                "dt": dt_actual,
                "n_steps": n_steps,
                "obs": _finalize_obs_data(obs_data),
            },
        )
        if recorded_times:
            result.times = np.asarray(recorded_times, dtype=float)
        return result

    def _build_state(self):
        NetworkState = self._api.NetworkState
        extents = (self.local_dim,) * self.spec.N
        config = self._mps_config()
        options = self._options()
        dtype = "complex128"
        try:
            state = NetworkState(extents, dtype=dtype, config=config, options=options)
        except TypeError:
            try:
                state = NetworkState(extents, dtype=dtype, config=config)
            except TypeError:
                state = NetworkState(extents, dtype=dtype)

        self.state = state
        labels = _state_labels_2d(self.spec, self.psi0)
        for site, label in enumerate(labels):
            target_index = self.levels.index(label)
            if target_index == 0:
                continue
            gate = _basis_preparation_gate(self.local_dim, target_index, self.xp)
            self._apply_tensor_operator((site,), gate)
        return self.state

    def _mps_config(self):
        MPSConfig = self._api.MPSConfig
        if MPSConfig is None:
            return None
        attempts = (
            {"max_extent": self.chi_max, "abs_cutoff": self.svd_min},
            {"max_extent": self.chi_max},
            {},
        )
        for kwargs in attempts:
            try:
                return MPSConfig(**kwargs)
            except TypeError:
                continue
        return None

    def _options(self):
        if self.device_id is None:
            return None
        options_cls = getattr(self._api, "NetworkOptions", None)
        if options_cls is None:
            return None
        try:
            return options_cls(device_id=int(self.device_id))
        except TypeError:
            return None

    def _apply_onsite(self, gates: list[Any]) -> None:
        for site, gate in enumerate(gates):
            self._apply_tensor_operator((site,), gate)

    def _apply_interactions(self, dt_piece: float) -> None:
        for i, j, gate in self._interaction_gates(dt_piece):
            self._apply_tensor_operator((i, j), gate)

    def _interaction_gates(self, dt_piece: float) -> list:
        """Constant rank-4 two-body phase gates for a Trotter slice, built once per dt_piece."""
        gates = self._gate_cache.get(dt_piece)
        if gates is None:
            d = self.local_dim
            gates = [
                (
                    int(i),
                    int(j),
                    _two_body_rydberg_phase_gate(
                        d, self.r_index, float(self.spec.V_nn) * float(v_rel), dt_piece, self.xp,
                    ).reshape((d,) * 4),
                )
                for i, j, v_rel in self.spec.vdw_pairs
            ]
            self._gate_cache[dt_piece] = gates
        return gates

    def _apply_tensor_operator(self, modes: tuple[int, ...], tensor) -> None:
        try:
            self.state.apply_tensor_operator(modes, tensor, unitary=True)
        except TypeError:
            self.state.apply_tensor_operator(modes, tensor)
        self._operators_applied = True

    def _record_observables(
        self,
        observables: list[str],
        obs_data: dict[str, list],
        recorded_times: list[float],
        t_value: float,
    ) -> None:
        recorded_times.append(float(t_value))
        occ_r = None
        for name in observables:
            if name in {"n_i", "n_r", "n_mean", "sigma_z", "z_i", "m_s", "czz_centerline"}:
                occ_r = self._rydberg_occupations() if occ_r is None else occ_r
            if name == "m_s":
                obs_data[name].append(float(np.sum(self.spec.sublattice * (2.0 * occ_r - 1.0)) / self.spec.N))
            elif name == "n_mean":
                obs_data[name].append(float(np.mean(occ_r)))
            elif name in {"n_i", "n_r"}:
                obs_data[name].append(occ_r.copy())
            elif name in {"sigma_z", "z_i"}:
                obs_data[name].append(2.0 * occ_r - 1.0)
            elif name == "czz_centerline":
                obs_data[name].append(self._centerline_connected_zz(occ_r))
            elif name in {"n_0", "n_1"}:
                obs_data[name].append(self._level_occupations(name[-1]))

    def _rydberg_occupations(self) -> np.ndarray:
        return self._level_occupations("r")

    def _level_occupations(self, level: str) -> np.ndarray:
        if level not in self.levels:
            return np.zeros(self.spec.N, dtype=float)
        if not self._operators_applied:
            return np.asarray([1.0 if label == level else 0.0 for label in self._initial_labels], dtype=float)
        level_index = self.levels.index(level)
        occ = np.empty(self.spec.N, dtype=float)
        for site in range(self.spec.N):
            rdm = _as_numpy(self._site_rdm(site), self.xp)
            occ[site] = float(np.real(rdm[level_index, level_index]))
        return occ

    def _centerline_connected_zz(self, occ_r: np.ndarray) -> np.ndarray:
        from ryd_gate.analysis.observables import line_pairs_from_reference

        pairs = line_pairs_from_reference(self.spec.Lx, self.spec.Ly, axis="horizontal")
        if not self._operators_applied:
            return np.zeros(len(pairs), dtype=float)

        values = []
        for i, j in pairs:
            rdm = _as_numpy(self._pair_rdm(int(i), int(j)), self.xp)
            rdm_mat = rdm.reshape(self.local_dim * self.local_dim, self.local_dim * self.local_dim)
            rr_index = self.r_index * self.local_dim + self.r_index
            nn = float(np.real(rdm_mat[rr_index, rr_index]))
            values.append(4.0 * (nn - occ_r[int(i)] * occ_r[int(j)]))
        return np.asarray(values, dtype=float)

    def _site_rdm(self, site: int):
        return self._compute_reduced_density_matrix((int(site),))

    def _pair_rdm(self, i: int, j: int):
        return self._compute_reduced_density_matrix((int(i), int(j)))

    def _compute_reduced_density_matrix(self, modes: tuple[int, ...]):
        for name in ("compute_reduced_density_matrix", "reduced_density_matrix"):
            method = getattr(self.state, name, None)
            if method is None:
                continue
            try:
                return _first_result(method(modes))
            except TypeError:
                return _first_result(method(list(modes)))
        raise GPUTNKernelError(
            "Installed cuQuantum NetworkState does not expose a reduced-density-matrix API."
        )

    def _final_state(self):
        if not self.return_state_vector:
            return self.state
        for name in ("compute_state_vector", "compute_output_state", "get_state_vector"):
            method = getattr(self.state, name, None)
            if method is not None:
                return method()
        return self.state


def _onsite_hamiltonians(spec, protocol, params: dict, t_mid: float) -> list[np.ndarray]:
    coeffs = protocol.get_drive_coefficients(t_mid, params)
    pin = pin_deltas_from_params(params, spec.N)
    levels = tuple(spec.level_spec.levels)
    dim = int(spec.level_spec.local_dim)

    if spec.level_structure == "1r":
        omega_t, delta_t, channel_pin = two_level_drive_and_detuning_from_coeffs(coeffs, spec)
        omega_profile = _as_profile(omega_t, spec.N)
        detuning_profile = np.full(spec.N, float(delta_t), dtype=float)
        if channel_pin is not None:
            detuning_profile += np.asarray(channel_pin, dtype=float)
        if pin is not None:
            detuning_profile += pin

        idx_1 = levels.index("1")
        idx_r = levels.index("r")
        hamiltonians = []
        for site in range(spec.N):
            h = np.zeros((dim, dim), dtype=complex)
            h[idx_1, idx_r] += 0.5 * float(omega_profile[site])
            h[idx_r, idx_1] += 0.5 * float(omega_profile[site])
            h[idx_r, idx_r] += -float(detuning_profile[site])
            hamiltonians.append(h)
        return hamiltonians

    profiles = three_level_profiles_from_coeffs(coeffs, spec)
    if pin is not None:
        profiles["delta_R"] = profiles["delta_R"] + pin
    idx = {level: levels.index(level) for level in levels}
    hamiltonians = []
    for site in range(spec.N):
        h = np.zeros((dim, dim), dtype=complex)
        if "1" in idx and "r" in idx:
            value = 0.5 * float(profiles["omega_R"][site])
            h[idx["1"], idx["r"]] += value
            h[idx["r"], idx["1"]] += value
        if "0" in idx and "1" in idx:
            value = 0.5 * float(profiles["omega_hf"][site])
            h[idx["0"], idx["1"]] += value
            h[idx["1"], idx["0"]] += value
        if "r" in idx:
            h[idx["r"], idx["r"]] += -float(profiles["delta_R"][site])
        if "1" in idx:
            h[idx["1"], idx["1"]] += -float(profiles["delta_hf"][site])
        hamiltonians.append(h)
    return hamiltonians


def _as_profile(value, n_sites: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_sites, float(arr), dtype=float)
    if arr.shape != (n_sites,):
        raise ValueError(f"Profile must be scalar or length-{n_sites}; got {arr.shape}.")
    return arr.astype(float, copy=False)


def _product_or_dense_state_tensor(spec, psi0: object, xp, *, dtype) -> Any:
    if isinstance(psi0, str) and psi0 == "plus":
        from ryd_gate.core.states import plus_local_amplitudes, product_superposition_state

        dim = int(spec.level_spec.local_dim)
        dense = product_superposition_state(plus_local_amplitudes(spec.level_spec.levels), spec.N)
        return _to_xp(dense.astype(dtype, copy=False).reshape((dim,) * spec.N), xp)

    labels = None
    if isinstance(psi0, str):
        labels = _state_labels_2d(spec, psi0)
    elif isinstance(psi0, (list, tuple)):
        labels = _state_labels_2d(spec, psi0)
    else:
        arr = np.asarray(psi0)
        if _looks_like_product_config(arr, spec.N):
            labels = _state_labels_2d(spec, arr)
        else:
            dim = int(spec.level_spec.local_dim)
            if arr.size != dim ** spec.N:
                raise ValueError(
                    f"Dense psi0 must have {dim ** spec.N} amplitudes for local_dim={dim}, "
                    f"N={spec.N}; got {arr.size}."
                )
            tensor = arr.astype(dtype, copy=False).reshape((dim,) * spec.N)
            norm = np.linalg.norm(tensor.reshape(-1))
            if norm == 0:
                raise ValueError("Initial state cannot have zero norm.")
            return _to_xp(tensor / norm, xp)

    dim = int(spec.level_spec.local_dim)
    levels = tuple(spec.level_spec.levels)
    tensor = np.zeros((dim,) * spec.N, dtype=dtype)
    index = tuple(levels.index(label) for label in labels)
    tensor[index] = 1.0
    return _to_xp(tensor, xp)


def _looks_like_product_config(arr: np.ndarray, n_sites: int) -> bool:
    if arr.shape != (n_sites,):
        return False
    if arr.dtype.kind in {"U", "S", "O", "b", "i", "u"}:
        return True
    if arr.dtype.kind == "f" and np.all(np.isin(arr, [0.0, 1.0])):
        return True
    return False


def _state_labels_2d(spec, config: np.ndarray | list | tuple | str) -> list[str]:
    if isinstance(config, str):
        return _named_state_labels_2d(spec, config)

    arr = np.asarray(config)
    if arr.shape != (spec.N,):
        raise ValueError(f"config must have shape ({spec.N},), got {arr.shape}.")
    if arr.dtype.kind in {"U", "S", "O"}:
        labels = [str(x) for x in arr]
        _validate_level_labels(spec, labels)
        return labels
    occ = arr.astype(int)
    labels = ["r" if int(c) == 1 else "1" for c in occ]
    _validate_level_labels(spec, labels)
    return labels


def _named_state_labels_2d(spec, name: str) -> list[str]:
    if name in {"all_ground", "all_1"}:
        labels = ["1"] * spec.N
    elif name in {"all_0", "all_zero"}:
        labels = ["0"] * spec.N
    elif name == "all_r":
        labels = ["r"] * spec.N
    elif name == "af1":
        labels = ["r" if s > 0 else "1" for s in spec.sublattice]
    elif name == "af2":
        labels = ["r" if s < 0 else "1" for s in spec.sublattice]
    else:
        raise ValueError(f"Unknown config string: {name!r}")
    _validate_level_labels(spec, labels)
    return labels


def _validate_level_labels(spec, labels: list[str]) -> None:
    allowed = set(spec.level_spec.levels)
    unknown = sorted(set(labels) - allowed)
    if unknown:
        raise ValueError(f"Unknown level label(s) for {spec.level_structure}: {unknown}.")


def _basis_preparation_gate(local_dim: int, target_index: int, xp):
    gate = np.eye(local_dim, dtype=complex)
    gate[[0, target_index], :] = gate[[target_index, 0], :]
    return _to_xp(gate, xp)


def _two_body_rydberg_phase_gate(local_dim: int, r_index: int, strength: float, dt_piece: float, xp):
    gate = np.eye(local_dim * local_dim, dtype=complex)
    rr_index = r_index * local_dim + r_index
    gate[rr_index, rr_index] = np.exp(-1j * float(strength) * float(dt_piece))
    return _to_xp(gate, xp)


def _apply_single_site_gate(psi, gate, site: int, xp):
    psi_perm = xp.moveaxis(psi, int(site), 0)
    dim = psi_perm.shape[0]
    rest_shape = psi_perm.shape[1:]
    updated = xp.tensordot(gate, psi_perm.reshape(dim, -1), axes=([1], [0]))
    updated = updated.reshape((dim,) + rest_shape)
    return xp.moveaxis(updated, 0, int(site))


def _apply_two_site_gate(psi, gate, i: int, j: int, xp):
    i = int(i)
    j = int(j)
    if i == j:
        raise ValueError("Two-site gate requires distinct sites.")
    psi_perm = xp.moveaxis(psi, (i, j), (0, 1))
    dim = psi_perm.shape[0]
    rest_shape = psi_perm.shape[2:]
    gate_matrix = gate.reshape(dim * dim, dim * dim)
    updated = xp.tensordot(gate_matrix, psi_perm.reshape(dim * dim, -1), axes=([1], [0]))
    updated = updated.reshape((dim, dim) + rest_shape)
    return xp.moveaxis(updated, (0, 1), (i, j))


def _record_steps(t_eval: np.ndarray | None, dt_actual: float, n_steps: int) -> set[int]:
    if t_eval is None:
        return set()
    record_at = set()
    for t_req in np.asarray(t_eval, dtype=float):
        step = int(round(float(t_req) / dt_actual))
        step = max(0, min(step, n_steps))
        record_at.add(step)
    return record_at


def _init_obs_data(observables: list[str] | None) -> dict[str, list]:
    return {name: [] for name in (observables or [])}


def _finalize_obs_data(obs_data: dict[str, list]) -> dict[str, np.ndarray]:
    return {name: np.asarray(values) for name, values in obs_data.items()}


def _to_xp(arr, xp):
    if hasattr(xp, "asarray"):
        return xp.asarray(arr)
    return np.asarray(arr)


def _as_numpy(arr, xp) -> np.ndarray:
    if hasattr(xp, "asnumpy"):
        return xp.asnumpy(arr)
    return np.asarray(arr)


def _first_result(value):
    if isinstance(value, tuple):
        return value[0]
    return value


def _network_state_api_available(cuquantum) -> bool:
    try:
        _network_state_api(cuquantum)
    except GPUTNKernelError:
        return False
    return True


def _is_cutensornet_not_supported(exc: Exception) -> bool:
    return exc.__class__.__name__ == "cuTensorNetError" and "NOT_SUPPORTED" in str(exc)


def _network_state_api(cuquantum):
    for attr in ("tensornet", "cutensornet"):
        namespace = getattr(cuquantum, attr, None)
        experimental = getattr(namespace, "experimental", None)
        NetworkState = getattr(experimental, "NetworkState", None)
        if NetworkState is not None:
            return _NetworkStateAPI(
                NetworkState=NetworkState,
                MPSConfig=getattr(experimental, "MPSConfig", None),
                NetworkOptions=getattr(experimental, "NetworkOptions", None),
            )

    candidates = (
        "cuquantum.tensornet.experimental",
        "cuquantum.cutensornet.experimental",
    )
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        NetworkState = getattr(module, "NetworkState", None)
        if NetworkState is None:
            continue
        return _NetworkStateAPI(
            NetworkState=NetworkState,
            MPSConfig=getattr(module, "MPSConfig", None),
            NetworkOptions=getattr(module, "NetworkOptions", None),
        )

    raise GPUTNKernelError("cuQuantum NetworkState API is unavailable.")


@dataclass(frozen=True)
class _NetworkStateAPI:
    NetworkState: Any
    MPSConfig: Any | None
    NetworkOptions: Any | None


# ── Backend entry points (no CUDA imports at module import time) ────────────


class GPUTNDependencyError(ImportError):
    """Raised when the GPU TN backend dependencies or CUDA device are unavailable."""


def _missing_dependency_message(missing: list[str]) -> str:
    missing_text = ", ".join(missing)
    return (
        f"GPUTN backend requires CUDA tensor-network dependencies: {missing_text}. "
        "For CUDA 12, install the optional extra with "
        "`pip install -e '.[gputn-cu12]'` or install NVIDIA cuQuantum Python "
        "manually for your CUDA version. The CPU TeNPy backend remains available "
        "with `backend='mps'`."
    )


def _require_gputn_dependencies(
    *,
    require_gpu: bool = True,
    device_id: int | None = 0,
) -> SimpleNamespace:
    """Return GPU dependency modules after checking importability and device access."""
    missing = [
        module_name
        for module_name in ("cupy", "cuquantum")
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        raise GPUTNDependencyError(_missing_dependency_message(missing))

    import cupy
    import cuquantum

    n_devices: int | None = None
    if require_gpu:
        try:
            n_devices = int(cupy.cuda.runtime.getDeviceCount())
        except Exception as exc:  # pragma: no cover - depends on local CUDA runtime
            raise GPUTNDependencyError(
                "GPUTN backend found CuPy/cuQuantum but could not query CUDA devices. "
                "Check the NVIDIA driver, CUDA runtime libraries, and LD_LIBRARY_PATH."
            ) from exc
        if n_devices <= 0:
            raise GPUTNDependencyError("GPUTN backend requires at least one CUDA device.")
        if device_id is not None and not 0 <= int(device_id) < n_devices:
            raise GPUTNDependencyError(
                f"GPUTN backend requested CUDA device {device_id}, "
                f"but only {n_devices} device(s) are visible."
            )

    return SimpleNamespace(cupy=cupy, cuquantum=cuquantum, n_devices=n_devices)


def gputn_available(*, require_gpu: bool = True, device_id: int | None = 0) -> bool:
    """Return whether the optional GPU TN dependencies and device are available."""
    try:
        _require_gputn_dependencies(require_gpu=require_gpu, device_id=device_id)
    except GPUTNDependencyError:
        return False
    return True


@dataclass
class GPUTNTDVPBackend:
    """GPU tensor-network lattice-evolution backend adapter.

    The public adapter is wired into :func:`tn_common.simulate_tn` so callers
    can switch from ``backend="mps"`` to ``backend="gputn"`` without changing
    protocol/notebook code.  When ``engine`` is omitted, the built-in
    :class:`CuTensorNetRydbergEngine` runs a cuTensorNet MPS Trotter kernel
    when available, with a small-system CuPy state-vector fallback.  Custom
    engines can still be injected by exposing an ``evolve(...)`` method with
    the signature below.
    """

    chi_max: int = 256
    dt: float = 0.2
    svd_min: float = 1e-10
    device_id: int | None = 0
    require_gpu: bool = True
    engine: object | None = None
    kernel: str = "auto"
    trotter_order: int = 2
    statevector_max_sites: int | None = 24
    return_state_vector: bool = False

    def evolve_ir(
        self,
        ir,
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        """Evolve a compiled TN IR with the configured GPU TN engine."""
        return self.evolve_compiled(
            ir.spec,
            ir.protocol,
            ir.params,
            initial_state,
            t_eval=t_eval,
            observables=observables,
        )

    def evolve_compiled(
        self,
        spec: TNLatticeSpec,
        protocol: Protocol,
        params: dict,
        psi0: object,
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        """Evolve with already-unpacked protocol parameters."""
        deps = _require_gputn_dependencies(
            require_gpu=self.require_gpu,
            device_id=self.device_id,
        )
        engine = self.engine
        if engine is None:
            engine = CuTensorNetRydbergEngine(
                kernel=self.kernel,
                trotter_order=self.trotter_order,
                statevector_max_sites=self.statevector_max_sites,
                return_state_vector=self.return_state_vector,
            )

        result = engine.evolve(
            spec,
            protocol,
            params,
            psi0,
            t_eval=t_eval,
            observables=observables,
            chi_max=self.chi_max,
            dt=self.dt,
            svd_min=self.svd_min,
            device_id=self.device_id,
            xp=deps.cupy,
            cuquantum=deps.cuquantum,
        )
        result.metadata.setdefault("backend", "gputn")
        result.metadata.setdefault("accelerator", "cuda")
        if self.engine is None:
            result.metadata.setdefault("engine_package", "gputn")
        return result


@dataclass(frozen=True)
class GPUTNOptions:
    """Options for ``backend="gputn"`` TDVP evolution.

    ``None`` means "use the backend default".
    """

    chi_max: int | None = None
    dt: float | None = None
    svd_min: float | None = None
    device_id: int | None = None
    require_gpu: bool | None = None
    kernel: str | None = None
    trotter_order: int | None = None
    statevector_max_sites: int | None = None
    return_state_vector: bool | None = None
