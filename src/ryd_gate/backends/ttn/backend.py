"""PyTreeNet TTN-TDVP backend for lattice Rydberg dynamics.

This backend vendors PyTreeNet as the concrete Python TTN kernel for
``backend="ttn"``.  PyTreeNet 1.0.0 is NumPy/SciPy based, so this adapter is a
CPU implementation.  It deliberately rejects CUDA/GPU options instead of
silently pretending to use an accelerator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ryd_gate._vendor import import_pytreenet
from ryd_gate.backends.tenpy_mps.backends import _merge_pin_deltas, _pin_deltas_from_params
from ryd_gate.core.channel_lowering import (
    three_level_profiles_from_coeffs,
    two_level_drive_and_detuning_from_coeffs,
)
from ryd_gate.ir.evolution import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.backends.tn_common.compiler import TNEvolutionIR
    from ryd_gate.backends.tn_common.lattice_spec import TNLatticeSpec


@dataclass(frozen=True)
class _BinaryNode:
    identifier: str
    children: tuple["_BinaryNode", ...] = ()
    site_index: int | None = None

    @property
    def is_leaf(self) -> bool:
        return self.site_index is not None


class PyTreeNetTTNTDVPBackend:
    """One-site binary-tree TDVP backend using vendored PyTreeNet.

    Parameters
    ----------
    chi_max
        Fixed virtual bond dimension of the TTNS. One-site TDVP keeps this
        bond dimension fixed; it is not a dynamic truncation cap.
    dt
        TDVP step size in the same dimensionless time units as the TN IR.
    tdvp_order
        ``1`` uses PyTreeNet ``FirstOrderOneSiteTDVP``; ``2`` uses
        ``SecondOrderOneSiteTDVP``.
    initial_noise
        Optional small complex noise injected into otherwise zero virtual
        sectors. This can make fixed-chi one-site TDVP less singular when
        ``chi_max > 1`` while keeping the product state dominant.
    seed
        Random seed for ``initial_noise``.
    """

    def __init__(
        self,
        chi_max: int = 8,
        dt: float = 0.05,
        tdvp_order: int = 1,
        initial_noise: float = 0.0,
        seed: int | None = 0,
        progress: bool = False,
        use_gpu: bool = False,
        use_cuda: bool = False,
        device: str | None = None,
        **options: Any,
    ) -> None:
        if chi_max < 1:
            raise ValueError("chi_max must be positive.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if tdvp_order not in {1, 2}:
            raise ValueError("PyTreeNet TTN supports tdvp_order=1 or 2.")
        if use_gpu or use_cuda or (device is not None and str(device).lower() != "cpu"):
            raise ValueError(
                "backend='ttn' uses vendored PyTreeNet, which is NumPy/SciPy "
                "CPU-only in this repo. Use backend='2dtn', 'gputn', or 'nqs' "
                "for GPU-capable simulations."
            )
        self.chi_max = int(chi_max)
        self.dt = float(dt)
        self.tdvp_order = int(tdvp_order)
        self.initial_noise = float(initial_noise)
        self.seed = seed
        self.progress = bool(progress)
        self.options = dict(options)

    def evolve_ir(
        self,
        ir: "TNEvolutionIR",
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        """Evolve a TN IR with PyTreeNet one-site TDVP."""
        spec = ir.spec
        if spec.bc != "open":
            raise ValueError("PyTreeNet TTN backend currently supports open boundaries only.")

        ptn = import_pytreenet()
        tree = _build_balanced_tree(spec.N)
        site_ids = [f"site_{i}" for i in range(spec.N)]
        state_indices = _state_indices_2d(spec, initial_state)
        psi = _product_ttns(
            ptn,
            tree,
            state_indices,
            local_dim=spec.level_spec.local_dim,
            bond_dim=self.chi_max,
            initial_noise=self.initial_noise,
            seed=self.seed,
        )
        _normalize_ttn(psi)

        t_gate = float(ir.params["t_gate"])
        n_steps = int(np.ceil(t_gate / self.dt))
        if n_steps < 1:
            n_steps = 1
        dt_actual = t_gate / n_steps

        record_at = _record_steps(t_eval, dt_actual, n_steps)
        if observables is None and t_eval is not None:
            observables = ["m_s", "n_mean"]
        obs_data = {name: [] for name in observables or []}
        recorded_times: list[float] = []

        def record(t_value: float) -> None:
            if not observables:
                return
            recorded_times.append(float(t_value))
            measured = _measure_observables(ptn, psi, spec, site_ids, observables)
            for name, value in measured.items():
                obs_data[name].append(value)

        if 0 in record_at:
            record(0.0)

        algo = None
        static_ttno = None
        if _protocol_coefficients_static(ir.protocol, ir.params, t_gate):
            static_ttno = _build_ttno_for_time(ptn, spec, ir.protocol, ir.params, site_ids, 0.5 * dt_actual)
            if static_ttno is not None:
                algo = _tdvp_algorithm(
                    ptn,
                    psi,
                    static_ttno,
                    dt_actual,
                    order=self.tdvp_order,
                    progress=self.progress,
                )

        for k in range(n_steps):
            if static_ttno is None:
                ttno = _build_ttno_for_time(
                    ptn,
                    spec,
                    ir.protocol,
                    ir.params,
                    site_ids,
                    (k + 0.5) * dt_actual,
                )
                if ttno is not None:
                    algo = _tdvp_algorithm(
                        ptn,
                        psi,
                        ttno,
                        dt_actual,
                        order=self.tdvp_order,
                        progress=self.progress,
                    )
                    algo.run_one_time_step()
                    psi = algo.state
            elif algo is not None:
                algo.run_one_time_step()
                psi = algo.state

            step_num = k + 1
            if step_num in record_at:
                record(step_num * dt_actual)

        for name in obs_data:
            obs_data[name] = np.asarray(obs_data[name])

        max_bond_dim = 1
        if spec.N > 1:
            max_bond_dim = int(psi.max_bond_dim())

        result = EvolutionResult(
            psi_final=psi,
            metadata={
                **(ir.metadata or {}),
                "backend": "ttn",
                "engine_package": "pytreenet",
                "engine_source": "vendored",
                "method": "ttn_tdvp",
                "tdvp_order": self.tdvp_order,
                "tree": "balanced_binary",
                "gpu": False,
                "chi_max": self.chi_max,
                "chi": max_bond_dim,
                "dt": dt_actual,
                "n_steps": n_steps,
                "initial_noise": self.initial_noise,
                "obs": obs_data,
            },
        )
        if recorded_times:
            result.times = np.asarray(recorded_times)
        return result


def _build_balanced_tree(n_sites: int) -> _BinaryNode:
    counter = 0

    def build(indices: Sequence[int]) -> _BinaryNode:
        nonlocal counter
        if len(indices) == 1:
            site = int(indices[0])
            return _BinaryNode(f"site_{site}", site_index=site)
        mid = len(indices) // 2
        left = build(indices[:mid])
        right = build(indices[mid:])
        identifier = f"node_{counter}"
        counter += 1
        return _BinaryNode(identifier, children=(left, right))

    if n_sites < 1:
        raise ValueError("TTN needs at least one site.")
    return build(tuple(range(n_sites)))


def _product_ttns(
    ptn,
    root: _BinaryNode,
    state_indices: Sequence[int],
    *,
    local_dim: int,
    bond_dim: int,
    initial_noise: float,
    seed: int | None,
):
    rng = np.random.default_rng(seed)
    ttn = ptn.TreeTensorNetworkState()

    def node_tensor(node: _BinaryNode, is_root: bool) -> np.ndarray:
        phys_dim = local_dim if node.is_leaf else 1
        n_virtual = len(node.children) + (0 if is_root else 1)
        virtual_shape = [bond_dim] * n_virtual
        shape = tuple(virtual_shape + [phys_dim])
        tensor = np.zeros(shape, dtype=complex)
        if initial_noise > 0 and bond_dim > 1:
            noise = rng.normal(size=shape) + 1j * rng.normal(size=shape)
            tensor += initial_noise * noise
        physical_index = int(state_indices[node.site_index]) if node.is_leaf else 0
        tensor[(0,) * n_virtual + (physical_index,)] += 1.0
        return tensor

    def add_rec(node: _BinaryNode, parent_id: str | None = None, parent_leg: int | None = None) -> None:
        tensor = node_tensor(node, is_root=parent_id is None)
        ptn_node = ptn.Node(identifier=node.identifier)
        if parent_id is None:
            ttn.add_root(ptn_node, tensor)
        else:
            ttn.add_child_to_parent(ptn_node, tensor, 0, parent_id, int(parent_leg))
        for child_index, child in enumerate(node.children):
            child_parent_leg = child_index if parent_id is None else child_index + 1
            add_rec(child, node.identifier, child_parent_leg)

    add_rec(root)
    return ttn


def _normalize_ttn(psi) -> None:
    norm_sq = psi.scalar_product()
    norm = float(np.sqrt(abs(norm_sq)))
    if norm == 0:
        raise ValueError("Initial TTNS has zero norm.")
    root_id = psi.root_id
    psi.tensors[root_id] = psi.tensors[root_id] / norm


def _state_indices_2d(spec: "TNLatticeSpec", config: np.ndarray | Sequence[str] | str) -> list[int]:
    labels = _state_labels_2d(spec, config)
    return [spec.level_spec.index(label) for label in labels]


def _state_labels_2d(spec: "TNLatticeSpec", config: np.ndarray | Sequence[str] | str) -> list[str]:
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
    return ["r" if c == 1 else "1" for c in occ]


def _named_state_labels_2d(spec: "TNLatticeSpec", name: str) -> list[str]:
    if name in {"all_ground", "all_1"}:
        labels = ["1"] * spec.N
    elif name in {"all_0", "all_zero"}:
        if "0" not in spec.level_spec.levels:
            raise ValueError("'all_0' requires a TN lattice spec with a |0> level.")
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


def _validate_level_labels(spec: "TNLatticeSpec", labels: Sequence[str]) -> None:
    allowed = set(spec.level_spec.levels)
    unknown = sorted(set(labels) - allowed)
    if unknown:
        raise ValueError(f"Unknown level label(s) for {spec.level_structure}: {unknown}.")


def _record_steps(t_eval: np.ndarray | None, dt_actual: float, n_steps: int) -> set[int]:
    if t_eval is None:
        return set()
    record_at: set[int] = set()
    for t_req in np.asarray(t_eval, dtype=float):
        step = int(round(float(t_req) / dt_actual))
        record_at.add(max(0, min(step, n_steps)))
    return record_at


def _protocol_coefficients_static(protocol, params: dict, t_gate: float) -> bool:
    if t_gate <= 0:
        return True
    samples = [
        protocol.get_drive_coefficients(frac * t_gate, params)
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    first = samples[0]
    for current in samples[1:]:
        if set(first) != set(current):
            return False
        if not all(np.allclose(first[key], current[key]) for key in first):
            return False
    return True


def _build_ttno_for_time(
    ptn,
    spec: "TNLatticeSpec",
    protocol,
    params: dict,
    site_ids: Sequence[str],
    t_mid: float,
):
    hamiltonian = _build_hamiltonian_for_time(ptn, spec, protocol, params, site_ids, t_mid)
    if len(hamiltonian.terms) == 0:
        return None
    reference_tree = _product_ttns(
        ptn,
        _build_balanced_tree(spec.N),
        [0] * spec.N,
        local_dim=spec.level_spec.local_dim,
        bond_dim=1,
        initial_noise=0.0,
        seed=None,
    )
    return ptn.TreeTensorNetworkOperator.from_hamiltonian(hamiltonian, reference_tree)


def _build_hamiltonian_for_time(
    ptn,
    spec: "TNLatticeSpec",
    protocol,
    params: dict,
    site_ids: Sequence[str],
    t_mid: float,
):
    coeffs = protocol.get_drive_coefficients(float(t_mid), params)
    ops = _local_ops(spec)
    terms = []
    conversion: dict[str, np.ndarray] = {}
    conversion[f"I{spec.level_spec.local_dim}"] = np.eye(spec.level_spec.local_dim, dtype=complex)
    conversion["I1"] = np.eye(1, dtype=complex)

    def label(matrix: np.ndarray) -> str:
        key = f"op_{len(conversion)}"
        conversion[key] = np.asarray(matrix, dtype=complex)
        return key

    def add_term(matrix_by_site: dict[int, np.ndarray]) -> None:
        nonzero = {
            site_ids[int(site)]: label(matrix)
            for site, matrix in matrix_by_site.items()
            if not np.allclose(matrix, 0.0)
        }
        if nonzero:
            terms.append(ptn.TensorProduct(nonzero))

    if spec.level_structure == "01r":
        profiles = three_level_profiles_from_coeffs(coeffs, spec)
        pin = _pin_deltas_from_params(params, spec.N)
        if pin is not None:
            profiles["delta_R"] = profiles["delta_R"] + pin
        for i in range(spec.N):
            add_term({i: 0.5 * profiles["omega_R"][i] * ops["x_1r"]})
            add_term({i: 0.5 * profiles["omega_hf"][i] * ops["x_01"]})
            add_term({i: -profiles["delta_R"][i] * ops["n_r"]})
            add_term({i: -profiles["delta_hf"][i] * ops["n_1"]})
    else:
        omega_t, delta_t, channel_pin_deltas = two_level_drive_and_detuning_from_coeffs(coeffs, spec)
        omega_profile = _as_profile(omega_t, spec.N)
        pin = _merge_pin_deltas(
            _pin_deltas_from_params(params, spec.N),
            channel_pin_deltas,
            n_sites=spec.N,
        )
        delta_profile = np.full(spec.N, float(delta_t), dtype=float)
        if pin is not None:
            delta_profile = delta_profile + pin
        for i in range(spec.N):
            add_term({i: 0.5 * omega_profile[i] * ops["sigma_x"]})
            add_term({i: -delta_profile[i] * ops["n_r"]})

    for i, j, v_rel in spec.vdw_pairs:
        strength = float(spec.V_nn) * float(v_rel)
        add_term({int(i): strength * ops["n_r"], int(j): ops["n_r"]})

    return ptn.Hamiltonian(terms, conversion_dictionary=conversion)


def _as_profile(value: float | np.ndarray, n_sites: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_sites, float(arr))
    if arr.shape != (n_sites,):
        raise ValueError(f"Profile must be scalar or shape ({n_sites},), got {arr.shape}.")
    return arr


def _local_ops(spec: "TNLatticeSpec") -> dict[str, np.ndarray]:
    levels = spec.level_spec.levels
    dim = spec.level_spec.local_dim
    identity = np.eye(dim, dtype=complex)

    def projector(level: str) -> np.ndarray:
        mat = np.zeros((dim, dim), dtype=complex)
        mat[spec.level_spec.index(level), spec.level_spec.index(level)] = 1.0
        return mat

    def x_between(lower: str, upper: str) -> np.ndarray:
        mat = np.zeros((dim, dim), dtype=complex)
        lo = spec.level_spec.index(lower)
        up = spec.level_spec.index(upper)
        mat[lo, up] = 1.0
        mat[up, lo] = 1.0
        return mat

    ops = {"identity": identity, "n_r": projector("r")}
    if spec.level_structure == "1r":
        ops["sigma_x"] = x_between(levels[0], "r")
        ops["sigma_z"] = 2.0 * ops["n_r"] - identity
    else:
        ops["n_0"] = projector("0")
        ops["n_1"] = projector("1")
        ops["x_01"] = x_between("0", "1")
        ops["x_1r"] = x_between("1", "r")
        ops["sigma_z"] = 2.0 * ops["n_r"] - identity
    return ops


def _tdvp_algorithm(ptn, psi, ttno, dt: float, *, order: int, progress: bool):
    config = ptn.TTNTimeEvolutionConfig(record_bond_dim=False)
    cls = ptn.FirstOrderOneSiteTDVP if order == 1 else ptn.SecondOrderOneSiteTDVP
    algo = cls(psi, ttno, dt, dt, {}, config=config)
    algo._progress = progress
    return algo


def _measure_observables(
    ptn,
    psi,
    spec: "TNLatticeSpec",
    site_ids: Sequence[str],
    observables: Sequence[str],
) -> dict[str, Any]:
    ops = _local_ops(spec)
    sigma_z = None
    n_r = None
    measured: dict[str, Any] = {}

    def site_expect(site: int, op: np.ndarray) -> float:
        value = psi.operator_expectation_value(ptn.TensorProduct({site_ids[int(site)]: op}))
        return float(np.real_if_close(value))

    def all_sigma_z() -> np.ndarray:
        nonlocal sigma_z
        if sigma_z is None:
            sigma_z = np.asarray([site_expect(i, ops["sigma_z"]) for i in range(spec.N)])
        return sigma_z

    def all_n_r() -> np.ndarray:
        nonlocal n_r
        if n_r is None:
            n_r = np.asarray([site_expect(i, ops["n_r"]) for i in range(spec.N)])
        return n_r

    for name in observables:
        if name == "m_s":
            measured[name] = float(np.sum(spec.sublattice * all_sigma_z()) / spec.N)
        elif name == "n_mean":
            measured[name] = float(np.mean(all_n_r()))
        elif name in {"n_i", "n_r"}:
            measured[name] = all_n_r().copy()
        elif name in {"sigma_z", "z_i"}:
            measured[name] = all_sigma_z().copy()
        elif name == "czz_centerline":
            measured[name] = _measure_centerline_czz(ptn, psi, spec, site_ids, ops["sigma_z"], all_sigma_z())
        elif name in {"n_0", "n_1"}:
            if name not in ops:
                if name == "n_1":
                    measured[name] = 1.0 - all_n_r()
                else:
                    measured[name] = np.zeros(spec.N)
            else:
                measured[name] = np.asarray([site_expect(i, ops[name]) for i in range(spec.N)])
        else:
            raise ValueError(f"Unknown TTN observable {name!r}.")
    return measured


def _measure_centerline_czz(
    ptn,
    psi,
    spec: "TNLatticeSpec",
    site_ids: Sequence[str],
    sigma_z_op: np.ndarray,
    sigma_z_values: np.ndarray,
) -> np.ndarray:
    from ryd_gate.analysis.spin_observables import line_pairs_from_reference

    values = []
    for i, j in line_pairs_from_reference(spec.Lx, spec.Ly, axis="horizontal"):
        op = ptn.TensorProduct({
            site_ids[int(i)]: sigma_z_op,
            site_ids[int(j)]: sigma_z_op,
        })
        zz = psi.operator_expectation_value(op)
        values.append(float(np.real_if_close(zz)) - sigma_z_values[int(i)] * sigma_z_values[int(j)])
    return np.asarray(values)
