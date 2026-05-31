"""Universal Rydberg system model.

A :class:`RydbergSystem` is built from three things: a lattice geometry,
a local energy-level structure (``1r`` / ``01r`` / ``1er`` / ``rb87_7`` /
``analog_3``), and a pulse protocol (sweep, CZ gate, digital-analog).
The class owns the time-independent Hamiltonian blocks, the per-site
observables, and the routines that combine them with the protocol into
the time-dependent ``HamiltonianIR`` consumed by ``ryd_gate.solvers``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.sparse import csc_matrix, diags as spdiags

from ryd_gate.core.basis import BasisSpec
from ryd_gate.core.blocks import BlockRegistry
from ryd_gate.core.interactions import vdw_couplings
from ryd_gate.core.observables import ObservableRegistry
from ryd_gate.core.operators import embed_site_op
from ryd_gate.core.system_model import SystemModel
from ryd_gate.lattice.geometry import LatticeGeometry, make_chain

if TYPE_CHECKING:
    from ryd_gate.protocols.base import Protocol
    from ryd_gate.solvers.ir import HamiltonianIR


_HC_CHANNELS = frozenset({"global_X", "drive_R", "drive_hf"})
_STATIC_BLOCKS = ("H_const", "H_vdw", "H_1013", "H_1013_conj")


DEFAULT_C6 = 2 * np.pi * 874e9


@dataclass(frozen=True)
class TransitionSpec:
    """Single-site transition block definition.

    ``operator`` is ``|upper><lower|``.  Protocols decide whether to add
    the Hermitian conjugate through the compiler channel map.
    """

    name: str
    lower: str
    upper: str
    channel: str


@dataclass(frozen=True)
class LevelStructureSpec:
    """Local energy-level structure for every site."""

    name: str
    levels: tuple[str, ...]
    rydberg_levels: tuple[str, ...]
    transitions: tuple[TransitionSpec, ...] = ()
    detuning_levels: dict[str, str] = field(default_factory=dict)

    @property
    def local_dim(self) -> int:
        return len(self.levels)

    def index(self, level: str) -> int:
        try:
            return self.levels.index(level)
        except ValueError:
            raise ValueError(f"Unknown level '{level}' for {self.name}: {self.levels}") from None


@dataclass(frozen=True)
class InteractionSpec:
    """Pairwise Rydberg interaction construction."""

    C6: float = DEFAULT_C6
    max_range_um: float | None = None
    mode: Literal["all", "nn", "nnn"] = "all"


def level_structure(name: str) -> LevelStructureSpec:
    """Return a built-in level-structure preset."""
    presets = {
        "1r": LevelStructureSpec(
            name="1r",
            levels=("1", "r"),
            rydberg_levels=("r",),
            transitions=(TransitionSpec("1_r", "1", "r", "global_X"),),
            detuning_levels={"global_n": "r"},
        ),
        "01r": LevelStructureSpec(
            name="01r",
            levels=("0", "1", "r"),
            rydberg_levels=("r",),
            transitions=(
                TransitionSpec("R", "1", "r", "drive_R"),
                TransitionSpec("hf", "0", "1", "drive_hf"),
            ),
            detuning_levels={"delta_R": "r", "delta_hf": "1"},
        ),
        "1er": LevelStructureSpec(
            name="1er",
            levels=("1", "e", "r"),
            rydberg_levels=("r",),
            transitions=(
                TransitionSpec("420", "1", "e", "drive_420"),
                TransitionSpec("1013", "e", "r", "H_1013"),
            ),
            detuning_levels={"delta_e": "e", "delta_R": "r"},
        ),
        "rb87_7": LevelStructureSpec(
            name="rb87_7",
            levels=("0", "1", "e1", "e2", "e3", "r", "r_garb"),
            rydberg_levels=("r", "r_garb"),
        ),
    }
    try:
        return presets[name]
    except KeyError:
        raise ValueError(f"Unknown level-structure preset '{name}'.") from None


class RydbergSystem(SystemModel):
    """Universal Rydberg-lattice system: geometry + level structure + protocol.

    Built via :meth:`from_lattice` or :meth:`from_preset`. Once a protocol
    is attached (constructor arg or :meth:`with_protocol`), the system can
    emit the time-dependent Hamiltonian description consumed by solvers
    through :meth:`compile_ir` / :meth:`hamiltonian`.
    """

    def __init__(
        self,
        *,
        param_set: str,
        basis: BasisSpec,
        blocks: BlockRegistry,
        observables: ObservableRegistry,
        protocol: "Protocol | None" = None,
        metadata: dict[str, Any] | None = None,
        geometry: LatticeGeometry | None = None,
        is_sparse: bool = True,
        amplitude_scale: float = 1.0,
    ) -> None:
        self._param_set = param_set
        self._basis = basis
        self._blocks = blocks
        self._observables = observables
        self.protocol = protocol
        self.metadata = metadata or {}
        self.geometry = geometry
        self.is_sparse = is_sparse
        self.amplitude_scale = amplitude_scale

    @property
    def basis(self) -> BasisSpec:
        return self._basis

    @property
    def blocks(self) -> BlockRegistry:
        return self._blocks

    @property
    def observables(self) -> ObservableRegistry:
        return self._observables

    @property
    def param_set(self) -> str:
        return self._param_set

    @property
    def system(self) -> "RydbergSystem":
        """Compatibility with dispatch unwrapping: the model is the system."""
        return self

    @property
    def N(self) -> int:
        return self.basis.n_sites

    @property
    def dim(self) -> int:
        return self.basis.total_dim

    def meta(self, name: str, default=None):
        return self.metadata.get(name, default)

    # -- Protocol binding --------------------------------------------------

    def with_protocol(self, protocol: "Protocol") -> "RydbergSystem":
        """Return a shallow copy with ``protocol`` (re)bound."""
        new = copy.copy(self)
        new.protocol = protocol
        return new

    def with_amplitude_scale(self, amplitude_scale: float) -> "RydbergSystem":
        """Return a shallow copy with ``amplitude_scale`` replaced (noise sweeps)."""
        new = copy.copy(self)
        new.amplitude_scale = amplitude_scale
        return new

    def _require_protocol(self) -> "Protocol":
        if self.protocol is None:
            raise ValueError(
                "RydbergSystem has no protocol bound. Construct with "
                "`protocol=...` or call `.with_protocol(...)` before compiling."
            )
        return self.protocol

    # -- Solver-facing API -------------------------------------------------

    def unpack_params(self, x) -> dict:
        """Translate protocol parameter vector ``x`` into a params dict."""
        return self._require_protocol().unpack_params(x, self)

    def compile_ir(self, params: dict) -> "HamiltonianIR":
        """Combine static blocks + protocol drives into a HamiltonianIR."""
        from ryd_gate.solvers.ir import HamiltonianIR, HamiltonianTerm

        protocol = self._require_protocol()

        static_terms: list[HamiltonianTerm] = []
        for name in _STATIC_BLOCKS:
            if self._blocks.has(name):
                static_terms.append(HamiltonianTerm(name, self._blocks.get(name), 1.0))

        overlay_dense, overlay_pin = self._static_overlays(params, protocol)
        if overlay_dense is not None:
            if static_terms and static_terms[0].name == "H_const":
                static_terms[0] = HamiltonianTerm(
                    "H_const",
                    static_terms[0].operator + overlay_dense,
                    1.0,
                )
            else:
                static_terms.insert(0, HamiltonianTerm("H_const_addition", overlay_dense, 1.0))
        if overlay_pin is not None:
            static_terms.append(HamiltonianTerm("pinning", overlay_pin, 1.0))

        amplitude_scale = self.amplitude_scale
        drive_terms: list[HamiltonianTerm] = []
        for channel in protocol.required_channels:
            if not self._blocks.has(channel):
                continue

            def coeff_fn(t, channel=channel):
                coeffs = protocol.get_drive_coefficients(t, params)
                coeff = coeffs[channel]
                if channel == "lightshift_zero":
                    return amplitude_scale ** 2 * coeff
                return amplitude_scale * coeff

            drive_terms.append(
                HamiltonianTerm(
                    channel,
                    self._blocks.get(channel),
                    coeff_fn,
                    add_hermitian_conjugate=channel in _HC_CHANNELS,
                )
            )

        return HamiltonianIR(
            static_terms=static_terms,
            drive_terms=drive_terms,
            dim=self._basis.total_dim,
            is_sparse=self.is_sparse,
            metadata={
                "t_gate": params["t_gate"],
                "param_set": self._param_set,
                "n_sites": self._basis.n_sites,
                "local_dim": self._basis.local_dim,
            },
        )

    def _static_overlays(self, params: dict, protocol):
        """Return ``(dense_overlay, sparse_pinning)`` matrices from params dict.

        ``dense_overlay`` lumps protocol-supplied additions onto ``H_const``
        (matches the legacy ``get_ham_const_additions`` channel). ``sparse_pinning``
        uses the registered per-site occupation blocks (lattice path).
        """
        pin_deltas = params.get("pin_deltas")
        scatter_rates = params.get("scatter_rates")
        static_overlays = params.get("static_overlays")

        if pin_deltas is None and scatter_rates is None and static_overlays is None:
            dense = _protocol_static_addition(self, protocol)
            sparse_pin = _sparse_pin_from_protocol(self, protocol)
            return dense, sparse_pin

        n_sites = self._basis.n_sites
        n_levels = self._basis.local_dim
        ground_label = self._basis.local_levels[0]
        ryd_label = self._basis.local_levels[-1]

        dense_overlay = None
        sparse_pin = None
        pin_deltas = pin_deltas or {}
        scatter_rates = scatter_rates or {}
        static_overlays = static_overlays or []

        for idx, delta in pin_deltas.items():
            if abs(delta) <= 1e-15 or idx >= n_sites:
                continue
            block_name = f"n_{ryd_label}_{idx}"
            if self._blocks.has(block_name):
                term = -delta * self._blocks.get(block_name)
                sparse_pin = term if sparse_pin is None else sparse_pin + term
            else:
                from ryd_gate.core.operators import build_atom_projector
                proj = build_atom_projector(idx, n_levels - 1, n_sites, n_levels)
                term = -delta * proj
                dense_overlay = term if dense_overlay is None else dense_overlay + term

        for idx, rate in scatter_rates.items():
            if rate <= 0 or idx >= n_sites:
                continue
            block_name = f"n_{ground_label}_{idx}"
            if self._blocks.has(block_name):
                term = (-1j * rate / 2) * self._blocks.get(block_name)
                sparse_pin = term if sparse_pin is None else sparse_pin + term
            else:
                from ryd_gate.core.operators import build_atom_projector
                proj = build_atom_projector(idx, 0, n_sites, n_levels)
                term = (-1j * rate / 2) * proj
                dense_overlay = term if dense_overlay is None else dense_overlay + term

        for block_name, coeff in static_overlays:
            if self._blocks.has(block_name):
                term = coeff * self._blocks.get(block_name)
                if hasattr(term, "toarray"):
                    sparse_pin = term if sparse_pin is None else sparse_pin + term
                else:
                    dense_overlay = term if dense_overlay is None else dense_overlay + term

        return dense_overlay, sparse_pin

    def hamiltonian(self, t: float, params: dict):
        """Assemble the time-dependent Hamiltonian H(t) as a matrix."""
        ir = self.compile_ir(params)
        H = None
        for term in ir.static_terms:
            coeff = term.coefficient if not callable(term.coefficient) else term.coefficient(0)
            contrib = coeff * term.operator
            H = contrib if H is None else H + contrib
        for term in ir.drive_terms:
            coeff = term.coefficient(t) if callable(term.coefficient) else term.coefficient
            contrib = coeff * term.operator
            H = contrib if H is None else H + contrib
            if term.add_hermitian_conjugate:
                H = H + np.conjugate(coeff) * term.operator.conj().T
        if H is None:
            return np.zeros((self._basis.total_dim, self._basis.total_dim), dtype=complex)
        return H

    def product_state(self, config: str | list[str] | tuple[str, ...]) -> np.ndarray:
        """Return a computational product state in this model's basis."""
        labels = list(config) if not isinstance(config, str) else list(config)
        if len(labels) != self.basis.n_sites:
            raise ValueError(f"config must have length {self.basis.n_sites}, got {len(labels)}.")
        idx = 0
        d = self.basis.local_dim
        for site_i, label in enumerate(labels):
            idx += self.basis.level_index(label) * d ** (self.basis.n_sites - 1 - site_i)
        psi = np.zeros(self.basis.total_dim, dtype=complex)
        psi[idx] = 1.0
        return psi

    def ground_state(self) -> np.ndarray:
        return self.product_state([self.basis.local_levels[0]] * self.basis.n_sites)

    def expectation(self, observable: str, psi: np.ndarray) -> float:
        return self.observables.measure(observable, psi)

    @classmethod
    def from_lattice(
        cls,
        geometry: LatticeGeometry,
        level_structure: str | LevelStructureSpec = "1r",
        interaction: InteractionSpec | None = None,
        *,
        protocol: "Protocol | None" = None,
        param_set: str | None = None,
        Omega: float = 1.0,
    ) -> "RydbergSystem":
        spec = (
            level_structure
            if isinstance(level_structure, LevelStructureSpec)
            else globals()["level_structure"](level_structure)
        )
        interaction = interaction or InteractionSpec()
        d = spec.local_dim
        N = geometry.N
        dim = d ** N
        basis = BasisSpec(
            site_labels=tuple(str(i) for i in range(N)),
            local_levels=spec.levels,
            local_dim=d,
            total_dim=dim,
        )

        blocks = BlockRegistry()
        observables = ObservableRegistry()

        n_site: dict[str, list[csc_matrix]] = {}
        sum_n: dict[str, csc_matrix] = {}
        for level in spec.levels:
            local = np.zeros((d, d), dtype=complex)
            local[spec.index(level), spec.index(level)] = 1.0
            ops = [embed_site_op(csc_matrix(local), i, N, d=d).tocsc() for i in range(N)]
            n_site[level] = ops
            total = _sum_sparse(ops, dim)
            sum_n[level] = total
            blocks.register(f"sum_n_{level}", total, description=f"total |{level}> occupation")
            for i, op in enumerate(ops):
                blocks.register(f"n_{level}_{i}", op, description=f"|{level}><{level}| on site {i}")
                observables.register(
                    f"n_{level}_{i}",
                    op,
                    description=f"|{level}> population on site {i}",
                    per_site=True,
                )
            observables.register(f"sum_n_{level}", total, description=f"total |{level}> population")

        ryd_site_ops = [_sum_sparse([n_site[lvl][i] for lvl in spec.rydberg_levels], dim) for i in range(N)]
        H_vdw = spdiags([np.zeros(dim, dtype=complex)], [0], shape=(dim, dim), format="csc")
        pairs = _interaction_pairs(geometry, interaction)
        ryd_diags = [op.diagonal() for op in ryd_site_ops]
        h_diag = np.zeros(dim, dtype=complex)
        for i, j, V_ij in pairs:
            h_diag += V_ij * (ryd_diags[i] * ryd_diags[j])
        H_vdw = spdiags([h_diag], [0], shape=(dim, dim), format="csc")
        blocks.register("H_vdw", H_vdw, description="Rydberg VdW interaction")

        for transition in spec.transitions:
            local = np.zeros((d, d), dtype=complex)
            local[spec.index(transition.upper), spec.index(transition.lower)] = 1.0
            per_site = [
                embed_site_op(csc_matrix(local), i, N, d=d).tocsc()
                for i in range(N)
            ]
            global_op = _sum_sparse(per_site, dim)
            blocks.register(transition.channel, global_op, description=transition.name, hermitian=False)
            for i, op in enumerate(per_site):
                blocks.register(f"{transition.channel}_{i}", op, description=f"{transition.name} on site {i}", hermitian=False)

        for channel, level in spec.detuning_levels.items():
            blocks.register(channel, sum_n[level], description=f"detuning projector for |{level}>")

        if "r" in sum_n:
            blocks.register("sum_nr", sum_n["r"], description="total Rydberg occupation")
            observables.register("sum_nr", sum_n["r"], description="total Rydberg population")
        if spec.name == "1r":
            blocks.register("global_n", sum_n["r"], description="2-level Rydberg occupation")

        if geometry.sublattice is not None and np.any(geometry.sublattice):
            if "r" in n_site:
                staggered = _sum_sparse(
                    [geometry.sublattice[i] * n_site["r"][i] for i in range(N)],
                    dim,
                )
                observables.register("staggered_rydberg", staggered, description="staggered Rydberg occupation")

        metadata = {
            "level_structure": spec.name,
            "interaction_pairs": pairs,
            "Omega": Omega,
            "local_dim": d,
            "n_sites": N,
        }
        return cls(
            param_set=param_set or f"lattice_{spec.name}",
            basis=basis,
            blocks=blocks,
            observables=observables,
            protocol=protocol,
            metadata=metadata,
            geometry=geometry,
            is_sparse=True,
        )

    @classmethod
    def from_preset(
        cls,
        name: str,
        protocol: "Protocol | None" = None,
        geometry: LatticeGeometry | None = None,
        **params,
    ) -> "RydbergSystem":
        """Build a named model preset."""
        if name in {"1r", "01r", "1er"}:
            if geometry is None:
                geometry = make_chain(params.pop("N", 2), spacing_um=params.pop("spacing_um", 4.0))
            interaction = InteractionSpec(
                C6=params.pop("C6", DEFAULT_C6),
                max_range_um=params.pop("max_range_um", None),
                mode=params.pop("interaction_mode", "all"),
            )
            return cls.from_lattice(
                geometry,
                level_structure=name,
                interaction=interaction,
                protocol=protocol,
                param_set=params.pop("param_set", f"lattice_{name}"),
                Omega=params.pop("Omega", 1.0),
            )
        if name == "analog_3":
            return _analog_3level_model(protocol=protocol, **params)
        if name in {"rb87_7", "our", "lukin"}:
            param_set = "our" if name == "rb87_7" else name
            return _rb87_7level_model(param_set=param_set, protocol=protocol, **params)
        raise ValueError(f"Unknown RydbergSystem preset '{name}'.")


# Backward-compatible alias (one release).
RydbergSystemModel = RydbergSystem


def _protocol_static_addition(system: RydbergSystem, protocol):
    """Resolve protocol-supplied time-independent terms (legacy hook).

    Used when ``unpack_params`` did not emit the new ``pin_deltas`` /
    ``scatter_rates`` / ``static_overlays`` dict keys.
    """
    if not hasattr(protocol, "get_ham_const_additions"):
        return None
    return protocol.get_ham_const_additions(
        n_atoms=system.basis.n_sites,
        n_levels=system.basis.local_dim,
    )


def _sparse_pin_from_protocol(system: RydbergSystem, protocol):
    """Legacy fallback for protocols that haven't migrated to params dict."""
    if not hasattr(protocol, "get_pin_deltas"):
        return None
    pin = protocol.get_pin_deltas(system.basis.n_sites)
    if not np.any(pin):
        return None
    sparse_pin = None
    for i, delta_i in enumerate(pin):
        if abs(delta_i) <= 1e-15:
            continue
        block_name = f"n_r_{i}"
        if system.blocks.has(block_name):
            term = -delta_i * system.blocks.get(block_name)
            sparse_pin = term if sparse_pin is None else sparse_pin + term
    return sparse_pin


def _sum_sparse(ops: list[csc_matrix], dim: int) -> csc_matrix:
    if not ops:
        return csc_matrix((dim, dim), dtype=complex)
    return sum(ops[1:], start=ops[0]).tocsc()


def _interaction_pairs(geometry: LatticeGeometry, interaction: InteractionSpec) -> tuple:
    if interaction.mode == "all":
        return vdw_couplings(geometry.coords, interaction.C6, interaction.max_range_um)

    coords = np.asarray(geometry.coords, dtype=float)
    spacing = geometry.spacing_um or _nearest_spacing(coords)
    max_dist = spacing * (1.01 if interaction.mode == "nn" else np.sqrt(2) * 1.01)
    max_range = interaction.max_range_um if interaction.max_range_um is not None else max_dist
    return vdw_couplings(coords, interaction.C6, max_range)


def _nearest_spacing(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 0.0
    dists = [
        float(np.linalg.norm(coords[i] - coords[j]))
        for i in range(len(coords))
        for j in range(i + 1, len(coords))
    ]
    return min(dists)


def _register_dense_observables(
    observables: ObservableRegistry,
    basis: BasisSpec,
    *,
    include_joint: bool = False,
) -> None:
    d = basis.local_dim
    for i, level in enumerate(basis.local_levels):
        per_site_ops = []
        for site_i, site in enumerate(basis.site_labels):
            op = _dense_site_projector(site_i, i, basis.n_sites, d)
            per_site_ops.append(op)
            observables.register(
                f"pop_{site}_{level}",
                op,
                description=f"site {site} in |{level}>",
                per_site=True,
            )
        observables.register(f"pop_{level}", sum(per_site_ops), description=f"total |{level}> population")

    if include_joint and basis.n_sites == 2:
        for i, a in enumerate(basis.local_levels):
            va = np.zeros(d, dtype=complex)
            va[i] = 1.0
            for j, b in enumerate(basis.local_levels):
                vb = np.zeros(d, dtype=complex)
                vb[j] = 1.0
                vec = np.kron(va, vb)
                observables.register(f"pop_{a}{b}", np.outer(vec, vec.conj()), description=f"joint |{a}{b}> population")


def _dense_site_projector(site_idx: int, level_idx: int, n_sites: int, local_dim: int) -> np.ndarray:
    local = np.zeros((local_dim, local_dim), dtype=np.complex128)
    local[level_idx, level_idx] = 1.0
    op = np.eye(1, dtype=np.complex128)
    for i in range(n_sites):
        op = np.kron(op, local if i == site_idx else np.eye(local_dim, dtype=np.complex128))
    return op


def _analog_3level_model(protocol: "Protocol | None" = None, **kwargs) -> RydbergSystem:
    from ryd_gate.legacy.atomic_system import create_analog_system

    system = create_analog_system(**kwargs)
    basis = BasisSpec(site_labels=tuple(str(i) for i in range(system.n_atoms)), local_levels=("g", "e", "r"), local_dim=3, total_dim=3 ** system.n_atoms)
    blocks = BlockRegistry()
    blocks.register("H_const", system.tq_ham_const)
    blocks.register("H_1013", system.tq_ham_1013, hermitian=False)
    blocks.register("H_1013_conj", system.tq_ham_1013_conj, hermitian=False)
    blocks.register("drive_420", system.tq_ham_420, hermitian=False)
    blocks.register("drive_420_dag", system.tq_ham_420_conj, hermitian=False)
    blocks.register("lightshift_zero", system.tq_ham_lightshift_zero)
    observables = ObservableRegistry()
    _register_dense_observables(observables, basis, include_joint=system.n_atoms == 2)
    return RydbergSystem(
        param_set="analog",
        basis=basis,
        blocks=blocks,
        observables=observables,
        protocol=protocol,
        metadata=_metadata_from_atomic(system),
        is_sparse=False,
    )


def _rb87_7level_model(
    param_set: str = "our",
    protocol: "Protocol | None" = None,
    **kwargs,
) -> RydbergSystem:
    from ryd_gate.legacy.atomic_system import create_lukin_system, create_our_system

    factory = create_lukin_system if param_set == "lukin" else create_our_system
    system = factory(**kwargs)
    levels = level_structure("rb87_7").levels
    basis = BasisSpec(site_labels=("A", "B"), local_levels=levels, local_dim=7, total_dim=49)
    blocks = BlockRegistry()
    blocks.register("H_const", system.tq_ham_const)
    blocks.register("H_1013", system.tq_ham_1013, hermitian=False)
    blocks.register("H_1013_conj", system.tq_ham_1013_conj, hermitian=False)
    blocks.register("drive_420", system.tq_ham_420, hermitian=False)
    blocks.register("drive_420_dag", system.tq_ham_420_conj, hermitian=False)
    blocks.register("lightshift_zero", system.tq_ham_lightshift_zero)
    observables = ObservableRegistry()
    _register_dense_observables(observables, basis)
    return RydbergSystem(
        param_set=param_set,
        basis=basis,
        blocks=blocks,
        observables=observables,
        protocol=protocol,
        metadata=_metadata_from_atomic(system),
        is_sparse=False,
    )


def _metadata_from_atomic(system) -> dict[str, Any]:
    return {
        "rabi_eff": system.rabi_eff,
        "time_scale": system.time_scale,
        "t_rise": system.t_rise,
        "blackmanflag": system.blackmanflag,
        "n_atoms": system.n_atoms,
        "n_levels": system.n_levels,
        "Delta": system.Delta,
        "v_ryd": system.v_ryd,
    }
