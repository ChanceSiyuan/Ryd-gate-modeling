"""Universal Rydberg system model.

A :class:`RydbergSystem` is built from three things: a lattice geometry,
a local energy-level structure (``1r`` / ``01r`` / ``ger`` / ``rb87_7``;
``analog_3`` is a physical ``ger`` preset), and a pulse protocol.
The class owns symbolic Hamiltonian blocks, observables, geometry metadata,
and the bound protocol. Backend-specific compilers materialize those symbolic
blocks into matrices, MPOs, or other solver inputs only when needed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ryd_gate.core.basis import BasisSpec
from ryd_gate.core.blocks import BlockRegistry
from ryd_gate.core.interactions import vdw_couplings
from ryd_gate.core.observables import ObservableRegistry
from ryd_gate.core.operator_spec import (
    LocalMatrixSumSpec,
    LocalProjectorSpec,
    RydbergPairInteractionSpec,
    SumProjectorSpec,
    TransitionOperatorSpec,
    WeightedProjectorSumSpec,
    is_operator_spec,
    measure_state_vector_operator,
)
from ryd_gate.core.system_model import SystemModel
from ryd_gate.lattice.geometry import LatticeGeometry

if TYPE_CHECKING:
    from ryd_gate.protocols.base import Protocol


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


@dataclass(frozen=True)
class _RB87PhysicalParams:
    param_set: str
    ryd_level: int
    Delta: float
    rabi_420: float
    rabi_1013: float
    rabi_eff: float
    time_scale: float
    rabi_420_garbage: float
    rabi_1013_garbage: float
    d_mid_ratio: float
    d_ryd_ratio: float
    v_ryd: float
    v_ryd_garb: float
    ryd_zeeman_shift: float
    detuning_sign: int
    mid_state_decay_rate: float
    mid_garb_decay_rate: float
    ryd_state_decay_rate: float
    ryd_RD_rate: float
    ryd_BBR_rate: float
    ryd_garb_decay_rate: float
    ryd_branch: dict
    mid_branch: dict
    t_rise: float
    blackmanflag: bool
    enable_rydberg_decay: bool
    enable_intermediate_decay: bool
    enable_0_scattering: bool
    enable_polarization_leakage: bool
    n_levels: int = 7
    rydberg_indices: tuple[int, ...] = (5, 6)
    n_atoms: int = 2


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
        "ger": LevelStructureSpec(
            name="ger",
            levels=("g", "e", "r"),
            rydberg_levels=("r",),
            transitions=(
                TransitionSpec("420", "g", "e", "drive_420"),
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

    Built via :meth:`from_lattice`. Once a protocol is attached (constructor
    arg or :meth:`with_protocol`), the system can be passed to
    :func:`ryd_gate.ir.compile_hamiltonian_ir`. Algorithm packages then lower
    that unified Hamiltonian IR into exact matrices, MPS/MPO data, TTN inputs,
    or external solver payloads.
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

    def hamiltonian(self, t: float, params: dict):
        """Materialized Hamiltonian access has moved to algorithm packages."""
        del t, params
        raise RuntimeError(
            "RydbergSystem no longer materializes algorithm-specific matrices. "
            "Use ryd_gate.compile_hamiltonian_ir(system, params) for the unified "
            "Hamiltonian IR, then pass that IR to an algorithm compiler."
        )

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
        obs = self.observables.get(observable)
        if is_operator_spec(obs.operator):
            return measure_state_vector_operator(obs.operator, self.basis, psi)
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
        **physical_params,
    ) -> "RydbergSystem":
        spec = (
            level_structure
            if isinstance(level_structure, LevelStructureSpec)
            else globals()["level_structure"](level_structure)
        )
        physical_model = _physical_model_for(spec.name, param_set)
        interaction = interaction or _default_interaction_for_physical_model(physical_model)
        d = spec.local_dim
        N = geometry.N
        dim = d**N
        basis = BasisSpec(
            site_labels=tuple(str(i) for i in range(N)),
            local_levels=spec.levels,
            local_dim=d,
            total_dim=dim,
        )

        blocks = BlockRegistry()
        observables = ObservableRegistry()

        for level in spec.levels:
            sum_spec = SumProjectorSpec(level)
            blocks.register(f"sum_n_{level}", sum_spec, description=f"total |{level}> occupation")
            for i in range(N):
                site_spec = LocalProjectorSpec(level, i)
                blocks.register(f"n_{level}_{i}", site_spec, description=f"|{level}><{level}| on site {i}")
                observables.register(
                    f"n_{level}_{i}",
                    site_spec,
                    description=f"|{level}> population on site {i}",
                    per_site=True,
                )
            observables.register(f"sum_n_{level}", sum_spec, description=f"total |{level}> population")

        pairs = _interaction_pairs(geometry, interaction)
        blocks.register(
            "H_vdw",
            RydbergPairInteractionSpec(pairs, spec.rydberg_levels),
            description="Rydberg VdW interaction",
        )

        for transition in spec.transitions:
            blocks.register(
                transition.channel,
                TransitionOperatorSpec(transition.lower, transition.upper),
                description=transition.name,
                hermitian=False,
            )
            for i in range(N):
                blocks.register(
                    f"{transition.channel}_{i}",
                    TransitionOperatorSpec(transition.lower, transition.upper, site=i),
                    description=f"{transition.name} on site {i}",
                    hermitian=False,
                )

        for channel, level in spec.detuning_levels.items():
            blocks.register(channel, SumProjectorSpec(level), description=f"detuning projector for |{level}>")

        if "r" in spec.levels:
            sum_r = SumProjectorSpec("r")
            blocks.register("sum_nr", sum_r, description="total Rydberg occupation")
            observables.register("sum_nr", sum_r, description="total Rydberg population")
        if spec.name == "1r":
            blocks.register("global_n", SumProjectorSpec("r"), description="2-level Rydberg occupation")

        if geometry.sublattice is not None and np.any(geometry.sublattice):
            if "r" in spec.levels:
                observables.register(
                    "staggered_rydberg",
                    WeightedProjectorSumSpec("r", tuple(float(x) for x in geometry.sublattice)),
                    description="staggered Rydberg occupation",
                )

        metadata = {
            "level_structure": spec.name,
            "level_spec": spec,
            "interaction_pairs": pairs,
            "Omega": Omega,
            "local_dim": d,
            "n_sites": N,
        }
        model = cls(
            param_set=param_set or f"lattice_{spec.name}",
            basis=basis,
            blocks=blocks,
            observables=observables,
            protocol=protocol,
            metadata=metadata,
            geometry=geometry,
            is_sparse=True,
        )
        if physical_model == "analog_3":
            _apply_analog_3_lattice_blocks(model, **physical_params)
        elif physical_model in {"our", "lukin"}:
            _apply_rb87_7_lattice_blocks(model, physical_model, **physical_params)
        return model


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
        float(np.linalg.norm(coords[i] - coords[j])) for i in range(len(coords)) for j in range(i + 1, len(coords))
    ]
    return min(dists)


def _physical_model_for(level_name: str, param_set: str | None) -> str | None:
    if level_name == "ger" and param_set in {"analog", "analog_3"}:
        return "analog_3"
    if level_name == "rb87_7" and param_set in {"our", "lukin"}:
        return param_set
    return None


def _default_interaction_for_physical_model(physical_model: str | None) -> InteractionSpec:
    if physical_model == "lukin":
        return InteractionSpec(C6=_rb87_default_c6("lukin"))
    return InteractionSpec(C6=DEFAULT_C6)


def _rb87_default_c6(param_set: str) -> float:
    if param_set == "lukin":
        return 2 * np.pi * 450e6 * 3.0**6
    return DEFAULT_C6


def _register_local_matrix_block(
    blocks: BlockRegistry,
    name: str,
    matrix: np.ndarray,
    *,
    hermitian: bool = True,
    description: str = "",
) -> None:
    blocks.register(
        name,
        LocalMatrixSumSpec(np.asarray(matrix, dtype=np.complex128)),
        description=description,
        hermitian=hermitian,
    )


def _apply_analog_3_lattice_blocks(
    model: RydbergSystem,
    *,
    detuning_sign: int = 1,
    blackmanflag: bool = True,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
    Delta_Hz: float | None = None,
    rabi_420_Hz: float | None = None,
    rabi_1013_Hz: float | None = None,
    **unused,
) -> None:
    _reject_unused(unused)
    ryd_level = 70
    Delta = detuning_sign * 2 * np.pi * (Delta_Hz if Delta_Hz is not None else 9.1e9)
    rabi_420 = 2 * np.pi * (rabi_420_Hz if rabi_420_Hz is not None else 491e6)
    rabi_1013 = 2 * np.pi * (rabi_1013_Hz if rabi_1013_Hz is not None else 491e6)
    rabi_eff = rabi_420 * rabi_1013 / (2 * abs(Delta))
    time_scale = 2 * np.pi / rabi_eff
    mid_state_decay_rate = 1 / 110.7e-9
    ryd_state_decay_rate = 1 / 151.55e-6
    ryd_RD_rate = 1 / 410.41e-6
    ryd_BBR_rate = ryd_state_decay_rate - ryd_RD_rate

    mid_decay = mid_state_decay_rate if enable_intermediate_decay else 0.0
    ryd_decay = ryd_state_decay_rate if enable_rydberg_decay else 0.0

    h_const = np.zeros((3, 3), dtype=np.complex128)
    h_const[1, 1] = Delta - 1j * mid_decay / 2
    h_const[2, 2] = -1j * ryd_decay / 2

    h1013 = np.zeros((3, 3), dtype=np.complex128)
    h1013[2, 1] = rabi_1013 / 2
    h420 = np.zeros((3, 3), dtype=np.complex128)
    h420[1, 0] = rabi_420 / 2
    lightshift_zero = np.zeros((3, 3), dtype=np.complex128)

    _register_local_matrix_block(model.blocks, "H_const", h_const, description="single-atom ger energies")
    _register_local_matrix_block(model.blocks, "H_1013", h1013, hermitian=False, description="static e-r coupling")
    _register_local_matrix_block(model.blocks, "H_1013_conj", h1013.conj().T, hermitian=False)
    _register_local_matrix_block(model.blocks, "drive_420", h420, hermitian=False, description="g-e drive")
    _register_local_matrix_block(model.blocks, "drive_420_dag", h420.conj().T, hermitian=False)
    _register_local_matrix_block(model.blocks, "lightshift_zero", lightshift_zero)

    model.metadata.update(
        {
            "physical_model": "analog_3",
            "rabi_eff": rabi_eff,
            "time_scale": time_scale,
            "t_rise": 20e-9,
            "blackmanflag": blackmanflag,
            "n_atoms": model.N,
            "n_levels": 3,
            "rabi_420": rabi_420,
            "rabi_1013": rabi_1013,
            "rabi_420_garbage": 0.0,
            "rabi_1013_garbage": 0.0,
            "Delta": Delta,
            "v_ryd": _nearest_pair_strength(model.metadata.get("interaction_pairs", ())),
            "v_ryd_garb": 0.0,
            "ryd_level": ryd_level,
            "ryd_state_decay_rate": ryd_state_decay_rate,
            "ryd_RD_rate": ryd_RD_rate,
            "ryd_BBR_rate": ryd_BBR_rate,
            "mid_state_decay_rate": mid_state_decay_rate,
            "ryd_branch": {},
            "mid_branch": {},
            "rydberg_indices": (2,),
            "enable_rydberg_decay": enable_rydberg_decay,
            "enable_intermediate_decay": enable_intermediate_decay,
            "enable_0_scattering": False,
            "enable_polarization_leakage": False,
        }
    )


def _apply_rb87_7_lattice_blocks(
    model: RydbergSystem,
    param_set: str,
    *,
    detuning_sign: int = 1,
    blackmanflag: bool = True,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
    enable_0_scattering: bool = True,
    enable_polarization_leakage: bool = False,
    **unused,
) -> None:
    _reject_unused(unused)
    physical = _rb87_physical_params(
        param_set,
        detuning_sign=detuning_sign,
        blackmanflag=blackmanflag,
        enable_rydberg_decay=enable_rydberg_decay,
        enable_intermediate_decay=enable_intermediate_decay,
        enable_0_scattering=enable_0_scattering,
        enable_polarization_leakage=enable_polarization_leakage,
    )

    h_const = _rb87_local_h_const(
        physical.Delta,
        physical.ryd_zeeman_shift,
        physical.mid_state_decay_rate if enable_intermediate_decay else 0.0,
        physical.ryd_state_decay_rate if enable_rydberg_decay else 0.0,
    )
    h420 = _rb87_local_h420(param_set, physical.rabi_420, physical.rabi_420_garbage)
    h1013 = _rb87_local_h1013(param_set, physical.rabi_1013, physical.rabi_1013_garbage)
    lightshift_zero = _rb87_local_zero_lightshift(
        param_set,
        physical.Delta,
        physical.rabi_420,
        physical.rabi_420_garbage,
        physical.mid_state_decay_rate,
        enable_intermediate_decay,
        enable_0_scattering,
    )

    _register_local_matrix_block(model.blocks, "H_const", h_const, description="single-atom rb87_7 energies")
    _register_local_matrix_block(model.blocks, "H_1013", h1013, hermitian=False, description="static 1013nm coupling")
    _register_local_matrix_block(model.blocks, "H_1013_conj", h1013.conj().T, hermitian=False)
    _register_local_matrix_block(model.blocks, "drive_420", h420, hermitian=False, description="420nm drive")
    _register_local_matrix_block(model.blocks, "drive_420_dag", h420.conj().T, hermitian=False)
    _register_local_matrix_block(model.blocks, "lightshift_zero", lightshift_zero)

    model.metadata.update(_metadata_from_rb87_params(physical))
    model.metadata.update(
        {
            "physical_model": param_set,
            "n_atoms": model.N,
            "n_sites": model.N,
            "level_structure": "rb87_7",
            "level_spec": level_structure("rb87_7"),
            "v_ryd": _nearest_pair_strength(model.metadata.get("interaction_pairs", ())),
            "ryd_level": physical.ryd_level,
        }
    )


def _rb87_physical_params(
    param_set: str,
    *,
    detuning_sign: int,
    blackmanflag: bool,
    enable_rydberg_decay: bool,
    enable_intermediate_decay: bool,
    enable_0_scattering: bool,
    enable_polarization_leakage: bool,
) -> _RB87PhysicalParams:
    from arc import Rubidium87
    from ryd_gate.physics.branching import _mid_branching_ratios, _rydberg_branching_ratios

    atom = Rubidium87()
    if param_set == "our":
        ryd_level = 70
        Delta = detuning_sign * 2 * np.pi * 9.1e9
        rabi_420 = 2 * np.pi * 491e6
        rabi_1013 = 2 * np.pi * 185e6
        d_mid_ratio = atom.getDipoleMatrixElement(5, 0, 0.5, 0.5, 6, 1, 1.5, -0.5, -1) / atom.getDipoleMatrixElement(
            5, 0, 0.5, -0.5, 6, 1, 1.5, -1.5, -1
        )
        d_ryd_ratio = atom.getDipoleMatrixElement(
            6, 1, 1.5, -0.5, ryd_level, 0, 0.5, 0.5, 1
        ) / atom.getDipoleMatrixElement(6, 1, 1.5, -1.5, ryd_level, 0, 0.5, -0.5, 1)
        v_ryd = 2 * np.pi * 874e9 / 3**6
        v_ryd_garb = v_ryd
        ryd_zeeman_shift = 2 * np.pi * 56e6 if enable_polarization_leakage else 2 * np.pi * 56e9
        mid_state_decay_rate = 1 / 110.7e-9
        ryd_state_decay_rate = 1 / 151.55e-6
        ryd_RD_rate = 1 / 410.41e-6
        ryd_branch = _rydberg_branching_ratios(atom, ryd_level, "our")
        mid_branch = {F: _mid_branching_ratios(atom, F, mF=-1) for F in (1, 2, 3)}
    elif param_set == "lukin":
        ryd_level = 53
        Delta = detuning_sign * 2 * np.pi * 7.8e9
        rabi_420 = 2 * np.pi * 237e6
        rabi_1013 = 2 * np.pi * 303e6
        d_mid_ratio = atom.getDipoleMatrixElement(5, 0, 0.5, -0.5, 6, 1, 1.5, 0.5, 1) / atom.getDipoleMatrixElement(
            5, 0, 0.5, 0.5, 6, 1, 1.5, 1.5, 1
        )
        d_ryd_ratio = atom.getDipoleMatrixElement(
            6, 1, 1.5, 0.5, ryd_level, 0, 0.5, -0.5, -1
        ) / atom.getDipoleMatrixElement(6, 1, 1.5, 1.5, ryd_level, 0, 0.5, 0.5, -1)
        v_ryd = 2 * np.pi * 450e6
        v_ryd_garb = v_ryd
        ryd_zeeman_shift = 2 * np.pi * 2.4e9 if enable_polarization_leakage else 2 * np.pi * 2.4e12
        mid_state_decay_rate = 1 / 110e-9
        ryd_state_decay_rate = 1 / 88e-6
        ryd_RD_rate = 1 / 147.64e-6
        ryd_branch = _rydberg_branching_ratios(atom, ryd_level, "lukin")
        mid_branch = {F: _mid_branching_ratios(atom, F, mF=1) for F in (1, 2, 3)}
    else:
        raise ValueError(f"Unknown rb87_7 parameter set '{param_set}'.")

    rabi_420_garbage = rabi_420 * d_mid_ratio
    rabi_1013_garbage = rabi_1013 * d_ryd_ratio
    rabi_eff = rabi_420 * rabi_1013 / (2 * abs(Delta))
    time_scale = 2 * np.pi / rabi_eff
    ryd_BBR_rate = ryd_state_decay_rate - ryd_RD_rate

    return _RB87PhysicalParams(
        param_set=param_set,
        ryd_level=ryd_level,
        Delta=Delta,
        rabi_420=rabi_420,
        rabi_1013=rabi_1013,
        rabi_eff=rabi_eff,
        time_scale=time_scale,
        rabi_420_garbage=rabi_420_garbage,
        rabi_1013_garbage=rabi_1013_garbage,
        d_mid_ratio=d_mid_ratio,
        d_ryd_ratio=d_ryd_ratio,
        v_ryd=v_ryd,
        v_ryd_garb=v_ryd_garb,
        ryd_zeeman_shift=ryd_zeeman_shift,
        detuning_sign=detuning_sign,
        mid_state_decay_rate=mid_state_decay_rate,
        mid_garb_decay_rate=mid_state_decay_rate,
        ryd_state_decay_rate=ryd_state_decay_rate,
        ryd_RD_rate=ryd_RD_rate,
        ryd_BBR_rate=ryd_BBR_rate,
        ryd_garb_decay_rate=ryd_state_decay_rate,
        ryd_branch=ryd_branch,
        mid_branch=mid_branch,
        t_rise=20e-9,
        blackmanflag=blackmanflag,
        enable_rydberg_decay=enable_rydberg_decay,
        enable_intermediate_decay=enable_intermediate_decay,
        enable_0_scattering=enable_0_scattering,
        enable_polarization_leakage=enable_polarization_leakage,
    )


def _rb87_local_h_const(
    Delta: float,
    ryd_zeeman_shift: float,
    middecay: float,
    ryddecay: float,
) -> np.ndarray:
    h = np.zeros((7, 7), dtype=np.complex128)
    h[2, 2] = Delta - 2 * np.pi * 51e6 - 1j * middecay / 2
    h[3, 3] = Delta - 1j * middecay / 2
    h[4, 4] = Delta + 2 * np.pi * 87e6 - 1j * middecay / 2
    h[5, 5] = -1j * ryddecay / 2
    h[6, 6] = ryd_zeeman_shift - 1j * ryddecay / 2
    return h


def _rb87_local_h420(param_set: str, rabi_420: float, rabi_420_garbage: float) -> np.ndarray:
    from arc.wigner import CG

    h = np.zeros((7, 7), dtype=np.complex128)
    if param_set == "our":
        for row, F in zip((2, 3, 4), (1, 2, 3)):
            h[row, 1] = (
                rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, F, -1)
                + rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, F, -1)
            ) / 2
    else:
        for row, F in zip((2, 3, 4), (1, 2, 3)):
            h[row, 1] = (
                rabi_420 * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, F, 1)
                + rabi_420_garbage * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, F, 1)
            ) / 2
    return h


def _rb87_local_h1013(param_set: str, rabi_1013: float, rabi_1013_garbage: float) -> np.ndarray:
    from arc.wigner import CG

    h = np.zeros((7, 7), dtype=np.complex128)
    if param_set == "our":
        for col, F in zip((2, 3, 4), (1, 2, 3)):
            h[5, col] = (rabi_1013 / 2) * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, F, -1)
            h[6, col] = (rabi_1013_garbage / 2) * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, F, -1)
    else:
        for col, F in zip((2, 3, 4), (1, 2, 3)):
            h[5, col] = (rabi_1013 / 2) * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, F, 1)
            h[6, col] = (rabi_1013_garbage / 2) * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, F, 1)
    return h


def _rb87_local_zero_lightshift(
    param_set: str,
    Delta: float,
    rabi_420: float,
    rabi_420_garbage: float,
    mid_state_decay_rate: float,
    enable_intermediate_decay: bool,
    enable_0_scattering: bool,
) -> np.ndarray:
    from arc.wigner import CG

    E_0 = -2 * np.pi * 6.835e9
    mid_energies = np.array(
        [
            Delta - 2 * np.pi * 51e6,
            Delta,
            Delta + 2 * np.pi * 87e6,
        ],
        dtype=np.float64,
    )
    if param_set == "our":
        cg_ratio_main = CG(1 / 2, -1 / 2, 3 / 2, 1 / 2, 1, 0) / CG(1 / 2, -1 / 2, 3 / 2, 1 / 2, 2, 0)
        cg_ratio_garb = CG(1 / 2, 1 / 2, 3 / 2, -1 / 2, 1, 0) / CG(1 / 2, 1 / 2, 3 / 2, -1 / 2, 2, 0)
        couplings = [
            (
                cg_ratio_main * rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, F, -1)
                + cg_ratio_garb * rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, F, -1)
            )
            / 2
            for F in (1, 2, 3)
        ]
    else:
        cg_ratio_main = CG(1 / 2, 1 / 2, 3 / 2, -1 / 2, 1, 0) / CG(1 / 2, 1 / 2, 3 / 2, -1 / 2, 2, 0)
        cg_ratio_garb = CG(1 / 2, -1 / 2, 3 / 2, 1 / 2, 1, 0) / CG(1 / 2, -1 / 2, 3 / 2, 1 / 2, 2, 0)
        couplings = [
            (
                cg_ratio_main * rabi_420 * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, F, 1)
                + cg_ratio_garb * rabi_420_garbage * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, F, 1)
            )
            / 2
            for F in (1, 2, 3)
        ]

    local = np.zeros((7, 7), dtype=np.complex128)
    total_shift = 0.0
    scatter_rate = 0.0
    gamma = mid_state_decay_rate if enable_intermediate_decay else 0.0
    for idx, (g_i, E_e) in enumerate(zip(couplings, mid_energies), start=2):
        detuning = E_e - E_0
        shift = (np.abs(g_i) ** 2) / detuning
        local[idx, idx] = shift
        total_shift += shift
        scatter_rate += (np.abs(g_i) ** 2) * gamma / (detuning**2)
    local[0, 0] = -total_shift - 1j * scatter_rate / 2 if enable_0_scattering else -total_shift
    return local


def _nearest_pair_strength(pairs: tuple) -> float:
    if not pairs:
        return 0.0
    return float(max(abs(strength) for _, _, strength in pairs))


def _reject_unused(unused: dict) -> None:
    if unused:
        names = ", ".join(sorted(unused))
        raise TypeError(f"Unused physical parameter(s): {names}")


def _metadata_from_rb87_params(system: _RB87PhysicalParams) -> dict[str, Any]:
    return {
        "rabi_eff": system.rabi_eff,
        "time_scale": system.time_scale,
        "t_rise": system.t_rise,
        "blackmanflag": system.blackmanflag,
        "n_atoms": system.n_atoms,
        "n_levels": system.n_levels,
        "rabi_420": system.rabi_420,
        "rabi_1013": system.rabi_1013,
        "rabi_420_garbage": system.rabi_420_garbage,
        "rabi_1013_garbage": system.rabi_1013_garbage,
        "Delta": system.Delta,
        "v_ryd": system.v_ryd,
        "v_ryd_garb": system.v_ryd_garb,
        "ryd_state_decay_rate": system.ryd_state_decay_rate,
        "ryd_RD_rate": system.ryd_RD_rate,
        "ryd_BBR_rate": system.ryd_BBR_rate,
        "mid_state_decay_rate": system.mid_state_decay_rate,
        "ryd_branch": system.ryd_branch,
        "mid_branch": system.mid_branch,
        "rydberg_indices": system.rydberg_indices,
        "enable_rydberg_decay": system.enable_rydberg_decay,
        "enable_intermediate_decay": system.enable_intermediate_decay,
        "enable_0_scattering": system.enable_0_scattering,
        "enable_polarization_leakage": system.enable_polarization_leakage,
    }
