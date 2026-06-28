"""Universal Rydberg system model and its lattice construction.

A :class:`RydbergSystem` is built from three things: a lattice geometry,
a local energy-level structure (``1r`` / ``01r`` / ``analog_3`` /
``rb87_7``, or a hand-built :class:`LevelStructureSpec`), and a pulse
protocol.
The class owns symbolic Hamiltonian blocks, observables, geometry metadata,
and the bound protocol. Backend-specific compilers materialize those symbolic
blocks into matrices, MPOs, or other solver inputs only when needed.

Level-structure/interaction specs live in :mod:`ryd_gate.core.level_structures`
and the Rb87 physical parameter sets and local matrix blocks in
:mod:`ryd_gate.core.physical_models`. A system is built with the fluent
``RydbergSystem.set_atom_level(...).set_atom_geom(...).set_protocol(...)``
builder; the underlying engine (:func:`_build_from_lattice`) lives below the
class and registers symbolic blocks and observables for a geometry + level
structure, with interaction-pair and physical-model resolution helpers.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from ryd_gate.core.level_structures import (
    DEFAULT_C6,
    InteractionSpec,
    LevelStructureSpec,
)
from ryd_gate.core.level_structures import level_structure as level_structure_preset
from ryd_gate.core.model import (
    BasisSpec,
    BlockRegistry,
    ObservableRegistry,
    SystemModel,
)
from ryd_gate.core.operators import (
    LocalProjectorSpec,
    RydbergPairInteractionSpec,
    SumProjectorSpec,
    TransitionOperatorSpec,
    WeightedProjectorSumSpec,
    is_operator_spec,
    measure_state_vector_operator,
)
from ryd_gate.core.physical_models import (
    _apply_analog_3_lattice_blocks,
    _apply_rb87_7_lattice_blocks,
    _rb87_default_c6,
    vdw_couplings,
)
from ryd_gate.lattice import Register

if TYPE_CHECKING:
    from ryd_gate.protocols.base import Protocol


class RydbergSystem(SystemModel):
    """Universal Rydberg-lattice system: geometry + level structure + protocol.

    Built via :meth:`set_atom_level` (then ``.set_atom_geom(...)`` and
    ``.set_protocol(...)``). Once a protocol is attached (via
    :meth:`_RydbergSystemBuilder.set_protocol` or :meth:`with_protocol`), the
    system can be passed to
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
        geometry: Register | None = None,
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
            "Use ryd_gate.ir.compile_hamiltonian_ir(system, params) for the unified "
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
    def set_atom_level(
        cls,
        level_structure: str | LevelStructureSpec = "1r",
        *,
        param_set: str | None = None,
        Omega: float = 1.0,
        **level_kwargs,
    ) -> "_RydbergSystemBuilder":
        """Start building a system by declaring the single-atom level structure.

        Returns a builder; chain :meth:`_RydbergSystemBuilder.set_atom_geom`
        (atom positions + Rydberg interaction) and
        :meth:`_RydbergSystemBuilder.set_protocol` (the pulse).  ``level_kwargs``
        are atom-level physics flags (``detuning_sign``, ``enable_*_decay`` ...)
        and the laser operating point (``Delta_Hz``; for
        analog_3 also ``rabi_420_Hz`` / ``rabi_1013_Hz``).  The rb87_7 420/1013
        Rabi scale is owned by the CZ protocol (``omega_*_max``), not the system.
        Only the keys you pass are forwarded.
        """
        return _RydbergSystemBuilder(
            level_structure,
            param_set=param_set,
            Omega=Omega,
            level_kwargs=dict(level_kwargs),
        )


class _RydbergSystemBuilder:
    """Accumulates atom-level + geometry + protocol config, materializing once.

    Returned by :meth:`RydbergSystem.set_atom_level`.  The single-atom numeric
    blocks depend on the atom-level laser parameters (``Delta_Hz``; analog_3 also
    its Rabis), so the system is materialized at the terminal step
    (:meth:`set_protocol` or :meth:`build`).  :meth:`RydbergSystem.with_protocol`
    then swaps the pulse freely; for rb87_7 the 420/1013 Rabi scale rides on the
    CZ protocol (``omega_*_max``) against unit blocks, so changing it needs no
    rebuild.
    """

    def __init__(
        self,
        level_structure: str | LevelStructureSpec,
        *,
        param_set: str | None,
        Omega: float,
        level_kwargs: dict,
    ) -> None:
        self._level_structure = level_structure
        self._param_set = param_set
        self._Omega = Omega
        self._level_kwargs = level_kwargs
        self._geometry: Register | None = None
        self._interaction: InteractionSpec | None = None
        self._protocol: "Protocol | None" = None

    def set_atom_geom(
        self, geometry: Register, interaction: InteractionSpec | None = None
    ) -> "_RydbergSystemBuilder":
        """Place the atoms, adding the Rydberg van der Waals interaction."""
        self._geometry = geometry
        self._interaction = interaction
        return self

    def set_protocol(self, protocol: "Protocol") -> "RydbergSystem":
        """Attach the 420/1013 nm drive protocol and materialize the system."""
        self._protocol = protocol
        return self.build()

    def build(self) -> "RydbergSystem":
        """Materialize the system (undriven if no protocol was attached)."""
        geometry = self._geometry if self._geometry is not None else Register.chain(1)
        return _build_from_lattice(
            RydbergSystem,
            geometry,
            self._level_structure,
            self._interaction,
            protocol=self._protocol,
            param_set=self._param_set,
            Omega=self._Omega,
            **self._level_kwargs,
        )


# ── system construction engine ───────────────────────────────────────────────


def _build_from_lattice(
    cls,
    geometry: Register,
    level_structure: str | LevelStructureSpec = "1r",
    interaction: InteractionSpec | None = None,
    *,
    protocol: "Protocol | None" = None,
    param_set: str | None = None,
    Omega: float = 1.0,
    **physical_params,
):
    spec = (
        level_structure
        if isinstance(level_structure, LevelStructureSpec)
        else level_structure_preset(level_structure)
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
    if spec.rydberg_levels:
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


def _interaction_pairs(geometry: Register, interaction: InteractionSpec) -> tuple:
    if interaction.mode == "all":
        return vdw_couplings(geometry.coords, interaction.C6, interaction.max_range_um)

    coords = np.asarray(geometry.coords, dtype=float)
    spacing = geometry.spacing_um or min(
        (d for _, _, d in geometry.distance_pairs()), default=0.0
    )
    max_dist = spacing * (1.01 if interaction.mode == "nn" else np.sqrt(2) * 1.01)
    max_range = interaction.max_range_um if interaction.max_range_um is not None else max_dist
    return vdw_couplings(coords, interaction.C6, max_range)


def _physical_model_for(level_name: str, param_set: str | None) -> str | None:
    # Preset *names* carry Hamiltonian semantics (stageplans/README D11/D13):
    # physical analog-3 construction is the `analog_3` preset only; custom
    # LevelStructureSpec instances always build symbolically.
    if level_name == "analog_3":
        return "analog_3"
    if level_name == "rb87_7" and param_set in {"our", "lukin"}:
        return param_set
    return None


def _default_interaction_for_physical_model(physical_model: str | None) -> InteractionSpec:
    if physical_model == "lukin":
        return InteractionSpec(C6=_rb87_default_c6("lukin"))
    return InteractionSpec(C6=DEFAULT_C6)
