"""Universal Rydberg system model and its lattice construction.

A :class:`RydbergSystem` is built from three things: a lattice geometry,
a local energy-level structure (one of the five built-in models in
:data:`_ATOM_LEVELS` — ``01`` / ``1r`` / ``01r`` / ``analog_3`` / ``rb87_7`` —
or a hand-built :class:`LevelStructureSpec`), and a pulse protocol.
The class owns symbolic Hamiltonian blocks, observables, geometry metadata,
and the bound protocol. Backend-specific compilers materialize those symbolic
blocks into matrices, MPOs, or other solver inputs only when needed.

Level-structure/interaction specs live in :mod:`ryd_gate.core.level_structures`
and the Rb87 physical parameter sets and local matrix blocks in
:mod:`ryd_gate.core.physical_models`. A system is built with the fluent
``RydbergSystem.set_atom_level(...).set_atom_geom(...).set_protocol(...)``
chain: :meth:`RydbergSystem.set_atom_level` returns a *pending* system, and the
terminal :meth:`RydbergSystem.build` resolves the level spec against
:data:`_ATOM_LEVELS` (kind, allowed atom-level kwargs, allowed ``param_set``),
validates the construction parameters, and registers the symbolic blocks and
observables — mounting the physical Rb87 blocks for ``analog_3`` / ``rb87_7``.
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


# ── Built-in atom-level models ───────────────────────────────────────────────
# The five built-in single-atom level structures and their construction
# contracts.  ``kind`` selects the build path: "symbolic" registers only the
# preset's symbolic blocks, while "analog_3"/"rb87_7" additionally mount the
# physical Rb87 local Hamiltonian blocks.  ``level_kwargs`` is the exact set of
# atom-level physics arguments the model accepts (it mirrors the ``_apply_*``
# signatures in physical_models.py); anything else raises.  ``param_sets`` is
# ``None`` for symbolic models (``param_set`` is then only a label) or a tuple of
# allowed values whose first element is the default.  Physical mounting is keyed
# off the preset *name*, never ``param_set``, so a hand-built LevelStructureSpec
# (absent from this table) is always symbolic.
_ANALOG3_KWARGS = frozenset({
    "detuning_sign", "Delta_Hz", "rabi_420_Hz", "rabi_1013_Hz",
    "enable_rydberg_decay", "enable_intermediate_decay",
})
_RB87_7_KWARGS = frozenset({
    "detuning_sign", "Delta_Hz",
    "enable_rydberg_decay", "enable_intermediate_decay", "enable_polarization_leakage",
})

_ATOM_LEVELS: dict[str, dict[str, Any]] = {
    "01": {
        "kind": "symbolic", "level_kwargs": frozenset(), "param_sets": None,
        "description": "qubit |0>,|1>, no Rydberg (stabilizer-capable)",
    },
    "1r": {
        "kind": "symbolic", "level_kwargs": frozenset(), "param_sets": None,
        "description": "two-level |1>,|r> Rydberg drive",
    },
    "01r": {
        "kind": "symbolic", "level_kwargs": frozenset(), "param_sets": None,
        "description": "three-level |0>,|1>,|r> effective CZ subspace",
    },
    "analog_3": {
        "kind": "analog_3", "level_kwargs": _ANALOG3_KWARGS, "param_sets": ("analog_3",),
        "description": "physical Rb87 g/e/r ladder with static H_1013",
    },
    "rb87_7": {
        "kind": "rb87_7", "level_kwargs": _RB87_7_KWARGS, "param_sets": ("our", "lukin"),
        "description": "physical Rb87 seven-level model (param_set our|lukin)",
    },
}


class RydbergSystem(SystemModel):
    """Universal Rydberg-lattice system: geometry + level structure + protocol.

    Built via :meth:`set_atom_level` (then ``.set_atom_geom(...)`` and
    ``.set_protocol(...)`` / ``.build()``). Once a protocol is attached (via
    :meth:`set_protocol` or :meth:`with_protocol`), the system can be passed to
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
    ) -> "RydbergSystem":
        """Start building a system by declaring the single-atom level structure.

        Returns a *pending* :class:`RydbergSystem`; chain :meth:`set_atom_geom`
        (atom positions + Rydberg interaction) then :meth:`set_protocol` (the
        pulse) or :meth:`build`.  ``level_structure`` is one of the built-in names
        in :data:`_ATOM_LEVELS` or a hand-built :class:`LevelStructureSpec`.
        ``level_kwargs`` are atom-level physics arguments (``detuning_sign``,
        ``Delta_Hz``, ``enable_*_decay`` ...) and are validated in :meth:`build`
        against the model's allowed set — ``analog_3``/``rb87_7`` accept their
        laser/decay flags, symbolic models accept none.  ``param_set`` selects the
        numeric set for ``rb87_7`` (``our``/``lukin``); for symbolic models it is
        only a label.  The rb87_7 420/1013 Rabi scale is owned by the CZ protocol
        (``omega_*_max``), not the system.
        """
        obj = cls.__new__(cls)
        obj._pending_level_structure = level_structure
        obj._pending_param_set = param_set
        obj._pending_omega = Omega
        obj._pending_level_kwargs = dict(level_kwargs)
        obj._pending_geometry = None
        obj._pending_interaction = None
        obj._pending_protocol = None
        obj._built = False
        return obj

    def set_atom_geom(
        self, geometry: Register, interaction: InteractionSpec | None = None
    ) -> "RydbergSystem":
        """Place the atoms, adding the Rydberg van der Waals interaction."""
        self._require_pending("set_atom_geom")
        self._pending_geometry = geometry
        self._pending_interaction = interaction
        return self

    def set_protocol(self, protocol: "Protocol") -> "RydbergSystem":
        """Attach the drive protocol and materialize the system."""
        self._require_pending("set_protocol")
        self._pending_protocol = protocol
        return self.build()

    def _require_pending(self, method: str) -> None:
        """Guard the builder methods against use on an already-built system."""
        if getattr(self, "_built", True):
            raise RuntimeError(
                f"{method}() is only valid on a pending builder returned by "
                "RydbergSystem.set_atom_level(...); this system is already built."
            )

    def build(self) -> "RydbergSystem":
        """Materialize the system from the pending config (undriven if no protocol).

        Resolves the level spec against :data:`_ATOM_LEVELS`, validates the
        atom-level kwargs and ``param_set``, registers the symbolic blocks and
        observables, and mounts the physical Rb87 blocks for ``analog_3`` /
        ``rb87_7``.
        """
        self._require_pending("build")
        self._built = True

        level_structure = self._pending_level_structure
        spec = (
            level_structure
            if isinstance(level_structure, LevelStructureSpec)
            else level_structure_preset(level_structure)
        )
        entry = _ATOM_LEVELS.get(spec.name)
        kind = entry["kind"] if entry is not None else "symbolic"
        allowed_kwargs = entry["level_kwargs"] if entry is not None else frozenset()
        param_sets = entry["param_sets"] if entry is not None else None

        level_kwargs = self._pending_level_kwargs
        unknown = set(level_kwargs) - allowed_kwargs
        if unknown:
            allowed = ", ".join(sorted(allowed_kwargs)) or "none"
            raise TypeError(
                f"{spec.name} does not accept atom-level parameter(s): "
                f"{', '.join(sorted(unknown))}. Allowed parameters: {allowed}."
            )

        param_set = self._pending_param_set
        if param_sets is not None:
            if param_set is None:
                param_set = param_sets[0]
            elif param_set not in param_sets:
                raise ValueError(
                    f"{spec.name} param_set must be one of {param_sets}, "
                    f"got {param_set!r}."
                )

        geometry = (
            self._pending_geometry
            if self._pending_geometry is not None
            else Register.chain(1)
        )
        interaction = self._pending_interaction
        if interaction is None:
            interaction = (
                InteractionSpec(C6=_rb87_default_c6("lukin"))
                if kind == "rb87_7" and param_set == "lukin"
                else InteractionSpec(C6=DEFAULT_C6)
            )

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
            "Omega": self._pending_omega,
            "local_dim": d,
            "n_sites": N,
        }
        model = RydbergSystem(
            param_set=param_set or f"lattice_{spec.name}",
            basis=basis,
            blocks=blocks,
            observables=observables,
            protocol=self._pending_protocol,
            metadata=metadata,
            geometry=geometry,
            is_sparse=True,
        )
        if kind == "analog_3":
            _apply_analog_3_lattice_blocks(model, **level_kwargs)
        elif kind == "rb87_7":
            _apply_rb87_7_lattice_blocks(model, param_set, **level_kwargs)
        return model


# ── interaction-pair resolution ──────────────────────────────────────────────


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
