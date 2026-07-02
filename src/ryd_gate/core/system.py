"""Universal Rydberg system model and its lattice construction.

A :class:`RydbergSystem` is built from three things: a lattice geometry,
a local energy-level structure (one of the built-in models in
:data:`_ATOM_LEVELS` — ``01`` / ``1r`` / ``01r`` / ``analog_3`` / ``rb87_7_mp``
/ ``rb87_7_pm`` — or a hand-built :class:`LevelStructureSpec`), and a protocol.
The class owns symbolic Hamiltonian blocks, observables, geometry metadata,
and the bound protocol. Backend-specific compilers materialize those symbolic
blocks into matrices, MPOs, or other solver inputs only when needed.

Level-structure/interaction specs live in :mod:`ryd_gate.core.level_structures`
and the Rb87 physical parameter sets and local matrix blocks in
:mod:`ryd_gate.core.physical_models`. A system is built with the fluent
``RydbergSystem.set_atom_level(...).set_atom_geom(...).set_protocol(...)``
chain: every step returns a fully materialized, usable system.
:meth:`RydbergSystem.set_atom_level` resolves the level spec against
:data:`_ATOM_LEVELS` (kind, allowed atom-level kwargs), validates the
construction parameters, and registers the symbolic blocks and observables —
mounting the physical Rb87 blocks for ``analog_3`` / ``rb87_7_mp`` / ``rb87_7_pm``.
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
    ObservableRegistry,
    SystemModel,
)
from ryd_gate.core.operators import (
    BasisOperatorFactory,
    StaticHamiltonianTerm,
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
# The built-in single-atom level structures and their construction contracts.
# ``kind`` selects the build path: "symbolic" registers only the preset's
# symbolic blocks, while "analog_3"/"rb87_7" additionally mount the physical Rb87
# local Hamiltonian blocks (the seven-level manifold is encoded in the tag:
# rb87_7_mp / rb87_7_pm).  ``level_kwargs`` is the exact set of atom-level physics
# arguments the model accepts (it mirrors the ``_apply_*`` signatures in
# physical_models.py); anything else raises.  Physical mounting is keyed off the
# preset *name*, so a hand-built LevelStructureSpec (absent from this table) is
# always symbolic.
_ANALOG3_KWARGS = frozenset({
    "detuning_sign", "Delta_Hz", "rabi_420_Hz", "rabi_1013_Hz",
    "enable_rydberg_decay", "enable_intermediate_decay",
})
_RB87_7_KWARGS = frozenset({
    "detuning_sign", "Delta_Hz", "ryd_level", "C6_rad_s_um6", "t_rise",
    "enable_rydberg_decay", "enable_intermediate_decay", "magnetic_field_G",
})

_ATOM_LEVELS: dict[str, dict[str, Any]] = {
    "01": {
        "kind": "symbolic", "level_kwargs": frozenset(),
        "description": "qubit |0>,|1>, no Rydberg (stabilizer-capable)",
    },
    "1r": {
        "kind": "symbolic", "level_kwargs": frozenset(),
        "description": "two-level |1>,|r> Rydberg drive",
    },
    "01r": {
        "kind": "symbolic", "level_kwargs": frozenset(),
        "description": "three-level |0>,|1>,|r> effective CZ subspace",
    },
    "analog_3": {
        "kind": "analog_3", "level_kwargs": _ANALOG3_KWARGS,
        "description": "physical Rb87 g/e/r ladder with static 1013 nm e-r coupling",
    },
    # Seven-level Rb87 gate model split by manifold/polarization convention:
    # _mp = sigma-(420)/sigma+(1013) (was param_set="our"); _pm = sigma+/sigma-
    # (was param_set="lukin").  Static numbers are explicit kwargs, no param_set.
    "rb87_7_mp": {
        "kind": "rb87_7", "level_kwargs": _RB87_7_KWARGS,
        "description": "physical Rb87 seven-level model, sigma-/sigma+ manifold",
    },
    "rb87_7_pm": {
        "kind": "rb87_7", "level_kwargs": _RB87_7_KWARGS,
        "description": "physical Rb87 seven-level model, sigma+/sigma- manifold",
    },
}


class RydbergSystem(SystemModel):
    """Universal Rydberg-lattice system: geometry + level structure + protocol.

    Built via :meth:`set_atom_level` (then ``.set_atom_geom(...)`` and
    optionally ``.set_protocol(...)``); every step returns a fully materialized,
    usable system. Once a protocol is attached (via
    :meth:`set_protocol` or :meth:`with_protocol`), the system can be passed to
    :func:`ryd_gate.ir.compile_hamiltonian_ir`. Algorithm packages then lower
    that unified Hamiltonian IR into exact matrices, MPS/MPO data, TTN inputs,
    or external solver payloads.
    """

    def __init__(
        self,
        *,
        basis: BasisSpec,
        operators: BasisOperatorFactory,
        hamiltonian_channels: dict[str, Any],
        static_hamiltonian_terms: list[StaticHamiltonianTerm],
        observables: ObservableRegistry,
        protocol: "Protocol | None" = None,
        metadata: dict[str, Any] | None = None,
        geometry: Register | None = None,
        is_sparse: bool = True,
        amplitude_scale: float = 1.0,
    ) -> None:
        self._basis = basis
        self._operators = operators
        self._hamiltonian_channels = hamiltonian_channels
        self._static_hamiltonian_terms = static_hamiltonian_terms
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
    def operators(self) -> BasisOperatorFactory:
        """Primitive ``E[ket,bra]`` operator factory generated from the basis."""
        return self._operators

    @property
    def hamiltonian_channels(self) -> dict[str, Any]:
        """Driveable primitive operators keyed by ``E[ket,bra]`` channel name."""
        return self._hamiltonian_channels

    @property
    def static_hamiltonian_terms(self) -> list[StaticHamiltonianTerm]:
        """Static (protocol-independent) Hamiltonian terms (diagonal energies,
        Rydberg pair interaction, static couplings)."""
        return self._static_hamiltonian_terms

    @property
    def observables(self) -> ObservableRegistry:
        return self._observables

    @property
    def model_tag(self) -> str:
        """Level-structure tag identifying this system's atom model.

        The built-in/preset name (``rb87_7_mp``, ``analog_3``, ``1r`` ...) or a
        custom :class:`LevelStructureSpec`'s ``name`` — read from metadata, not a
        separate ``param_set`` selector.
        """
        return self.metadata.get("level_structure", "")

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
        Omega: float = 1.0,
        **level_kwargs,
    ) -> "RydbergSystem":
        """
        Declare the single-atom level structure, returning a usable system.
        """
        return cls._materialize(
            level_structure=level_structure,
            omega=Omega,
            level_kwargs=dict(level_kwargs),
            geometry=Register.chain(1),
            interaction=None,
            protocol=None,
        )

    def set_atom_geom(
        self, geometry: Register, interaction: InteractionSpec | None = None
    ) -> "RydbergSystem":
        """Place the atoms, adding the Rydberg van der Waals interaction.

        Returns a new, fully materialized system rebuilt from this system's
        atom-level config (level structure, physical flags) with only the
        geometry/interaction replaced and any bound protocol preserved. The
        receiver is left unchanged.
        """
        return type(self)._materialize(
            level_structure=self._level_structure,
            omega=self._omega,
            level_kwargs=self._level_kwargs,
            geometry=geometry,
            interaction=interaction,
            protocol=self.protocol,
        )

    def set_protocol(self, protocol: "Protocol") -> "RydbergSystem":
        """Bind the drive protocol, returning a usable system.

        Protocol binding never affects the blocks/metadata (the physical
        transition blocks are unit-Rabi; the protocol supplies the time-dependent
        ``A e^{i phi}`` coefficients at compile time), so this just delegates to
        :meth:`with_protocol`.
        """
        return self.with_protocol(protocol)

    @classmethod
    def _materialize(
        cls,
        *,
        level_structure: str | LevelStructureSpec,
        omega: float,
        level_kwargs: dict,
        geometry: Register,
        interaction: InteractionSpec | None,
        protocol: "Protocol | None",
    ) -> "RydbergSystem":
        """Construct a complete system from an explicit construction config.

        Resolves the level spec against :data:`_ATOM_LEVELS`, validates the
        atom-level kwargs, registers the symbolic blocks and observables, mounts
        the physical Rb87 blocks for ``analog_3`` / ``rb87_7_mp`` / ``rb87_7_pm``,
        and stashes the construction config on the returned system so
        :meth:`set_atom_geom` can rebuild from it.
        """
        spec = (
            level_structure
            if isinstance(level_structure, LevelStructureSpec)
            else level_structure_preset(level_structure)
        )
        entry = _ATOM_LEVELS.get(spec.name)
        kind = entry["kind"] if entry is not None else "symbolic"
        allowed_kwargs = entry["level_kwargs"] if entry is not None else frozenset()

        unknown = set(level_kwargs) - allowed_kwargs
        if unknown:
            allowed = ", ".join(sorted(allowed_kwargs)) or "none"
            raise TypeError(
                f"{spec.name} does not accept atom-level parameter(s): "
                f"{', '.join(sorted(unknown))}. Allowed parameters: {allowed}."
            )

        # rb87 manifold ("mp"/"pm") from the tag; static numbers are level_kwargs.
        manifold = spec.name.removeprefix("rb87_7_") if kind == "rb87_7" else None

        if interaction is None:
            if kind == "rb87_7":
                c6 = level_kwargs.get("C6_rad_s_um6")
                interaction = InteractionSpec(
                    C6=c6 if c6 is not None else _rb87_default_c6(manifold)
                )
            else:
                interaction = InteractionSpec(C6=DEFAULT_C6)

        d = spec.local_dim
        N = geometry.N
        dim = d**N
        basis = BasisSpec(
            site_labels=tuple(str(i) for i in range(N)),
            local_levels=spec.levels,
            local_dim=d,
            total_dim=dim,
        )

        operators = BasisOperatorFactory(basis)
        observables = ObservableRegistry()

        # Observables: friendly user-facing names, all built from the factory.
        for level in spec.levels:
            ee = f"E[{level},{level}]"
            for i in range(N):
                observables.register(
                    f"n_{level}_{i}",
                    operators.local(ee, i),
                    description=f"|{level}> population on site {i}",
                    per_site=True,
                )
            observables.register(
                f"sum_n_{level}", operators.sum(ee), description=f"total |{level}> population"
            )
        if "r" in spec.levels:
            observables.register("sum_nr", operators.sum("E[r,r]"), description="total Rydberg population")
        if geometry.sublattice is not None and np.any(geometry.sublattice) and "r" in spec.levels:
            observables.register(
                "staggered_rydberg",
                operators.weighted_sum("E[r,r]", tuple(float(x) for x in geometry.sublattice)),
                description="staggered Rydberg occupation",
            )

        # Hamiltonian channels: driveable primitive E[ket,bra] operators (summed
        # over sites; per-site variants E[...]_<i> are resolved by the IR).
        hamiltonian_channels: dict[str, Any] = {}
        for transition in spec.transitions:
            hamiltonian_channels[transition.channel] = operators.sum(transition.channel)
        for channel in spec.detuning_levels:
            hamiltonian_channels[channel] = operators.sum(channel)

        # Static terms: the Rydberg pair interaction (+ physical-model diagonal
        # energies / static couplings appended by the _apply_* helpers below).
        pairs = _interaction_pairs(geometry, interaction)
        static_hamiltonian_terms: list[StaticHamiltonianTerm] = []
        if spec.rydberg_levels:
            static_hamiltonian_terms.append(
                StaticHamiltonianTerm(
                    "H_pair", operators.pair_projector(pairs, spec.rydberg_levels), 1.0
                )
            )

        metadata = {
            "level_structure": spec.name,
            "level_spec": spec,
            "interaction_pairs": pairs,
            "Omega": omega,
            "local_dim": d,
            "n_sites": N,
        }
        model = RydbergSystem(
            basis=basis,
            operators=operators,
            hamiltonian_channels=hamiltonian_channels,
            static_hamiltonian_terms=static_hamiltonian_terms,
            observables=observables,
            protocol=protocol,
            metadata=metadata,
            geometry=geometry,
            is_sparse=True,
        )
        if kind == "analog_3":
            _apply_analog_3_lattice_blocks(model, **level_kwargs)
        elif kind == "rb87_7":
            _apply_rb87_7_lattice_blocks(model, manifold, **level_kwargs)

        # Construction config for set_atom_geom() rebuilds.
        model._level_structure = level_structure
        model._omega = omega
        model._level_kwargs = level_kwargs
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
