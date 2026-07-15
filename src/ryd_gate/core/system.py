"""Universal Rydberg system model.

A :class:`RydbergSystem` is constructed explicitly from four things::

    RydbergSystem(
        level_structure=level_structure("rb87_7_mp", ...),  # resolved preset
        register=Register.chain(2, spacing_um=3.0),          # geometry
        interaction=InteractionSpec(...),                    # optional override
        protocol=protocol,                                   # optional binding
    )

Everything else is derived and exposed as explicit read-only fields:
``basis``, ``observables``, ``interaction_pairs``, ``t_gate`` (the bound
protocol's duration resolved against this system), and the physical data on
``level_structure``.  Backend-specific compilers materialize the symbolic
blocks into matrices, MPOs, or other solver inputs only when needed.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from ryd_gate.core.level_structures import (
    DEFAULT_C6,
    InteractionSpec,
    LevelStructure,
)
from ryd_gate.core.model import BasisSpec
from ryd_gate.core.observables import ObservableFactory
from ryd_gate.core.operators import (
    BasisOperatorFactory,
    StaticHamiltonianTerm,
)
from ryd_gate.core.physical_models import (
    vdw_couplings,
    vdw_couplings_from_c6_function,
)
from ryd_gate.lattice import Register

if TYPE_CHECKING:
    from ryd_gate.protocols.base import Protocol


class RydbergSystem:
    """Universal Rydberg-lattice system: level structure + register + protocol.

    Constructed explicitly; every field is read-only after construction.  A
    system with a protocol bound can be passed to
    :func:`ryd_gate.ir.compile_hamiltonian_ir`; algorithm packages then lower
    that unified Hamiltonian IR into exact matrices, MPS/MPO data, or external
    solver payloads.  ``with_protocol`` returns a new system with another
    fully specified protocol bound.
    """

    def __init__(
        self,
        *,
        level_structure: LevelStructure,
        register: Register,
        interaction: InteractionSpec | None = None,
        protocol: "Protocol | None" = None,
    ) -> None:
        if not isinstance(level_structure, LevelStructure):
            raise TypeError(
                "level_structure must be a resolved preset from "
                "ryd_gate.level_structure(name, **physical_kwargs); got "
                f"{type(level_structure).__name__}."
            )
        if not isinstance(register, Register):
            raise TypeError(f"register must be a Register; got {type(register).__name__}.")
        if interaction is not None and not isinstance(interaction, InteractionSpec):
            raise TypeError(
                f"interaction must be an InteractionSpec or None; got {type(interaction).__name__}."
            )

        ls = level_structure
        n = register.N
        d = ls.local_dim
        self._level_structure = ls
        self._register = register
        self._protocol = protocol
        self._amplitude_scale = 1.0
        self._is_sparse = True

        self._basis = BasisSpec(
            site_labels=tuple(str(i) for i in range(n)),
            local_levels=ls.levels,
            local_dim=d,
            total_dim=d**n,
        )
        operators = BasisOperatorFactory(self._basis)
        self._operators = operators

        # Observables: an immutable expression factory bound to this basis
        # (system.observables.n("r", 0), .level_sum("r"), .identity(), ...).
        self._observables = ObservableFactory(self._basis)

        # Hamiltonian channels: driveable primitive E[ket,bra] operators (summed
        # over sites; per-site variants E[...]_<i> are resolved by the IR).
        channels: dict[str, Any] = {}
        for transition in ls.transitions:
            channels[transition.channel] = operators.sum(transition.channel)
        for channel in ls.detuning_levels:
            channels[channel] = operators.sum(channel)
        self._hamiltonian_channels = channels

        # Interaction pairs: explicit InteractionSpec wins; else the preset's
        # anisotropic pair-C6 function (297) or isotropic default C6.
        self._interaction_pairs = _resolve_interaction_pairs(ls, register, interaction)

        # Static terms: preset single-atom energies + static couplings + the
        # Rydberg pair interaction.
        static_terms: list[StaticHamiltonianTerm] = []
        if ls.rydberg_levels:
            static_terms.append(
                StaticHamiltonianTerm(
                    "H_pair",
                    operators.pair_projector(self._interaction_pairs, ls.rydberg_levels),
                    1.0,
                )
            )
        if ls.local_static is not None:
            for i, level in enumerate(ls.levels):
                coeff = complex(ls.local_static[i, i])
                if coeff != 0:
                    name = f"E[{level},{level}]"
                    static_terms.append(
                        StaticHamiltonianTerm(name, operators.sum(name), coeff)
                    )
        for channel, coeff in ls.static_couplings:
            static_terms.append(
                StaticHamiltonianTerm(
                    channel, operators.sum(channel), coeff, add_hermitian_conjugate=True
                )
            )
        self._static_hamiltonian_terms = static_terms

    # -- Explicit read-only fields -------------------------------------------

    @property
    def level_structure(self) -> LevelStructure:
        """The resolved level-structure preset (physical data lives here)."""
        return self._level_structure

    @property
    def register(self) -> Register:
        return self._register

    @property
    def interaction_pairs(self) -> tuple:
        """Resolved ``(i, j, V_ij)`` Rydberg pair interactions (rad/s)."""
        return self._interaction_pairs

    @property
    def protocol(self) -> "Protocol | None":
        return self._protocol

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
    def observables(self) -> ObservableFactory:
        """Read-only :class:`ObservableExpr` factory bound to this system's basis."""
        return self._observables

    @property
    def N(self) -> int:
        return self._basis.n_sites

    @property
    def dim(self) -> int:
        return self._basis.total_dim

    @property
    def is_sparse(self) -> bool:
        return self._is_sparse

    @property
    def amplitude_scale(self) -> float:
        """Internal noise-realization amplitude scale (1.0 outside ensembles)."""
        return self._amplitude_scale

    # Kept as an attribute-style alias some plotting/analysis code reads.
    @property
    def geometry(self) -> Register:
        return self._register

    # -- Protocol binding --------------------------------------------------

    def with_protocol(self, protocol: "Protocol") -> "RydbergSystem":
        """Return a shallow copy with another fully specified protocol bound."""
        new = copy.copy(self)
        new._protocol = protocol
        return new

    def _with_amplitude_scale(self, amplitude_scale: float) -> "RydbergSystem":
        """Return a shallow copy with ``amplitude_scale`` replaced.

        Private: noise-realization amplitude scaling is a temporary quantity
        inside ensemble execution, never public system state.
        """
        new = copy.copy(self)
        new._amplitude_scale = float(amplitude_scale)
        return new

    def _require_protocol(self) -> "Protocol":
        if self._protocol is None:
            raise ValueError(
                "RydbergSystem has no protocol bound. Construct with "
                "`protocol=...` or call `.with_protocol(...)` before compiling."
            )
        return self._protocol

    # -- Solver-facing API -------------------------------------------------

    @property
    def t_gate(self) -> float:
        """The bound protocol's gate duration resolved against this system (s).

        The protocol decides the duration; normalized quantities (e.g. a CZ
        ``duration_ratio``) resolve against this system's physical scales.
        Read-only — ``simulate()`` cannot override it.
        """
        return self._require_protocol().resolve_t_gate(self)

    def product_state(self, config: str | list[str] | tuple[str, ...]) -> np.ndarray:
        """Return a computational product state in this system's basis.

        The canonical seam for constructing bra/kets by the system's
        basis/site ordering (site 0 is the most significant digit): scripts
        use it to read CZ return amplitudes, logical leakage, and Bell/SSS
        overlaps off ``final_state``.
        """
        labels = list(config) if not isinstance(config, str) else list(config)
        if len(labels) != self._basis.n_sites:
            raise ValueError(
                f"config must have length {self._basis.n_sites}, got {len(labels)}."
            )
        idx = 0
        d = self._basis.local_dim
        for site_i, label in enumerate(labels):
            idx += self._basis.level_index(label) * d ** (self._basis.n_sites - 1 - site_i)
        psi = np.zeros(self._basis.total_dim, dtype=complex)
        psi[idx] = 1.0
        return psi

    def ground_state(self) -> np.ndarray:
        return self.product_state(
            [self._basis.local_levels[0]] * self._basis.n_sites
        )


# ── interaction-pair resolution ──────────────────────────────────────────────


def _resolve_interaction_pairs(
    ls: LevelStructure, register: Register, interaction: InteractionSpec | None
) -> tuple:
    if interaction is None:
        if ls.pair_c6_fn is not None:
            return vdw_couplings_from_c6_function(register.coords, ls.pair_c6_fn)
        interaction = InteractionSpec(
            C6=ls.default_c6 if ls.default_c6 is not None else DEFAULT_C6
        )

    if interaction.mode == "all":
        return vdw_couplings(register.coords, interaction.C6, interaction.max_range_um)

    coords = np.asarray(register.coords, dtype=float)
    spacing = register.spacing_um or min(
        (dist for _, _, dist in register.distance_pairs()), default=0.0
    )
    max_dist = spacing * (1.01 if interaction.mode == "nn" else np.sqrt(2) * 1.01)
    max_range = interaction.max_range_um if interaction.max_range_um is not None else max_dist
    return vdw_couplings(coords, interaction.C6, max_range)
