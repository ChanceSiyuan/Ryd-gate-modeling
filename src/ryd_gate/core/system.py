"""Universal Rydberg system model.

A :class:`RydbergSystem` is built from three things: a lattice geometry,
a local energy-level structure (``1r`` / ``01r`` / ``ger`` / ``rb87_7``;
``analog_3`` is a physical ``ger`` preset), and a pulse protocol.
The class owns symbolic Hamiltonian blocks, observables, geometry metadata,
and the bound protocol. Backend-specific compilers materialize those symbolic
blocks into matrices, MPOs, or other solver inputs only when needed.

Level-structure/interaction specs live in :mod:`ryd_gate.core.level_structures`,
the Rb87 physical parameter sets in :mod:`ryd_gate.core.rb87_params`, the local
matrix blocks in :mod:`ryd_gate.core.local_blocks`, and the ``from_lattice``
construction logic in :mod:`ryd_gate.core.factories`.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from ryd_gate.core.basis import BasisSpec
from ryd_gate.core.blocks import BlockRegistry
from ryd_gate.core.level_structures import InteractionSpec, LevelStructureSpec
from ryd_gate.core.observables import ObservableRegistry
from ryd_gate.core.operator_spec import (
    is_operator_spec,
    measure_state_vector_operator,
)
from ryd_gate.core.system_model import SystemModel
from ryd_gate.lattice.geometry import Register

if TYPE_CHECKING:
    from ryd_gate.protocols.base import Protocol


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
        geometry: Register,
        level_structure: str | LevelStructureSpec = "1r",
        interaction: InteractionSpec | None = None,
        *,
        protocol: "Protocol | None" = None,
        param_set: str | None = None,
        Omega: float = 1.0,
        **physical_params,
    ) -> "RydbergSystem":
        from ryd_gate.core.factories import build_from_lattice

        return build_from_lattice(
            cls,
            geometry,
            level_structure,
            interaction,
            protocol=protocol,
            param_set=param_set,
            Omega=Omega,
            **physical_params,
        )
