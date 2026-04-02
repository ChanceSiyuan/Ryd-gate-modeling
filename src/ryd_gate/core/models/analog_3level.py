"""3-level analog two-atom SystemModel wrapper."""

from __future__ import annotations

import numpy as np

from ryd_gate.core.atomic_system import AtomicSystem, create_analog_system
from ryd_gate.core.basis import BasisSpec
from ryd_gate.core.blocks import BlockRegistry
from ryd_gate.core.observables import ObservableRegistry
from ryd_gate.core.operators import (
    build_occ_operator,
    build_atom_a_projector,
    build_atom_b_projector,
    build_product_state_map,
)
from ryd_gate.core.system_model import SystemModel


class Analog3LevelModel(SystemModel):
    """3-level analog two-atom model wrapping an AtomicSystem (param_set ``"analog"``).

    Level structure (per atom):
        |g> (index 0), |e> (index 1), |r> (index 2)

    Parameters
    ----------
    system : AtomicSystem
        A pre-built 3-level AtomicSystem (param_set ``"analog"``).
    """

    def __init__(self, system: AtomicSystem) -> None:
        if system.n_levels != 3:
            raise ValueError(
                f"Analog3LevelModel requires a 3-level system, got n_levels={system.n_levels}. "
                "Use Rb87TwoAtomModel for 7-level systems."
            )
        self._system = system

        # Basis
        self._basis = BasisSpec(
            site_labels=("A", "B"),
            local_levels=("g", "e", "r"),
            local_dim=3,
            total_dim=9,
        )

        # Blocks
        self._blocks = BlockRegistry()
        self._blocks.register("H_const", system.tq_ham_const, description="constant Hamiltonian terms")
        self._blocks.register("H_1013", system.tq_ham_1013, description="1013nm coupling", hermitian=False)
        self._blocks.register("H_1013_conj", system.tq_ham_1013_conj, description="1013nm coupling conjugate", hermitian=False)
        self._blocks.register("drive_420", system.tq_ham_420, description="420nm drive", hermitian=False)
        self._blocks.register("drive_420_dag", system.tq_ham_420_conj, description="420nm drive conjugate transpose", hermitian=False)
        self._blocks.register("lightshift_zero", system.tq_ham_lightshift_zero, description="zero-state light shift")

        # Observables
        self._observables = ObservableRegistry()

        # Total population per level (summed over both atoms)
        self._observables.register("pop_g", build_occ_operator(0, n_levels=3), description="total population in |g>")
        self._observables.register("pop_e", build_occ_operator(1, n_levels=3), description="total population in |e>")
        self._observables.register("pop_r", build_occ_operator(2, n_levels=3), description="total population in |r>")

        # Per-atom projectors
        self._observables.register("pop_A_g", build_atom_a_projector(0, n_levels=3), description="atom A in |g>", per_site=True)
        self._observables.register("pop_A_r", build_atom_a_projector(2, n_levels=3), description="atom A in |r>", per_site=True)
        self._observables.register("pop_B_g", build_atom_b_projector(0, n_levels=3), description="atom B in |g>", per_site=True)
        self._observables.register("pop_B_r", build_atom_b_projector(2, n_levels=3), description="atom B in |r>", per_site=True)

        # Joint two-atom state projectors |ij><ij|
        product_states = build_product_state_map(n_levels=3)
        for label, vec in product_states.items():
            proj = np.outer(vec, vec.conj())
            self._observables.register(
                f"pop_{label}", proj,
                description=f"joint population in |{label}>",
            )

    # -- SystemModel interface --

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
        return self._system.param_set

    # -- Convenience --

    @property
    def system(self) -> AtomicSystem:
        """Access the underlying AtomicSystem dataclass."""
        return self._system

    @classmethod
    def from_defaults(cls, **kwargs) -> Analog3LevelModel:
        """Create an analog model with default parameters.

        Parameters
        ----------
        **kwargs
            Forwarded to ``create_analog_system``.
        """
        system = create_analog_system(**kwargs)
        return cls(system)
