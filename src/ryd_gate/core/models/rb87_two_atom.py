"""Rb87 7-level two-atom SystemModel wrapper."""

from __future__ import annotations

from ryd_gate.core.atomic_system import AtomicSystem, create_our_system, create_lukin_system
from ryd_gate.core.basis import BasisSpec
from ryd_gate.core.blocks import BlockRegistry
from ryd_gate.core.observables import ObservableRegistry
from ryd_gate.core.operators import build_occ_operator, build_atom_a_projector, build_atom_b_projector
from ryd_gate.core.system_model import SystemModel

_LEVEL_LABELS_7 = ("0", "1", "e1", "e2", "e3", "r", "r_garb")


class Rb87TwoAtomModel(SystemModel):
    """7-level two-atom Rb87 model wrapping an existing AtomicSystem.

    Suitable for param_set ``"our"`` or ``"lukin"`` (both use 7-level basis).
    For the 3-level analog system, use :class:`Analog3LevelModel` instead.

    Parameters
    ----------
    system : AtomicSystem
        A pre-built 7-level AtomicSystem (param_set ``"our"`` or ``"lukin"``).
    """

    def __init__(self, system: AtomicSystem) -> None:
        if system.n_levels != 7:
            raise ValueError(
                f"Rb87TwoAtomModel requires a 7-level system, got n_levels={system.n_levels}. "
                "Use Analog3LevelModel for 3-level systems."
            )
        self._system = system

        # Basis
        self._basis = BasisSpec(
            site_labels=("A", "B"),
            local_levels=_LEVEL_LABELS_7,
            local_dim=7,
            total_dim=49,
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

        # Per-level total population (summed over both atoms)
        for i, label in enumerate(_LEVEL_LABELS_7):
            self._observables.register(
                f"pop_{label}",
                build_occ_operator(i, n_levels=7),
                description=f"total population in level |{label}>",
            )

        # Per-atom projectors
        self._observables.register("pop_A_ground", build_atom_a_projector(0, n_levels=7), description="atom A in |0>", per_site=True)
        self._observables.register("pop_A_qubit", build_atom_a_projector(1, n_levels=7), description="atom A in |1>", per_site=True)
        self._observables.register("pop_B_ground", build_atom_b_projector(0, n_levels=7), description="atom B in |0>", per_site=True)
        self._observables.register("pop_B_qubit", build_atom_b_projector(1, n_levels=7), description="atom B in |1>", per_site=True)
        self._observables.register("pop_A_ryd", build_atom_a_projector(5, n_levels=7), description="atom A in |r>", per_site=True)
        self._observables.register("pop_B_ryd", build_atom_b_projector(5, n_levels=7), description="atom B in |r>", per_site=True)

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
    def from_param_set(cls, name: str, **kwargs) -> Rb87TwoAtomModel:
        """Create a model from a named parameter set.

        Parameters
        ----------
        name : str
            ``"our"`` or ``"lukin"``.
        **kwargs
            Forwarded to ``create_our_system`` or ``create_lukin_system``.
        """
        factories = {
            "our": create_our_system,
            "lukin": create_lukin_system,
        }
        if name not in factories:
            raise ValueError(f"Unknown param_set '{name}'. Choose from {list(factories.keys())}.")
        system = factories[name](**kwargs)
        return cls(system)
