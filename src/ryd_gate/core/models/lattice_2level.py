"""2-level lattice SystemModel wrapper."""

from __future__ import annotations

from ryd_gate.core.atomic_system import LatticeSystem
from ryd_gate.core.basis import BasisSpec
from ryd_gate.core.blocks import BlockRegistry
from ryd_gate.core.observables import ObservableRegistry
from ryd_gate.core.system_model import SystemModel


class Lattice2LevelModel(SystemModel):
    """N-atom 2-level lattice model wrapping an existing LatticeSystem.

    Parameters
    ----------
    system : LatticeSystem
        A pre-built lattice system from :func:`create_lattice_system`.
    """

    def __init__(self, system: LatticeSystem) -> None:
        self._system = system
        N = system.N

        # Basis
        self._basis = BasisSpec(
            site_labels=tuple(str(i) for i in range(N)),
            local_levels=("g", "r"),
            local_dim=2,
            total_dim=2 ** N,
        )

        # Blocks
        self._blocks = BlockRegistry()
        self._blocks.register("global_X", system.sum_X, description="global sigma_x drive")
        self._blocks.register("global_n", system.sum_n, description="global occupation number")
        self._blocks.register("H_vdw", system.H_vdw, description="van der Waals interaction")
        for i in range(N):
            self._blocks.register(f"n_{i}", system.n_list[i], description=f"occupation number on site {i}")

        # Observables
        self._observables = ObservableRegistry()
        for i in range(N):
            self._observables.register(
                f"n_r_{i}",
                system.n_list[i],
                description=f"Rydberg occupation on site {i}",
                per_site=True,
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
        return "lattice"

    # -- Convenience --

    @property
    def system(self) -> LatticeSystem:
        """Access the underlying LatticeSystem dataclass."""
        return self._system
