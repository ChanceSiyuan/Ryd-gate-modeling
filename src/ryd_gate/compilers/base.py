"""Abstract base class for Hamiltonian compilers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ryd_gate.compilers.ir import HamiltonianIR
    from ryd_gate.protocols.base import Protocol


class Compiler(ABC):
    """Compiles a system + protocol + params into a HamiltonianIR.

    The compiler maps protocol drive channels to system block operators,
    producing a solver-agnostic intermediate representation.
    """

    @abstractmethod
    def compile(self, system, protocol: Protocol, params: dict) -> HamiltonianIR:
        """Build a HamiltonianIR from system blocks and protocol coefficients.

        Parameters
        ----------
        system : AtomicSystem or LatticeSystem (or SystemModel in future)
            The physical system with precomputed operator blocks.
        protocol : Protocol
            The evolution protocol providing drive coefficients.
        params : dict
            Unpacked parameters from ``protocol.unpack_params(x, system)``.

        Returns
        -------
        HamiltonianIR
            Solver-ready Hamiltonian description.
        """
        ...
