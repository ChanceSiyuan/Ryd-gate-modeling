"""Block registry for Hamiltonian operator components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BlockInfo:
    """Metadata for a registered Hamiltonian block."""

    name: str
    operator: Any  # ndarray or sparse matrix
    description: str = ""
    hermitian: bool = True


class BlockRegistry:
    """Dict-like container mapping string names to operator matrices.

    Stores Hamiltonian building blocks (e.g. "drive_420", "H_const", "H_vdw")
    with metadata. Operators can be dense numpy arrays or scipy sparse matrices.
    """

    def __init__(self) -> None:
        self._blocks: dict[str, BlockInfo] = {}

    def register(
        self,
        name: str,
        operator: Any,
        description: str = "",
        hermitian: bool = True,
    ) -> None:
        """Register an operator block."""
        self._blocks[name] = BlockInfo(
            name=name,
            operator=operator,
            description=description,
            hermitian=hermitian,
        )

    def get(self, name: str) -> Any:
        """Get the operator matrix for a block. Raises KeyError if not found."""
        return self._blocks[name].operator

    def get_info(self, name: str) -> BlockInfo:
        """Get full BlockInfo for a block."""
        return self._blocks[name]

    def list(self) -> list[str]:
        """Return names of all registered blocks."""
        return list(self._blocks.keys())

    def has(self, name: str) -> bool:
        """Check if a block is registered."""
        return name in self._blocks

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __len__(self) -> int:
        return len(self._blocks)
