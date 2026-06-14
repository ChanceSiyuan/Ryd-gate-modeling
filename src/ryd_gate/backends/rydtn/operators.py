"""Local operators for the rydtn PEPS engine.

Pure NumPy ``(d, d)`` complex matrices built from the ``levels`` tuple, mirroring
``ryd_gate.backends.peps2d._YASTNPEPSOps`` (and ``_local_hamiltonian``).  Operator
algebra (scalar mult / add) stays in NumPy; gates and measurement convert to the
array backend at point of use.
"""

from __future__ import annotations

import numpy as np


class PEPSOps:
    """Dense local operators over a small level structure (``1r`` or ``01r``)."""

    def __init__(self, levels) -> None:
        self.levels = tuple(str(level) for level in levels)
        self.dim = len(self.levels)

    def index(self, label: str) -> int:
        try:
            return self.levels.index(label)
        except ValueError:
            raise ValueError(
                f"Unknown PEPS level label {label!r}; available levels are {self.levels}."
            ) from None

    # ---- state vectors ----
    def vector(self, label: str) -> np.ndarray:
        values = np.zeros(self.dim, dtype=np.complex128)
        values[self.index(label)] = 1.0
        return values

    def superposition_vector(self, amps) -> np.ndarray:
        values = np.asarray(amps, dtype=np.complex128)
        if values.shape != (self.dim,):
            raise ValueError(f"superposition amps must have shape ({self.dim},); got {values.shape}.")
        return values

    # ---- operator matrices ----
    def projector(self, label: str) -> np.ndarray:
        mat = np.zeros((self.dim, self.dim), dtype=np.complex128)
        if label in self.levels:
            idx = self.index(label)
            mat[idx, idx] = 1.0
        return mat

    def x_between(self, lower: str, upper: str) -> np.ndarray:
        mat = np.zeros((self.dim, self.dim), dtype=np.complex128)
        if lower in self.levels and upper in self.levels:
            lo, up = self.index(lower), self.index(upper)
            mat[lo, up] = 1.0
            mat[up, lo] = 1.0
        return mat

    @property
    def I(self) -> np.ndarray:  # noqa: E743 - conventional identity symbol
        return np.eye(self.dim, dtype=np.complex128)

    @property
    def X_01(self) -> np.ndarray:
        return self.x_between("0", "1")

    @property
    def X_1r(self) -> np.ndarray:
        return self.x_between("1", "r")

    @property
    def n_0(self) -> np.ndarray:
        return self.projector("0")

    @property
    def n_1(self) -> np.ndarray:
        return self.projector("1")

    @property
    def n_r(self) -> np.ndarray:
        return self.projector("r")

    @property
    def Z(self) -> np.ndarray:
        return 2.0 * self.n_r - self.I

    # ---- Hamiltonian terms ----
    def local_hamiltonian(self, omega_R, omega_hf, delta_R, delta_hf) -> np.ndarray:
        """``0.5 Ω_R X_1r + 0.5 Ω_hf X_01 - Δ_R n_r - Δ_hf n_1`` (mirror _local_hamiltonian)."""
        return (
            0.5 * float(omega_R) * self.X_1r
            + 0.5 * float(omega_hf) * self.X_01
            - float(delta_R) * self.n_r
            - float(delta_hf) * self.n_1
        )

    def nn_hamiltonian(self) -> np.ndarray:
        """``n_r ⊗ n_r`` as a 4-leg tensor with leg order ``(s0_out, s0_in, s1_out, s1_in)``.

        Matches ``yastn.tn.fpeps.gates.fkron(n_r, n_r)``.
        """
        nr = self.n_r
        return np.einsum("ab,cd->abcd", nr, nr)
