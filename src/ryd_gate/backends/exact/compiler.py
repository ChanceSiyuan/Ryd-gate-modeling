"""Exact matrix compilation from the canonical lowered drive representation.

Builds a matvec ``H(t) @ psi`` from a system's private static terms plus the
lowered drive channels (``ryd_gate.core.lowering.lower_drives``). Storage is
dense or sparse per ``hamiltonian_format``; the sparse path never densifies
(E04). All Hamiltonians are Hermitian (E08).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix

from ryd_gate.core.lowering import lower_drives
from ryd_gate.core.operators import materialize_sparse_operator, parse_E

# dim^2 * 16 bytes ~ 256 MiB at dim ~ 4096.
_DENSE_DIM_THRESHOLD = 4096


def _choose_dense(hamiltonian_format: str, dim: int) -> bool:
    if hamiltonian_format == "dense":
        return True
    if hamiltonian_format == "sparse":
        return False
    if hamiltonian_format == "auto":
        return dim <= _DENSE_DIM_THRESHOLD
    raise ValueError(
        f"hamiltonian_format must be 'auto', 'dense' or 'sparse'; got {hamiltonian_format!r}."
    )


class _DriveChannel:
    """One lowered channel with materialized operators (scalar + lazy per-site)."""

    __slots__ = (
        "coeff", "is_diag", "sum_op", "sum_op_hc",
        "_operators", "_basis", "_dense", "_channel", "_site_ops", "_site_ops_hc",
    )

    def __init__(self, lowered, operators, basis, dense: bool) -> None:
        self.coeff = lowered.coefficient
        self._channel = lowered.channel
        ket, bra = parse_E(lowered.channel)
        self.is_diag = ket == bra
        self._operators = operators
        self._basis = basis
        self._dense = dense
        sum_op = materialize_sparse_operator(operators.sum(lowered.channel), basis)
        self.sum_op = sum_op.toarray() if dense else sum_op
        self.sum_op_hc = None if self.is_diag else (
            self.sum_op.conj().T if dense else sum_op.conj().T.tocsc()
        )
        self._site_ops = None
        self._site_ops_hc = None

    def _ensure_site_ops(self):
        if self._site_ops is None:
            ops = [
                materialize_sparse_operator(self._operators.local(self._channel, i), self._basis)
                for i in range(self._basis.n_sites)
            ]
            self._site_ops = [o.toarray() if self._dense else o.tocsc() for o in ops]
            self._site_ops_hc = None if self.is_diag else [o.conj().T for o in self._site_ops]
        return self._site_ops, self._site_ops_hc


class _ExactHamiltonian:
    """Callable ``apply(t, psi) = H(t) @ psi`` for the exact ODE RHS."""

    __slots__ = ("h_static", "channels", "dim", "dense", "n_sites")

    def __init__(self, h_static, channels, dim, dense, n_sites) -> None:
        self.h_static = h_static
        self.channels = channels
        self.dim = dim
        self.dense = dense
        self.n_sites = n_sites

    def apply(self, t: float, psi: np.ndarray) -> np.ndarray:
        y = self.h_static @ psi
        for ch in self.channels:
            cv = ch.coeff(t)
            arr = np.asarray(cv)
            if arr.ndim == 0:
                # Diagonal channels are Hermitian energies (E08): use the real
                # part (matching the TN compiler) so H stays Hermitian even if a
                # coefficient carries a spurious imaginary component.
                c = complex(cv)
                if ch.is_diag:
                    c = complex(c.real)
                if c != 0.0:
                    y = y + c * (ch.sum_op @ psi)
                    if not ch.is_diag:
                        y = y + np.conj(c) * (ch.sum_op_hc @ psi)
            else:
                ops, ops_hc = ch._ensure_site_ops()
                for i in range(self.n_sites):
                    ci = complex(arr[i])
                    if ch.is_diag:
                        ci = complex(ci.real)  # Hermitian diagonal energy (E08)
                    if ci == 0.0:
                        continue
                    y = y + ci * (ops[i] @ psi)
                    if not ch.is_diag:
                        y = y + np.conj(ci) * (ops_hc[i] @ psi)
        return y


def compile_exact(system, *, hamiltonian_format: str = "auto"):
    """Return ``(_ExactHamiltonian, t_gate)`` for a protocol-bound system.

    A noise realization (laser amplitude/frequency noise) is read from the
    system's private ``_realization`` and injected during lowering; position
    noise is already baked into the system's static interaction terms.
    """
    realization = getattr(system, "_realization", None)
    basis = system._basis
    operators = system._operators
    dim = basis.total_dim
    dense = _choose_dense(hamiltonian_format, dim)

    h_static = np.zeros((dim, dim), dtype=complex) if dense else csc_matrix((dim, dim), dtype=complex)
    for term in system._static_terms:
        op = materialize_sparse_operator(term.operator, basis)
        m = op.toarray() if dense else op
        c = complex(term.coefficient)
        h_static = h_static + c * m
        if term.add_hermitian_conjugate:
            h_static = h_static + np.conj(c) * m.conj().T

    t_gate, lowered = lower_drives(system, realization=realization)
    channels = [_DriveChannel(lc, operators, basis, dense) for lc in lowered]
    return _ExactHamiltonian(h_static, channels, dim, dense, basis.n_sites), float(t_gate)
