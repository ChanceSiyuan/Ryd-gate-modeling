"""Sparse operator construction for many-body Rydberg Hamiltonians (2-level and 3-level)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix, diags as spdiags, eye as speye, kron as spkron


def _embed_site_op(op, site, N, d=None):
    """Embed a d×d single-atom operator on `site` into d^N Hilbert space.

    Uses I_{d^site} ⊗ op ⊗ I_{d^(N-1-site)}.
    The local dimension d is inferred from op.shape[0] if not given.
    """
    if d is None:
        d = op.shape[0]
    result = op
    if site > 0:
        result = spkron(speye(d**site, format='csc', dtype=complex),
                        result, format='csc')
    if site < N - 1:
        result = spkron(result,
                        speye(d**(N - 1 - site), format='csc', dtype=complex),
                        format='csc')
    return result


def _apply_local_detuning(H, delta_local, n_list):
    """Subtract per-site detunings: H -= sum_i delta_i * n_i."""
    for i, d_i in enumerate(delta_local):
        if abs(d_i) > 1e-15:
            H = H - d_i * n_list[i]
    return H


def build_operators(N, vdw_pairs, V_nn, verbose=False):
    """Precompute all sparse operators needed for the Hamiltonian.

    Parameters
    ----------
    N : int
        Number of atoms.
    vdw_pairs : list of (i, j, V_rel)
        Interaction pairs with relative strengths.
    V_nn : float
        Nearest-neighbor interaction strength.
    verbose : bool
        Print progress/timing information.

    Returns
    -------
    dict with keys 'sum_X', 'sum_n', 'n_list', 'H_vdw'.
    """
    import time as _time

    dim = 2 ** N
    if verbose:
        print(f"  Building operators for {N} atoms (dim = {dim})...")
    t0 = _time.time()

    sx = csc_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    nr = csc_matrix(np.array([[0, 0], [0, 1]], dtype=complex))

    n_list = []
    sum_X = csc_matrix((dim, dim), dtype=complex)

    for i in range(N):
        n_list.append(_embed_site_op(nr, i, N))
        sum_X = sum_X + _embed_site_op(sx, i, N)

    # n_i operators are diagonal — build sum_n and H_vdw as dense diag vectors
    n_diags = [ni.diagonal() for ni in n_list]

    sum_n_diag = np.zeros(dim, dtype=complex)
    for d in n_diags:
        sum_n_diag += d
    sum_n = spdiags([sum_n_diag], offsets=[0], shape=(dim, dim), format='csc')

    h_vdw_diag = np.zeros(dim, dtype=complex)
    for (i, j, v_rel) in vdw_pairs:
        h_vdw_diag += V_nn * v_rel * (n_diags[i] * n_diags[j])
    H_vdw = spdiags([h_vdw_diag], offsets=[0], shape=(dim, dim), format='csc')

    if verbose:
        print(f"  Operators built in {_time.time() - t0:.1f}s")
    return {
        'sum_X': sum_X,
        'sum_n': sum_n,
        'n_list': n_list,
        'H_vdw': H_vdw,
    }


def build_hamiltonian(Omega, Delta, delta_local, ops):
    """Assemble the full Hamiltonian as a sparse matrix (hbar = 1).

    H = (Omega/2)*sum_X - Delta*sum_n - sum_i(delta_i * n_i) + H_vdw
    """
    H = (Omega / 2) * ops['sum_X'] - Delta * ops['sum_n'] + ops['H_vdw']
    return _apply_local_detuning(H, delta_local, ops['n_list'])


def build_hamiltonian_base(delta_local, ops):
    """Precompute the time-independent part: H_vdw + local pinning.

    During a sweep only Omega and Delta change, so the per-step Hamiltonian
    is H(t) = (Omega_t/2)*sum_X - Delta_t*sum_n + H_base.
    """
    H_base = ops['H_vdw'].copy()
    return _apply_local_detuning(H_base, delta_local, ops['n_list'])


# ── 3-level operators (N-atom, dim = 3^N) ────────────────────────────


@dataclass(frozen=True)
class ThreeLevelOps:
    """Precomputed sparse operators for N-atom 3-level Hamiltonian."""

    N: int
    dim: int
    H_const: csc_matrix
    H_1013: csc_matrix
    H_1013_dag: csc_matrix
    H_420_uniform: csc_matrix
    H_420_uniform_dag: csc_matrix
    H_420_per_atom: list[csc_matrix]
    n_r_list: list[csc_matrix]



def build_3level_operators(
    geom,
    Delta: float,
    Omega_1013: float,
    Omega_420: float | np.ndarray,
    mid_decay: float = 0.0,
    ryd_decay: float = 0.0,
    verbose: bool = False,
) -> ThreeLevelOps:
    """Build all sparse operators for the N-atom 3-level Hamiltonian."""
    import time as _time
    N = geom.N
    dim = 3**N
    if verbose:
        print(f"  Building 3-level operators for {N} atoms (dim={dim})...")
    t0 = _time.time()

    eg = csc_matrix(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=complex))
    re = csc_matrix(np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=complex))
    nr = csc_matrix(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=complex))
    ne = csc_matrix(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=complex))

    if np.isscalar(Omega_420):
        omega_420_arr = np.full(N, Omega_420)
    else:
        omega_420_arr = np.asarray(Omega_420)

    n_r_list = []
    H_420_per_atom = []
    H_1013_sum = csc_matrix((dim, dim), dtype=complex)
    H_420_sum = csc_matrix((dim, dim), dtype=complex)

    for i in range(N):
        n_r_i = _embed_site_op(nr, i, N)
        n_r_list.append(n_r_i)
        h420_i = (omega_420_arr[i] / 2) * _embed_site_op(eg, i, N)
        H_420_per_atom.append(h420_i)
        H_420_sum = H_420_sum + h420_i
        H_1013_sum = H_1013_sum + (Omega_1013 / 2) * _embed_site_op(re, i, N)

    n_r_diags = [n_r_i.diagonal() for n_r_i in n_r_list]
    h_const_diag = np.zeros(dim, dtype=complex)
    for i in range(N):
        ne_i = _embed_site_op(ne, i, N)
        h_const_diag += (Delta - 1j * mid_decay / 2) * ne_i.diagonal()
        h_const_diag += (-1j * ryd_decay / 2) * n_r_diags[i]
    for (i, j, V_ij) in geom.vdw_couplings:
        h_const_diag += V_ij * (n_r_diags[i] * n_r_diags[j])

    H_const = spdiags([h_const_diag], offsets=[0], shape=(dim, dim), format='csc')

    if verbose:
        print(f"  Operators built in {_time.time() - t0:.1f}s")
    return ThreeLevelOps(
        N=N, dim=dim, H_const=H_const,
        H_1013=H_1013_sum, H_1013_dag=H_1013_sum.conj().T.tocsc(),
        H_420_uniform=H_420_sum, H_420_uniform_dag=H_420_sum.conj().T.tocsc(),
        H_420_per_atom=H_420_per_atom, n_r_list=n_r_list,
    )
