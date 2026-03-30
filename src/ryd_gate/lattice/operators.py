"""Sparse operator construction for many-body Rydberg Hamiltonians."""

import numpy as np
from scipy.sparse import csc_matrix, diags as spdiags, eye as speye, kron as spkron


def _single_site_op(op_2x2, site, N):
    """Embed a 2x2 operator on `site` into the N-atom Hilbert space.

    Uses I_{2^site} kron op kron I_{2^(N-1-site)} (2 kron calls, not N-1).
    """
    result = op_2x2
    if site > 0:
        result = spkron(speye(2**site, format='csc', dtype=complex),
                        result, format='csc')
    if site < N - 1:
        result = spkron(result,
                        speye(2**(N - 1 - site), format='csc', dtype=complex),
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
        n_list.append(_single_site_op(nr, i, N))
        sum_X = sum_X + _single_site_op(sx, i, N)

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
