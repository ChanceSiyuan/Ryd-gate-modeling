"""NTU-NN bond truncation for the rydtn engine.

Ports YASTN's neighborhood tensor update (``EnvNTU(which='NN')`` +
``truncate_optimize_`` with ``initialization="SVD"``) to dense arrays.  For a
two-site bond we:

1. QR/LQ-reduce the two (gate-enlarged) site tensors to a small bond matrix
   ``RR`` and residual isometries ``Q0, Q1`` (``reduce_bond``).
2. Build the NTU-NN bond metric ``G`` by an exact double-layer contraction of the
   6-neighbour ring around the bond, with missing edge neighbours padded by a
   trivial dim-1 tensor (``nn_bond_metric``).
3. Truncate ``RR`` to ``chi_max`` via an SVD initialization refined by
   alternating least squares in the ``G`` metric (``truncate_optimize``).
4. Reinsert into ``Q0, Q1`` (``writeback_bond``).

Leg order is ``[t, l, b, r, s]`` throughout (see ``peps.py``).  Spectral steps
run in complex128 (via ``ArrayBackend``) for single-precision stability.
"""

from __future__ import annotations

import numpy as np

from .peps import FinitePEPS
from .tensors import ArrayBackend

_PINV_CUTOFFS = (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)


# --------------------------------------------------------------------------- #
# QR reduction and write-back
# --------------------------------------------------------------------------- #
def reduce_bond(ab: ArrayBackend, A0, A1, dirn: str):
    """Reduce the two site tensors of a bond to ``(Q0, Q1, RR)``.

    For ``dirn='lr'``: ``Q0=[t,l,b,rr,s]``, ``Q1=[t,ll,b,r,s]``, ``RR=[rr,ll]``.
    For ``dirn='tb'``: ``Q0=[t,l,bb,r,s]``, ``Q1=[tt,l,b,r,s]``, ``RR=[bb,tt]``.
    """
    if dirn == "lr":
        Dt, Dl, Db, Dr, d = A0.shape
        M = ab.transpose(A0, (0, 1, 2, 4, 3)).reshape(Dt * Dl * Db * d, Dr)
        Q, R0 = ab.qr(M)
        k0 = Q.shape[1]
        Q0 = ab.transpose(Q.reshape(Dt, Dl, Db, d, k0), (0, 1, 2, 4, 3))  # t l b rr s
        Et, El, Eb, Er, d1 = A1.shape
        Mr = ab.transpose(A1, (1, 0, 2, 3, 4)).reshape(El, Et * Eb * Er * d1)
        Q2, R2 = ab.qr(ab.transpose(Mr, (1, 0)))  # (rest,k1), (k1,El)
        k1 = Q2.shape[1]
        R1 = ab.transpose(R2, (1, 0))  # l ll
        Q1 = ab.transpose(ab.transpose(Q2, (1, 0)).reshape(k1, Et, Eb, Er, d1), (1, 0, 2, 3, 4))  # t ll b r s
        return Q0, Q1, R0 @ R1
    if dirn == "tb":
        Dt, Dl, Db, Dr, d = A0.shape
        M = ab.transpose(A0, (0, 1, 3, 4, 2)).reshape(Dt * Dl * Dr * d, Db)
        Q, R0 = ab.qr(M)
        k0 = Q.shape[1]
        Q0 = ab.transpose(Q.reshape(Dt, Dl, Dr, d, k0), (0, 1, 4, 2, 3))  # t l bb r s
        Et, El, Eb, Er, d1 = A1.shape
        Mr = A1.reshape(Et, El * Eb * Er * d1)
        Q2, R2 = ab.qr(ab.transpose(Mr, (1, 0)))  # (rest,k1), (k1,Et)
        k1 = Q2.shape[1]
        R1 = ab.transpose(R2, (1, 0))  # t tt
        Q1 = ab.transpose(Q2, (1, 0)).reshape(k1, El, Eb, Er, d1)  # tt l b r s
        return Q0, Q1, R0 @ R1
    raise ValueError(f"reduce_bond: unexpected dirn {dirn!r}.")


def writeback_bond(ab: ArrayBackend, psi: FinitePEPS, s0, s1, dirn, Q0, M0, M1, Q1) -> None:
    """Contract truncated bond matrices ``M0, M1`` back into the PEPS tensors."""
    if dirn == "lr":
        psi[s0] = ab.einsum("tlbqs,qx->tlbxs", Q0, ab.asarray(M0))
        psi[s1] = ab.einsum("xl,tlbrs->txbrs", ab.asarray(M1), Q1)
    elif dirn == "tb":
        psi[s0] = ab.einsum("tlqrs,qx->tlxrs", Q0, ab.asarray(M0))
        psi[s1] = ab.einsum("xt,tlbrs->xlbrs", ab.asarray(M1), Q1)
    else:
        raise ValueError(f"writeback_bond: unexpected dirn {dirn!r}.")


# --------------------------------------------------------------------------- #
# NTU-NN bond metric (dense double-layer contraction of the 6-neighbour ring)
# --------------------------------------------------------------------------- #
def _neighbor(ab: ArrayBackend, psi: FinitePEPS, x, y):
    if 0 <= x < psi.Lx and 0 <= y < psi.Ly:
        return psi[(x, y)]
    return ab.asarray(np.ones((1, 1, 1, 1, 1)))


def nn_bond_metric(ab: ArrayBackend, psi: FinitePEPS, s0, s1, dirn, Q0, Q1):
    """Dense NTU-NN bond metric as a Hermitian ``(k0*k1, k0*k1)`` matrix.

    The metric acts on the vectorized bond ``vec(RR)`` ordered ``(bond0, bond1)``
    row-major; ``RRgRR = vec(RR)^H G vec(RR)`` equals the cluster norm.
    """
    cj = ab.conj
    ein = ab.einsum
    x0, y0 = s0

    def nb(dr, dc):
        return _neighbor(ab, psi, x0 + dr, y0 + dc)

    if dirn == "lr":
        AL, AR = nb(0, -1), nb(0, 2)
        AT0, AT1 = nb(-1, 0), nb(-1, 1)
        AB0, AB1 = nb(1, 0), nb(1, 1)
        # side hairs (expose the leg facing Q0 / Q1)
        nl = ein("tlbrs,tlbZs->rZ", AL, cj(AL))  # -> Q0.l
        nr = ein("tlbrs,tLbrs->lL", AR, cj(AR))  # -> Q1.r
        Dt0, Dl0, Db0, k0, _ = Q0.shape
        Dt1, k1, Db1, Dr1, _ = Q1.shape
        EL = ein("tlbrs,TLBRs,lL->bBrRtT", Q0, cj(Q0), nl).reshape(Db0 * Db0, k0 * k0, Dt0 * Dt0)
        ER = ein("tlbrs,TLBRs,rR->tTlLbB", Q1, cj(Q1), nr).reshape(Dt1 * Dt1, k1 * k1, Db1 * Db1)
        ctl = _cor(ab, AT0, "br")   # (Q0.t),(top0.r)
        ctr = _cor(ab, AT1, "lb")   # (top1.l),(Q1.t)
        cbl = _cor(ab, AB0, "rt")   # (bot0.r),(Q0.b)
        cbr = _cor(ab, AB1, "tl")   # (Q1.b),(bot1.l)
        Lm = ein("ab,bcd->acd", ein("ab,bc->ac", cbr, cbl), EL)  # (Q1.b),(Q0.rr),(Q0.t)
        Rm = ein("ab,bcd->acd", ein("ab,bc->ac", ctl, ctr), ER)  # (Q0.t),(Q1.ll),(Q1.b)
        g = ein("xiy,yjx->ij", Lm, Rm)  # (Q0.rr fused),(Q1.ll fused)
        return _finalize_metric(ab, g, k0, k1)
    if dirn == "tb":
        AT = nb(-1, 0)
        AL0, AL1 = nb(0, -1), nb(1, -1)
        AR0, AR1 = nb(0, 1), nb(1, 1)
        AB = nb(2, 0)
        ht = ein("tlbrs,tlBrs->bB", AT, cj(AT))  # -> Q0.t
        hb = ein("tlbrs,Tlbrs->tT", AB, cj(AB))  # -> Q1.b
        Dt0, Dl0, k0, Dr0, _ = Q0.shape
        k1, Dl1, Db1, Dr1, _ = Q1.shape
        ET = ein("tlbrs,TLBRs,tT->lLbBrR", Q0, cj(Q0), ht).reshape(Dl0 * Dl0, k0 * k0, Dr0 * Dr0)
        EB = ein("tlbrs,TLBRs,bB->rRtTlL", Q1, cj(Q1), hb).reshape(Dr1 * Dr1, k1 * k1, Dl1 * Dl1)
        cbl = _cor(ab, AL1, "rt")   # (Q1.l),(left1.t)
        ctl = _cor(ab, AL0, "br")   # (left0.b),(Q0.l)
        ctr = _cor(ab, AR0, "lb")   # (Q0.r),(right0.b)
        cbr = _cor(ab, AR1, "tl")   # (right1.t),(Q1.r)
        Lm = ein("ab,bcd->acd", ein("ab,bc->ac", cbl, ctl), ET)  # (Q1.l),(Q0.bb),(Q0.r)
        Rm = ein("ab,bcd->acd", ein("ab,bc->ac", ctr, cbr), EB)  # (Q0.r),(Q1.tt),(Q1.l)
        g = ein("xiy,yjx->ij", Lm, Rm)  # (Q0.bb fused),(Q1.tt fused)
        return _finalize_metric(ab, g, k0, k1)
    raise ValueError(f"nn_bond_metric: unexpected dirn {dirn!r}.")


def _cor(ab: ArrayBackend, A, keep: str):
    """Corner cap: trace physical and the two legs not in ``keep`` (ket+bra).

    Returns a ket-major fused ``(D_keep0^2, D_keep1^2)`` matrix with axis order
    matching ``keep`` (e.g. ``keep='br'`` -> ``[(b B),(r R)]``).
    """
    cj = ab.conj
    specs = {
        "br": ("tlbrs,tlBRs->bBrR", (2, 3)),
        "lb": ("tlbrs,tLBrs->lLbB", (1, 2)),
        "rt": ("tlbrs,TlbRs->rRtT", (3, 0)),
        "tl": ("tlbrs,TLbrs->tTlL", (0, 1)),
    }
    spec, (ax0, ax1) = specs[keep]
    out = ab.einsum(spec, A, cj(A))
    D0, D1 = A.shape[ax0], A.shape[ax1]
    return out.reshape(D0 * D0, D1 * D1)


def _finalize_metric(ab: ArrayBackend, g, k0: int, k1: int):
    g = g.reshape(k0, k0, k1, k1)              # rr, Rr, ll, Ll
    G = ab.transpose(g, (1, 3, 0, 2)).reshape(k0 * k1, k0 * k1)  # (Rr,Ll),(rr,ll)
    G = ab.to_numpy(G).astype(np.complex128)
    return (G + G.conj().T) / 2


# --------------------------------------------------------------------------- #
# SVD initialization + ALS optimization (initialization="SVD")
# --------------------------------------------------------------------------- #
def _svd_split(RR_np, chi_max, svd_min, normalize=False):
    U, s, Vh = np.linalg.svd(RR_np, full_matrices=False)
    s0 = s[0] if s.size else 1.0
    k = max(1, min(int(chi_max), int(np.count_nonzero(s > svd_min * s0))))
    U, s, Vh = U[:, :k], s[:k], Vh[:k, :]
    if normalize and s.size:
        s = s / s.max()
    sq = np.sqrt(s)
    return U * sq, sq[:, None] * Vh  # M0 (k0,k), M1 (k,k1)


def _err2(M0, M1, G, fRR, RRgRR):
    delta = fRR - (M0 @ M1).reshape(-1)
    return abs(np.vdot(delta, G @ delta)) / RRgRR


def _optimal_pinv(g, j, error_fun):
    w, U = np.linalg.eigh(g)
    smax = w.max() if w.size else 1.0
    UHj = U.conj().T @ j
    best_M, best_err = None, np.inf
    seen = -1
    for cutoff in _PINV_CUTOFFS:
        keep = w > cutoff * smax
        nkeep = int(keep.sum())
        if nkeep == seen:
            continue
        seen = nkeep
        inv = np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0)
        M = (U @ (inv * UHj))
        err = error_fun(M)
        if err < best_err:
            best_M, best_err = M, err
    return best_M, best_err


def _als(M0, M1, err2, G, fRR, RRgRR, max_iter, tol_iter):
    k0 = M0.shape[0]
    k1 = M1.shape[1]
    chi = M0.shape[1]
    Garr = G.reshape(k0, k1, k0, k1)
    Gv = (G @ fRR).reshape(k0, k1)
    err_old = err2
    for _ in range(max(1, int(max_iter))):
        # fix M1, optimize M0
        g0 = np.einsum("cp,apbq,dq->acbd", M1.conj(), Garr, M1, optimize=True).reshape(k0 * chi, k0 * chi)
        j0 = np.einsum("cp,ap->ac", M1.conj(), Gv, optimize=True).reshape(k0 * chi)
        vec0, _ = _optimal_pinv(g0, j0, lambda x: _err2(x.reshape(k0, chi), M1, G, fRR, RRgRR))
        M0 = vec0.reshape(k0, chi)
        # fix M0, optimize M1
        g1 = np.einsum("ic,ibjB,jC->cbCB", M0.conj(), Garr, M0, optimize=True).reshape(chi * k1, chi * k1)
        j1 = np.einsum("ic,ib->cb", M0.conj(), Gv, optimize=True).reshape(chi * k1)
        vec1, err2 = _optimal_pinv(g1, j1, lambda x: _err2(M0, x.reshape(chi, k1), G, fRR, RRgRR))
        M1 = vec1.reshape(chi, k1)
        if abs(err2 - err_old) < tol_iter:
            break
        err_old = err2
    return M0, M1, err2


def truncate_optimize(ab: ArrayBackend, G, RR, chi_max, svd_min, max_iter, tol_iter, normalize=True):
    """Truncate bond matrix ``RR`` to ``chi_max`` in the ``G`` metric.

    Returns ``(M0, M1, truncation_error)`` with ``M0=[k0,chi]``, ``M1=[chi,k1]``
    (numpy complex128).  ``truncation_error`` is the relative N-metric residual
    (computed before the final rescale), matching YASTN's per-bond
    ``info.truncation_error``.  With ``normalize=True`` (the default, as in YASTN)
    the final bond is rescaled to unit largest singular value.
    """
    RR_np = ab.to_numpy(RR).astype(np.complex128)
    fRR = RR_np.reshape(-1)
    RRgRR = abs(np.vdot(fRR, G @ fRR))
    if RRgRR == 0:
        RRgRR = 1.0
    # SVD initialization, then ALS refinement.
    M0, M1 = _svd_split(RR_np, chi_max, svd_min, normalize=False)
    err2 = _err2(M0, M1, G, fRR, RRgRR)
    M0o, M1o, err2o = _als(M0, M1, err2, G, fRR, RRgRR, max_iter, tol_iter)
    if err2o < err2:
        M0, M1, err2 = M0o, M1o, err2o
    # final symmetric re-split (optionally normalized)
    M0, M1 = _svd_split(M0 @ M1, chi_max, svd_min, normalize=normalize)
    return M0, M1, float(np.sqrt(max(err2, 0.0)))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def truncate_bond(ab: ArrayBackend, psi: FinitePEPS, s0, s1, dirn, chi_max, svd_min, max_iter, tol_iter) -> float:
    """Reduce, build NTU-NN metric, truncate to ``chi_max``, write back.

    Assumes the gate (if any) is already applied to ``psi[s0], psi[s1]``.
    Returns the relative truncation error for this bond.
    """
    Q0, Q1, RR = reduce_bond(ab, psi[s0], psi[s1], dirn)
    G = nn_bond_metric(ab, psi, s0, s1, dirn, Q0, Q1)
    M0, M1, err = truncate_optimize(ab, G, RR, chi_max, svd_min, max_iter, tol_iter)
    writeback_bond(ab, psi, s0, s1, dirn, Q0, M0, M1, Q1)
    return err
