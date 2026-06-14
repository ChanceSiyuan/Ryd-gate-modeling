"""Boundary-MPS contraction of the finite double-layer network for measurement.

The single contraction engine for all observables (1-site, 2-site correlators,
and the O(N) ground-state energy), replacing YASTN's ``EnvBP``/``EnvCTM`` /
``EnvBoundaryMPS``.  Both ``measurement_environment="bp"`` and ``"ctm"`` route
here.

Each site becomes a rank-4 *double tensor* ``a[u, l, d, r]`` (fused ket+bra legs,
ket-major) obtained by tracing the physical leg of ``A`` against ``conj(A)`` with
an optional operator inserted.  The 2D network of double tensors is contracted
row by row with a boundary MPS compressed to ``chi`` (``None`` = exact).

All expectation values are divided by the norm ``Z`` from the same construction.
"""

from __future__ import annotations

import numpy as np

from .operators import PEPSOps
from .peps import FinitePEPS
from .tensors import ArrayBackend


def double_tensor(ab: ArrayBackend, A, O=None):
    """Rank-4 double tensor ``a[u, l, d, r]`` (fused ket+bra, ket-major).

    ``u<-(t,t')``, ``l<-(l,l')``, ``d<-(b,b')``, ``r<-(r,r')``.  With ``O`` the
    physical legs are traced as ``<bra|O|ket>``: ``sum A[..,s] O[S,s] conj(A)[..,S]``.
    """
    Ac = ab.conj(A)
    Dt, Dl, Db, Dr, _ = A.shape
    if O is None:
        out = ab.einsum("tlbrs,TLBRs->tTlLbBrR", A, Ac)
    else:
        out = ab.einsum("tlbrs,TLBRS,Ss->tTlLbBrR", A, Ac, ab.asarray(O))
    return out.reshape(Dt * Dt, Dl * Dl, Db * Db, Dr * Dr)


def _absorb_row(ab: ArrayBackend, B, row, chi, svd_min):
    """Absorb a row of MPO double tensors into boundary MPS ``B`` via a
    left-to-right zip-up, truncating each new bond to ``chi`` (``None`` = exact).

    ``B[y] = [a, pin, b]``; ``row[y] = [pin, l, pout, r]``.  Returns a new MPS
    ``[a', pout, b']`` with bonds ``<= chi``.  Unlike a naive contract-then-compress,
    this never forms the full ``(a*l, pout, b*r)`` product tensor: the peak cost is
    ``O(chi^2 D^4)`` (``D^2`` = doubled bond), which is what makes moderate ``D``
    feasible.
    """
    Ly = len(B)
    cap = chi if chi else (1 << 30)
    carry = ab.asarray(np.ones((1, 1, 1)))  # (x_new_left, a_old_mps_bond, m_mpo_left)
    out = []
    for y in range(Ly):
        By, Ry = B[y], row[y]
        # T[x, pout, b, r] = sum_{a, pin, l=m} carry[x,a,m] By[a,pin,b] Ry[pin,l,pout,r]
        T = ab.einsum("xam,apb,pmqr->xqbr", carry, By, Ry)
        x, q, b, r = T.shape
        if y == Ly - 1:  # right boundary: b, r are dim 1
            out.append(T.reshape(x, q, b * r))
            break
        U, s, Vh, _ = ab.svd_truncate(T.reshape(x * q, b * r), cap, svd_min)
        k = s.shape[0]
        out.append(ab.asarray(U).reshape(x, q, k))
        carry = (ab.asarray(s)[:, None] * ab.asarray(Vh)).reshape(k, b, r)
    return out


def _contract_scalar(ab: ArrayBackend, M):
    res = None
    for t in M:
        a, p, b = t.shape
        mat = t.reshape(a, b)  # bottom-boundary phys is dim 1
        res = mat if res is None else res @ mat
    return complex(ab.to_numpy(res).reshape(-1)[0])


def _sandwich(ab: ArrayBackend, top, row, bot):
    """Contract ``top`` (MPS, phys=row.u) / ``row`` (MPO) / ``bot`` (MPS, phys=row.d)."""
    E = ab.asarray(np.ones((1, 1, 1)))
    for ty, ry, by in zip(top, row, bot):
        E = ab.einsum("tlb,tuT,uldr,bdB->TrB", E, ty, ry, by)
    return complex(ab.to_numpy(E).reshape(-1)[0])


class BoundaryMPS:
    """Boundary-MPS measurement environment over a finite PEPS."""

    def __init__(self, psi: FinitePEPS, ab: ArrayBackend, chi=None, svd_min=1e-12) -> None:
        self.psi = psi
        self.ab = ab
        self.Lx, self.Ly = psi.Lx, psi.Ly
        self.chi = chi
        self.svd_min = svd_min
        self._I = ab.asarray(np.ones((1, 1, 1)))
        self._top = None
        self._bot = None
        self._Z = None

    # ---- environments ----
    def _row(self, x, ops_by_col=None, flip=False):
        """Double tensors for row ``x`` as MPO tensors ``[pin, l, pout, r]``."""
        ops_by_col = ops_by_col or {}
        out = []
        for y in range(self.Ly):
            a = double_tensor(self.ab, self.psi[(x, y)], ops_by_col.get(y))  # [u,l,d,r]
            # reorder to [pin, l, pout, r]: top-down pin=u,pout=d; bottom-up pin=d,pout=u
            out.append(self.ab.transpose(a, (2, 1, 0, 3)) if flip else a)
        return out

    def _trivial_mps(self):
        return [self.ab.asarray(np.ones((1, 1, 1))) for _ in range(self.Ly)]

    def _build(self):
        if self._top is not None:
            return
        ab = self.ab
        top = [self._trivial_mps()]
        for x in range(self.Lx):
            top.append(_absorb_row(ab, top[-1], self._row(x), self.chi, self.svd_min))
        bot = [None] * (self.Lx + 1)
        bot[self.Lx] = self._trivial_mps()
        for x in range(self.Lx - 1, -1, -1):
            bot[x] = _absorb_row(ab, bot[x + 1], self._row(x, flip=True), self.chi, self.svd_min)
        self._top, self._bot = top, bot
        self._Z = _contract_scalar(ab, top[self.Lx])

    def norm(self) -> complex:
        self._build()
        return self._Z

    # ---- measurements ----
    def measure_1site(self, O) -> dict:
        """``{(x, y): <O>}`` for every site."""
        self._build()
        ab = self.ab
        out = {}
        for x in range(self.Lx):
            row = self._row(x)
            for y in range(self.Ly):
                rowO = list(row)
                rowO[y] = double_tensor(ab, self.psi[(x, y)], O)
                num = _sandwich(ab, self._top[x], rowO, self._bot[x + 1])
                out[(x, y)] = num / self._Z
        return out

    def measure_2site(self, O, P, si, sj) -> complex:
        """``<O_si P_sj>`` via a full boundary contraction with both insertions."""
        self._build()
        ab = self.ab
        ins = {tuple(si): O, tuple(sj): P}
        B = self._trivial_mps()
        for x in range(self.Lx):
            ops = {y: ins[(x, y)] for y in range(self.Ly) if (x, y) in ins}
            B = _absorb_row(ab, B, self._row(x, ops_by_col=ops), self.chi, self.svd_min)
        return _contract_scalar(ab, B) / self._Z


def contract_peps_dense(psi: FinitePEPS, ab: ArrayBackend) -> np.ndarray:
    """Contract a (small) finite PEPS into a dense statevector over all physical legs.

    Physical legs are ordered row-major ``(x, y)``.  For validation / exact
    small-system use only.
    """
    Lx, Ly = psi.Lx, psi.Ly
    T = {s: ab.to_numpy(psi[s]) for s in psi.sites()}
    nxt = [0]

    def lab():
        nxt[0] += 1
        return nxt[0]

    h_bond, v_bond, phys = {}, {}, {}
    for x in range(Lx):
        for y in range(Ly):
            phys[(x, y)] = lab()
            if y < Ly - 1:
                h_bond[(x, y)] = lab()
            if x < Lx - 1:
                v_bond[(x, y)] = lab()

    args = []
    for x in range(Lx):
        for y in range(Ly):
            t = v_bond[(x - 1, y)] if x > 0 else lab()  # unique (dim-1) when at boundary
            l = h_bond[(x, y - 1)] if y > 0 else lab()
            b = v_bond[(x, y)] if x < Lx - 1 else lab()
            r = h_bond[(x, y)] if y < Ly - 1 else lab()
            args.append(T[(x, y)])
            args.append([t, l, b, r, phys[(x, y)]])
    args.append([phys[(x, y)] for x in range(Lx) for y in range(Ly)])
    return np.einsum(*args, optimize=True)
