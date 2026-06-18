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


def _transpose_peps(psi: FinitePEPS, ab: ArrayBackend) -> FinitePEPS:
    """Reflect the PEPS across its main diagonal (swap rows<->columns).

    Site ``(x, y) -> (y, x)`` and legs ``[t,l,b,r,s] -> [l,t,r,b,s]`` so that a
    vertical bond of ``psi`` becomes a horizontal bond of the result.
    """
    out = FinitePEPS(psi.Ly, psi.Lx, ab)
    for (x, y), A in psi.tensors.items():
        out[(y, x)] = ab.transpose(A, (1, 0, 3, 2, 4))
    return out


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


def _absorb_row(ab: ArrayBackend, B, tfn, chi, svd_min):
    """Absorb a row of MPO double tensors into boundary MPS ``B`` via a
    left-to-right zip-up, truncating each new bond to ``chi`` (``None`` = exact).

    ``tfn(y)`` returns the column-``y`` double tensor ``[pin, l, pout, r]`` on
    demand, so only one column is materialised at a time -- at ``D^2`` doubled
    bonds this keeps a full ``Ly``-wide row (``Ly * O(D^8)`` floats) off the
    device.  ``B[y] = [a, pin, b]``.  Returns a new MPS ``[a', pout, b']`` with
    bonds ``<= chi``.  Unlike a naive contract-then-compress, this never forms
    the full ``(a*l, pout, b*r)`` product tensor: the peak cost is
    ``O(chi^2 D^4)`` (``D^2`` = doubled bond), which is what makes moderate ``D``
    feasible.
    """
    Ly = len(B)
    cap = chi if chi else (1 << 30)
    carry = ab.asarray(np.ones((1, 1, 1)))  # (x_new_left, a_old_mps_bond, m_mpo_left)
    out = []
    for y in range(Ly):
        By, Ry = B[y], tfn(y)
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
        self._dt_cache: dict = {}  # background (op=None) double tensors by site

    # ---- environments ----
    def _row_tensor(self, x, y, op=None, flip=False):
        """Column ``(x, y)`` double tensor as an MPO tensor ``[pin, l, pout, r]``.

        ``flip=False`` (top-down): ``pin=u, pout=d``; ``flip=True`` (bottom-up):
        ``pin=d, pout=u``.  The operator-free tensor is cached: it is rebuilt
        many times across the top/bottom build and the streaming L/R sweeps, and
        the double-layer ``D^4`` contraction is the dominant per-call cost.
        """
        if op is None:
            a = self._dt_cache.get((x, y))
            if a is None:
                a = double_tensor(self.ab, self.psi[(x, y)], None)  # [u,l,d,r]
                self._dt_cache[(x, y)] = a
        else:
            a = double_tensor(self.ab, self.psi[(x, y)], op)  # [u,l,d,r]
        return self.ab.transpose(a, (2, 1, 0, 3)) if flip else a

    def _trivial_mps(self):
        return [self.ab.asarray(np.ones((1, 1, 1))) for _ in range(self.Ly)]

    def _build(self):
        if self._top is not None:
            return
        ab = self.ab
        ab.empty_cache()
        top = [self._trivial_mps()]
        for x in range(self.Lx):
            top.append(_absorb_row(
                ab, top[-1], lambda y, x=x: self._row_tensor(x, y), self.chi, self.svd_min))
            ab.empty_cache()
        bot = [None] * (self.Lx + 1)
        bot[self.Lx] = self._trivial_mps()
        for x in range(self.Lx - 1, -1, -1):
            bot[x] = _absorb_row(
                ab, bot[x + 1], lambda y, x=x: self._row_tensor(x, y, flip=True),
                self.chi, self.svd_min)
            ab.empty_cache()
        self._top, self._bot = top, bot
        self._Z = _contract_scalar(ab, top[self.Lx])

    def norm(self) -> complex:
        self._build()
        return self._Z

    # ---- measurements ----
    def measure_1site(self, O) -> dict:
        """``{(x, y): <O>}`` for every site, via streaming left/right environments.

        For each row the background double tensors are swept once from the left
        and once from the right into ``O(chi^2 D^2)`` environment tensors; each
        site value is then a single ``L . <O> . R`` contraction.  No full row of
        ``Ly`` double tensors is ever materialised (peak memory ~ a few columns),
        and the cost drops from ``O(Lx Ly^2)`` sandwiches to ``O(Lx Ly)``.
        """
        self._build()
        ab = self.ab
        eye = ab.asarray(np.ones((1, 1, 1)))
        out = {}
        for x in range(self.Lx):
            top, bot = self._top[x], self._bot[x + 1]
            L = [eye] + [None] * self.Ly  # L[y] = environment left of column y
            for y in range(self.Ly):
                d = self._row_tensor(x, y)
                L[y + 1] = ab.einsum("tlb,tuT,uldr,bdB->TrB", L[y], top[y], d, bot[y])
            R = [None] * self.Ly + [eye]  # R[y] = environment right of column y
            for y in range(self.Ly - 1, -1, -1):
                d = self._row_tensor(x, y)
                R[y] = ab.einsum("tuT,uldr,bdB,TrB->tlb", top[y], d, bot[y], R[y + 1])
            for y in range(self.Ly):
                dO = self._row_tensor(x, y, op=O)
                LO = ab.einsum("tlb,tuT,uldr,bdB->TrB", L[y], top[y], dO, bot[y])
                num = ab.einsum("TrB,TrB->", LO, R[y + 1])
                out[(x, y)] = complex(ab.to_numpy(num).reshape(-1)[0]) / self._Z
            ab.empty_cache()
        return out

    def measure_2site(self, O, P, si, sj) -> complex:
        """``<O_si P_sj>`` via a full boundary contraction with both insertions."""
        self._build()
        ab = self.ab
        ins = {tuple(si): O, tuple(sj): P}
        B = self._trivial_mps()
        for x in range(self.Lx):
            B = _absorb_row(
                ab, B, lambda y, x=x: self._row_tensor(x, y, op=ins.get((x, y))),
                self.chi, self.svd_min)
        return _contract_scalar(ab, B) / self._Z

    def _nn_horizontal(self, O, P) -> dict:
        """``{((x,y),(x,y+1)): <O P>}`` for every horizontal (same-row) bond.

        One streaming left/right sweep per row over the cached top/bottom
        environments -- the same machinery as ``measure_1site``, with no
        per-bond boundary rebuild or SVD.
        """
        self._build()
        ab = self.ab
        eye = ab.asarray(np.ones((1, 1, 1)))
        out: dict = {}
        for x in range(self.Lx):
            top, bot = self._top[x], self._bot[x + 1]
            L = [eye] + [None] * self.Ly  # L[y] = env left of column y
            for y in range(self.Ly):
                d = self._row_tensor(x, y)
                L[y + 1] = ab.einsum("tlb,tuT,uldr,bdB->TrB", L[y], top[y], d, bot[y])
            R = [None] * self.Ly + [eye]  # R[y] = env right of column y
            for y in range(self.Ly - 1, -1, -1):
                d = self._row_tensor(x, y)
                R[y] = ab.einsum("tuT,uldr,bdB,TrB->tlb", top[y], d, bot[y], R[y + 1])
            for y in range(self.Ly - 1):
                dO = self._row_tensor(x, y, op=O)
                LO = ab.einsum("tlb,tuT,uldr,bdB->TrB", L[y], top[y], dO, bot[y])
                dP = self._row_tensor(x, y + 1, op=P)
                LOP = ab.einsum("tlb,tuT,uldr,bdB->TrB", LO, top[y + 1], dP, bot[y + 1])
                num = ab.einsum("TrB,TrB->", LOP, R[y + 2])
                out[((x, y), (x, y + 1))] = complex(ab.to_numpy(num).reshape(-1)[0]) / self._Z
            ab.empty_cache()
        return out

    def measure_nn(self, O, P) -> dict:
        """``{((xi,yi),(xj,yj)): <O_i P_j>}`` for every nearest-neighbour bond.

        Horizontal bonds come from a streaming sweep on ``psi``; vertical bonds
        are horizontal bonds of the transposed PEPS, so they reuse the *same*
        streaming on ``psi^T``.  Two boundary builds + cheap streaming replace
        the per-bond full-boundary rebuilds of ``measure_2site`` -- cost is
        independent of the number of bonds.
        """
        out = self._nn_horizontal(O, P)
        envT = BoundaryMPS(_transpose_peps(self.psi, self.ab), self.ab,
                           chi=self.chi, svd_min=self.svd_min)
        for ((a, b), (a, b1)), val in envT._nn_horizontal(O, P).items():
            # psi^T site (a,b) == psi site (b,a); a horizontal bond (a,b)-(a,b+1)
            # on psi^T is the vertical bond (b,a)-(b+1,a) on psi.
            out[((b, a), (b1, a))] = val
        return out


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
