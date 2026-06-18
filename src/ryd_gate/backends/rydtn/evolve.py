"""Second-order Trotter evolution for the rydtn engine.

Builds and applies the gate layers ``local(½) + nn + local(½)`` per step (mirror
``_yastn_gates`` / ``_yastn_imag_gates``), with NN bonds truncated by the NTU-NN
update.  Gate construction is cached by field tuple / bond strength, exactly like
``peps2d._local_gate_list`` / ``_nn_gate_list``.
"""

from __future__ import annotations

import numpy as np

from . import gates as G
from .ntu import truncate_bond
from .operators import PEPSOps
from .peps import FinitePEPS
from .tensors import ArrayBackend


def apply_local_gate_(ab: ArrayBackend, psi: FinitePEPS, coord, gate) -> None:
    """``A'[..., a] = sum_s gate[a, s] A[..., s]`` on the physical leg."""
    psi[coord] = ab.einsum("as,tlbrs->tlbra", ab.asarray(gate), psi[coord])


def _split_nn_gate(G_nn4, svd_min=1e-14):
    """SVD-split ``G[s0o,s0i,s1o,s1i]`` into ``g0[s0o,s0i,a]``, ``g1[s1o,s1i,a]``."""
    d = G_nn4.shape[0]
    M = G_nn4.reshape(d * d, d * d)  # (s0o,s0i),(s1o,s1i)
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    k = max(1, int(np.count_nonzero(s > svd_min * s[0])))
    U, s, Vh = U[:, :k], s[:k], Vh[:k, :]
    sq = np.sqrt(s)
    g0 = (U * sq).reshape(d, d, k)            # s0o, s0i, a
    g1 = np.transpose((sq[:, None] * Vh).reshape(k, d, d), (1, 2, 0))  # s1o, s1i, a
    return g0, g1


def apply_nn_gate_and_truncate_(ab, psi, s0, s1, dirn, G_nn4, chi_max, svd_min, max_iter, tol_iter) -> float:
    """Apply a 2-site gate across bond ``(s0, s1)`` and NTU-truncate it back."""
    g0, g1 = _split_nn_gate(G_nn4)
    g0b, g1b = ab.asarray(g0), ab.asarray(g1)
    A0, A1 = psi[s0], psi[s1]
    Dt0, Dl0, Db0, Dr0, _ = A0.shape
    Et1, El1, Eb1, Er1, _ = A1.shape
    aux = g0.shape[2]
    if dirn == "lr":  # s0 left (enlarge r), s1 right (enlarge l)
        psi[s0] = ab.einsum("tlbri,oia->tlbrao", A0, g0b).reshape(Dt0, Dl0, Db0, Dr0 * aux, -1)
        psi[s1] = ab.einsum("tlbri,oia->tlabro", A1, g1b).reshape(Et1, El1 * aux, Eb1, Er1, -1)
    else:  # 'tb': s0 top (enlarge b), s1 bottom (enlarge t)
        psi[s0] = ab.einsum("tlbri,oia->tlbaro", A0, g0b).reshape(Dt0, Dl0, Db0 * aux, Dr0, -1)
        psi[s1] = ab.einsum("tlbri,oia->talbro", A1, g1b).reshape(Et1 * aux, El1, Eb1, Er1, -1)
    return truncate_bond(ab, psi, s0, s1, dirn, chi_max, svd_min, max_iter, tol_iter)


def _local_layer(ab, psi, ops: PEPSOps, payload, step_data, coeff):
    lat = payload["lattice"]
    Ly = int(lat["Ly"])
    lb = lat.get("local_blocks")
    if lb is not None:
        # analog_3: one spatially-uniform 3x3 gate per layer (static blocks + complex drive).
        H = ops.matrix_hamiltonian(lb.static, lb.drive_420, step_data["drive_coeffs"]["drive_420"])
        gate = G.gate_local_exp(coeff, H, hermitian=lb.hermitian)
        for site_2d in range(int(lat["N"])):
            apply_local_gate_(ab, psi, (site_2d // Ly, site_2d % Ly), gate)
        return
    inv = np.asarray(lat["inv_snake"], dtype=int)
    oR = np.asarray(step_data["omega_R_1d"], dtype=float)
    ohf = np.asarray(step_data["omega_hf_1d"], dtype=float)
    dR = np.asarray(step_data["delta_R_1d"], dtype=float)
    dhf = np.asarray(step_data["delta_hf_1d"], dtype=float)
    cache: dict = {}
    for site_2d in range(int(lat["N"])):
        pos = int(inv[site_2d])
        coord = (site_2d // Ly, site_2d % Ly)
        key = (oR[pos], ohf[pos], dR[pos], dhf[pos])
        gate = cache.get(key)
        if gate is None:
            gate = G.gate_local_exp(coeff, ops.local_hamiltonian(*key))
            cache[key] = gate
        apply_local_gate_(ab, psi, coord, gate)


def _nn_layer(ab, psi, ops: PEPSOps, payload, coeff, chi_max, svd_min, max_iter, tol_iter):
    lat = payload["lattice"]
    Ly = int(lat["Ly"])
    s2d = np.asarray(lat["snake_to_2d"], dtype=int)
    Hnn = ops.nn_hamiltonian()
    cache: dict = {}
    errs = []
    for i_pos, j_pos, strength in lat["vdw_pairs_1d"]:
        i2 = int(s2d[int(i_pos) - 1])
        j2 = int(s2d[int(j_pos) - 1])
        if i2 == j2 or abs(float(strength)) == 0:
            continue
        ci = (i2 // Ly, i2 % Ly)
        cj = (j2 // Ly, j2 % Ly)
        dirn = psi.nn_bond_dirn(ci, cj)  # raises if not lattice-adjacent (PEPS nn-gate limit)
        if dirn in ("rl", "bt"):
            ci, cj, dirn = cj, ci, dirn[::-1]
        gate = cache.get(float(strength))
        if gate is None:
            gate = G.gate_nn_exp(coeff, float(strength) * Hnn)
            cache[float(strength)] = gate
        errs.append(
            apply_nn_gate_and_truncate_(ab, psi, ci, cj, dirn, gate, chi_max, svd_min, max_iter, tol_iter)
        )
    return errs


def trotter_step(ab, psi, ops, payload, step_data, *, dt, chi_max, svd_min, max_iter, tol_iter, imag=False):
    """One 2nd-order Trotter step; returns the max bond truncation error.

    Real time: ``local(0.5j dt) + nn(1j dt) + local(0.5j dt)``.
    Imag time: ``local(0.5 dt) + nn(dt) + local(0.5 dt)`` (``dt`` is ``dtau``);
    PEPS tensors are renormalized after the non-unitary local layers.
    """
    if imag:
        loc, nn = 0.5 * dt, dt
    else:
        loc, nn = 0.5j * dt, 1j * dt
    _local_layer(ab, psi, ops, payload, step_data, loc)
    if imag:
        _renormalize(ab, psi)
    errs = _nn_layer(ab, psi, ops, payload, nn, chi_max, svd_min, max_iter, tol_iter)
    _local_layer(ab, psi, ops, payload, step_data, loc)
    if imag:
        _renormalize(ab, psi)
    return max(errs, default=0.0)


def _renormalize(ab, psi: FinitePEPS) -> None:
    """Rescale each site tensor to unit max-abs (keeps imag-time amplitudes bounded)."""
    for s in psi.sites():
        A = psi[s]
        m = ab.norm(A)
        if m > 0:
            psi[s] = A / m
