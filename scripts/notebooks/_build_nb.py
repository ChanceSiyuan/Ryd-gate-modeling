"""Rebuild 01r_tfim_critical_field.ipynb as a flat exact/MPS/rydtn-PEPS notebook.

Run once: `python scripts/notebooks/_build_nb.py`, then delete this file.
"""
import nbformat as nbf

C0 = nbf.v4.new_markdown_cell(r'''# 01r — 2D square-lattice TFIM critical field $g_c$ (exact / MPS / rydtn-PEPS)

2D square transverse-field Ising model $H=-J\sum_{\langle ij\rangle}Z_iZ_j-gJ\sum_iX_i$;
QMC (Blöte–Deng) gives $g_c\simeq3.04438$.

**`1r` → TFIM mapping.** $J=V/4$, $g=2\Omega/V$. We fix $J=1$ via $V_{\rm nn}=4$
(so $g=h_x$, $\Omega=2g$), use nearest-neighbour-only Ising, and set per-site
detunings so $h_z=0$. The Rydberg $V n_i n_j$ is antiferromagnetic, so the order
parameter is the checkerboard $m_s=\frac1N\sum_i s_iZ_i$, $s_i=(-1)^{i_x+i_y}$.
At $h_z=0$, $\langle m_s\rangle=0$, so we use the structure factor
$S(\mathbf q)=\frac1N\sum_{ij}e^{i\mathbf q\cdot(\mathbf r_i-\mathbf r_j)}\langle Z_iZ_j\rangle$
and the correlation ratio $R=1-S(\mathbf Q+\delta\mathbf q)/S(\mathbf Q)$, $\mathbf Q=(\pi,\pi)$.

**Part 1.** $4\times4$ ground-state **energy** by exact diagonalization, MPS (DMRG),
and rydtn-PEPS (imaginary time), swept over $g$ — the three must agree.

**Part 2.** Having confirmed the agreement at $4\times4$, compute $R(g)$ at $L=4,5,6$
(exact only anchors $4\times4$; MPS and rydtn-PEPS scale up) and read $g_c$ from the
correlation-ratio **crossing** of the two largest sizes.

Everything below is written flat — each method is called directly, no helper functions.''')

C1 = nbf.v4.new_code_cell(r'''# Pin BLAS/OpenMP to 1 thread before importing numpy/scipy/tenpy (small 2D-TN tensors).
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

import ryd_gate as rg
from ryd_gate.core.level_structures import InteractionSpec
from ryd_gate.lattice import Register
from ryd_gate.core.operators import (
    materialize_sparse_operator, WeightedProjectorSumSpec, is_operator_spec,
)
from ryd_gate.protocols.lattice_dynamics import TFIMQuenchProtocol, tfim_to_rydberg_controls
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.protocol_context import TNProtocolContext, pin_deltas_from_params
from ryd_gate.backends.tn_common.compiler import TNEvolutionIR
from ryd_gate.backends.tenpy_mps.backends import TenpyDMRGBackend
from ryd_gate.backends.rydtn import RydTNPEPSBackend

plt.rcParams.update({"figure.dpi": 120})
J, V_NN, G_C_REF = 1.0, 4.0, 3.04438
print("setup done; QMC reference g_c =", G_C_REF)''')

C2 = nbf.v4.new_markdown_cell(r'''## Part 1 — $4\times4$ ground-state energy: exact vs MPS vs rydtn-PEPS

All three solve the *same* nearest-neighbour TFIM at field $g$ on a $4\times4$ open
cluster (with $h_z=0$ pins) and should return the same ground-state energy.''')

C3 = nbf.v4.new_code_cell(r'''G1 = np.round(np.linspace(2.6, 3.5, 6), 4)
E_exact, E_mps, E_peps = [], [], []
for g in G1:
    t0 = time.perf_counter()

    # --- exact: assemble the full 1r Hamiltonian and take the Lanczos ground state ---
    geom = Register.rectangle(4, 4, spacing_um=1.0)
    proto = TFIMQuenchProtocol(hx=g * J, hz=0.0, t_gate=1.0)
    system = rg.RydbergSystem.from_lattice(
        geom, "1r", interaction=InteractionSpec(C6=V_NN, mode="nn"), protocol=proto)
    irh = rg.compile_hamiltonian_ir(system, system.unpack_params([]))
    H = None
    for term in list(irh.static_terms) + list(irh.drive_terms):
        c = term.coefficient(0) if callable(term.coefficient) else term.coefficient
        op = materialize_sparse_operator(term.operator, system.basis) if is_operator_spec(term.operator) else term.operator
        contrib = c * op
        if getattr(term, "add_hermitian_conjugate", False):
            contrib = contrib + np.conj(c) * op.conj().T
        H = contrib if H is None else H + contrib
    Eex = float(eigsh(H.tocsc(), k=1, which="SA")[0][0])

    # --- MPS (DMRG) ---
    spec = create_tn_lattice_spec(Lx=4, Ly=4, V_nn=V_NN, Omega=1.0, bc="open",
                                  level_structure="1r", interaction_mode="nn")
    ctrl = tfim_to_rydberg_controls(TNProtocolContext(spec), hx=g * J, hz=0.0)
    pin = pin_deltas_from_params({"pin_deltas": ctrl.pin_deltas}, spec.N)
    res_m = TenpyDMRGBackend(chi_max=128, n_sweeps=14, svd_min=1e-10).find_ground_state(
        replace(spec, Omega=ctrl.Omega), ctrl.Delta, pin_deltas=pin, initial_state="af1")
    Em = float(res_m.metadata["energy"])

    # --- rydtn-PEPS (imaginary-time NTU ground state) ---
    params = proto.unpack_params([], TNProtocolContext(spec))
    ir = TNEvolutionIR(spec=spec, protocol=proto, params=params, metadata={})
    res_p = RydTNPEPSBackend(chi_max=8, dt=0.05).find_ground_state(
        ir, dtau_schedule=((0.1, 40), (0.03, 40), (0.02, 50)), initial_state="af1")
    Ep = float(res_p.metadata["energy"])

    E_exact.append(Eex); E_mps.append(Em); E_peps.append(Ep)
    print(f"g={g:.4f}  E_exact={Eex:.4f}  E_mps={Em:.4f}  E_peps={Ep:.4f}  "
          f"d(mps)={Em - Eex:+.1e}  d(peps)={Ep - Eex:+.1e}  t={time.perf_counter() - t0:.0f}s",
          flush=True)

E_exact, E_mps, E_peps = np.array(E_exact), np.array(E_mps), np.array(E_peps)
print(f"\nmax |E_mps  - E_exact| = {np.max(np.abs(E_mps - E_exact)):.2e}")
print(f"max |E_peps - E_exact| = {np.max(np.abs(E_peps - E_exact)):.2e}")''')

C4 = nbf.v4.new_code_cell(r'''plt.figure(figsize=(6.2, 4.2))
plt.plot(G1, E_exact, "k-o", label="exact (eigsh)")
plt.plot(G1, E_mps, "C0-s", label="MPS (DMRG, $\\chi$=128)")
plt.plot(G1, E_peps, "C1-^", label="rydtn-PEPS (D=8)")
plt.axvline(G_C_REF, color="gray", ls="--", lw=1, label=f"$g_c$={G_C_REF}")
plt.xlabel(r"$g = 2\Omega/V$"); plt.ylabel("ground-state energy $E$")
plt.title(r"$4\times4$ TFIM ground-state energy")
plt.legend(); plt.tight_layout(); plt.show()''')

C5 = nbf.v4.new_markdown_cell(r'''## Part 2 — correlation ratio $R(g)$ and $g_c$ from the crossing

The three methods agree at $4\times4$, so we now scale up the correlation ratio
$R=1-S(\mathbf Q+\delta\mathbf q)/S(\mathbf Q)$ ($\mathbf Q=(\pi,\pi)$,
$\delta\mathbf q=(2\pi/L,0)$). $R$ is RG-invariant, so $R(g)$ curves for different
$L$ cross at $g_c$. We run MPS at $L=4,5,6$ and rydtn-PEPS at $L=4,5$; exact anchors
$L=4$. (Open clusters bias the crossing slightly above the bulk $g_c$.)''')

C6 = nbf.v4.new_code_cell(r'''G2 = np.array([2.8, 3.0, 3.04438, 3.15, 3.3])
MPS_CHI = {4: 128, 5: 200, 6: 300}
R_mps = {4: [], 5: [], 6: []}
R_peps = {4: [], 5: []}
R_exact = {4: []}

for L in (4, 5, 6):
    spec = create_tn_lattice_spec(Lx=L, Ly=L, V_nn=V_NN, Omega=1.0, bc="open",
                                  level_structure="1r", interaction_mode="nn")
    N = spec.N
    coords = spec.coords                          # (N, 2) in 2D site order
    idx = spec.snake_to_2d                          # snake(1D) -> 2D site
    Q = np.array([np.pi, np.pi]); dq = np.array([2 * np.pi / L, 0.0])
    phQ = np.exp(1j * (coords @ Q)); phQd = np.exp(1j * (coords @ (Q + dq)))
    for g in G2:
        t0 = time.perf_counter()
        # --- MPS correlation matrix <Z_i Z_j> = 4<Sz_i Sz_j> ---
        ctrl = tfim_to_rydberg_controls(TNProtocolContext(spec), hx=g * J, hz=0.0)
        pin = pin_deltas_from_params({"pin_deltas": ctrl.pin_deltas}, N)
        res_m = TenpyDMRGBackend(chi_max=MPS_CHI[L], n_sweeps=14, svd_min=1e-10).find_ground_state(
            replace(spec, Omega=ctrl.Omega), ctrl.Delta, pin_deltas=pin, initial_state="af1")
        Csnake = 4.0 * np.asarray(res_m.psi_final.correlation_function("Sz", "Sz"))
        Cm = np.empty_like(Csnake); Cm[np.ix_(idx, idx)] = Csnake     # -> 2D site order
        sQ = float(np.real(np.conj(phQ) @ Cm @ phQ) / N)
        sQd = float(np.real(np.conj(phQd) @ Cm @ phQd) / N)
        R_mps[L].append(1.0 - sQd / sQ if abs(sQ) > 1e-12 else np.nan)

        # --- rydtn-PEPS correlation matrix (only up to L=5 here) ---
        if L <= 5:
            proto = TFIMQuenchProtocol(hx=g * J, hz=0.0, t_gate=1.0)
            params = proto.unpack_params([], TNProtocolContext(spec))
            ir = TNEvolutionIR(spec=spec, protocol=proto, params=params, metadata={})
            res_p = RydTNPEPSBackend(chi_max=6, dt=0.05, measure_chi=48).find_ground_state(
                ir, dtau_schedule=((0.1, 30), (0.03, 30), (0.02, 30)),
                observables=["czz_full"], initial_state="af1")
            Cp = res_p.metadata["obs"]["czz_full"]                    # already 2D site order
            sQ = float(np.real(np.conj(phQ) @ Cp @ phQ) / N)
            sQd = float(np.real(np.conj(phQd) @ Cp @ phQd) / N)
            R_peps[L].append(1.0 - sQd / sQ if abs(sQ) > 1e-12 else np.nan)

        # --- exact correlation matrix (anchor at L=4 only) ---
        if L == 4:
            geom = Register.rectangle(4, 4, spacing_um=1.0)
            proto = TFIMQuenchProtocol(hx=g * J, hz=0.0, t_gate=1.0)
            system = rg.RydbergSystem.from_lattice(
                geom, "1r", interaction=InteractionSpec(C6=V_NN, mode="nn"), protocol=proto)
            irh = rg.compile_hamiltonian_ir(system, system.unpack_params([]))
            H = None
            for term in list(irh.static_terms) + list(irh.drive_terms):
                c = term.coefficient(0) if callable(term.coefficient) else term.coefficient
                op = materialize_sparse_operator(term.operator, system.basis) if is_operator_spec(term.operator) else term.operator
                contrib = c * op
                if getattr(term, "add_hermitian_conjugate", False):
                    contrib = contrib + np.conj(c) * op.conj().T
                H = contrib if H is None else H + contrib
            psi0 = eigsh(H.tocsc(), k=1, which="SA")[1][:, 0]
            Zpsi = np.empty((N, psi0.size), dtype=complex)            # Z_i |psi>, Z_i = 2 n_r,i - I
            for i in range(N):
                ei = np.zeros(N); ei[i] = 1.0
                Pri = materialize_sparse_operator(WeightedProjectorSumSpec("r", tuple(ei)), system.basis)
                Zpsi[i] = (2.0 * Pri) @ psi0 - psi0
            Cx = np.real(Zpsi.conj() @ Zpsi.T)                        # <Z_i Z_j>
            sQ = float(np.real(np.conj(phQ) @ Cx @ phQ) / N)
            sQd = float(np.real(np.conj(phQd) @ Cx @ phQd) / N)
            R_exact[4].append(1.0 - sQd / sQ if abs(sQ) > 1e-12 else np.nan)
        print(f"  [L={L}] g={g:.4f}  t={time.perf_counter() - t0:.0f}s", flush=True)''')

C7 = nbf.v4.new_code_cell(r'''plt.figure(figsize=(7.2, 4.6))
for L in (4, 5, 6):
    plt.plot(G2, R_mps[L], "-o", label=f"MPS L={L}")
for L in (4, 5):
    plt.plot(G2, R_peps[L], "--s", label=f"PEPS L={L}")
plt.plot(G2, R_exact[4], "k:^", lw=1.5, label="exact L=4")
plt.axvline(G_C_REF, color="gray", ls="--", lw=1, label=f"$g_c$={G_C_REF}")
plt.xlabel(r"$g = 2\Omega/V$"); plt.ylabel(r"$R = 1 - S(Q+\delta q)/S(Q)$")
plt.title("correlation ratio (curves cross at $g_c$)")
plt.legend(fontsize=8); plt.tight_layout(); plt.show()

# g_c from the crossing of two R(g) curves (inline linear interpolation)
gg = np.linspace(G2.min(), G2.max(), 800)
d_mps = np.interp(gg, G2, R_mps[6]) - np.interp(gg, G2, R_mps[5])
sc = np.where(np.sign(d_mps[:-1]) != np.sign(d_mps[1:]))[0]
gc_mps = float(gg[sc[0]] - d_mps[sc[0]] * (gg[sc[0] + 1] - gg[sc[0]]) / (d_mps[sc[0] + 1] - d_mps[sc[0]])) if sc.size else float("nan")
d_peps = np.interp(gg, G2, R_peps[5]) - np.interp(gg, G2, R_peps[4])
scp = np.where(np.sign(d_peps[:-1]) != np.sign(d_peps[1:]))[0]
gc_peps = float(gg[scp[0]] - d_peps[scp[0]] * (gg[scp[0] + 1] - gg[scp[0]]) / (d_peps[scp[0] + 1] - d_peps[scp[0]])) if scp.size else float("nan")

print(f"g_c  (MPS  L=5x6 crossing) = {gc_mps:.4f}")
print(f"g_c  (PEPS L=4x5 crossing) = {gc_peps:.4f}")
print(f"g_c  (QMC reference)       = {G_C_REF}")''')

nb = nbf.v4.new_notebook()
nb.cells = [C0, C1, C2, C3, C4, C5, C6, C7]
nb.metadata = {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
}
nbf.write(nb, "scripts/notebooks/01r_tfim_critical_field.ipynb")
print("wrote scripts/notebooks/01r_tfim_critical_field.ipynb with", len(nb.cells), "cells")
