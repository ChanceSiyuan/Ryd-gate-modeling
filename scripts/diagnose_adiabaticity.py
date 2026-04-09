#!/usr/bin/env python3
"""Diagnose why P_gg / P_rg do NOT shrink with longer gate time.

Three companion experiments. The reference operating point matches the
experiments stored in ``results/exp{1,2}_addressing_opt_grid.csv``: the
two-atom analog 3-level model with distance = 4 μm, beam waist = 1 μm,
addressing wavelength λ = 783.5 nm, single-beam power P = 320 μW, and a
±40 MHz Landau–Zener detuning ramp on the global drive. All decay channels
are disabled (default) so we look at *coherent* errors only.

Experiment 1 — Adiabatic-fidelity tracking
    For T ∈ {4.5, 9, 50, 200} μs, capture ψ(t) at every step and compute
    F_adiab(t) = max_j |⟨e_j(t) | ψ(t)⟩|², the overlap of the wavefunction
    with the closest *instantaneous* eigenstate of H(t). If F_adiab stays
    pinned at 1, the evolution is on an eigen-branch (i.e. perfectly
    adiabatic) and any leftover P_gg / P_rg cannot be a Landau–Zener
    diabatic excitation — it must be a property of the terminal eigenstate
    itself or of the post-eigenbasis projection at t = T.

Experiment 2 — Extreme-slow-sweep test
    Sweep T from 1 μs up to 200 μs at the same operating point. If the
    residual errors come from LZ diabaticity, P_gg should drop exponentially
    in T. If they come from finite-sweep boundary impurity (the terminal
    |+⟩ eigenstate is not a pure |gr⟩) or from the "wrong-direction"
    pinning crossing on atom A (slower → atom A follows the eigenstate
    onto |r⟩_A more accurately = pin LEAKS more), they will not.

Experiment 3 — Boundary-truncation test
    Hold the sweep rate α = (Δ_end − Δ_start) / T constant and vary the
    half-width Δ_max ∈ {20, 30, 40, 60, 80, 120, 160} MHz. With α fixed,
    the LZ diabatic probability is fixed, so any change in residual P_gg
    is purely a finite-sweep boundary effect. Theory predicts the residual
    impurity of the terminal eigenstate scales as Ω_eff² / (4 Δ_max²).

Usage
-----
    uv run --active python scripts/diagnose_adiabaticity.py

Outputs (under ``results/``):
    diagnose_exp1_adiabatic_fidelity.png
    diagnose_exp2_slow_sweep.{png,csv}
    diagnose_exp3_boundary.{png,csv}
"""
from __future__ import annotations

import os
import time as _time

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi

from ryd_gate import simulate
from ryd_gate.compilers.dense_atomic import DenseAtomicCompiler
from ryd_gate.core.atomic_system import POWER_REF_UW, compute_shift_scatter
from ryd_gate.core.models.analog_3level import Analog3LevelModel
from ryd_gate.protocols.sweep import SweepProtocol


# ──────────────────────────────────────────────────────────────────────────
# Reference physical parameters (must match scan_local_addressing.py)
# ──────────────────────────────────────────────────────────────────────────
DISTANCE_UM = 4.0
WAIST_UM = 1.0
DELTA_HZ = 2.4e9
RABI_420_HZ = 135e6
RABI_1013_HZ = 135e6

# Reference operating point (≈ exp1's best (λ, P) cell)
WL_REF_NM = 783.5
P_REF_UW = 320.0
DELTA_START_MHZ = -40.0
DELTA_END_MHZ = 40.0

OUTDIR = "results"


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
_PARENT_MODEL = None  # built once in main(), inherited by fork-based workers


def build_model() -> Analog3LevelModel:
    """Pure-unitary 3-level model (decay channels are off by default).

    Returns the parent-process pre-built model when one exists (so fork-based
    workers don't each race on ARC's SQLite literature-values cache).
    """
    global _PARENT_MODEL
    if _PARENT_MODEL is not None:
        return _PARENT_MODEL
    _PARENT_MODEL = Analog3LevelModel.from_defaults(
        detuning_sign=1, blackmanflag=True,
        distance_um=DISTANCE_UM,
        Delta_Hz=DELTA_HZ,
        rabi_420_Hz=RABI_420_HZ,
        rabi_1013_Hz=RABI_1013_HZ,
    )
    return _PARENT_MODEL


def make_protocol_and_x(model, wl_nm, power_uw, t_gate_us,
                        delta_start_mhz, delta_end_mhz):
    """Build a fresh SweepProtocol + x_sweep for one (λ, P, T, Δ_window) cell."""
    system = model.system
    eta = np.exp(-2 * (DISTANCE_UM / WAIST_UM) ** 2)
    shift_ref, _ = compute_shift_scatter(wl_nm)
    scale = power_uw / POWER_REF_UW
    delta_A = 2 * pi * float(shift_ref) * scale  # rad/s
    delta_B = eta * delta_A
    ac_stark_peak = system.rabi_420 ** 2 / (4 * abs(system.Delta))

    protocol = SweepProtocol(
        addressing={0: delta_A, 1: delta_B},
        ac_stark_shift=ac_stark_peak,
    )
    t_gate = t_gate_us * 1e-6
    x_sweep = [
        2 * pi * delta_start_mhz * 1e6 / system.rabi_eff,
        2 * pi * delta_end_mhz * 1e6 / system.rabi_eff,
        t_gate / system.time_scale,
    ]
    return protocol, x_sweep, delta_A, delta_B


def measure_pops(model, psi):
    return {
        "P_gg": model.observables.measure("pop_gg", psi),
        "P_gr": model.observables.measure("pop_gr", psi),
        "P_rg": model.observables.measure("pop_rg", psi),
        "P_rr": model.observables.measure("pop_rr", psi),
    }


def build_H_at_t(ir, t):
    """Reconstruct the dense Hamiltonian H(t) from a compiled IR."""
    H = np.zeros((ir.dim, ir.dim), dtype=complex)
    for term in ir.static_terms:
        c = term.coefficient(0) if callable(term.coefficient) else term.coefficient
        H += c * np.asarray(term.operator)
    for term in ir.drive_terms:
        c = term.coefficient(t) if callable(term.coefficient) else term.coefficient
        H += c * np.asarray(term.operator)
    return H


def adiabatic_fidelity_history(ir, times, states):
    """For each step compute F(t) = max_j |⟨e_j(t)|ψ(t)⟩|²."""
    F = np.empty(len(times))
    for i, (t, psi) in enumerate(zip(times, states.T)):
        H = build_H_at_t(ir, t)
        H = 0.5 * (H + H.conj().T)  # symmetrize numerically
        evals, evecs = np.linalg.eigh(H)
        overlaps = np.abs(evecs.conj().T @ psi) ** 2
        F[i] = float(overlaps.max())
    return F


# Top-level worker functions (must be picklable for ProcessPoolExecutor.map).

_EXP1_N_PTS = 301


def _exp1_one_T(T_us, n_pts=None):
    if n_pts is None:
        n_pts = _EXP1_N_PTS
    """Run one (λ, P, T) cell for experiment 1 and return everything needed."""
    model = build_model()
    psi0 = np.zeros(9, dtype=complex); psi0[0] = 1.0
    protocol, x, _, _ = make_protocol_and_x(
        model, WL_REF_NM, P_REF_UW, T_us, DELTA_START_MHZ, DELTA_END_MHZ)
    params = protocol.unpack_params(x, model.system)
    t_gate = params["t_gate"]
    t_eval = np.linspace(0.0, t_gate, n_pts)
    t0 = _time.time()
    result = simulate(model, protocol, x, psi0, t_eval=t_eval)
    ir = DenseAtomicCompiler().compile(model.system, protocol, params)
    F = adiabatic_fidelity_history(ir, result.times, result.states)
    elapsed = _time.time() - t0
    m = measure_pops(model, result.psi_final)
    return T_us, result.times, F, m, elapsed


def _exp2_one_T(T_us):
    """Run one (λ, P, T) cell for experiment 2 (final populations only)."""
    model = build_model()
    psi0 = np.zeros(9, dtype=complex); psi0[0] = 1.0
    protocol, x, _, _ = make_protocol_and_x(
        model, WL_REF_NM, P_REF_UW, T_us, DELTA_START_MHZ, DELTA_END_MHZ)
    t0 = _time.time()
    result = simulate(model, protocol, x, psi0)
    elapsed = _time.time() - t0
    m = measure_pops(model, result.psi_final)
    return T_us, m, elapsed


def _exp3_one_dmax(dmax_T):
    dmax, T_us = dmax_T
    model = build_model()
    psi0 = np.zeros(9, dtype=complex); psi0[0] = 1.0
    protocol, x, _, _ = make_protocol_and_x(
        model, WL_REF_NM, P_REF_UW, T_us, -dmax, dmax)
    t0 = _time.time()
    result = simulate(model, protocol, x, psi0)
    elapsed = _time.time() - t0
    m = measure_pops(model, result.psi_final)
    return dmax, T_us, m, elapsed


_MAX_WORKERS = int(os.environ.get("DIAG_WORKERS", "5"))


def _parallel_map(fn, items):
    """Run fn over items in parallel using fork-based ProcessPoolExecutor."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    n_workers = min(len(items), _MAX_WORKERS)
    if n_workers <= 1:
        return [fn(it) for it in items]
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        return list(ex.map(fn, items))


# ──────────────────────────────────────────────────────────────────────────
# Experiment 1 — Adiabatic-fidelity tracking
# ──────────────────────────────────────────────────────────────────────────
def experiment1():
    print("=" * 64)
    print("Experiment 1: adiabatic fidelity tracking")
    print("=" * 64)

    Ts_us = [4.5, 9.0, 20.0]   # T>20 μs starves under cpu contention
    n_pts = 251  # samples per trajectory

    fig, axes = plt.subplots(len(Ts_us), 1,
                             figsize=(9, 2.4 * len(Ts_us)),
                             sharex=False)

    global _EXP1_N_PTS
    _EXP1_N_PTS = n_pts
    print(f"  running {len(Ts_us)} T values in parallel...")
    results_per_T = _parallel_map(_exp1_one_T, Ts_us)

    summary = []
    for ax, (T_us, times, F, m, elapsed) in zip(axes, results_per_T):
        max_dropout = float((1.0 - F).max())
        summary.append((T_us, max_dropout, m))
        print(f"  T={T_us:6.1f} μs  max(1-F)={max_dropout:.4e}  "
              f"P_gg={m['P_gg']:.4e}  P_gr={m['P_gr']:.6f}  "
              f"P_rg={m['P_rg']:.4e}  ({elapsed:.1f}s)")

        ax.semilogy(times * 1e6, np.maximum(1.0 - F, 1e-16),
                    color='k', lw=1.4)
        ax.set_ylim(1e-14, 1.0)
        ax.set_xlabel("time (μs)")
        ax.set_ylabel(r"$1 - F_{\rm adiab}(t)$")
        ax.set_title(
            f"T = {T_us:.1f} μs   "
            f"max(1−F) = {max_dropout:.2e}   "
            f"P_gg = {m['P_gg']:.2e}   "
            f"P_gr = {m['P_gr']:.4f}   "
            f"P_rg = {m['P_rg']:.2e}",
            fontsize=10)
        ax.grid(alpha=0.3, which='both')

    fig.suptitle(
        "Exp 1 — Adiabatic fidelity tracking "
        f"(λ={WL_REF_NM} nm, P={P_REF_UW:.0f} μW, ±{abs(DELTA_START_MHZ):.0f} MHz sweep)",
        y=1.0)
    fig.tight_layout()
    path = os.path.join(OUTDIR, "diagnose_exp1_adiabatic_fidelity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}\n")
    return summary


# ──────────────────────────────────────────────────────────────────────────
# Experiment 2 — Extreme-slow-sweep test
# ──────────────────────────────────────────────────────────────────────────
def experiment2():
    print("=" * 64)
    print("Experiment 2: extreme-slow-sweep test (pure unitary)")
    print("=" * 64)

    Ts_us = np.array([1.0, 2.0, 4.5, 9.0, 20.0])
    print(f"  running {len(Ts_us)} T values in parallel...")
    out = _parallel_map(_exp2_one_T, list(Ts_us))
    out = sorted(out, key=lambda r: r[0])
    rows = []
    for T_us, m, elapsed in out:
        rows.append((T_us, m["P_gg"], m["P_gr"], m["P_rg"], m["P_rr"]))
        print(f"  T={T_us:6.1f} μs   "
              f"P_gg={m['P_gg']:.4e}  P_gr={m['P_gr']:.6f}  "
              f"P_rg={m['P_rg']:.4e}  P_rr={m['P_rr']:.4e}  ({elapsed:.1f}s)")
    rows = np.array(rows)

    # Theory: pure-LZ diabatic prediction for atom B
    Omega_eff = build_model().system.rabi_eff  # rad/s
    delta_window_rad = 2 * pi * (DELTA_END_MHZ - DELTA_START_MHZ) * 1e6
    alpha = delta_window_rad / (Ts_us * 1e-6)
    P_LZ = np.exp(-pi * Omega_eff ** 2 / (2 * alpha))

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.loglog(Ts_us, np.maximum(rows[:, 1], 1e-16),  'o-', label="P_gg (crosstalk)")
    ax.loglog(Ts_us, np.maximum(rows[:, 3], 1e-16),  's-', label="P_rg (atom A pinning leak)")
    ax.loglog(Ts_us, np.maximum(rows[:, 4], 1e-16),  '^-', label="P_rr")
    ax.loglog(Ts_us, np.maximum(1.0 - rows[:, 2], 1e-16), 'x--',
              label="1 − P_gr (target infidelity)")
    ax.loglog(Ts_us, P_LZ, 'k:', alpha=0.7,
              label=r"pure-LZ prediction $e^{-\pi\Omega^2/2\alpha}$")
    ax.set_xlabel("Gate time T (μs)")
    ax.set_ylabel("Population")
    ax.set_title(
        f"Exp 2 — Pure unitary slow-sweep test\n"
        f"λ={WL_REF_NM} nm, P={P_REF_UW:.0f} μW, ±{abs(DELTA_START_MHZ):.0f} MHz sweep")
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    path = os.path.join(OUTDIR, "diagnose_exp2_slow_sweep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    np.savetxt(os.path.join(OUTDIR, "diagnose_exp2_slow_sweep.csv"), rows,
               fmt="%.6e", delimiter=",",
               header="T_us,P_gg,P_gr,P_rg,P_rr")
    print(f"Saved {path}\n")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# Experiment 3 — Boundary-truncation test
# ──────────────────────────────────────────────────────────────────────────
def experiment3():
    print("=" * 64)
    print("Experiment 3: boundary-truncation test (constant α)")
    print("=" * 64)

    T_ref_us = 4.5
    delta_max_ref = abs(DELTA_START_MHZ)
    alpha = (2.0 * delta_max_ref) / T_ref_us  # MHz/μs, kept constant
    delta_max_list = np.array([20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0])

    items = [(dmax, (2.0 * dmax) / alpha) for dmax in delta_max_list]
    print(f"  running {len(items)} Δ_max values in parallel...")
    out = _parallel_map(_exp3_one_dmax, items)
    out = sorted(out, key=lambda r: r[0])
    rows = []
    for dmax, T_us, m, elapsed in out:
        rows.append((dmax, T_us, m["P_gg"], m["P_gr"], m["P_rg"], m["P_rr"]))
        print(f"  Δmax={dmax:6.1f} MHz  T={T_us:6.2f} μs   "
              f"P_gg={m['P_gg']:.4e}  P_rg={m['P_rg']:.4e}  ({elapsed:.1f}s)")
    rows = np.array(rows)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    dmax = rows[:, 0]
    ax.loglog(dmax, np.maximum(rows[:, 2], 1e-16), 'o-', label="P_gg")
    ax.loglog(dmax, np.maximum(rows[:, 4], 1e-16), 's-', label="P_rg")
    # 1/Δ² reference, anchored to first point
    inv_sq = (dmax[0] / dmax) ** 2 * rows[0, 2]
    ax.loglog(dmax, inv_sq, 'k--', alpha=0.5, label=r"$\propto 1/\Delta_{\max}^{\,2}$")
    ax.set_xlabel(r"Sweep half-width $\Delta_\max$ (MHz)")
    ax.set_ylabel("Population")
    ax.set_title(
        f"Exp 3 — Boundary scaling at constant sweep rate\n"
        f"α = 2Δ/T = {alpha:.1f} MHz/μs")
    ax.legend(loc='best')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    path = os.path.join(OUTDIR, "diagnose_exp3_boundary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    np.savetxt(os.path.join(OUTDIR, "diagnose_exp3_boundary.csv"), rows,
               fmt="%.6e", delimiter=",",
               header="delta_max_mhz,T_us,P_gg,P_gr,P_rg,P_rr")
    print(f"Saved {path}\n")
    return rows


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    # Pre-build the model in the parent so fork-based worker pools inherit
    # it instead of each racing on ARC's SQLite cache.
    print("Pre-building model in parent process (warms ARC cache)...")
    build_model()
    print(f"Reference operating point: λ={WL_REF_NM} nm, P={P_REF_UW} μW, "
          f"sweep ±{abs(DELTA_START_MHZ)} MHz")
    print(f"  distance={DISTANCE_UM} μm  waist={WAIST_UM} μm  "
          f"Δ_int/(2π)={DELTA_HZ/1e9} GHz  Ω_420={RABI_420_HZ/1e6} MHz\n")

    s1 = experiment1()
    r2 = experiment2()
    r3 = experiment3()

    print("=" * 64)
    print("Summary")
    print("=" * 64)
    print("\nExp 1 max(1−F_adiab) per T:")
    for T_us, drop, m in s1:
        print(f"  T={T_us:6.1f} μs  max(1-F)={drop:.4e}  "
              f"P_gg={m['P_gg']:.4e}  P_rg={m['P_rg']:.4e}")
    print(f"\nExp 2 final populations vs T (see CSV).")
    print(f"Exp 3 final populations vs Δ_max (see CSV).")
    print("\nDone.")


if __name__ == "__main__":
    main()
