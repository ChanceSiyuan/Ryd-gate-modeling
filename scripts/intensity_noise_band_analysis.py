#!/usr/bin/env python3
"""Intensity-noise (RIN) filter-function analysis of the optimal 297 nm gate.

Frequency noise enters the gate as H -> H_0 + 2 pi dnu(t) N_r; relative intensity
noise rho(t) = dI/I enters as H -> H_0 + rho(t) B(t) with B(t) = H_drive(t) / 2,
because Omega scales with sqrt(I) and this model carries no other
intensity-dependent term (AC-Stark of the off-resonant r_garb leg emerges from the
drive dynamics, so it is inside B automatically).  Writing B = 2 pi M with
M(t) = H_drive(t) / (4 pi) puts intensity noise in exactly the phase-noise form, so
:func:`ryd_gate.phase_noise.filter_kernel` and

    eps_int^s = 2 pi^2 sum_b S_rho(f_b) K_b^{int,s}

apply verbatim with S_rho the one-sided RIN PSD (1/Hz) at 297 nm.  The measured
RIN is of the 1180/1187 nm fundamental; each undepleted SHG stage doubles dP/P, so
the quadrupler multiplies the RIN PSD by 16 — the same factor as frequency noise,
for a different reason — and that is a lower bound (doubling cavities and power
servos add noise the fundamental measurement cannot see).

Cross-checks run every time:
  1. the same pipeline with M = N_r must reproduce the stored phase-noise kernel
     of the optimal point (filter/ series of the a3.0 store);
  2. the intensity kernel must be converged in the trajectory sampling n_t
     (off-diagonal B picks up blockade/detuning-scale phases that N_r cancels);
  3. the f -> 0 kernel must match a finite-difference propagation with a static
     intensity offset, eps = 4 pi^2 rho0^2 ||Q G(0)||^2, which pins every factor.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(
    os.environ.get("RYD_GATE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, os.fspath(ROOT / "src"))
sys.path.insert(0, os.fspath(ROOT / "scripts"))

import max_leakage_297_sweep as mls  # noqa: E402
import sweeplib  # noqa: E402
from ryd_gate.phase_noise import PhaseNoisePSD, filter_kernel  # noqa: E402

RESULT_ROOT = ROOT / "results" / "297_laser_noise"
TAU = 2.0 * math.pi

# The optimal computed point of the a3.0 store (README section 2/3) and the
# production tolerances its filter/ series was generated with.
RYD_N = 73
T_GATE = 1.0e-6
OMEGA_297 = TAU * 14.25e6
D_SWEEP = TAU * 17.5e6
RTOL, ATOL = 1e-9, 1e-12
RAMP = 0.15
HARMONIC_PSD_FACTOR = 16.0          # two undepleted SHG stages: (2 x 2)^2 in PSD
LOGICAL_INPUTS = mls.LOGICAL_INPUTS
COLORS = {"ECDL": "#c75b32", "seed": "#087f76", "RIN": "#7a4fa3"}


# ── Forward + adjoint trajectories (states kept, unlike the sweep's pass) ────


def trajectory_legs(ops, n_t: int):
    """Forward logical states and backward-propagated basis states on one grid."""
    dim = ops.h_static_diag.size
    times = np.linspace(0.0, T_GATE, n_t)
    om = np.asarray([OMEGA_297])
    dsw = np.asarray([D_SWEEP])

    fwd = sweeplib.integrate_batch(
        ops, T_GATE, {"omega_297": om, "d_sweep": dsw},
        LOGICAL_INPUTS, rhs_factory=mls._297_rhs_factory, dim=dim,
        rtol=RTOL, atol=ATOL, ramp=RAMP, t_eval=times)
    basis = np.arange(dim)
    adj = sweeplib.integrate_batch(
        ops, T_GATE, {"omega_297": om, "d_sweep": dsw},
        tuple(str(i) for i in basis),
        rhs_factory=mls._297_adjoint_rhs_factory, dim=dim,
        rtol=RTOL, atol=ATOL, ramp=RAMP, t_eval=times,
        initial_indices=basis, reverse_time=True)
    if not (np.array_equal(fwd.times, times) and np.array_equal(adj.times, times)):
        raise RuntimeError("integrate_batch did not sample the requested t_eval grid")
    phi = adj.states[::-1]          # tau = T - t flipped onto the forward axis
    psi = fwd.states
    return times, phi, psi, fwd.psi_final


def drive_coefficient(ops, times: np.ndarray) -> np.ndarray:
    """c297(t) of the aggregated Hamiltonian H_drive = Re(c) X + Im(c) Y."""
    s = times / T_GATE
    amp = np.sqrt(mls.envelope(s, RAMP))
    phase = (-T_GATE / TAU) * np.sin(TAU * s) * D_SWEEP
    return ops.amplitude_scale * OMEGA_297 * amp * np.exp(-1j * phase)


def noise_components(ops, times, phi, psi, psi_final):
    """<phi_q(t)|M(t)|psi_s(t)> and its projection for both noise operators.

    Returns ``{"phase": (comp, proj), "int": (comp, proj)}`` with comp shaped
    (n_points, 4, n_t, dim); ``phase`` uses M = N_r, ``int`` M = H_drive / (4 pi).
    """
    dim = ops.h_static_diag.size
    n_r = mls._rydberg_number_diag(dim)
    comp_nr = np.einsum("tpqi,i,tpsi->pstq", phi.conj(), n_r, psi)

    c = drive_coefficient(ops, times) / (2.0 * TAU)      # c297 / (4 pi)
    xpsi = np.einsum("ij,tpsj->tpsi", ops.x297, psi)
    ypsi = np.einsum("ij,tpsj->tpsi", ops.y297, psi)
    comp_int = (
        c.real[None, None, :, None]
        * np.einsum("tpqi,tpsi->pstq", phi.conj(), xpsi)
        + c.imag[None, None, :, None]
        * np.einsum("tpqi,tpsi->pstq", phi.conj(), ypsi)
    )
    out = {}
    for name, comp in (("phase", comp_nr), ("int", comp_int)):
        proj = np.einsum("psq,pstq->pst", psi_final.conj(), comp)
        out[name] = (comp, proj)
    return out


def binned_kernels(times, comp, proj):
    """(4, n_bins) kernels on the store's global frequency bins."""
    f_bins, df_bins = mls.kernel_frequency_bins()
    fine = mls.kernel_fine_per_decade(T_GATE)
    kernels = np.empty((4, f_bins.size))
    for s in range(4):
        kernels[s] = filter_kernel(times, comp[0, s], f_bins, df_bins,
                                   subtract=proj[0, s], fine_per_decade=fine)
    return f_bins, kernels


# ── RIN spectrum ─────────────────────────────────────────────────────────────


def load_rin():
    """(f_meas, S_meas_297) samples and the S_rho^297(f) interpolant (1/Hz).

    dB values interpolate linearly in log10(f) and hold flat outside the measured
    10 Hz - 10 MHz band: below, the 1-10 Hz decade carries ~1e-3 of the error so
    the hold is immaterial; above, holding the -148 dBc/Hz floor is the
    conservative choice (the phase-noise power law would instead decay).
    """
    rows = np.genfromtxt(RESULT_ROOT / "rin_fundamental.csv", delimiter=",",
                         comments="#", skip_header=1)
    f_meas, rin_db = rows[:, 0], rows[:, 1]

    def s_rho(f):
        db = np.interp(np.log10(np.asarray(f, dtype=float)),
                       np.log10(f_meas), rin_db)
        return HARMONIC_PSD_FACTOR * 10.0 ** (db / 10.0)

    return f_meas, s_rho


# ── Cross-checks ─────────────────────────────────────────────────────────────


def check_stored_phase_kernel(f_bins, k_phase) -> float:
    """Max relative deviation from the stored optimal-point phase kernel."""
    stored_bins, stored = _load_stored_kernel()
    if not np.allclose(stored_bins, f_bins):
        raise RuntimeError("frequency bins differ from the stored filter series")
    scale = np.abs(stored).max()
    return float(np.abs(k_phase - stored).max() / scale)


def _load_stored_kernel():
    filter_root = ROOT / "results" / "max_leakage_297" / "a3.0" / "filter"
    for path in sorted(filter_root.glob("filter_*.npz")):
        with np.load(path) as data:
            mask = (
                (np.asarray(data["n_idx"]) == 7)
                & (np.asarray(data["t_idx"]) == 0)
                & np.isclose(np.asarray(data["omega297_mhz"]), 14.25)
                & np.isclose(np.asarray(data["dsweep_mhz"]), 17.5)
            )
            rows = np.flatnonzero(mask)
            if rows.size:
                return (np.asarray(data["f_bins"], dtype=float),
                        np.asarray(data["kernel"])[int(rows[0])].copy())
    raise RuntimeError("optimal-point filter kernel not found")


def check_static_offset(ops, times, comp_int, proj_int, psi_final) -> list[str]:
    """eps from a static Omega*sqrt(1+rho0) propagation vs 4 pi^2 rho0^2 ||Q G(0)||^2."""
    weights = np.gradient(times)
    weights[[0, -1]] *= 0.5
    g0 = np.einsum("t,pstq->psq", weights, comp_int)
    g0_sub = np.einsum("t,pst->ps", weights, proj_int)
    qg0_sq = (np.abs(g0[0]) ** 2).sum(axis=1) - np.abs(g0_sub[0]) ** 2

    lines = []
    for rho0 in (1e-3, -1e-3, 3e-3):
        run = mls.integrate_batch(
            ops, T_GATE, np.asarray([OMEGA_297 * math.sqrt(1.0 + rho0)]),
            np.asarray([D_SWEEP]), rtol=RTOL, atol=ATOL, ramp=RAMP)
        overlap = np.einsum("sq,sq->s", psi_final[0].conj(), run.psi_final[0])
        eps_fd = 1.0 - np.abs(overlap) ** 2
        eps_pred = (TAU ** 2) * rho0 ** 2 * qg0_sq
        worst = max(
            (abs(a - b) / b for a, b in zip(eps_fd[1:], eps_pred[1:])), default=0.0
        )  # inputs 01/10/11; 00 is dark and both sides are 0
        lines.append(f"  rho0={rho0:+.0e}: eps_fd={eps_fd[1]:.4e}/{eps_fd[3]:.4e} "
                     f"pred={eps_pred[1]:.4e}/{eps_pred[3]:.4e} (01/11) "
                     f"worst rel dev {worst:.2%}")
    return lines


# ── Reporting ────────────────────────────────────────────────────────────────


def error_table(f_bins, kernels, s_rho, f_min: float):
    contrib = 2.0 * np.pi ** 2 * kernels * s_rho(f_bins)[None, :]
    contrib[:, f_bins < f_min] = 0.0
    return contrib


def print_summary(f_bins, contrib, eps_phase):
    totals = contrib.sum(axis=1)
    s = int(np.argmax(totals))
    eps_int = float(totals[s])
    cum = np.cumsum(contrib[s]) / eps_int
    f50 = f_bins[np.searchsorted(cum, 0.5)]
    f90 = f_bins[np.searchsorted(cum, 0.9)]
    measured = float(contrib[s, f_bins <= 1e7].sum())
    print("\nper-input eps_int:",
          "  ".join(f"{LOGICAL_INPUTS[i]}={totals[i]:.3e}" for i in range(4)))
    print(f"worst input {LOGICAL_INPUTS[s]}: eps_int = {eps_int:.3e}")
    print(f"  50% / 90% of the error below {f50:.3g} / {f90:.3g} Hz")
    print(f"  measured band (<=10 MHz) share: {measured / eps_int:.1%}; "
          f"flat-floor extension: {1 - measured / eps_int:.1%}")
    for laser, eps in eps_phase.items():
        print(f"  vs {laser} phase noise {eps:.3e}: ratio {eps_int / eps:.2e} "
              f"({10 * math.log10(eps / eps_int):.1f} dB headroom)")
    return s, eps_int


def render(f_bins, f_meas, s_rho, contribs):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    fig.patch.set_facecolor("#f3efe4")
    omega_hz = OMEGA_297 / TAU
    near_lo, near_hi = omega_hz / 2.0, 2.0 * omega_hz

    ax = axes[0]
    ax.set_facecolor("#fffdf7")
    f_lo = np.logspace(0, 7, 400)
    f_hi = np.logspace(7, np.log10(f_bins[-1]), 100)
    ax.loglog(f_lo, s_rho(f_lo), color=COLORS["RIN"], linewidth=2.2,
              label="measured (x16 to 297 nm)")
    ax.loglog(f_hi, s_rho(f_hi), color=COLORS["RIN"], linewidth=2.2,
              linestyle="--", label="flat-floor hold")
    ax.axvline(1e7, color="#4e4a42", linestyle="--", linewidth=1.2,
               label="RIN measurement edge")
    ax.axvspan(near_lo, near_hi, color="#d8c9a7", alpha=0.35, linewidth=0)
    ax.axvline(omega_hz, color="#8c7a52", linestyle=":", linewidth=1.5,
               label=r"$\Omega/2\pi$")
    ax.set_xlabel("Offset frequency (Hz)")
    ax.set_ylabel(r"$S_\rho^{297}$ (1/Hz)")
    ax.set_title("RIN spectrum at 297 nm (undepleted-SHG scaling)", weight="bold")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    ax.set_facecolor("#fffdf7")
    for name, (contrib, style) in contribs.items():
        s = int(np.argmax(contrib.sum(axis=1)))
        keep = contrib[s] > 0.0
        ax.loglog(f_bins[keep], contrib[s][keep], color=COLORS[name],
                  linewidth=2.2, linestyle=style,
                  label=f"{name}, input {LOGICAL_INPUTS[s]}")
    ax.axvspan(near_lo, near_hi, color="#d8c9a7", alpha=0.35, linewidth=0,
               label=r"$[\Omega/2,2\Omega]/2\pi$")
    ax.axvline(1e7, color="#4e4a42", linestyle="--", linewidth=1.2,
               label="RIN measurement edge")
    ax.axvline(omega_hz, color="#8c7a52", linestyle=":", linewidth=1.5)
    ax.set_ylim(1e-13, None)
    ax.set_xlabel("Offset frequency (Hz)")
    ax.set_ylabel(r"$\Delta\varepsilon$ per log bin")
    ax.set_title("Fidelity-loss contribution per frequency bin", weight="bold")
    ax.legend(frameon=False, fontsize=8.5)

    for ax in axes:
        ax.grid(which="major", color="#d8d1c3", linewidth=0.8, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Intensity noise is orders below both phase-noise budgets",
                 fontsize=14.5, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "intensity_noise_filter_overlap.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    system = mls.build_system(mls.ScanConfig(), RYD_N)
    ops = mls.aggregate_operators(system, RYD_N)

    kernels = {}
    for n_t in (mls.KERNEL_N_T, 4 * mls.KERNEL_N_T):
        times, phi, psi, psi_final = trajectory_legs(ops, n_t)
        comps = noise_components(ops, times, phi, psi, psi_final)
        f_bins, k_int = binned_kernels(times, *comps["int"])
        kernels[n_t] = k_int
        if n_t == mls.KERNEL_N_T:
            _, k_phase = binned_kernels(times, *comps["phase"])
            dev = check_stored_phase_kernel(f_bins, k_phase)
            print(f"[check 1] phase kernel vs stored filter series: "
                  f"max rel dev {dev:.3e}")
            print("[check 3] static intensity offset vs f->0 kernel:")
            for line in check_static_offset(ops, times, *comps["int"], psi_final):
                print(line)
        del phi, psi, comps

    f_meas, s_rho = load_rin()
    coarse, dense = (kernels[n] for n in sorted(kernels))
    eps_c = error_table(f_bins, coarse, s_rho, 1.0).sum(axis=1).max()
    eps_d = error_table(f_bins, dense, s_rho, 1.0).sum(axis=1).max()
    print(f"[check 2] n_t convergence: eps_int {eps_c:.4e} (n_t={mls.KERNEL_N_T}) "
          f"vs {eps_d:.4e} (n_t={4 * mls.KERNEL_N_T}): "
          f"rel dev {abs(eps_c - eps_d) / eps_d:.2%}")

    k_int = dense
    eps_phase, phase_contrib = {}, {}
    _, stored_phase = _load_stored_kernel()
    for laser in ("ECDL", "seed"):
        psd = PhaseNoisePSD.from_csv(RESULT_ROOT / f"psd_{laser}.csv",
                                     harmonic=4, extrapolation="power")
        contrib = 2.0 * np.pi ** 2 * stored_phase * psd.s_dnu(f_bins)[None, :]
        phase_contrib[laser] = contrib
        eps_phase[laser] = float(contrib.sum(axis=1).max())

    for f_min in (1.0, 10.0):
        print(f"\n=== f_min = {f_min:g} Hz ===")
        contrib = error_table(f_bins, k_int, s_rho, f_min)
        s, eps_int = print_summary(f_bins, contrib, eps_phase)

    contrib = error_table(f_bins, k_int, s_rho, 1.0)
    np.savez(
        RESULT_ROOT / "intensity_noise_kernel.npz",
        f_bins=f_bins, kernel_int=k_int, kernel_phase=stored_phase,
        eps_int_per_input=contrib.sum(axis=1),
        eps_phase_ecdl=eps_phase["ECDL"], eps_phase_seed=eps_phase["seed"],
        ryd_n=RYD_N, t_gate_us=T_GATE * 1e6, omega297_mhz=OMEGA_297 / TAU / 1e6,
        dsweep_mhz=D_SWEEP / TAU / 1e6, rtol=RTOL, atol=ATOL,
        n_t=4 * mls.KERNEL_N_T, harmonic_psd_factor=HARMONIC_PSD_FACTOR,
    )
    render(f_bins, f_meas, s_rho,
           {"ECDL": (phase_contrib["ECDL"], "-"),
            "seed": (phase_contrib["seed"], "-"),
            "RIN": (contrib, "-")})
    print(f"\nwrote {RESULT_ROOT / 'intensity_noise_kernel.npz'}")
    print(f"wrote {RESULT_ROOT / 'intensity_noise_filter_overlap.png'}")


if __name__ == "__main__":
    main()
