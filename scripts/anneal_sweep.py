#!/usr/bin/env python
"""Parallel launcher for 10x10 TFIM anneal parameter sweeps (notebook 03 physics).

Each point runs on a single BLAS thread (pinned below, before any import);
throughput comes from ``--max-workers`` concurrent processes (default 30 on the
40-core DGX). A point whose ``.npz`` already exists in ``results/anneal_sweep/``
is skipped, so an interrupted sweep just resumes on relaunch.

Edit ``PARAM_GRID`` for the sweep at hand. Fidelity is the scan tier from
notebook 03 §2 (``D=6``, ``dt=0.1/w0``, 200 shots); re-check keeper points at
``D=8``. (``dt=0.2/w0`` was tried and rejected: the V*dt ~ 4.8 rad Trotter
gates collapse the final density 0.42 -> 0.23, 2026-07-21 diagnostics.)

Grid points may override the tier with ``dt_w0`` (default 0.1) and ``D``
(default 6); non-default values are appended to the point tag, so existing
default-tier ``.npz`` files keep their names. ``--report`` prints an acceptance
table over the finished points: BP observables, shot-derived <m_s^2> and the
long-range connected correlator (truncation-sensitive per
``sun2026quantumclassical``'s global
Pauli-string diagnostic), and the cumulative NTU truncation error against the
bar calibrated by ``scripts/calibrate_anneal_3x3.py``.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np

from anneal_model import (  # noqa: E402
    A_UM,
    build_anneal_system,
    peps_options,
    rydberg_c6,
    staggered_magnetization,
)
from ryd_gate import Register, simulate

LX, LY = 10, 10
SHOTS, SEED = 200, 7
OUT_DIR = REPO_ROOT / "results" / "anneal_sweep"

# Sweep axes, all in the notebook's natural units (w0 = V_nn/24):
#   hz_i_w0    -- h_z start point; 24 = the classical phase boundary (Delta=0),
#                 >24 starts inside the gapped trivial phase
#   hx_peak_w0 -- transverse-field plateau h_x/w0 (J = 6*w0)
#   t_hold_w0  -- h_z ramp duration in units of 1/w0 (notebook: 12)
#   dt_w0, D   -- optional tier overrides (defaults below)
DT_W0_DEFAULT, D_DEFAULT = 0.1, 6

PARAM_GRID = [
    dict(hz_i_w0=hz, hx_peak_w0=hx, t_hold_w0=th)
    for hz, hx, th in itertools.product([28.0], [1.0, 2.0, 4.0], [12.0, 24.0, 48.0])
]
# Keeper rechecks on the notebook-03 canonical schedule: dt-convergence at the
# scan tier plus the D=6/8/10 trend (fixed-chi runs can look smooth while far
# from converged -- ``sun2026quantumclassical``'s central caveat).
PARAM_GRID += [
    dict(hz_i_w0=24.0, hx_peak_w0=1.0, t_hold_w0=12.0),
    dict(hz_i_w0=24.0, hx_peak_w0=1.0, t_hold_w0=12.0, dt_w0=0.05),
    dict(hz_i_w0=24.0, hx_peak_w0=1.0, t_hold_w0=12.0, D=8),
    dict(hz_i_w0=24.0, hx_peak_w0=1.0, t_hold_w0=12.0, D=10),
]

# Acceptance bar for --report: flag a point once its cumulative NTU truncation
# error exceeds this value. Anchored at the 10x10 keeper measurements
# (2026-07-21): at cum_ntu 3.4-4.5 the D=6 scan tier was measured against D=8
# to carry ~0.07 systematic error on n_mean and an unphysical m_s bias
# (0.2 at dt=0.1, 0.9 at dt=0.05); beyond cum_ntu ~4.5 a point extrapolates
# past the region where the tier error was quantified. The 3x3 exact_ode
# calibration (results/anneal_sweep/calibration_3x3.npz) pins the Trotter
# component: dt=0.1/w0 alone is a ~4.5e-2 site-density error at 3x3, ~0.008 on
# n_mean at 10x10 D=6.
NTU_CUM_BAR = 4.5

# Pairs at least this many lattice spacings apart enter the long-range
# connected correlator in --report.
LONG_RANGE_A = 6.0

# Computed once at import; forked workers inherit it (warm ARC cache anyway).
C6 = rydberg_c6()


def point_tag(p):
    tag = f"hz{p['hz_i_w0']:g}_hx{p['hx_peak_w0']:g}_th{p['t_hold_w0']:g}"
    if p.get("dt_w0", DT_W0_DEFAULT) != DT_W0_DEFAULT:
        tag += f"_dt{p['dt_w0']:g}"
    if p.get("D", D_DEFAULT) != D_DEFAULT:
        tag += f"_D{p['D']}"
    return tag


def run_point(p):
    t_start = time.perf_counter()
    dt_w0, D = p.get("dt_w0", DT_W0_DEFAULT), p.get("D", D_DEFAULT)
    system, n, t_gate, w0, subl = build_anneal_system(
        LX, LY, C6, hz_i_w0=p["hz_i_w0"],
        hx_peak_w0=p["hx_peak_w0"], t_hold_w0=p["t_hold_w0"])
    obs = {f"n_r_{i}": system.observables.n("r", i) for i in range(n)}
    result = simulate(
        system, None, backend="peps",
        t_eval=np.linspace(0.0, t_gate, 9), observables=obs,
        backend_options=peps_options(
            dt_w0=dt_w0, w0=w0, bond_dimension=D,
            measurement_method="belief_propagation"),
    )
    n_r_t = np.array([result.expectation(f"n_r_{i}") for i in range(n)])
    m_s_t = staggered_magnetization(n_r_t, subl)

    counts = result.sample(shots=SHOTS, seed=SEED)
    shot_occ = np.array([[lbl == "r" for lbl in cfg] for cfg in counts], dtype=np.uint8)
    shot_mult = np.array(list(counts.values()), dtype=np.int64)

    ev = result.peps_evidence.to_dict()
    out = OUT_DIR / f"{point_tag(p)}.npz"
    tmp = out.with_name(out.stem + ".tmp.npz")   # savez appends .npz to other suffixes
    np.savez_compressed(
        tmp,
        hz_i_w0=p["hz_i_w0"], hx_peak_w0=p["hx_peak_w0"], t_hold_w0=p["t_hold_w0"],
        dt_w0=dt_w0, D=D,
        w0_rad_s=w0, t_gate_s=t_gate, shots=SHOTS, seed=SEED,
        times=result.times, n_r_t=n_r_t, m_s_t=m_s_t,
        shot_occ=shot_occ, shot_mult=shot_mult, subl=subl,
        max_ntu_truncation_error=ev["max_ntu_truncation_error"],
        cumulative_ntu_truncation_error=ev["cumulative_ntu_truncation_error"],
        wall_s=time.perf_counter() - t_start,
    )
    tmp.rename(out)
    return point_tag(p), time.perf_counter() - t_start


def report():
    coords = Register.rectangle(LX, LY, spacing_um=A_UM).coords
    dist = np.hypot(*(coords[:, None, :] - coords[None, :, :]).transpose(2, 0, 1))
    far = dist >= LONG_RANGE_A * A_UM

    files = [f for f in sorted(OUT_DIR.glob("*.npz")) if not f.name.startswith("calibration")]
    if not files:
        print(f"no point files in {OUT_DIR}")
        return
    print(f"{'tag':>28} {'D':>3} {'dt':>5} {'n_mean':>7} {'m_s':>7} {'<m_s^2>':>8} "
          f"{'C_long':>8} {'max_ntu':>9} {'cum_ntu':>9} {'wall':>7}  verdict")
    for f in files:
        z = np.load(f)
        sig = z["subl"] * (2.0 * z["shot_occ"].astype(float) - 1.0)
        w = z["shot_mult"].astype(float)
        w /= w.sum()
        ms2 = float(w @ sig.mean(axis=1) ** 2)
        m1 = w @ sig
        conn = sig.T @ (sig * w[:, None]) - np.outer(m1, m1)
        c_long = float(conn[far].mean())
        cum = float(z["cumulative_ntu_truncation_error"])
        if NTU_CUM_BAR is None:
            verdict = "n/a (bar uncalibrated)"
        else:
            verdict = "ok" if cum <= NTU_CUM_BAR else f"FLAG (> {NTU_CUM_BAR:g})"
        print(f"{f.stem:>28} {int(z['D']):3d} {float(z['dt_w0']):5.3f} "
              f"{z['n_r_t'][:, -1].mean():7.4f} {z['m_s_t'][-1]:7.4f} "
              f"{ms2:8.4f} {c_long:8.4f} {float(z['max_ntu_truncation_error']):9.2e} "
              f"{cum:9.2e} {float(z['wall_s']) / 60:6.1f}m  {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-workers", type=int, default=30)
    ap.add_argument("--report", action="store_true",
                    help="print the acceptance table over finished points and exit")
    args = ap.parse_args()

    if args.report:
        report()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pending = [p for p in PARAM_GRID if not (OUT_DIR / f"{point_tag(p)}.npz").exists()]
    done_already = len(PARAM_GRID) - len(pending)
    print(f"{len(PARAM_GRID)} points, {done_already} already on disk, "
          f"{len(pending)} to run on {args.max_workers} workers", flush=True)

    n_done = 0
    with ProcessPoolExecutor(max_workers=args.max_workers,
                             mp_context=mp.get_context("fork")) as ex:
        futures = {ex.submit(run_point, p): p for p in pending}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                tag, wall = fut.result()
                n_done += 1
                print(f"[{n_done}/{len(pending)}] {tag}: {wall / 60:.1f} min", flush=True)
            except Exception as e:
                print(f"[FAIL] {point_tag(p)}: {e!r}", flush=True)
    print("sweep finished", flush=True)


if __name__ == "__main__":
    main()
