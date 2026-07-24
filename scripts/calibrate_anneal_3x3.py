#!/usr/bin/env python
"""3x3 exact-vs-PEPS calibration for the notebook-03 anneal (arxiv_v1.0 protocol).

Same physics construction as ``scripts/anneal_sweep.py`` (NN-cutoff TFIM with
boundary pins) on 3x3, canonical schedule hz_i=24*w0, hx_peak=1*w0, t_hold=12/w0.
One ``exact_ode`` reference (DOP853, rtol 1e-10) plus a PEPS ladder:

  Trotter split   -- D=8 at dt/w0 in {0.1, 0.05, 0.025} (D=8 is truncation-free
                     at 3x3, so the residual vs exact is the Strang error)
  truncation split -- D in {2, 3, 4, 6} at dt=0.05/w0, where the Strang error
                     (~4.5e-3) no longer buries truncation; the same ladder at
                     dt=0.1 is kept for the record but is Trotter-dominated
                     (everything sits at the ~4.5e-2 dt=0.1 floor)

Writes everything to ``results/anneal_sweep/calibration_3x3.npz`` and prints a
summary table. Re-run whenever the sweep tiers (D, dt) change.

Readout defaults to CTM: belief-propagation readout carries a ~4-6e-2 bias on
site densities at 3x3 (probe 2026-07-21, results/anneal_sweep/probe_ctm_3x3.log)
that buries the evolution error being calibrated. ``--measurement
belief_propagation`` reproduces the archived ``calibration_3x3_bp.npz`` table.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.physics import arc_pair_c6_rad_s_um6
from ryd_gate.protocols import SweepProtocol

RYD_LEVEL, A_UM = 70, 6.0
LX, LY = 3, 3
HZ_I_W0, HX_PEAK_W0, T_HOLD_W0 = 24.0, 1.0, 12.0
N_EVAL = 9
OUT_DIR = REPO_ROOT / "results" / "anneal_sweep"

PEPS_RUNS = [dict(D=8, dt_w0=0.1), dict(D=8, dt_w0=0.05), dict(D=8, dt_w0=0.025),
             dict(D=6, dt_w0=0.1), dict(D=4, dt_w0=0.1), dict(D=3, dt_w0=0.1),
             dict(D=2, dt_w0=0.1),
             dict(D=6, dt_w0=0.05), dict(D=4, dt_w0=0.05), dict(D=3, dt_w0=0.05),
             dict(D=2, dt_w0=0.05)]

C6 = arc_pair_c6_rad_s_um6(n1=RYD_LEVEL, l1=0, j1=0.5, mj1=-0.5, mj2=-0.5,
                           theta=0.0, phi=0.0, degenerate=False)


def build_system():
    geom = Register.rectangle(LX, LY, spacing_um=A_UM)
    cutoff_um = 1.1 * A_UM

    coords = geom.coords
    n = geom.N
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.hypot(*(coords[j] - coords[i])))
            if r <= cutoff_um * (1 + 1e-9):
                V[i, j] = V[j, i] = C6 / r ** 6
    shift = 0.25 * V.sum(axis=1)
    shift_ref = float(shift.mean())
    pins = 2.0 * (shift - shift_ref)

    V_nn = C6 / A_UM ** 6
    w0 = V_nn / 24.0
    hx_peak = HX_PEAK_W0 * w0
    hz_i, hz_f = HZ_I_W0 * w0, 0.0
    t_rise, t_hold, t_fall = 2.0 / w0, T_HOLD_W0 / w0, 2.0 / w0
    t_gate = t_rise + t_hold + t_fall

    def hx(t):
        if t < t_rise:
            return hx_peak * (t / t_rise)
        if t < t_rise + t_hold:
            return hx_peak
        return hx_peak * max(0.0, 1.0 - (t - t_rise - t_hold) / t_fall)

    def hz(t):
        if t < t_rise:
            return hz_i
        if t < t_rise + t_hold:
            return hz_i + (hz_f - hz_i) * (t - t_rise) / t_hold
        return hz_f

    proto = SweepProtocol(
        t_gate_s=t_gate,
        omega_half_rad_s=lambda t: hx(t),
        detuning_rad_s=lambda t: 2.0 * (shift_ref - hz(t)),
        local_detuning_rad_s=lambda t, i, pins=pins: float(pins[i]),
    )
    system = RydbergSystem(
        level_structure=level_structure("1r", ryd_level=RYD_LEVEL),
        register=geom,
        protocol=proto,
        interaction_cutoff_um=cutoff_um,
    )
    subl = np.array([(-1.0) ** (ix + iy)
                     for ix, iy in np.round(coords / A_UM).astype(int)])
    return system, n, t_gate, w0, subl


def stag_mag(n_r_t, subl):
    return (subl[:, None] * (2.0 * n_r_t - 1.0)).mean(axis=0)


def run_one(spec):
    t_start = time.perf_counter()
    system, n, t_gate, w0, subl = build_system()
    obs = {f"n_r_{i}": system.observables.n("r", i) for i in range(n)}
    t_eval = np.linspace(0.0, t_gate, N_EVAL)

    if spec["kind"] == "exact":
        result = simulate(system, None, backend="exact_ode", t_eval=t_eval,
                          observables=obs,
                          backend_options={"rtol": 1e-10, "atol": 1e-12})
        extra = {}
    else:
        result = simulate(
            system, None, backend="peps", t_eval=t_eval, observables=obs,
            backend_options={
                "time_step_s": spec["dt_w0"] / w0,
                "bond_dimension": spec["D"],
                "svd_tolerance": 1e-8,
                "ntu_max_iterations": 20,
                "ntu_iteration_tolerance": 1e-10,
                "measurement_method": spec["measurement"],
                "environment_bond_dimension": 32,
                "environment_tolerance": 1e-8,
                "environment_max_iterations": 50,
                "device": "cpu",
            },
        )
        ev = result.peps_evidence.to_dict()
        extra = {"max_ntu": ev["max_ntu_truncation_error"],
                 "cum_ntu": ev["cumulative_ntu_truncation_error"]}

    n_r_t = np.array([result.expectation(f"n_r_{i}") for i in range(n)])
    return spec, {"times": result.times, "n_r_t": n_r_t,
                  "m_s_t": stag_mag(n_r_t, subl),
                  "wall_s": time.perf_counter() - t_start, **extra}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measurement", default="ctm",
                    choices=("ctm", "belief_propagation"))
    args = ap.parse_args()
    out = OUT_DIR / ("calibration_3x3.npz" if args.measurement == "ctm"
                     else "calibration_3x3_bp.npz")

    specs = [dict(kind="exact")] + [dict(kind="peps", measurement=args.measurement, **r)
                                    for r in PEPS_RUNS]
    with ProcessPoolExecutor(max_workers=len(specs),
                             mp_context=mp.get_context("fork")) as ex:
        outs = list(ex.map(run_one, specs))

    by_tag = {}
    for spec, data in outs:
        tag = "exact" if spec["kind"] == "exact" else f"D{spec['D']}_dt{spec['dt_w0']:g}"
        by_tag[tag] = data

    ref = by_tag["exact"]
    payload = {
        "lx": LX, "ly": LY, "hz_i_w0": HZ_I_W0, "hx_peak_w0": HX_PEAK_W0,
        "t_hold_w0": T_HOLD_W0, "measurement": args.measurement, "times": ref["times"],
        "exact_n_r_t": ref["n_r_t"], "exact_m_s_t": ref["m_s_t"],
        "exact_wall_s": ref["wall_s"],
    }
    print(f"{'run':>12} {'max|dn_r|':>10} {'fin|dn_r|':>10} {'fin|dm_s|':>10} "
          f"{'max_ntu':>9} {'cum_ntu':>9} {'wall':>7}")
    for tag, d in by_tag.items():
        if tag == "exact":
            continue
        dn = np.abs(d["n_r_t"] - ref["n_r_t"])
        dms = np.abs(d["m_s_t"] - ref["m_s_t"])
        payload.update({
            f"{tag}_n_r_t": d["n_r_t"], f"{tag}_m_s_t": d["m_s_t"],
            f"{tag}_err_n_r_max": dn.max(), f"{tag}_err_n_r_final": dn[:, -1].max(),
            f"{tag}_err_m_s_final": dms[-1],
            f"{tag}_max_ntu": d["max_ntu"], f"{tag}_cum_ntu": d["cum_ntu"],
            f"{tag}_wall_s": d["wall_s"],
        })
        print(f"{tag:>12} {dn.max():10.2e} {dn[:, -1].max():10.2e} {dms[-1]:10.2e} "
              f"{d['max_ntu']:9.2e} {d['cum_ntu']:9.2e} {d['wall_s'] / 60:6.1f}m")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"saved {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
