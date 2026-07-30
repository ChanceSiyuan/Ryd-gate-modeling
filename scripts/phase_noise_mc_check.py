#!/usr/bin/env python
"""Validate the 297 nm filter kernels against direct Monte Carlo on the real gate.

``scripts/max_leakage_297_sweep.py`` prices laser phase noise perturbatively: one
adjoint pass over the noiseless trajectory gives the binned kernel ``K_b``, and any
PSD is then ``eps_phase = 2 pi^2 sum_b S_dnu(f_b) K_b``.  That chain is validated
against Phys. Rev. A **107**, 042611 (2023) on a two-level Rabi pulse
(``tests/test_phase_noise.py``), but a toy system cannot show that the *real*
two-atom gate responds the way the kernel says it does.

This script closes that gap the only way that does not reuse the kernel: it puts the
noise back on the Hamiltonian as ``H -> H_0 + 2 pi dnu(t) N_r`` (``N_r`` counting
atoms in ``r`` and ``r_garb`` -- one laser drives both 297 legs) and integrates real
``phase_trace`` realizations through the same block-max DOP853 kernel the sweep
uses.  The measured quantity is the ensemble-mean *increase* in terminal nonlogical
leakage over the noise-free run, per logical input; the prediction is
``error_from_kernel`` on a kernel computed here, point by point, by the same
``filter_kernels`` the ``filter`` subcommand calls.  Nothing is read from the store
and the only file written under it is ``reports/phase_noise_mc.json``.

Estimator
---------
Realizations are drawn in antithetic ``+/- dnu`` pairs and each *pair mean* is one
sample.  ``L(dnu) = L_0 + L_1[dnu] + L_2[dnu, dnu] + ...``, so a pair mean cancels
every odd order exactly and leaves ``L_0 + L_2 + O(dnu^4)``: the same expectation as
plain sampling (``<L_1> = 0`` already), with the term that dominates the *variance*
removed.  The first-order term fluctuates by ``~2 sqrt(L_0 <Delta L>)`` per shot, so
on a point whose coherent leakage exceeds its noise increase it would swamp the
signal and leave an acceptance band wide enough to accept anything.  The pair mean
over ``shots/2`` pairs is exactly the plain mean over all ``shots`` solves; only the
standard error differs, and the pairs are mutually independent, so it is the honest
one for this design.

Acceptance, per point and per logical input with a nonzero prediction: the Monte
Carlo mean is within four Monte Carlo standard errors *or* 10% of the prediction,
whichever is looser.  ``|00>`` is excluded throughout -- ``|0>`` is a dark spectator
carrying no 297 leg, so ``N_r psi_00 = 0``, the kernel is identically zero and there
is nothing to compare.

What the first campaign found
-----------------------------
The kernel's ``G`` is exact -- its ``f = 0`` value reproduces the gate's measured
first-order response to a static detuning to seven digits
(``tests/test_max_leakage_297_sweep.py``).  The *error formula* built on it is not.
``<Delta L> = 2 pi^2 int S ||Q G||^2 df`` keeps ``<||Q chi_1||^2>`` and drops the
second-order term ``2 Re <Q psi_0 | Q chi_2>``, which vanishes only when the
noiseless gate does not leak -- the setting of both literature checks, whose metric
is infidelity against the *noiseless* final state.  Here ``L_0 != 0`` and the
dropped term is the same order in the noise: against a deterministic single tone it
cancels ~99% of the quasi-static response and moves the drive-scale response by a
few percent.  So the ``mc / kernel`` ratio is independent of the noise amplitude
(checked over four decades of ``--psd-scale``) and lands anywhere from -0.1 to 1.4,
tracking how much of a point's prediction sits below ~1 MHz.

Usage
-----
    python scripts/phase_noise_mc_check.py --n-points 20 --shots 200 \
        --laser ECDL --extrapolation flat
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO_ROOT, "src"), os.path.join(REPO_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import laser_noise_psd as lnp  # noqa: E402
import max_leakage_297_sweep as mls  # noqa: E402
import sweeplib  # noqa: E402
from ryd_gate.phase_noise import (  # noqa: E402
    PhaseNoisePSD, PhaseTrace, error_from_kernel, phase_trace)

# The traces must carry exactly the band the stored kernel integrates, or the two
# legs are pricing different noise.
TRACE_F_MIN_HZ = mls.KERNEL_F_MIN_HZ
TRACE_F_MAX_HZ = mls.KERNEL_F_MAX_HZ
TRACE_SAMPLES_PER_PERIOD = 80

# Checked points, spread over the panel family rather than clustered: both ends of
# the n axis, both ends of the T axis, and every node of the level-1 Omega and
# D_sweep axes at least twice.  ``--n-points N`` takes the first N, so the head is
# the n = 53, T = 1 us centre point the convergence gate also uses.
# (n_idx, t_idx, omega coord, dsweep coord) in the sweep's rational axis units.
MC_POINTS = (
    (1, 0, (3, 2), (3, 2)),      # n=53  T=1.0  Om=13.5  D=15
    (0, 0, (0, 1), (1, 1)),      # n=50  T=1.0  Om=9.0   D=10
    (0, 0, (3, 1), (3, 1)),      # n=50  T=1.0  Om=18.0  D=30
    (7, 0, (1, 2), (1, 2)),      # n=73  T=1.0  Om=10.5  D=6
    (7, 0, (5, 2), (5, 2)),      # n=73  T=1.0  Om=16.5  D=25
    (0, 0, (2, 1), (3, 2)),      # n=50  T=1.0  Om=15.0  D=15
    (2, 1, (2, 1), (2, 1)),      # n=56  T=1.2  Om=15.0  D=20
    (5, 1, (5, 2), (1, 2)),      # n=68  T=1.2  Om=16.5  D=6
    (1, 2, (1, 2), (3, 2)),      # n=53  T=1.5  Om=10.5  D=15
    (4, 2, (3, 1), (0, 1)),      # n=64  T=1.5  Om=18.0  D=2
    (6, 2, (0, 1), (5, 2)),      # n=71  T=1.5  Om=9.0   D=25
    (3, 3, (0, 1), (0, 1)),      # n=60  T=2.0  Om=9.0   D=2   (worst coherent corner)
    (1, 4, (2, 1), (3, 2)),      # n=53  T=2.5  Om=15.0  D=15
    (4, 4, (3, 2), (3, 1)),      # n=64  T=2.5  Om=13.5  D=30
    (6, 4, (1, 1), (2, 1)),      # n=71  T=2.5  Om=12.0  D=20
    (2, 6, (5, 2), (3, 2)),      # n=56  T=3.5  Om=16.5  D=15
    (5, 6, (1, 2), (2, 1)),      # n=68  T=3.5  Om=10.5  D=20
    (0, 8, (3, 2), (3, 2)),      # n=50  T=4.5  Om=13.5  D=15
    (3, 8, (1, 1), (1, 1)),      # n=60  T=4.5  Om=12.0  D=10
    (7, 8, (5, 2), (2, 1)),      # n=73  T=4.5  Om=16.5  D=20
)

SIGMA_TOL = 4.0        # accept within four Monte Carlo standard errors ...
REL_TOL = 0.10         # ... or 10% of the prediction, whichever is looser


def trace_samples(t_gate: float) -> int:
    """Phase samples per trace: ~80 per period at ``TRACE_F_MAX_HZ``.

    A cost knob, not an accuracy one.  The terminal leakage is already converged to
    ~1e-6 relative at 20 samples per period, but an under-resolved cubic spline
    hands DOP853 knot-scale structure that it then chases: over a 1 us gate, 4096
    samples cost 3.7x the function evaluations of 16384 for the same answer to eight
    digits.
    """
    return int(TRACE_SAMPLES_PER_PERIOD * TRACE_F_MAX_HZ * t_gate) + 1


def _noisy_rhs_factory(trace, n_r_diag):
    """The sweep's 297 RHS with the laser phase noise added back on.

    ``H -> H_0 + 2 pi dnu(t) N_r`` and ``d(phi)/dt = 2 pi dnu(t)``, so the added term
    is ``-i (d phi/dt) N_r psi``.  ``PhaseTrace.derivative`` is the guarded public
    accessor: ``CubicSpline.derivative()`` inherits ``extrapolate=True`` and would
    silently continue a quadratic past ``t_gate``, where the solver's last step does
    land, instead of raising as ``PhaseTrace`` does everywhere else.
    """
    def make(ops, cols, t_gate, ramp):
        base = mls._297_rhs_factory(ops, cols, t_gate, ramp)
        n_cols = cols["shift"].size

        def rhs(t, y):
            out = base(t, y)
            ym = y.reshape(n_cols, n_r_diag.size)
            return out - 1j * (float(trace.derivative(t))
                               * (n_r_diag[None, :] * ym)).ravel()
        return rhs
    return make


def _leakage(ops, t_gate, omega_297, d_sweep, trace, ramp, rtol, atol) -> np.ndarray:
    """(4,) terminal nonlogical population per logical input under one trace."""
    dim = ops.h_static_diag.size
    labels = ("00", "01", "11") if ops.swap_symmetric else mls.LOGICAL_INPUTS
    res = sweeplib.integrate_batch(
        ops, t_gate, {"omega_297": np.asarray([omega_297]),
                      "d_sweep": np.asarray([d_sweep])},
        labels,
        rhs_factory=_noisy_rhs_factory(trace, mls._rydberg_number_diag(dim)),
        dim=dim, rtol=rtol, atol=atol, ramp=ramp)
    return res.leakage[0]


def _pair(args) -> np.ndarray:
    """One antithetic realization pair (pool entry): (2, 4) leakages, ``+`` then ``-``."""
    ops, t_gate, omega_297, d_sweep, ramp, rtol, atol, psd, seed = args
    tr = phase_trace(psd, t_gate, seed=seed, f_min=TRACE_F_MIN_HZ,
                     f_max=TRACE_F_MAX_HZ, n_samples=trace_samples(t_gate))
    anti = PhaseTrace(tr.times, -tr.values, -tr.dnu_0, tr.f_grid, tr.df_grid)
    return np.stack([_leakage(ops, t_gate, omega_297, d_sweep, t, ramp, rtol, atol)
                     for t in (tr, anti)])


def check_point(ops, cfg, key, psd, *, shots: int, seed: int,
                rtol: float, atol: float, pool=None) -> dict:
    """Monte Carlo vs filter-kernel prediction for one grid point.

    Returns the record written to ``reports/phase_noise_mc.json``: the noise-free
    leakage, the antithetic-pair mean increase and its standard error, and the
    kernel prediction, each a length-4 vector over the logical inputs 00/01/10/11.
    """
    # Two pairs is the minimum that has a standard error at all; one would make the
    # ddof=1 sample deviation NaN and quietly report an unfalsifiable point.
    if shots < 4 or shots % 2:
        raise ValueError(f"shots must be an even number >= 4; got {shots}")
    t0 = time.time()
    t_gate = cfg.t_gate_us[key.t_idx] * 1e-6
    omega_297 = float(key.omega_mhz()) * 1e6 * mls.TAU
    d_sweep = float(key.dsweep_mhz()) * 1e6 * mls.TAU

    n_r = mls._rydberg_number_diag(ops.h_static_diag.size)
    if ops.swap_symmetric and not np.array_equal(n_r[ops.swap_perm], n_r):
        raise RuntimeError("N_r is not atom-swap invariant; the |10> column cannot "
                           "be reconstructed from |01> under the noise term")

    base = mls.integrate_batch(ops, t_gate, np.asarray([omega_297]),
                               np.asarray([d_sweep]), rtol=rtol, atol=atol,
                               ramp=cfg.ramp_frac)
    f_bins, _df = mls.kernel_frequency_bins()
    kernels = mls.filter_kernels(ops, t_gate, np.asarray([omega_297]),
                                 np.asarray([d_sweep]), rtol=rtol, atol=atol,
                                 ramp=cfg.ramp_frac)[0]
    prediction = np.asarray([error_from_kernel(psd, f_bins, kernels[s])
                             for s in range(4)])

    n_pairs = shots // 2
    tasks = [(ops, t_gate, omega_297, d_sweep, cfg.ramp_frac, rtol, atol, psd,
              seed + i) for i in range(n_pairs)]
    runner = map if pool is None else pool.map
    pairs = np.stack(list(runner(_pair, tasks)))            # (n_pairs, 2, 4)
    increase = pairs.mean(axis=1) - base.leakage[0]         # (n_pairs, 4)

    mc_mean = increase.mean(axis=0)
    mc_stderr = increase.std(axis=0, ddof=1) / np.sqrt(n_pairs)
    tested = prediction > 0.0
    passed = bool(np.all(np.abs(mc_mean - prediction)[tested]
                         <= np.maximum(SIGMA_TOL * mc_stderr,
                                       REL_TOL * prediction)[tested]))
    return {
        "point_id": key.id(),
        "ryd_n": int(cfg.ryd_n[key.n_idx]),
        "t_gate_us": float(cfg.t_gate_us[key.t_idx]),
        "omega297_mhz": float(key.omega_mhz()),
        "dsweep_mhz": float(key.dsweep_mhz()),
        "logical_inputs": list(mls.LOGICAL_INPUTS),
        "leakage_noise_free": base.leakage[0].tolist(),
        "mc_mean": mc_mean.tolist(),
        "mc_stderr": mc_stderr.tolist(),
        "filter_prediction": prediction.tolist(),
        "passed": passed,
        "n_pairs": n_pairs,
        "trace_samples": trace_samples(t_gate),
        "runtime_s": time.time() - t0,
    }


def _format(rec: dict) -> str:
    """One console line per point: every logical input with a nonzero prediction."""
    parts = []
    for i, s in enumerate(rec["logical_inputs"]):
        pred = rec["filter_prediction"][i]
        if pred <= 0.0:
            continue
        parts.append(f"{s}: mc {rec['mc_mean'][i]:.3e}+-{rec['mc_stderr'][i]:.1e} "
                     f"kern {pred:.3e} r={rec['mc_mean'][i] / pred:5.2f}")
    return (f"{rec['point_id']:<22s} n={rec['ryd_n']:<3d} T={rec['t_gate_us']:.1f}us "
            f"Om={rec['omega297_mhz']:<5.1f} D={rec['dsweep_mhz']:<4.1f} | "
            + "  ".join(parts) + f"  {'PASS' if rec['passed'] else 'FAIL'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase_noise_mc_check",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--output", default=mls._default_output(3.0),
                   help="scan store whose reports/ receives phase_noise_mc.json "
                        "(nothing else under it is read or written)")
    p.add_argument("--n-points", type=int, default=len(MC_POINTS),
                   help=f"how many of the {len(MC_POINTS)} checked points to run")
    p.add_argument("--shots", type=int, default=200,
                   help="realizations per point; halved into antithetic pairs")
    p.add_argument("--laser", default="ECDL", choices=sorted(lnp.AXES))
    p.add_argument("--extrapolation", default="flat", choices=["flat", "power"],
                   help="S_dnu above the 1 MHz measurement edge")
    p.add_argument("--psd-scale", type=float, default=1.0,
                   help="multiply S_dnu by this factor.  The kernel prediction is "
                        "exactly linear in it, so sweeping it separates a "
                        "discrepancy that is higher order in the noise (the ratio "
                        "moves towards 1 as the scale falls) from one that is the "
                        "same order as the term the kernel keeps (it does not)")
    p.add_argument("--workers", type=int, default=min(40, os.cpu_count() or 1))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rtol", type=float, default=1e-9)
    p.add_argument("--atol", type=float, default=1e-12)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if not 0 < args.n_points <= len(MC_POINTS):
        raise SystemExit(f"--n-points must be in 1..{len(MC_POINTS)}")
    specs = MC_POINTS[:args.n_points]
    psd = lnp.load_psds(os.path.join(lnp.NOISE_DIR,
                                     f"psd_{args.laser}.csv"))[args.extrapolation]
    if args.psd_scale != 1.0:
        psd = PhaseNoisePSD(psd.f_hz, psd.s_meas * args.psd_scale,
                            harmonic=psd.harmonic, extrapolation=psd.extrapolation,
                            power_law_fit_hz=psd.power_law_fit_hz)

    cfg = mls.ScanConfig()
    keys = [mls.make_key(*spec) for spec in specs]
    ops = {n_idx: mls.aggregate_operators(
        mls.build_system(cfg, cfg.ryd_n[n_idx]), cfg.ryd_n[n_idx])
        for n_idx in sorted({k.n_idx for k in keys})}
    print(f"[mc] {args.laser}/{args.extrapolation} x{args.psd_scale:g}: "
          f"sigma_nu(1 Hz, 200 MHz) = "
          f"{psd.sigma_nu(TRACE_F_MIN_HZ, TRACE_F_MAX_HZ) / 1e3:.1f} kHz | "
          f"{len(keys)} points x {args.shots} shots on {args.workers} workers",
          flush=True)

    t0 = time.time()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context("fork")) as pool:
        for i, key in enumerate(keys, start=1):
            rec = check_point(ops[key.n_idx], cfg, key, psd, shots=args.shots,
                              seed=args.seed + 10_000 * i, rtol=args.rtol,
                              atol=args.atol, pool=pool)
            records.append(rec)
            print(f"[{i:2d}/{len(keys)}] {_format(rec)}", flush=True)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "laser": args.laser,
        "extrapolation": args.extrapolation,
        "psd_scale": args.psd_scale,
        "harmonic": lnp.HARMONIC,
        "shots": args.shots,
        "seed": args.seed,
        "rtol": args.rtol,
        "atol": args.atol,
        "trace_f_min_hz": TRACE_F_MIN_HZ,
        "trace_f_max_hz": TRACE_F_MAX_HZ,
        "estimator": "mean over antithetic +/- trace pairs of the terminal "
                     "nonlogical-leakage increase over the noise-free run",
        "acceptance": f"|mc_mean - prediction| <= max({SIGMA_TOL:g} * mc_stderr, "
                      f"{REL_TOL:g} * prediction), over the logical inputs with a "
                      f"nonzero prediction",
        "n_points": len(records),
        "n_passed": sum(r["passed"] for r in records),
        "wall_seconds": time.time() - t0,
        "points": records,
    }
    reports_dir = os.path.join(args.output, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, "phase_noise_mc.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(path + ".tmp", path)
    print(f"passed: {report['n_passed']}/{report['n_points']}\nwrote {path}")


if __name__ == "__main__":
    main()
