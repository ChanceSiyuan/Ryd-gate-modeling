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
uses.  The measured quantity is the noise-induced fidelity loss against the
noiseless final state, ``1 - |<psi_0^s(T)|psi^s(T)>|^2``, per logical input; the
prediction is ``error_from_kernel`` on a kernel computed here, point by point, by
the same ``filter_kernels`` the ``filter`` subcommand calls.  Nothing is read from
the store and the only file written under it is ``reports/phase_noise_mc.json``.

Its first campaign is why that metric is the fidelity loss.  The original
``<Delta L> = 2 pi^2 int S ||Q G||^2 df`` over the fixed nonlogical projector passed
7 of 20 points, worst ratio ``-0.116``; the kernel object was exact (its ``G(0)``
reproduces the gate's measured static-detuning response to seven digits) but the
metric dropped ``2 Re <Q psi_0 | Q chi_2>``, the same order in the noise whenever
the noiseless gate already leaks.  ``Q = 1 - |psi_0(T)><psi_0(T)|`` annihilates
``psi_0`` and kills that term identically.  See the design doc's "Why not the
leakage increase".

Estimator
---------
Plain sampling: ``shots`` independent realizations, mean and standard error over
them.  Unlike the leakage increase, this metric has **no first-order term at all** --
``<psi_0(T)|chi_1> = -i int <psi_0(t)|V(t)|psi_0(t)> dt`` is purely imaginary because
``V`` is Hermitian, so the real part that would enter at first order vanishes for
*every* realization, not merely in the mean.  Antithetic ``+/-`` pairs therefore buy
nothing here -- they would only cancel the O(dnu^3) term, ~``||chi_1||`` of the
signal -- while halving the number of independent samples, so they are not used.
Every sample is non-negative by construction.

Acceptance, per point and per logical input with a nonzero prediction: the Monte
Carlo mean is within four Monte Carlo standard errors *or* 10% of the prediction,
whichever is looser.  ``|00>`` is excluded throughout -- ``|0>`` is a dark spectator
carrying no 297 leg, so ``N_r psi_00 = 0``, ``A(t)`` vanishes and the kernel is
identically zero.

Predictions above ``REGIME_MAX`` are flagged: they are outside the weak-noise regime
the perturbative expansion assumes.  The measured ``sigma_nu(1 Hz, 200 MHz)`` of
718 kHz against a 13.5 MHz drive is ``2 pi dnu / Omega = 0.053``, five times the 0.01
the design originally assumed.

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
import hashlib
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
    PhaseNoisePSD, error_from_kernel, phase_trace)

# The traces must carry exactly the band the stored kernel integrates, or the two
# legs are pricing different noise.
TRACE_F_MIN_HZ = mls.KERNEL_F_MIN_HZ
TRACE_F_MAX_HZ = mls.KERNEL_F_MAX_HZ
TRACE_SAMPLES_PER_PERIOD = 80

# ``phase_trace`` sums tones on a logarithmic grid, so the Monte Carlo ensemble mean
# is a Riemann sum of ``S_dnu(f) W(f)`` on that grid where the kernel integrates it.
# ``W`` carries sinc fringes of width ``1/T``, and the library default of 40 tones
# per decade under-resolves them from ``f ~ 17/T`` upwards -- straddling the drive
# scale for most of this grid.  Measured deterministically (grid sum vs the kernel's
# integral) over the 20 checked points, 40 is off by up to 67% and tracks the Monte
# Carlo's own error cell for cell, while 320 is converged: every in-regime cell is
# within 0.5% of its 2560-tone limit.  This is a property of the *validator*, not of
# the stored kernel.
TRACE_POINTS_PER_DECADE = 320

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
REGIME_MAX = 0.1       # above this the perturbative expansion is out of range


def _script_sha256() -> str:
    """Fingerprint of this file, recorded in the report so a record cannot drift."""
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def trace_samples(t_gate: float) -> int:
    """Phase samples per trace: ~80 per period at ``TRACE_F_MAX_HZ``.

    A cost knob, not an accuracy one.  The terminal state is already converged to
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


def _infidelity(ops, t_gate, omega_297, d_sweep, trace, ramp, rtol, atol,
                psi_ref) -> np.ndarray:
    """(4,) ``1 - |<psi_0(T)|psi(T)>|^2`` per logical input, under one trace."""
    dim = ops.h_static_diag.size
    labels = ("00", "01", "11") if ops.swap_symmetric else mls.LOGICAL_INPUTS
    res = sweeplib.integrate_batch(
        ops, t_gate, {"omega_297": np.asarray([omega_297]),
                      "d_sweep": np.asarray([d_sweep])},
        labels,
        rhs_factory=_noisy_rhs_factory(trace, mls._rydberg_number_diag(dim)),
        dim=dim, rtol=rtol, atol=atol, ramp=ramp)
    overlap = np.einsum("si,si->s", psi_ref.conj(), res.psi_final[0])
    return 1.0 - np.abs(overlap) ** 2


def _shot(args) -> np.ndarray:
    """One noise realization (pool entry): the (4,) fidelity loss it causes."""
    ops, t_gate, omega_297, d_sweep, ramp, rtol, atol, psd, psi_ref, seed = args
    trace = phase_trace(psd, t_gate, seed=seed, f_min=TRACE_F_MIN_HZ,
                        f_max=TRACE_F_MAX_HZ, n_samples=trace_samples(t_gate),
                        points_per_decade=TRACE_POINTS_PER_DECADE)
    return _infidelity(ops, t_gate, omega_297, d_sweep, trace, ramp, rtol, atol,
                       psi_ref)


def check_point(ops, cfg, key, psd, *, shots: int, seed: int,
                rtol: float, atol: float, pool=None) -> dict:
    """Monte Carlo vs filter-kernel prediction for one grid point.

    Returns the record written to ``reports/phase_noise_mc.json``: the noiseless
    leakage (context only), the mean noise-induced fidelity loss and its standard
    error, and the kernel prediction, each a length-4 vector over the logical inputs
    00/01/10/11.
    """
    if shots < 2:
        raise ValueError(f"shots must be at least 2 for a standard error; got {shots}")
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

    tasks = [(ops, t_gate, omega_297, d_sweep, cfg.ramp_frac, rtol, atol, psd,
              base.psi_final[0], seed + i) for i in range(shots)]
    runner = map if pool is None else pool.map
    eps = np.stack(list(runner(_shot, tasks)))              # (shots, 4)

    mc_mean = eps.mean(axis=0)
    mc_stderr = eps.std(axis=0, ddof=1) / np.sqrt(shots)
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
        "out_of_regime": bool(np.any(prediction > REGIME_MAX)),
        "n_shots": int(shots),
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
                     f"kern {pred:.3e} r={rec['mc_mean'][i] / pred:5.2f}"
                     f"{'!' if pred > REGIME_MAX else ' '}")
    return (f"{rec['point_id']:<22s} n={rec['ryd_n']:<3d} T={rec['t_gate_us']:.1f}us "
            f"Om={rec['omega297_mhz']:<5.1f} D={rec['dsweep_mhz']:<4.1f} | "
            + "  ".join(parts) + f" {'PASS' if rec['passed'] else 'FAIL'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase_noise_mc_check",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--output", default=None,
                   help="scan store whose reports/ receives phase_noise_mc.json "
                        "(nothing else under it is read or written); defaults to "
                        f"{mls._default_output(3.0)}, which a partial or rescaled "
                        "run must not claim -- such a run has to name its own")
    p.add_argument("--n-points", type=int, default=len(MC_POINTS),
                   help=f"how many of the {len(MC_POINTS)} checked points to run")
    p.add_argument("--shots", type=int, default=200,
                   help="independent noise realizations per point")
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


def _resolve_output(args) -> str:
    """The store to write into, refusing to let a probe clobber the deliverable."""
    if args.output is not None:
        return args.output
    if args.n_points < len(MC_POINTS) or args.psd_scale != 1.0:
        raise SystemExit(
            f"refusing the default store: --n-points {args.n_points} of "
            f"{len(MC_POINTS)} / --psd-scale {args.psd_scale:g} is a diagnostic run, "
            "and its record would replace the full unit-scale deliverable.  Pass "
            "--output <dir> to say where it should go.")
    return mls._default_output(3.0)


def _load_psd(args) -> PhaseNoisePSD:
    psd = lnp.load_psds(os.path.join(lnp.NOISE_DIR,
                                     f"psd_{args.laser}.csv"))[args.extrapolation]
    if args.psd_scale == 1.0:
        return psd
    # Rebuilding from the measured samples alone would silently drop an analytic
    # white floor or servo bump.  load_psds sets neither; this is what says so.
    if psd.white_h0 or psd.servo_bump is not None:
        raise SystemExit("--psd-scale cannot rescale an analytic white/servo-bump "
                         "term; scale the construction parameters instead")
    return PhaseNoisePSD(psd.f_hz, psd.s_meas * args.psd_scale,
                         harmonic=psd.harmonic, extrapolation=psd.extrapolation,
                         power_law_fit_hz=psd.power_law_fit_hz)


def _write_report(output: str, report: dict) -> str:
    """Atomically place ``phase_noise_mc.json``, leaving no debris behind on a kill."""
    reports_dir = os.path.join(output, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, "phase_noise_mc.json")
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if not 0 < args.n_points <= len(MC_POINTS):
        raise SystemExit(f"--n-points must be in 1..{len(MC_POINTS)}")
    output = _resolve_output(args)
    psd = _load_psd(args)
    specs = MC_POINTS[:args.n_points]

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
        "script_sha256": _script_sha256(),
        "metric": "1 - |<psi_0(T)|psi(T)>|^2, the noise-induced fidelity loss "
                  "against the noiseless final state, per logical input",
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
        "estimator": "mean over independent realizations; this metric has no "
                     "first-order term, so no antithetic pairing is used",
        "acceptance": f"|mc_mean - prediction| <= max({SIGMA_TOL:g} * mc_stderr, "
                      f"{REL_TOL:g} * prediction), over the logical inputs with a "
                      f"nonzero prediction",
        "regime_max": REGIME_MAX,
        "n_points": len(records),
        "n_passed": sum(r["passed"] for r in records),
        "n_out_of_regime": sum(r["out_of_regime"] for r in records),
        "wall_seconds": time.time() - t0,
        "points": records,
    }
    path = _write_report(output, report)
    print(f"passed: {report['n_passed']}/{report['n_points']}  "
          f"(out of perturbative regime: {report['n_out_of_regime']})\nwrote {path}")


if __name__ == "__main__":
    main()
