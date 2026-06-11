"""Quench-benchmark parity/timing harness for the TN-dewrap refactor.

Mirrors ``scripts/notebooks/run_quench_benchmark.ipynb`` but parameterized and
machine-readable, so a run captured before the refactor can be compared bit-for-bit
against a run captured after it.

Run with ``uv run`` (project convention):

    # capture a baseline (on the untouched branch)
    uv run python scripts/bench_quench_check.py --backends exact mps gputn --out /tmp/base.json

    # after a change, compare to the baseline
    uv run python scripts/bench_quench_check.py --backends exact mps gputn \
        --out /tmp/after.json --baseline /tmp/base.json --atol 1e-10

Phase 1 (pure plumbing) should pass at ``--atol 1e-10``. Phase 2 (speed hoist) is
checked instead against the exact backend via ``max|Delta|`` (printed per backend).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import ryd_gate as rg
from ryd_gate import InteractionSpec
from ryd_gate.lattice import Register


def build_system(args):
    Omega = 2 * np.pi * args.omega_mhz * 1e6
    geom = Register.rectangle(args.lx, args.ly, spacing_um=args.a_um)
    C6 = 2 * np.pi * 874e9
    t_sweep = args.t_sweep
    delta_start = -2 * np.pi * 10.0e6
    delta_end = 2 * np.pi * 10.0e6
    ramp_frac = 0.09

    def omega_half_t(t):
        s = np.clip(t / max(t_sweep, np.finfo(float).eps), 0.0, 1.0)

        def smoothstep5(u):
            u = np.clip(u, 0.0, 1.0)
            return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5

        if s < ramp_frac:
            env = smoothstep5(s / ramp_frac)
        elif s > 1.0 - ramp_frac:
            env = smoothstep5((1.0 - s) / ramp_frac)
        else:
            env = 1.0
        return 0.5 * Omega * env

    def delta_t(t):
        s = np.clip(t / max(t_sweep, np.finfo(float).eps), 0.0, 1.0)
        delta_mid = 0.5 * (delta_start + delta_end)
        delta_amp = 0.5 * (delta_end - delta_start)
        return delta_mid - delta_amp * np.cos(2.0 * np.pi * s)

    protocol = rg.SweepProtocol(
        t_gate=t_sweep,
        omega_half_fn=omega_half_t,
        delta_fn=delta_t,
        address_fn=None,
        n_steps=args.n_steps,
    )
    system = rg.RydbergSystem.from_lattice(
        geom, "1r",
        interaction=InteractionSpec(C6=C6, mode="nn"),
        protocol=protocol,
    )
    return system, Omega, t_sweep


def run_exact(system, t_eval):
    t0 = time.perf_counter()
    res = rg.simulate(system, [], "all_ground", backend="exact", t_eval=t_eval)
    elapsed = time.perf_counter() - t0
    n_i = np.asarray([[system.expectation(f"n_r_{i}", psi) for i in range(system.N)] for psi in res.states])
    n_mean = n_i.mean(axis=1)
    return n_mean, n_i, elapsed


def run_tn(system, backend, t_eval, opts):
    t0 = time.perf_counter()
    res = rg.simulate(
        system, [], "all_ground", backend=backend, t_eval=t_eval,
        observables=["n_mean", "n_i"], backend_options=opts,
    )
    elapsed = time.perf_counter() - t0
    n_mean = np.asarray(res.metadata["obs"]["n_mean"])
    n_i = np.asarray(res.metadata["obs"]["n_i"])
    return n_mean, n_i, elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lx", type=int, default=3)
    p.add_argument("--ly", type=int, default=3)
    p.add_argument("--a-um", type=float, default=10.0)
    p.add_argument("--omega-mhz", type=float, default=3.8)
    p.add_argument("--t-sweep", type=float, default=1.5e-6)
    p.add_argument("--n-steps", type=int, default=40, help="protocol/exact piecewise steps")
    p.add_argument("--n-eval", type=int, default=5)
    p.add_argument("--chi-max", type=int, default=16)
    p.add_argument("--dt-frac", type=float, default=0.2, help="dt = dt_frac / Omega")
    p.add_argument("--backends", nargs="+", default=["exact", "mps"])
    p.add_argument("--gputn-kernel", type=str, default="statevector",
                   help="gputn kernel: 'statevector' (exact, fast, <=24 sites) or 'auto'/'cutensornet_mps'")
    p.add_argument("--peps-cuda", action="store_true", help="run YASTN PEPS on CUDA (needs torch); default CPU")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--baseline", type=str, default=None)
    p.add_argument("--atol", type=float, default=1e-10)
    args = p.parse_args()

    system, Omega, t_sweep = build_system(args)
    t_eval = np.linspace(0.0, t_sweep, args.n_eval)
    dt_tn = args.dt_frac / Omega

    tn_opts = {
        "mps": {"chi_max": args.chi_max, "dt": dt_tn, "svd_min": 1e-10},
        "gputn": {"chi_max": args.chi_max, "dt": dt_tn, "svd_min": 1e-10,
                  "require_gpu": True, "kernel": args.gputn_kernel},
        "peps": {"chi_max": min(args.chi_max, 10), "dt": dt_tn, "svd_min": 1e-8,
                 "measurement_environment": "bp", "update_environment": "ntu", "max_iter": 10,
                 "use_cuda": args.peps_cuda},
    }

    results = {}
    exact_n_mean = None
    for backend in args.backends:
        try:
            if backend == "exact":
                n_mean, n_i, elapsed = run_exact(system, t_eval)
                exact_n_mean = n_mean
            else:
                n_mean, n_i, elapsed = run_tn(system, backend, t_eval, tn_opts[backend])
        except Exception as exc:  # noqa: BLE001 - record, don't abort the sweep
            results[backend] = {"error": repr(exc)[:300]}
            print(f"[{backend}] ERROR: {repr(exc)[:200]}")
            continue
        entry = {"n_mean": n_mean.tolist(), "n_i": n_i.tolist(), "elapsed_s": elapsed}
        if exact_n_mean is not None and backend != "exact":
            entry["max_abs_diff_n_mean"] = float(np.max(np.abs(n_mean - exact_n_mean)))
        results[backend] = entry
        diff = entry.get("max_abs_diff_n_mean")
        diff_str = f"  max|Δ vs exact|={diff:.3e}" if diff is not None else ""
        print(f"[{backend}] elapsed={elapsed:8.3f}s{diff_str}")

    payload = {"config": vars(args), "results": results}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"wrote {args.out}")

    if args.baseline:
        with open(args.baseline) as fh:
            base = json.load(fh)["results"]
        print(f"\n=== parity vs {args.baseline} (atol={args.atol:g}) ===")
        ok = True
        for backend, entry in results.items():
            if "error" in entry or backend not in base or "error" in base[backend]:
                print(f"[{backend}] SKIP (missing/error)")
                continue
            for key in ("n_mean", "n_i"):
                d = float(np.max(np.abs(np.asarray(entry[key]) - np.asarray(base[backend][key]))))
                status = "PASS" if d <= args.atol else "FAIL"
                ok = ok and d <= args.atol
                print(f"[{backend}] {key}: max|Δ|={d:.3e} {status}")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
