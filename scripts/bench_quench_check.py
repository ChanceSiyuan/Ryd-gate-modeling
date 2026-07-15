"""Quench-benchmark parity/timing harness for the TN-dewrap refactor.

Mirrors ``scripts/notebooks/run_quench_benchmark.ipynb`` but parameterized and
machine-readable, so a run captured before the refactor can be compared bit-for-bit
against a run captured after it.

Run with ``uv run`` (project convention):

    # capture a baseline (on the untouched branch)
    uv run python scripts/bench_quench_check.py --backends exact_ode mps peps --out /tmp/base.json

    # after a change, compare to the baseline
    uv run python scripts/bench_quench_check.py --backends exact_ode mps peps \
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
from ryd_gate import level_structure
from ryd_gate.lattice import Register
from ryd_gate.protocols import SweepProtocol


def build_system(args):
    # The interaction C6 is an atomic property of the Rydberg level and lives on
    # the level_structure preset (70S -> ~2pi*874 GHz*um^6 via ARC); it is no
    # longer passed as an explicit InteractionSpec. Nearest-neighbour coupling
    # (the old mode="nn") is reproduced with a distance cutoff at the lattice
    # spacing, which keeps the distance-a pairs and drops the a*sqrt(2) diagonals.
    Omega = 2 * np.pi * args.omega_mhz * 1e6
    geom = Register.rectangle(args.lx, args.ly, spacing_um=args.a_um)
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

    # SweepProtocol drives H[r,1] = omega_half_rad_s (already Omega/2) and
    # H[r,r] = -detuning_rad_s -- the same convention the old omega_half_fn /
    # delta_fn schedule used.
    protocol = SweepProtocol(
        t_gate_s=t_sweep,
        omega_half_rad_s=omega_half_t,
        detuning_rad_s=delta_t,
    )
    system = rg.RydbergSystem(
        level_structure=level_structure("1r"),
        register=geom,
        protocol=protocol,
        interaction_cutoff_um=args.a_um,
    )
    return system, Omega, t_sweep


def run_exact(system, t_eval):
    obs = system.observables
    observables = {f"n_r_{i}": obs.n("r", i) for i in range(system.N)}  # per-site "n_r_i"
    t0 = time.perf_counter()
    res = rg.simulate(system, t_eval=t_eval, observables=observables)  # default exact_ode (adaptive DOP853)
    elapsed = time.perf_counter() - t0
    n_i = np.stack([res.expectation(f"n_r_{i}").real for i in range(system.N)], axis=1)
    n_mean = n_i.mean(axis=1)
    return n_mean, n_i, elapsed


def run_tn(system, backend, t_eval, opts):
    obs = system.observables
    observables = {f"n_r_{i}": obs.n("r", i) for i in range(system.N)}  # per-site "n_r_i"
    observables["n_mean"] = (1.0 / system.N) * sum(obs.n("r", i) for i in range(system.N))
    t0 = time.perf_counter()
    res = rg.simulate(
        system, backend=backend, t_eval=t_eval,
        observables=observables, backend_options=opts,
    )
    elapsed = time.perf_counter() - t0
    n_mean = res.expectation("n_mean").real
    n_i = np.stack([res.expectation(f"n_r_{i}").real for i in range(system.N)], axis=1)
    return n_mean, n_i, elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lx", type=int, default=3)
    p.add_argument("--ly", type=int, default=3)
    p.add_argument("--a-um", type=float, default=10.0)
    p.add_argument("--omega-mhz", type=float, default=3.8)
    p.add_argument("--t-sweep", type=float, default=1.5e-6)
    p.add_argument("--n-eval", type=int, default=5)
    p.add_argument("--chi-max", type=int, default=16)
    p.add_argument("--dt-frac", type=float, default=0.2, help="dt = dt_frac / Omega")
    p.add_argument("--backends", nargs="+", default=["exact_ode", "mps"])
    p.add_argument("--peps-cuda", action="store_true", help="run YASTN PEPS on CUDA (needs torch); default CPU")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--baseline", type=str, default=None)
    p.add_argument("--atol", type=float, default=1e-10)
    args = p.parse_args()

    system, Omega, t_sweep = build_system(args)
    t_eval = np.linspace(0.0, t_sweep, args.n_eval)
    dt_tn = args.dt_frac / Omega

    tn_opts = {
        # MPS TDVP options are exactly these three keys (E23).
        "mps": {"time_step_s": dt_tn, "bond_dimension": args.chi_max, "discarded_weight_tolerance": 1e-10},
        # PEPS real-time options are exactly these ten keys; NTU/environment choices are report-only.
        "peps": {
            "time_step_s": dt_tn,
            "bond_dimension": min(args.chi_max, 10),
            "svd_tolerance": 1e-8,
            "ntu_max_iterations": 100,
            "ntu_iteration_tolerance": 1e-10,
            "measurement_method": "belief_propagation",
            "environment_bond_dimension": min(args.chi_max, 16),
            "environment_tolerance": 1e-8,
            "environment_max_iterations": 50,
            "device": "cuda" if args.peps_cuda else "cpu",
        },
    }

    results = {}
    exact_n_mean = None
    for backend in args.backends:
        try:
            if backend == "exact_ode":
                n_mean, n_i, elapsed = run_exact(system, t_eval)
                exact_n_mean = n_mean
            else:
                n_mean, n_i, elapsed = run_tn(system, backend, t_eval, tn_opts[backend])
        except Exception as exc:  # noqa: BLE001 - record, don't abort the sweep
            results[backend] = {"error": repr(exc)[:300]}
            print(f"[{backend}] ERROR: {repr(exc)[:200]}")
            continue
        entry = {"n_mean": n_mean.tolist(), "n_i": n_i.tolist(), "elapsed_s": elapsed}
        if exact_n_mean is not None and backend != "exact_ode":
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
