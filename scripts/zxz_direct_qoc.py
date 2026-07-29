"""ZXZ synthesis study (arXiv:2508.19075 Fig. 3 reproduction; spec 2026-07-28 §3-§4).

Direct quantum optimal control (qoc.direct/IPOPT) vs GRAPE (qoc.grape +
qoc.minimize) on the 3-atom 1r analog chain: synthesize
U_target = expm(-i*0.8*Z1 X2 Z3) under Aquila hardware constraints.

Units: rad/us and us everywhere; SI (rad/s, s) only at the simulate boundary.
Subcommands: model-check | direct | grape | validate | plot. All artifacts go
to results/zxz_direct_qoc/ as npz so plotting replays without recompute.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
from scipy.linalg import expm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

TAU = 2.0 * np.pi
RYD_LEVEL = 70
SPACING_UM = 8.9
N_ATOMS = 3
DT_US = 0.05
U_OMEGA_MAX = TAU * 2.4 / 2.0        # channel value = Omega/2      (rad/us)
DELTA_MAX = TAU * 20.0               # |channel value| = |Delta|    (rad/us)
SLEW_U_OMEGA = 250.0 / 2.0           # Omega slew 250 rad/us^2 -> channel
SLEW_U_DELTA = 2500.0
TAU_JEFF = 0.8
DURATIONS = {"pulse1": 1.2, "pulse2": 3.6}
CH_OM = "E[r,1]:x"
CH_DE = "E[r,r]"
LABELS = [tuple(p) for p in product("1r", repeat=N_ATOMS)]
RESULTS_DIR = REPO_ROOT / "results" / "zxz_direct_qoc"


def build_model():
    """8-dim bilinear model (rad/us) of the 3-atom 1r chain + basis index map."""
    from ryd_gate import Register, RydbergSystem, bilinear_control_model, level_structure
    from ryd_gate.protocols import SweepProtocol

    reference = SweepProtocol(
        t_gate_s=1e-6,
        omega_half_rad_s=lambda t: 0.0,
        detuning_rad_s=lambda t: 0.0,
    )
    system = RydbergSystem(
        level_structure=level_structure("1r", ryd_level=RYD_LEVEL),
        register=Register.chain(N_ATOMS, spacing_um=SPACING_UM),
        protocol=reference,
    )
    h0_si, channels, states = bilinear_control_model(system, states=[list(l) for l in LABELS])
    index = {}
    for lab in LABELS:
        vec = states[lab]
        idx = int(np.argmax(np.abs(vec)))
        if abs(abs(vec[idx]) - 1.0) > 1e-12:
            raise RuntimeError(f"product state {lab} is not a coordinate vector")
        index[lab] = idx
    return {
        "h0": np.asarray(h0_si, dtype=complex) * 1e-6,   # rad/s -> rad/us
        "controls": {CH_OM: np.asarray(channels[CH_OM]), CH_DE: np.asarray(channels[CH_DE])},
        "index": index,
        "labels": LABELS,
    }


def build_zxz(index):
    """Z1 X2 Z3 in the basis ordering given by index (Z = |1><1| - |r><r|)."""
    op = np.zeros((8, 8), dtype=complex)
    for lab, col in index.items():
        flipped = (lab[0], "r" if lab[1] == "1" else "1", lab[2])
        sign = (1.0 if lab[0] == "1" else -1.0) * (1.0 if lab[2] == "1" else -1.0)
        op[index[flipped], col] = sign
    return op


def build_target(index):
    return expm(-1j * TAU_JEFF * build_zxz(index))


def fidelity(u, target):
    return float(abs(np.trace(target.conj().T @ u)) ** 2) / target.shape[0] ** 2


def unitary_infidelity(u_final, target):
    """(1 - F, G) with G = dL/d(conj U) per the qoc costate convention."""
    d2 = target.shape[0] ** 2
    c = np.trace(target.conj().T @ u_final)
    return 1.0 - float(abs(c) ** 2) / d2, -(c / d2) * target


U_BOUNDS = {CH_OM: (0.0, U_OMEGA_MAX), CH_DE: (-DELTA_MAX, DELTA_MAX)}
DU_BOUNDS = {CH_OM: SLEW_U_OMEGA, CH_DE: SLEW_U_DELTA}
DDU_BOUNDS = {CH_OM: SLEW_U_OMEGA / DT_US, CH_DE: SLEW_U_DELTA / DT_US}


def cmd_model_check(args):
    model = build_model()
    index = model["index"]
    v_nn = model["h0"][index[("r", "r", "1")], index[("r", "r", "1")]].real
    target = build_target(index)
    print(f"basis order: {[lab for lab, _ in sorted(index.items(), key=lambda kv: kv[1])]}")
    print(f"V_NN/2pi = {v_nn / TAU:.4f} MHz (expect ~ +1.736)")
    print(f"F(identity) = {fidelity(np.eye(8, dtype=complex), target):.6f}")
    print(f"channels: {list(model['controls'])}, dim = {model['h0'].shape[0]}")


def _smooth_random_knots(rng, k, lo, hi):
    u = rng.uniform(lo, hi, k + 1)
    u[0] = u[-1] = 0.0
    u = np.convolve(u, np.ones(3) / 3.0, mode="same")
    u[0] = u[-1] = 0.0
    return np.clip(u, lo, hi)


def run_direct_seed(seed, tag, duration_us, maxiter, model_arrays):
    """One IPOPT solve from one smooth random start. Top-level: picklable."""
    from qoc import direct

    h0, ops_om, ops_de, target = model_arrays
    controls = {CH_OM: ops_om, CH_DE: ops_de}
    k = int(round(duration_us / DT_US))
    rng = np.random.default_rng(seed)
    initial = {
        CH_OM: _smooth_random_knots(rng, k, 0.0, U_OMEGA_MAX),
        CH_DE: _smooth_random_knots(rng, k, -DELTA_MAX, DELTA_MAX),
    }
    result = direct.optimize(
        h0,
        controls,
        n_slices=k,
        dt=DT_US,
        terminal_objective=partial(unitary_infidelity, target=target),
        u_bounds=U_BOUNDS,
        du_bounds=DU_BOUNDS,
        ddu_bounds=DDU_BOUNDS,
        initial_controls=initial,
        maxiter=maxiter,
    )
    out = RESULTS_DIR / f"direct_{tag}_seed{seed}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        u_omega=result.controls[CH_OM],
        u_delta=result.controls[CH_DE],
        du_omega=result.du[CH_OM],
        du_delta=result.du[CH_DE],
        ddu_omega=result.ddu[CH_OM],
        ddu_delta=result.ddu[CH_DE],
        fidelity=1.0 - result.objective,
        objective=result.objective,
        max_defect=result.max_defect,
        accepted=result.accepted,
        ipopt_status=result.ipopt_status,
        n_iter=result.n_iter,
        seed=seed,
        K=k,
        dt_us=DT_US,
        duration_us=duration_us,
    )
    return {
        "seed": seed,
        "fidelity": 1.0 - result.objective,
        "accepted": bool(result.accepted),
        "ipopt_status": int(result.ipopt_status),
        "n_iter": int(result.n_iter),
    }


def cmd_direct(args):
    model = build_model()
    target = build_target(model["index"])
    model_arrays = (model["h0"], model["controls"][CH_OM], model["controls"][CH_DE], target)
    tags = list(DURATIONS) if args.tag == "all" else [args.tag]
    for tag in tags:
        duration = DURATIONS[tag]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(run_direct_seed, seed, tag, duration, args.maxiter, model_arrays)
                for seed in range(args.seeds)
            ]
            for fut in futures:
                row = fut.result()
                rows.append(row)
                print(f"[{tag}] seed {row['seed']}: F={row['fidelity']:.4f} "
                      f"accepted={row['accepted']} status={row['ipopt_status']} it={row['n_iter']}")
        accepted = [r for r in rows if r["accepted"]]
        best = max(accepted or rows, key=lambda r: r["fidelity"])
        summary = {"tag": tag, "duration_us": duration, "runs": rows, "best": best}
        (RESULTS_DIR / f"direct_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[{tag}] best: seed {best['seed']} F={best['fidelity']:.4f} "
              f"({len(accepted)}/{len(rows)} accepted)")


def _knots_from_params(named, k):
    om = np.concatenate([[0.0], np.asarray(named["omega"], dtype=float), [0.0]])
    de = np.concatenate([[0.0], np.asarray(named["delta"], dtype=float), [0.0]])
    return om, de


def _penalty_and_grad(knots, lo, hi, slew_max):
    """lambda-weighted quadratic penalties on values and knot slopes."""
    value = 0.0
    grad = np.zeros_like(knots)
    over = np.maximum(0.0, knots - hi)
    under = np.maximum(0.0, lo - knots)
    value += float(over @ over + under @ under)
    grad += 2.0 * over - 2.0 * under
    slopes = np.diff(knots) / DT_US
    excess = np.maximum(0.0, np.abs(slopes) - slew_max)
    value += float(excess @ excess)
    d_slope = 2.0 * excess * np.sign(slopes) / DT_US
    grad[1:] += d_slope
    grad[:-1] -= d_slope
    return value, grad


def _smoothness_and_grad(knots):
    """mean of squared second differences (rad/us^3 units)."""
    d2 = (knots[:-2] - 2.0 * knots[1:-1] + knots[2:]) / DT_US**2
    n = max(d2.size, 1)
    value = float(d2 @ d2) / n
    grad = np.zeros_like(knots)
    coeff = 2.0 * d2 / (n * DT_US**2)
    grad[:-2] += coeff
    grad[1:-1] += -2.0 * coeff
    grad[2:] += coeff
    return value, grad


def run_grape_seed(seed, r_weight, k, model_arrays, maxiter):
    """One penalized L-BFGS-B GRAPE run. Top-level: picklable."""
    import qoc
    from qoc import grape

    h0, ops_om, ops_de, target = model_arrays
    controls = {CH_OM: ops_om, CH_DE: ops_de}
    time_grid = np.linspace(0.0, k * DT_US, k + 1)
    basis_states = [np.eye(8, dtype=complex)[:, j] for j in range(8)]
    lam = 100.0

    def terminal_objective(final_states):
        u = np.column_stack(final_states)
        value, g = unitary_infidelity(u, target)
        return value, [np.array(g[:, j]) for j in range(8)]

    def control_map(named):
        om, de = _knots_from_params(named, k)
        return {CH_OM: 0.5 * (om[:-1] + om[1:]), CH_DE: 0.5 * (de[:-1] + de[1:])}

    def control_pullback(named, channel_gradients):
        out = {}
        for name, key in ((CH_OM, "omega"), (CH_DE, "delta")):
            g = np.asarray(channel_gradients[name], dtype=float)
            knots = np.zeros(k + 1)
            knots[:-1] += 0.5 * g
            knots[1:] += 0.5 * g
            out[key] = knots[1:-1]
        return out

    def full_loss_and_grad(named):
        fid_value, fid_grad = grape.value_and_grad(
            named, h0=h0, controls=controls, initial_states=basis_states,
            time_grid=time_grid, control_map=control_map,
            control_pullback=control_pullback, terminal_objective=terminal_objective,
        )
        value = fid_value
        grads = {key: np.asarray(fid_grad[key], dtype=float).copy() for key in ("omega", "delta")}
        om, de = _knots_from_params(named, k)
        for knots, key, lo, hi, slew in (
            (om, "omega", 0.0, U_OMEGA_MAX, SLEW_U_OMEGA),
            (de, "delta", -DELTA_MAX, DELTA_MAX, SLEW_U_DELTA),
        ):
            p_val, p_grad = _penalty_and_grad(knots, lo, hi, slew)
            s_val, s_grad = _smoothness_and_grad(knots)
            value += lam * p_val + r_weight * s_val
            grads[key] += lam * p_grad[1:-1] + r_weight * s_grad[1:-1]
        return value, grads

    cache = {}

    def loss(named):
        key = (np.asarray(named["omega"]).tobytes(), np.asarray(named["delta"]).tobytes())
        if key not in cache:
            cache.clear()
            cache[key] = full_loss_and_grad(named)
        return cache[key][0]

    def gradient(named):
        loss(named)
        key = (np.asarray(named["omega"]).tobytes(), np.asarray(named["delta"]).tobytes())
        return cache[key][1]

    rng = np.random.default_rng(seed)
    x0 = {
        "omega": rng.uniform(0.0, U_OMEGA_MAX, k - 1),
        "delta": rng.uniform(-TAU * 5.0, TAU * 5.0, k - 1),
    }
    result = qoc.minimize(
        loss, x0, method="l-bfgs-b",
        scales={"omega": U_OMEGA_MAX, "delta": DELTA_MAX},
        options={"gradient": gradient, "maxiter": maxiter},
    )
    om, de = _knots_from_params(result.best_parameters, k)
    fid_value = grape.value(
        result.best_parameters, h0=h0, controls=controls, initial_states=basis_states,
        time_grid=time_grid, control_map=control_map, terminal_objective=terminal_objective,
    )
    viol = max(
        float(np.max(np.maximum(0.0, om - U_OMEGA_MAX))),
        float(np.max(np.maximum(0.0, -om))),
        float(np.max(np.maximum(0.0, np.abs(de) - DELTA_MAX))),
        float(np.max(np.maximum(0.0, np.abs(np.diff(om)) / DT_US - SLEW_U_OMEGA))),
        float(np.max(np.maximum(0.0, np.abs(np.diff(de)) / DT_US - SLEW_U_DELTA))),
    )
    return {
        "seed": seed, "fidelity": 1.0 - fid_value, "final_loss": result.best_loss,
        "max_violation": viol, "u_omega": om, "u_delta": de,
    }


def cmd_grape(args):
    model = build_model()
    target = build_target(model["index"])
    model_arrays = (model["h0"], model["controls"][CH_OM], model["controls"][CH_DE], target)
    k = int(round(args.duration_us / DT_US))
    r_values = [float(v) for v in args.r_values.split(",")]
    seeds = list(range(args.seeds))
    fids = np.zeros((len(r_values), len(seeds)))
    losses = np.zeros_like(fids)
    viols = np.zeros_like(fids)
    om_all = np.zeros((len(r_values), len(seeds), k + 1))
    de_all = np.zeros_like(om_all)
    for i, r_weight in enumerate(r_values):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(run_grape_seed, seed, r_weight, k, model_arrays, args.maxiter)
                for seed in seeds
            ]
            for j, fut in enumerate(futures):
                row = fut.result()
                fids[i, j] = row["fidelity"]
                losses[i, j] = row["final_loss"]
                viols[i, j] = row["max_violation"]
                om_all[i, j] = row["u_omega"]
                de_all[i, j] = row["u_delta"]
        print(f"[grape r={r_weight:g}] median F={np.median(fids[i]):.4f} "
              f"best F={np.max(fids[i]):.4f} max_viol={np.max(viols[i]):.3g}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        RESULTS_DIR / f"grape_T{args.duration_us}.npz",
        fidelities=fids, final_losses=losses, max_violation=viols,
        u_omega=om_all, u_delta=de_all, r_values=np.asarray(r_values),
        seeds=np.asarray(seeds), K=k, dt_us=DT_US, duration_us=args.duration_us,
        lambda_penalty=100.0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("model-check").set_defaults(func=cmd_model_check)
    p = sub.add_parser("direct")
    p.add_argument("--tag", default="all")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--maxiter", type=int, default=4000)
    p.set_defaults(func=cmd_direct)
    p = sub.add_parser("grape")
    p.add_argument("--duration-us", type=float, default=1.2)
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--maxiter", type=int, default=1500)
    p.add_argument("--r-values", default="0,1e-8,1e-7,1e-6")
    p.set_defaults(func=cmd_grape)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
