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
LAMBDA_PENALTY = 100.0
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


def run_direct_seed(seed, tag, duration_us, maxiter, model_arrays, warm):
    """One IPOPT solve from one smooth random (or warm) start. Top-level: picklable."""
    from qoc import direct

    h0, ops_om, ops_de, target = model_arrays
    controls = {CH_OM: ops_om, CH_DE: ops_de}
    k = int(round(duration_us / DT_US))
    out = RESULTS_DIR / f"direct_{tag}_seed{seed}.npz"
    if warm and out.exists():
        prev = np.load(out)
        initial = {CH_OM: prev["u_omega"], CH_DE: prev["u_delta"]}
    else:
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
    u_om = result.controls[CH_OM]
    u_de = result.controls[CH_DE]
    u_chain = np.eye(8, dtype=complex)
    for kk in range(k):
        h_k = h0 + 0.5 * (u_om[kk] + u_om[kk + 1]) * ops_om + 0.5 * (u_de[kk] + u_de[kk + 1]) * ops_de
        u_chain = expm(-1j * DT_US * h_k) @ u_chain
    fidelity_rollout = fidelity(u_chain, target)
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
        fidelity_rollout=fidelity_rollout,
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
        "fidelity_rollout": fidelity_rollout,
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
                pool.submit(run_direct_seed, seed, tag, duration, args.maxiter,
                            model_arrays, args.warm_start)
                for seed in range(args.seeds)
            ]
            for fut in futures:
                row = fut.result()
                rows.append(row)
                print(f"[{tag}] seed {row['seed']}: F_var={row['fidelity']:.4f} "
                      f"F_roll={row['fidelity_rollout']:.4f} "
                      f"accepted={row['accepted']} status={row['ipopt_status']} it={row['n_iter']}")
        accepted = [r for r in rows if r["accepted"]]
        best = max(accepted or rows, key=lambda r: r["fidelity_rollout"])
        summary = {"tag": tag, "duration_us": duration, "runs": rows, "best": best}
        (RESULTS_DIR / f"direct_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[{tag}] best: seed {best['seed']} F_var={best['fidelity']:.4f} "
              f"F_roll={best['fidelity_rollout']:.4f} ({len(accepted)}/{len(rows)} accepted)")


def _knots_from_params(named):
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
    """mean of squared second differences (units (rad/us^3)^2)."""
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
    lam = LAMBDA_PENALTY

    def terminal_objective(final_states):
        u = np.column_stack(final_states)
        value, g = unitary_infidelity(u, target)
        return value, [np.array(g[:, j]) for j in range(8)]

    def control_map(named):
        om, de = _knots_from_params(named)
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
        om, de = _knots_from_params(named)
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
    om, de = _knots_from_params(result.best_parameters)
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
        RESULTS_DIR / f"grape_T{args.duration_us:g}.npz",
        fidelities=fids, final_losses=losses, max_violation=viols,
        u_omega=om_all, u_delta=de_all, r_values=np.asarray(r_values),
        seeds=np.asarray(seeds), K=k, dt_us=DT_US, duration_us=args.duration_us,
        lambda_penalty=LAMBDA_PENALTY,
    )


def _replay_unitary(model, u_om_knots, u_de_knots, k, mode):
    """Exact-ODE propagator of the knot waveform (mode: 'zoh'|'linear')."""
    from ryd_gate import Register, RydbergSystem, level_structure, simulate
    from ryd_gate.protocols import SweepProtocol

    t_total_us = k * DT_US
    knots_t = np.arange(k + 1) * DT_US
    mid_om = 0.5 * (u_om_knots[:-1] + u_om_knots[1:])
    mid_de = 0.5 * (u_de_knots[:-1] + u_de_knots[1:])

    def value_at(t_s, knots, mids):
        t_us = min(max(t_s * 1e6, 0.0), t_total_us - 1e-12)
        if mode == "zoh":
            return float(mids[int(t_us / DT_US)])
        return float(np.interp(t_us, knots_t, knots))

    protocol = SweepProtocol(
        t_gate_s=t_total_us * 1e-6,
        omega_half_rad_s=lambda t: value_at(t, u_om_knots, mid_om) * 1e6,
        detuning_rad_s=lambda t: -value_at(t, u_de_knots, mid_de) * 1e6,
    )
    system = RydbergSystem(
        level_structure=level_structure("1r", ryd_level=RYD_LEVEL),
        register=Register.chain(N_ATOMS, spacing_um=SPACING_UM),
        protocol=protocol,
    )
    index = model["index"]
    ordered = [lab for lab, _ in sorted(index.items(), key=lambda kv: kv[1])]
    results = simulate(system, [list(lab) for lab in ordered], backend="exact_ode")
    u_ode = np.zeros((8, 8), dtype=complex)
    for j, res in enumerate(results):
        for i, lab in enumerate(ordered):
            u_ode[i, j] = res.amplitude(list(lab))
    return u_ode


def cmd_validate(args):
    model = build_model()
    target = build_target(model["index"])
    if args.grape:
        g = np.load(RESULTS_DIR / f"grape_T{args.duration_us:g}.npz")
        fids = g["fidelities"]
        i_best = np.unravel_index(np.argmax(fids), fids.shape)
        r_value = float(g["r_values"][i_best[0]])
        seed = int(g["seeds"][i_best[1]])
        k = int(g["K"])
        f_disc = float(fids[i_best])
        u_zoh = _replay_unitary(model, g["u_omega"][i_best], g["u_delta"][i_best], k, "zoh")
        u_lin = _replay_unitary(model, g["u_omega"][i_best], g["u_delta"][i_best], k, "linear")
        f_zoh = fidelity(u_zoh, target)
        f_lin = fidelity(u_lin, target)
        np.savez(
            RESULTS_DIR / f"validate_grape_T{args.duration_us:g}.npz",
            f_discrete=f_disc, f_ode_zoh=f_zoh, f_ode_linear=f_lin,
            r_value=r_value, seed=seed,
        )
        print(f"[grape T={args.duration_us:g} r={r_value:g} seed={seed}] "
              f"F_discrete={f_disc:.5f} F_ode_zoh={f_zoh:.5f} F_ode_linear={f_lin:.5f}")
        return
    if args.tag is None:
        raise ValueError("validate requires --tag (direct path) or --grape")
    if args.seed is None:
        summary = json.loads((RESULTS_DIR / f"direct_{args.tag}_summary.json").read_text())
        args.seed = int(summary["best"]["seed"])
    data = np.load(RESULTS_DIR / f"direct_{args.tag}_seed{args.seed}.npz")
    k = int(data["K"])
    f_disc = float(data["fidelity_rollout"]) if "fidelity_rollout" in data.files else float(data["fidelity"])
    u_zoh = _replay_unitary(model, data["u_omega"], data["u_delta"], k, "zoh")
    u_lin = _replay_unitary(model, data["u_omega"], data["u_delta"], k, "linear")
    f_zoh = fidelity(u_zoh, target)
    f_lin = fidelity(u_lin, target)
    gate_pass = bool(abs(f_zoh - f_disc) < 1e-3)
    np.savez(
        RESULTS_DIR / f"validate_{args.tag}.npz",
        f_discrete=f_disc, f_ode_zoh=f_zoh, f_ode_linear=f_lin,
        gate_pass=gate_pass, seed=args.seed,
    )
    print(f"[{args.tag} seed {args.seed}] F_discrete={f_disc:.5f} "
          f"F_ode_zoh={f_zoh:.5f} (gate {'PASS' if gate_pass else 'FAIL'}) "
          f"F_ode_linear={f_lin:.5f}")


def cmd_plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = RESULTS_DIR / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    g = np.load(RESULTS_DIR / f"grape_T{args.duration_us:g}.npz")
    fids, r_values = g["fidelities"], g["r_values"]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.violinplot([fids[i] for i in range(len(r_values))],
                  positions=np.arange(len(r_values)), showmedians=True)
    ax.set_xticks(np.arange(len(r_values)))
    ax.set_xticklabels([f"r={v:g}" for v in r_values])
    colors = {"pulse1": "tab:red", "pulse2": "tab:purple"}
    for tag, color in colors.items():
        path = RESULTS_DIR / f"direct_{tag}_summary.json"
        if path.exists():
            best = json.loads(path.read_text())["best"]
            f_best = best.get("fidelity_rollout", best["fidelity"])
            ax.axhline(f_best, color=color, ls="--", lw=1.2,
                       label=f"direct {tag} (T={DURATIONS[tag]} us): F={f_best:.3f}")
    ax.set_ylabel("unitary fidelity (discrete knot model)")
    ax.set_xlabel(f"GRAPE regularization (T={args.duration_us} us, "
                  f"{fids.shape[1]} seeds)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("ZXZ synthesis: direct method vs GRAPE (Fig. 3b style)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plots / f"fig3b_violin.{ext}", dpi=200)
    plt.close(fig)

    pulse1_summary = RESULTS_DIR / "direct_pulse1_summary.json"
    if not pulse1_summary.exists():
        print(f"note: {pulse1_summary.name} missing; wrote fig3b, skipping fig3c pulse figure")
        return
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.5), sharex=True)
    best1 = json.loads(pulse1_summary.read_text())["best"]
    d = np.load(RESULTS_DIR / f"direct_pulse1_seed{best1['seed']}.npz")
    t = np.arange(int(d["K"]) + 1) * DT_US
    axes[0].plot(t, 2.0 * d["u_omega"] / TAU, label="direct pulse1")
    axes[1].plot(t, -d["u_delta"] / TAU, label="direct pulse1")
    i_best = np.unravel_index(np.argmax(fids), fids.shape)
    viol = float(g["max_violation"][i_best])
    tg = np.arange(int(g["K"]) + 1) * DT_US
    grape_label = f"best GRAPE (r={r_values[i_best[0]]:g}, viol={viol:.1e})"
    axes[0].plot(tg, 2.0 * g["u_omega"][i_best] / TAU, alpha=0.7, label=grape_label)
    axes[1].plot(tg, -g["u_delta"][i_best] / TAU, alpha=0.7, label=grape_label)
    axes[0].set_ylabel("Omega/2pi (MHz)")
    axes[1].set_ylabel("Delta/2pi (MHz)")
    axes[1].set_xlabel("t (us)")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Optimized pulses (Fig. 3c style)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plots / f"fig3c_pulses.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {plots}/fig3b_violin.png|pdf and fig3c_pulses.png|pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("model-check").set_defaults(func=cmd_model_check)
    p = sub.add_parser("direct")
    p.add_argument("--tag", default="all")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--maxiter", type=int, default=4000)
    p.add_argument("--warm-start", action="store_true")
    p.set_defaults(func=cmd_direct)
    p = sub.add_parser("grape")
    p.add_argument("--duration-us", type=float, default=1.2)
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--maxiter", type=int, default=1500)
    p.add_argument("--r-values", default="0,1e-8,1e-7,1e-6")
    p.set_defaults(func=cmd_grape)
    p = sub.add_parser("validate")
    p.add_argument("--tag", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--grape", action="store_true")
    p.add_argument("--duration-us", type=float, default=1.2)
    p.set_defaults(func=cmd_validate)
    p = sub.add_parser("plot")
    p.add_argument("--duration-us", type=float, default=1.2)
    p.set_defaults(func=cmd_plot)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
