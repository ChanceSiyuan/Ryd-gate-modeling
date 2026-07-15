"""Re-optimize AR CZ pulse parameters on the current kernel.

Optimization workflow script (repo convention: search loops and objectives
live in ``scripts/``, only system/protocol/evolution live in ``src/``). Each
start runs a theta-projection warm start + Nelder-Mead polish on the 3-state
Nielsen infidelity, both written out below; this script owns the objective,
the multi-start loop, and checkpointing.

The AR landscape has a non-entangling local minimum on ``mp`` (~0.45, wrong
conditional phase; see ``scripts/diagnose_ar_target.py``), so escaping it needs
multiple starts -- ``--restarts`` (curated + random seeds). The theta projection
also makes each random restart far more effective (it snaps onto the
optimal-theta ridge before polishing).

The manifold is selected by the level-structure tag suffix: ``mp`` (σ⁻/σ⁺, was
``our``) or ``pm`` (σ⁺/σ⁻, was ``lukin``).  ``pm`` is not the protocol default,
so its canonical Rabis are passed to ``ARProtocol`` explicitly.

Usage:
    # single start from the legacy seed (or --resume from checkpoint best)
    OMP_NUM_THREADS=1 uv run python scripts/optimize_ar_cz.py mp [maxiter] [--resume]
    # multi-start: legacy seed + pm optimum (cross-system warm start) + N random
    OMP_NUM_THREADS=1 uv run python scripts/optimize_ar_cz.py mp --restarts 16 [--seed 0]

The global best across all starts is checkpointed to
``results/cz_gate/ar_optimization/ar_opt_<manifold>.json`` after every improvement, so partial results
survive interruption. ``--resume`` adds the previous checkpoint best as an
extra start.

Runtime: each Nielsen evaluation (three 7-level two-atom states on the
adaptive exact_ode solver) takes ~3 min single-threaded, so a Nelder-Mead
polish is a long (multi-day) job per start.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.protocols import ARProtocol

# Two-atom computational-basis labels: level "0" = |0>, level "1" = |1>.
_LABELS = {"00": ["0", "0"], "01": ["0", "1"], "11": ["1", "1"]}


def _ar_pulse(fixed, x):
    """Concrete AR pulse from the non-theta entries of the optimizer x-vector.

    ``x`` layout: [modulation/Omega_eff, A1, phi1, A2, phi2, freq_offset/Omega_eff,
    T/T_scale, theta]; theta = x[7] is a scoring parameter, not pulse shape.
    ``fixed`` carries the manifold's Rabis, intermediate detuning and rise time.
    """
    return ARProtocol(
        modulation_frequency_ratio=x[0], phase_amplitude_1_rad=x[1],
        phase_offset_1_rad=x[2], phase_amplitude_2_rad=x[3],
        phase_offset_2_rad=x[4], frequency_offset_ratio=x[5],
        duration_ratio=x[6], **fixed,
    )


def average_gate_infidelity(system, fixed, x):
    """3-state Nielsen CZ infidelity, formulas written out.

    Build the concrete AR pulse from the non-theta entries of x (theta = x[7]
    is a local scoring parameter, not pulse shape), evolve |00>, |01>, |11>,
    apply the ideal single-qubit Rz corrections, and score with the Nielsen
    average-gate-fidelity formula (d = 4, with |10> folded into |01> by
    exchange symmetry).
    """
    theta = float(x[7])  # single-qubit Rz correction (AR x-vector, index 7)
    bound = system.with_protocol(_ar_pulse(fixed, x))
    corrections = {"00": 1.0, "01": np.exp(-1j * theta),
                   "11": np.exp(-2j * theta - 1j * np.pi)}
    a = {k: corrections[k] * simulate(bound, labels).amplitude(labels)
         for k, labels in _LABELS.items()}
    avg_f = (1 / 20) * (abs(a["00"] + 2 * a["01"] + a["11"]) ** 2
                        + abs(a["00"]) ** 2 + 2 * abs(a["01"]) ** 2 + abs(a["11"]) ** 2)
    return float(1 - avg_f)


def optimize_start(system, fixed, x0, maxiter):
    """Theta-projection warm start + Nelder-Mead polish.

    Theta (the single-qubit Z correction) is hyper-sensitive to the gate time
    under the explicit |0> model — |0> sits at the 6.835 GHz clock splitting,
    so the optimum winds rapidly with T and a wrong theta dominates the
    objective (~1e-2), masking the ~1e-6 leakage/conditional-phase gradients.
    A cheap bounded 1-D re-fit of theta snaps onto the optimal-theta ridge
    (typically 1e-2 -> 1e-6), then Nelder-Mead polishes all parameters.

    Returns ``(x, infidelity, theta_infidelity)``.
    """
    f = lambda xv: average_gate_infidelity(system, fixed, list(xv))
    x = [float(v) for v in x0]
    ti = 7  # theta position in the AR x-vector
    res_t = optimize.minimize_scalar(
        lambda t: f([*x[:ti], float(t), *x[ti + 1:]]),
        bounds=(x[ti] - np.pi, x[ti] + np.pi), method="bounded",
        options={"xatol": 1e-10},
    )
    x = [*x[:ti], float(res_t.x), *x[ti + 1:]]
    res = optimize.minimize(
        f, x, method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-13, "maxiter": maxiter},
    )
    return res.x.tolist(), float(res.fun), float(res_t.fun)

# Legacy seed: [omega/Omega_eff, A1, phi1, A2, phi2, delta/Omega_eff, T/T_scale, theta]
X_AR_LEGACY = [0.85973359, 0.39146974, 0.99181418, 0.1924498, -1.17123748, -0.00826712, 1.67429728, 0.28527346]

# Canonical single-photon Rabis + intermediate detuning + Blackman rise per
# manifold (rad/s, rad/s, s). Formerly the protocol/preset defaults; the new
# TO/AR constructors take them explicitly (P19/P20), so the caller supplies the
# manifold's fixed physics: mp is σ⁻/σ⁺ (491/185 MHz, Delta=+9.1 GHz), pm is
# σ⁺/σ⁻ (237/303 MHz, Delta=+7.8 GHz); both use the 20 ns dark-branch rise.
FIXED_MP = dict(
    intermediate_detuning_rad_s=2 * np.pi * 9.1e9,
    omega_420_max_rad_s=2 * np.pi * 491e6,
    omega_1013_max_rad_s=2 * np.pi * 185e6,
    rise_time_s=20e-9,
)
FIXED_PM = dict(
    intermediate_detuning_rad_s=2 * np.pi * 7.8e9,
    omega_420_max_rad_s=2 * np.pi * 237e6,
    omega_1013_max_rad_s=2 * np.pi * 303e6,
    rise_time_s=20e-9,
)


# AR x-vector bounds: [omega/Omega_eff, A1, phi1, A2, phi2, delta/Omega_eff, T/T_scale, theta].
AR_BOUNDS = ((-10, 10), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
             (-np.pi, np.pi), (-2, 2), (-np.inf, np.inf), (-np.pi, np.pi))


def _sample_bounds():
    """Finite sampling ranges for random restarts.

    AR_BOUNDS leaves T/T_scale (index 6) unbounded and omega/Omega_eff
    (index 0) at (-10, 10); narrow both to physical positive ranges so random
    seeds land in plausible gate configurations.
    """
    bounds = [list(b) for b in AR_BOUNDS]
    bounds[0] = [0.2, 3.0]   # omega/Omega_eff
    bounds[6] = [0.3, 3.0]   # T/T_scale (unbounded -> sane finite gate time)
    return bounds


def main() -> None:
    raw = sys.argv[1:]
    flags = {a for a in raw if a.startswith("--")}
    pos = [a for a in raw if not a.startswith("--")]

    def opt_value(name, default):
        if name in raw:
            return raw[raw.index(name) + 1]
        return default

    manifold = pos[0] if pos else "mp"
    maxiter = int(pos[1]) if len(pos) > 1 else 400
    resume = "--resume" in flags
    restarts = int(opt_value("--restarts", 0))
    rng_seed = int(opt_value("--seed", 0))
    no_curated = "--no-curated" in flags  # random restarts only (parallel workers)
    tag = str(opt_value("--tag", ""))      # per-worker checkpoint suffix
    if manifold not in {"mp", "pm"}:
        raise SystemExit(f"manifold must be 'mp' or 'pm', got {manifold!r}")

    # mp/pm each carry their own canonical Rabis + intermediate detuning + rise.
    fixed = FIXED_PM if manifold == "pm" else FIXED_MP
    system = RydbergSystem(
        level_structure=level_structure(f"rb87_7_{manifold}"),
        register=Register.chain(2, spacing_um=3.0),
        protocol=_ar_pulse(fixed, X_AR_LEGACY),
    )
    suffix = f"_{tag}" if tag else ""
    results_dir = Path("results") / "cz_gate" / "ar_optimization"
    out_path = results_dir / f"ar_opt_{manifold}{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- assemble the start list ----
    seeds: list[tuple[str, list[float]]] = []
    if not no_curated:
        seeds.append(("legacy", list(X_AR_LEGACY)))
        pm_path = results_dir / "ar_opt_pm.json"
        if pm_path.exists():
            pm_best = json.loads(pm_path.read_text()).get("best_x")
            if pm_best and manifold != "pm":
                seeds.append(("pm-opt", [float(v) for v in pm_best]))
        if resume and out_path.exists():
            prev = json.loads(out_path.read_text()).get("best_x")
            if prev:
                seeds.insert(0, ("resume", [float(v) for v in prev]))
    rng = np.random.default_rng(rng_seed)
    sb = _sample_bounds()
    for i in range(restarts):
        seeds.append((f"rand{i}", [float(rng.uniform(lo, hi)) for lo, hi in sb]))
    if not seeds:
        raise SystemExit("no starts: use --restarts N (and/or drop --no-curated).")

    state = {
        "manifold": manifold,
        "x_initial": seeds[0][1],
        "n_starts": len(seeds),
        "best_infidelity": float("inf"),
        "best_x": None,
        "best_seed": None,
        "done": False,
    }
    t_start = time.time()

    # ---- multi-start; each start uses the theta-projection optimizer ----
    for label, x0 in seeds:
        try:
            best_x, infidelity, theta_infidelity = optimize_start(
                system, fixed, x0, maxiter=maxiter
            )
        except Exception as exc:  # a bad random seed can throw inside the solver
            print(f"[{manifold}] start {label}: failed ({exc})", flush=True)
            continue
        if infidelity < state["best_infidelity"]:
            state["best_infidelity"] = infidelity
            state["best_x"] = best_x
            state["best_seed"] = label
            state["elapsed_s"] = round(time.time() - t_start, 1)
            out_path.write_text(json.dumps(state, indent=1))
        print(f"[{manifold}] start {label}: theta {theta_infidelity:.6e} -> "
              f"polish {infidelity:.6e} (global best {state['best_infidelity']:.6e})",
              flush=True)

    state["done"] = True
    state["elapsed_s"] = round(time.time() - t_start, 1)
    out_path.write_text(json.dumps(state, indent=1))
    print(f"[{manifold}] FINAL best infidelity {state['best_infidelity']:.6e} (seed {state['best_seed']})")
    print(f"[{manifold}] FINAL best_x = {state['best_x']}")


if __name__ == "__main__":
    main()
