"""Re-optimize AR CZ pulse parameters on the current kernel.

Optimization workflow script (repo convention: search loops stay in
``scripts/``, only reusable functions live in ``src/``). Each start is run
through the shared ``ryd_gate.gates.optimize_cz_parameters`` (theta-projection
warm start + Nelder-Mead polish); this script only owns the multi-start loop and
checkpointing.

The AR landscape has a non-entangling local minimum on ``our`` (~0.45, wrong
conditional phase; see ``scripts/diagnose_ar_target.py``), so escaping it needs
multiple starts -- ``--restarts`` (curated + random seeds). The theta projection
also makes each random restart far more effective (it snaps onto the
optimal-theta ridge before polishing).

Usage:
    # single start from the legacy seed (or --resume from checkpoint best)
    OMP_NUM_THREADS=1 uv run python scripts/optimize_ar_cz.py our [maxiter] [--resume]
    # multi-start: legacy seed + lukin optimum (cross-system warm start) + N random
    OMP_NUM_THREADS=1 uv run python scripts/optimize_ar_cz.py our --restarts 16 [--seed 0]

The global best across all starts is checkpointed to
``data/ar_opt_<param_set>.json`` after every improvement, so partial results
survive interruption. ``--resume`` adds the previous checkpoint best as an
extra start.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from ryd_gate import Register, RydbergSystem
from ryd_gate.gates import optimize_cz_parameters
from ryd_gate.protocols.gate_cz import ARProtocol

# Legacy seed: [omega/Omega_eff, A1, phi1, A2, phi2, delta/Omega_eff, T/T_scale, theta]
X_AR_LEGACY = [0.85973359, 0.39146974, 0.99181418, 0.1924498, -1.17123748, -0.00826712, 1.67429728, 0.28527346]


def _sample_bounds(protocol):
    """Finite sampling ranges for random restarts.

    The protocol bounds leave T/T_scale (index 6) unbounded and omega/Omega_eff
    (index 0) at (-10, 10); narrow both to physical positive ranges so random
    seeds land in plausible gate configurations.
    """
    bounds = [list(b) for b in protocol.get_optimization_bounds()]
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

    param_set = pos[0] if pos else "our"
    maxiter = int(pos[1]) if len(pos) > 1 else 400
    resume = "--resume" in flags
    restarts = int(opt_value("--restarts", 0))
    rng_seed = int(opt_value("--seed", 0))
    no_curated = "--no-curated" in flags  # random restarts only (parallel workers)
    tag = str(opt_value("--tag", ""))      # per-worker checkpoint suffix
    if param_set not in {"our", "lukin"}:
        raise SystemExit(f"param_set must be 'our' or 'lukin', got {param_set!r}")

    system = (
        RydbergSystem.set_atom_level("rb87_7", param_set=param_set)
        .set_atom_geom(Register.chain(2, spacing_um=3.0))
        .build()
    )
    protocol = ARProtocol()
    suffix = f"_{tag}" if tag else ""
    out_path = Path("data") / f"ar_opt_{param_set}{suffix}.json"
    out_path.parent.mkdir(exist_ok=True)

    # ---- assemble the start list ----
    seeds: list[tuple[str, list[float]]] = []
    if not no_curated:
        seeds.append(("legacy", list(X_AR_LEGACY)))
        lukin_path = Path("data") / "ar_opt_lukin.json"
        if lukin_path.exists():
            lukin_best = json.loads(lukin_path.read_text()).get("best_x")
            if lukin_best and param_set != "lukin":
                seeds.append(("lukin-opt", [float(v) for v in lukin_best]))
        if resume and out_path.exists():
            prev = json.loads(out_path.read_text()).get("best_x")
            if prev:
                seeds.insert(0, ("resume", [float(v) for v in prev]))
    rng = np.random.default_rng(rng_seed)
    sb = _sample_bounds(protocol)
    for i in range(restarts):
        seeds.append((f"rand{i}", [float(rng.uniform(lo, hi)) for lo, hi in sb]))
    if not seeds:
        raise SystemExit("no starts: use --restarts N (and/or drop --no-curated).")

    state = {
        "param_set": param_set,
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
            result = optimize_cz_parameters(system, protocol, x0, maxiter=maxiter)
        except Exception as exc:  # a bad random seed can throw inside the solver
            print(f"[{param_set}] start {label}: failed ({exc})", flush=True)
            continue
        if result.infidelity < state["best_infidelity"]:
            state["best_infidelity"] = result.infidelity
            state["best_x"] = result.x
            state["best_seed"] = label
            state["elapsed_s"] = round(time.time() - t_start, 1)
            out_path.write_text(json.dumps(state, indent=1))
        print(f"[{param_set}] start {label}: theta {result.theta_infidelity:.6e} -> "
              f"polish {result.infidelity:.6e} (global best {state['best_infidelity']:.6e})",
              flush=True)

    state["done"] = True
    state["elapsed_s"] = round(time.time() - t_start, 1)
    out_path.write_text(json.dumps(state, indent=1))
    print(f"[{param_set}] FINAL best infidelity {state['best_infidelity']:.6e} (seed {state['best_seed']})")
    print(f"[{param_set}] FINAL best_x = {state['best_x']}")


if __name__ == "__main__":
    main()
