"""Re-optimize AR CZ pulse parameters on the current kernel.

Optimization workflow script (repo convention: search loops stay in
``scripts/``, only reusable functions live in ``src/``). Reproduces the
removed legacy ``CZGateSimulator._optimization_AR`` loop: scipy Nelder-Mead
over ``analysis.gate_metrics.average_gate_infidelity`` within
``ARProtocol().get_optimization_bounds()``, starting from the legacy X_AR
point (which is *not* a high-fidelity optimum under the current protocol
conventions — see tests/gates/test_cz_benchmark_pins.py).

Usage:
    OMP_NUM_THREADS=1 uv run python scripts/optimize_ar_cz.py our [maxiter] [--resume]
    OMP_NUM_THREADS=1 uv run python scripts/optimize_ar_cz.py lukin [maxiter] [--resume]

The best-so-far point is checkpointed to ``data/ar_opt_<param_set>.json``
after every improvement, so partial results survive interruption.
``--resume`` restarts Nelder-Mead from the checkpoint's best point (a fresh
simplex around the old optimum often keeps descending after a maxiter stop).
"""

import json
import sys
import time
from pathlib import Path

from scipy.optimize import minimize

from ryd_gate import Register, RydbergSystem
from ryd_gate.analysis.gate_metrics import average_gate_infidelity
from ryd_gate.protocols.gate_cz import ARProtocol

# Legacy starting point: [omega/Omega_eff, A1, phi1, A2, phi2, delta/Omega_eff, T/T_scale, theta]
X_AR_LEGACY = [0.85973359, 0.39146974, 0.99181418, 0.1924498, -1.17123748, -0.00826712, 1.67429728, 0.28527346]


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--resume"]
    resume = "--resume" in sys.argv[1:]
    param_set = args[0] if args else "our"
    maxiter = int(args[1]) if len(args) > 1 else 400
    if param_set not in {"our", "lukin"}:
        raise SystemExit(f"param_set must be 'our' or 'lukin', got {param_set!r}")

    system = RydbergSystem.from_lattice(
        Register.chain(2, spacing_um=3.0), "rb87_7", param_set=param_set
    )
    protocol = ARProtocol()
    out_path = Path("data") / f"ar_opt_{param_set}.json"
    out_path.parent.mkdir(exist_ok=True)

    x_initial = list(X_AR_LEGACY)
    if resume:
        previous = json.loads(out_path.read_text())
        if previous.get("best_x"):
            x_initial = [float(v) for v in previous["best_x"]]
            print(f"[{param_set}] resuming from checkpoint best "
                  f"{previous['best_infidelity']:.6e}", flush=True)

    state = {
        "param_set": param_set,
        "x_initial": x_initial,
        "n_eval": 0,
        "best_infidelity": float("inf"),
        "best_x": None,
        "done": False,
    }
    t_start = time.time()

    def objective(x):
        val = float(average_gate_infidelity(system, protocol, [float(v) for v in x]))
        state["n_eval"] += 1
        if val < state["best_infidelity"]:
            state["best_infidelity"] = val
            state["best_x"] = [float(v) for v in x]
            state["elapsed_s"] = round(time.time() - t_start, 1)
            out_path.write_text(json.dumps(state, indent=1))
            print(f"[{param_set}] eval {state['n_eval']}: infidelity {val:.6e}", flush=True)
        return val

    result = minimize(
        objective,
        x_initial,
        method="Nelder-Mead",
        bounds=protocol.get_optimization_bounds(),
        options={"disp": True, "fatol": 1e-9, "maxiter": maxiter},
    )

    state["done"] = True
    state["final_x"] = [float(v) for v in result.x]
    state["final_infidelity"] = float(result.fun)
    state["converged"] = bool(result.success)
    state["elapsed_s"] = round(time.time() - t_start, 1)
    out_path.write_text(json.dumps(state, indent=1))
    print(f"[{param_set}] FINAL infidelity {result.fun:.6e}")
    print(f"[{param_set}] FINAL x = {[float(v) for v in result.x]}")


if __name__ == "__main__":
    main()
