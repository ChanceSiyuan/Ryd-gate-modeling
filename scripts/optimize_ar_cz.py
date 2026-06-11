"""Re-optimize AR CZ pulse parameters on the current kernel.

Optimization workflow script (repo convention: search loops stay in
``scripts/``, only reusable functions live in ``src/``). Reproduces the
removed legacy ``CZGateSimulator._optimization_AR`` loop: scipy Nelder-Mead
over ``analysis.gate_metrics.average_gate_infidelity`` within
``ARProtocol().get_optimization_bounds()``, starting from the legacy X_AR
point (which is *not* a high-fidelity optimum under the current protocol
conventions — see tests/gates/test_cz_benchmark_pins.py).

Usage:
    OMP_NUM_THREADS=1 uv run python scripts/optimize_ar_cz.py our [maxiter]
    OMP_NUM_THREADS=1 uv run python scripts/optimize_ar_cz.py lukin [maxiter]

The best-so-far point is checkpointed to ``data/ar_opt_<param_set>.json``
after every improvement, so partial results survive interruption.
"""

import json
import sys
import time
from pathlib import Path

from scipy.optimize import minimize

from ryd_gate import Register, RydbergSystem
from ryd_gate.analysis.gate_metrics import average_gate_infidelity
from ryd_gate.protocols.gate_cz_ar import ARProtocol

# Legacy starting point: [omega/Omega_eff, A1, phi1, A2, phi2, delta/Omega_eff, T/T_scale, theta]
X_AR_LEGACY = [0.85973359, 0.39146974, 0.99181418, 0.1924498, -1.17123748, -0.00826712, 1.67429728, 0.28527346]


def main() -> None:
    param_set = sys.argv[1] if len(sys.argv) > 1 else "our"
    maxiter = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    if param_set not in {"our", "lukin"}:
        raise SystemExit(f"param_set must be 'our' or 'lukin', got {param_set!r}")

    system = RydbergSystem.from_lattice(
        Register.chain(2, spacing_um=3.0), "rb87_7", param_set=param_set
    )
    protocol = ARProtocol()
    out_path = Path("data") / f"ar_opt_{param_set}.json"
    out_path.parent.mkdir(exist_ok=True)

    state = {
        "param_set": param_set,
        "x_initial": list(X_AR_LEGACY),
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
        X_AR_LEGACY,
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
