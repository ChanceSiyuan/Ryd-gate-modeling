"""Re-optimize the CZ optima for the explicit |0> model.

The perturbative-era optima are genuine entangling gates whose single-qubit Z
correction (theta) goes stale when |0> is modeled explicitly (|0>/|1> pick up
different light shifts). ``ryd_gate.gates.optimize_cz_parameters`` handles this
with a theta-projection warm start (re-fit theta, then polish all parameters).
The dark branch is clean (|0> leakage ~5e-7); the bright branch is NOT included
because its intermediate manifold sits 2.3 GHz from |0> (near-resonant, a hard
coherent-leakage floor ~9e-3 that optimization cannot fix).

Prints the theta-projection recovery (seed -> theta-only -> polish) and an
n_steps convergence check. Checkpoints best_x to data/<name>_explicit.json.
"""
import os
import json
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
Path("data").mkdir(exist_ok=True)

from ryd_gate import Register, RydbergSystem
from ryd_gate.analysis.gate_metrics import average_gate_infidelity
from ryd_gate.gates import optimize_cz_parameters
from ryd_gate.protocols.gate_cz import ARProtocol, TOProtocol

# Seeds: the perturbative-era optima (genuine gates with stale theta).
X_TO_DARK = [-0.6989301339711643, 1.0296229082590798, 0.3759232324550267,
             1.5710180991068543, 1.4454279613697887, 1.3406239758422793]
X_AR_OUR = [1.3417376355208614, -0.47514206631153455, 0.9374165061782775,
            -0.24539458137685877, 2.755618354280231, 1.0063163509211008,
            1.9975377136905927, -1.6351076243247586]
X_AR_LUKIN = [1.152444475742163, 0.5298612474274744, 1.0463537649409957,
              0.5247328111843467, -1.3495589677508941, -0.008058653524462491,
              1.4486084769604992, -0.1399323673945825]

# (name, protocol_cls, set_atom_level kwargs, seed)
GATES = [
    ("to_dark", TOProtocol, dict(param_set="our", detuning_sign=1), X_TO_DARK),
    ("ar_our", ARProtocol, dict(param_set="our"), X_AR_OUR),
    ("ar_lukin", ARProtocol, dict(param_set="lukin"), X_AR_LUKIN),
]

for name, Proto, kw, seed in GATES:
    sys = (RydbergSystem.set_atom_level("rb87_7", **kw)
           .set_atom_geom(Register.chain(2, spacing_um=3.0)).build())
    p = Proto(); p.n_steps = 200
    print(f"\n=== {name} ===", flush=True)
    result = optimize_cz_parameters(sys, p, seed)
    print(f"seed={result.seed_infidelity:.6e}  theta-only={result.theta_infidelity:.6e}  "
          f"polish={result.infidelity:.6e}  ({result.n_eval} evals)", flush=True)
    xf = result.x
    json.dump({"name": name, "f": result.infidelity, "x": xf},
              open(f"data/{name}_explicit.json", "w"), indent=1)
    print(f"best_x = {xf}", flush=True)
    for ns in (200, 1200, 4800):
        p.n_steps = ns
        print(f"  verify n_steps={ns}: {average_gate_infidelity(sys, p, xf):.6e}", flush=True)
