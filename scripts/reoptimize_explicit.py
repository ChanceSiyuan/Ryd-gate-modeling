"""Re-optimize the CZ optima for the explicit |0> model.

The perturbative-era optima are genuine entangling gates whose single-qubit Z
correction (theta) goes stale when |0> is modeled explicitly (|0>/|1> pick up
different light shifts). The optimizer below (written out in this script)
handles this with a theta-projection warm start: theta is hyper-sensitive to
the gate time (|0> sits at the 6.835 GHz clock splitting, so the optimum
winds rapidly with T) and a wrong theta dominates the Nielsen objective,
trapping a joint Nelder-Mead search — so re-fit theta alone first with a
cheap bounded 1-D minimization, then polish all parameters.
The dark branch is clean (|0> leakage ~5e-7); the bright branch is NOT included
because its intermediate manifold sits 2.3 GHz from |0> (near-resonant, a hard
coherent-leakage floor ~9e-3 that optimization cannot fix).

Prints the theta-projection recovery (seed -> theta-only -> polish) and a
solver-tolerance verification (default vs tightened rtol/atol on the adaptive
ODE solver). Checkpoints best_x to
results/cz_gate/ar_optimization/<name>_explicit.json.

Runtime: each Nielsen evaluation (three 7-level two-atom states on the
adaptive exact_ode solver) takes ~3 min single-threaded, so a full polish run
is a long (multi-day) job.
"""
import os
import json
from math import pi
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
AR_RESULTS_DIR = Path("results") / "cz_gate" / "ar_optimization"
AR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
from scipy import optimize

from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.backends.exact import simulate
from ryd_gate.protocols.gate_cz import ARProtocol, TOProtocol

# 7-level single-atom basis vectors: index 0 = |0>, index 1 = |1>.
_S0 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=complex)
_S1 = np.array([0, 1, 0, 0, 0, 0, 0], dtype=complex)
_BASIS = {"00": np.kron(_S0, _S0), "01": np.kron(_S0, _S1), "11": np.kron(_S1, _S1)}


def build_pulse(Proto, x, omega_kw):
    """Concrete TO/AR pulse from the non-theta entries of the optimizer x-vector.

    x layouts: TO [A, w, phi0, d, theta, T] (theta = x[4]);
    AR [w, A1, p1, A2, p2, d, T, theta] (theta = x[7]).
    """
    x = [float(v) for v in x]
    if Proto is TOProtocol:
        return TOProtocol(phase_amplitude=x[0], frequency_ratio=x[1],
                          phase_offset=x[2], detuning_ratio=x[3],
                          duration_ratio=x[5], **omega_kw)
    return ARProtocol(frequency_ratio=x[0], phase_amplitude_1=x[1],
                      phase_offset_1=x[2], phase_amplitude_2=x[3],
                      phase_offset_2=x[4], detuning_ratio=x[5],
                      duration_ratio=x[6], **omega_kw)


def average_gate_infidelity(system, Proto, omega_kw, x, backend_options=None):
    """3-state Nielsen CZ infidelity, formulas written out.

    Build the concrete pulse from x (theta is a local scoring parameter, not
    pulse shape), evolve |00>, |01>, |11>, apply the ideal single-qubit Rz
    corrections, and score with the Nielsen average-gate-fidelity formula
    (d = 4, with |10> folded into |01> by exchange symmetry).
    """
    pulse = build_pulse(Proto, x, omega_kw)
    theta = float(x[4] if Proto is TOProtocol else x[7])
    bound = system.with_protocol(pulse)
    corrections = {"00": 1.0, "01": np.exp(-1j * theta),
                   "11": np.exp(-2j * theta - 1j * np.pi)}
    a = {k: corrections[k] * complex(np.vdot(
             ini, simulate(bound, ini, backend_options=backend_options).final_state))
         for k, ini in _BASIS.items()}
    avg_f = (1 / 20) * (abs(a["00"] + 2 * a["01"] + a["11"]) ** 2
                        + abs(a["00"]) ** 2 + 2 * abs(a["01"]) ** 2 + abs(a["11"]) ** 2)
    return float(1 - avg_f)

# Seeds: the perturbative-era optima (genuine gates with stale theta).
X_TO_DARK = [-0.6989301339711643, 1.0296229082590798, 0.3759232324550267,
             1.5710180991068543, 1.4454279613697887, 1.3406239758422793]
X_AR_OUR = [1.3417376355208614, -0.47514206631153455, 0.9374165061782775,
            -0.24539458137685877, 2.755618354280231, 1.0063163509211008,
            1.9975377136905927, -1.6351076243247586]
X_AR_LUKIN = [1.152444475742163, 0.5298612474274744, 1.0463537649409957,
              0.5247328111843467, -1.3495589677508941, -0.008058653524462491,
              1.4486084769604992, -0.1399323673945825]

# Canonical σ⁺/σ⁻ (pm) single-photon Rabis (rad/s); pm is not the protocol default.
OMEGA_PM = dict(omega_420_max=2 * pi * 237e6, omega_1013_max=2 * pi * 303e6)

# (name, level-structure tag, protocol_cls, protocol omega kwargs, level_structure kwargs, seed)
GATES = [
    ("to_dark", "rb87_7_mp", TOProtocol, {}, dict(detuning_sign=1), X_TO_DARK),
    ("ar_our", "rb87_7_mp", ARProtocol, {}, {}, X_AR_OUR),
    ("ar_lukin", "rb87_7_pm", ARProtocol, OMEGA_PM, {}, X_AR_LUKIN),
]

for name, atom_tag, Proto, omega_kw, kw, seed in GATES:
    sys = RydbergSystem(level_structure=level_structure(atom_tag, **kw),
                        register=Register.chain(2, spacing_um=3.0))
    print(f"\n=== {name} ===", flush=True)

    # Theta-projection warm start + Nelder-Mead polish, written out.
    f = lambda xv: average_gate_infidelity(sys, Proto, omega_kw, list(xv))
    x = [float(v) for v in seed]
    seed_infidelity = f(x)
    ti = 4 if Proto is TOProtocol else 7  # theta position in the x-vector
    res_t = optimize.minimize_scalar(
        lambda t: f([*x[:ti], float(t), *x[ti + 1:]]),
        bounds=(x[ti] - pi, x[ti] + pi), method="bounded",
        options={"xatol": 1e-10},
    )
    x = [*x[:ti], float(res_t.x), *x[ti + 1:]]
    theta_infidelity = float(res_t.fun)
    res = optimize.minimize(
        f, x, method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-13, "maxiter": 4000},
    )
    xf, final_infidelity = res.x.tolist(), float(res.fun)
    print(f"seed={seed_infidelity:.6e}  theta-only={theta_infidelity:.6e}  "
          f"polish={final_infidelity:.6e}  ({int(res.nfev)} evals)", flush=True)
    with (AR_RESULTS_DIR / f"{name}_explicit.json").open("w") as f_out:
        json.dump({"name": name, "f": final_infidelity, "x": xf}, f_out, indent=1)
    print(f"best_x = {xf}", flush=True)
    # Solver verification: the adaptive ODE solver replaced the fixed-n_steps
    # expm backends, so the old n_steps convergence loop is meaningless. Re-score
    # once at tighter tolerances and print both against the default.
    inf_default = average_gate_infidelity(sys, Proto, omega_kw, xf)
    inf_tight = average_gate_infidelity(
        sys, Proto, omega_kw, xf, backend_options={"rtol": 1e-9, "atol": 1e-13})
    print(f"  verify default tol: {inf_default:.6e}  "
          f"rtol=1e-9/atol=1e-13: {inf_tight:.6e}", flush=True)
