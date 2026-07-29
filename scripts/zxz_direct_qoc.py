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
import sys
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("model-check").set_defaults(func=cmd_model_check)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
