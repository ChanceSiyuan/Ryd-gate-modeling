"""Maintained 10x10 finite-PEPS validation run (the single large DGX case).

Unconditional live YASTN PEPS real-time evolution of a 100-atom Rydberg array —
no saved-result fallback, no broad exception handler, no "under rewrite" guard.
The physical case and PEPS controls are the fixed notebook-04 case. Numerical
values are named constants, not a generic kwargs parser.

DGX acceptance::

    .venv/bin/python scripts/run_peps_10x10.py \\
        --device cuda --output-dir /tmp/ryd_gate_peps_validation/10x10

Success: exit zero, exactly seven finite real mean Rydberg-occupation records,
and JSON-compatible PEPS evidence whose parameters equal the controls below.
Arrays/evidence are written only beneath ``--output-dir`` (never repo results/).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ryd_gate import RydbergSystem, level_structure
from ryd_gate.lattice import Register
from ryd_gate.protocols.sweep import SweepProtocol
from ryd_gate.simulate import simulate

# ── fixed physical case (notebook-04) ────────────────────────────────────────
TWO_PI = 2.0 * np.pi
SIDE = 10
SPACING_UM = 5.0
RYD_LEVEL = 70
OMEGA_RAD_S = TWO_PI * 380e6          # constant Rabi frequency
DETUNING_AMP_RAD_S = TWO_PI * 10e6    # smooth round trip -10 -> +10 -> -10 MHz
T_GATE_S = 0.15e-6
N_EVAL = 7                            # equally spaced, both endpoints included

# ── fixed PEPS controls (the ten mandatory real-time keys) ───────────────────
_CONTROLS = {
    "time_step_s": 0.15e-6 / 250,
    "bond_dimension": 8,
    "svd_tolerance": 1e-8,
    "ntu_max_iterations": 20,
    "ntu_iteration_tolerance": 1e-10,
    "measurement_method": "belief_propagation",
    "environment_bond_dimension": 32,
    "environment_tolerance": 1e-8,
    "environment_max_iterations": 50,
    "device": "cuda",
}


def _detuning_rad_s(t: float) -> float:
    # -cos runs -amp -> +amp -> -amp smoothly over [0, t_gate].
    return -DETUNING_AMP_RAD_S * np.cos(TWO_PI * t / T_GATE_S)


def _build_system() -> RydbergSystem:
    return RydbergSystem(
        level_structure=level_structure("1r", ryd_level=RYD_LEVEL),
        register=Register.rectangle(SIDE, SIDE, spacing_um=SPACING_UM),
        protocol=SweepProtocol(
            t_gate_s=T_GATE_S,
            omega_half_rad_s=lambda t: OMEGA_RAD_S / 2.0,
            detuning_rad_s=_detuning_rad_s,
        ),
        interaction_cutoff_um=SPACING_UM,  # keep only Cartesian nearest neighbours
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Maintained 10x10 finite-PEPS validation run.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir", required=True,
                        help="Directory (outside the repo) for arrays + evidence JSON.")
    args = parser.parse_args()

    system = _build_system()
    n = system.N
    n_r_mean = sum(system.observables.n("r", i) for i in range(n)) * (1.0 / n)
    t_eval = np.linspace(0.0, T_GATE_S, N_EVAL)
    backend_options = {**_CONTROLS, "device": args.device}

    result = simulate(
        system, backend="peps", t_eval=t_eval,
        observables={"n_r_mean": n_r_mean}, backend_options=backend_options,
    )

    times = np.asarray(result.times, dtype=float)
    occupation = np.asarray(result.expectation("n_r_mean"), dtype=float)
    if occupation.shape != (N_EVAL,) or not np.all(np.isfinite(occupation)):
        raise SystemExit(f"expected {N_EVAL} finite occupation records; got {occupation!r}.")

    evidence = result.peps_evidence.to_dict()  # JSON-compatible provenance
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "times.npy"), times)
    np.save(os.path.join(args.output_dir, "n_r_mean.npy"), occupation)
    with open(os.path.join(args.output_dir, "peps_evidence.json"), "w") as fh:
        json.dump(evidence, fh, indent=2)

    print(f"10x10 finite-PEPS run on device={args.device}: {n} atoms, {N_EVAL} measurement times")
    for t, occ in zip(times, occupation):
        print(f"  t={t * 1e6:.4f} us  <n_r>_mean={occ:.6f}")
    print(f"max NTU truncation error: {evidence['max_ntu_truncation_error']:.3e}")
    print(f"wrote arrays + evidence to {args.output_dir}")


if __name__ == "__main__":
    main()
