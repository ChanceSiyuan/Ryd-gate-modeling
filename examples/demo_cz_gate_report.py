#!/usr/bin/env python3
"""CZ gate report demo: the flagship benchmark point in one call.

Reproduces the TO dark-detuning benchmark (infidelity ~7.8e-7). Runtime is a
few exact 49-dim solves (~15 s single-threaded).

Usage:
    OMP_NUM_THREADS=1 uv run python examples/demo_cz_gate_report.py
"""

import json

from ryd_gate import Register, RydbergSystem
from ryd_gate.gates import TOProtocol, cz_gate_report

X_TO_DARK = [-0.6989301339711643, 1.0296229082590798, 0.3759232324550267,
             1.5710180991068543, 1.4454279613697887, 1.3406239758422793]


def main() -> None:
    system = RydbergSystem.from_lattice(
        Register.chain(2, spacing_um=3.0), "rb87_7", param_set="our",
        blackmanflag=True, detuning_sign=1,
    )
    report = cz_gate_report(system, TOProtocol(), X_TO_DARK, include_error_budget=False)

    print(f"protocol:        {report.protocol}")
    print(f"infidelity:      {report.infidelity:.3e}")
    print(f"fidelity:        {report.fidelity:.7f}")
    print(f"phase error:     {report.phase_error_rad:+.3e} rad")
    print(f"theta:           {report.theta_rad:.6f} rad")
    print("residual leakage populations:")
    for level, value in report.residuals.items():
        print(f"   {level:9s} {value:.3e}")
    print("payload:", json.dumps(report.to_dict())[:80], "...")


if __name__ == "__main__":
    main()
