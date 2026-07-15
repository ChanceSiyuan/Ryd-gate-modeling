#!/usr/bin/env python3
"""NoiseModel demo: quasi-static noise ensembles with the statistics inline.

Runs a shortened TO gate window under shot-to-shot detuning + amplitude noise
via ``simulate_ensemble`` and computes the mean/std CZ infidelity from the raw
shot results (the library returns raw ensembles; aggregation is yours).

Runtime: 8 shots x 3 states = 24 solves of the shortened window on the
adaptive exact_ode solver, ~5 min single-threaded.

Usage:
    OMP_NUM_THREADS=1 uv run python examples/demo_noise_model.py
"""

import numpy as np

from ryd_gate import NoiseModel, Register, RydbergSystem, level_structure, simulate_ensemble
from ryd_gate.protocols import TOProtocol

# Historical layout [A, omega/Omega_eff, phi0, delta/Omega_eff, theta, T/T_scale];
# theta = x[4] is the single-qubit Rz scoring parameter, not pulse shape.
X_TO_SHORT = [-0.6894097925886826, 1.040962607910546, 0.3277877211544321,
              1.5639989822346387, 0.6689846026179691, 0.13]


def main() -> None:
    noise = NoiseModel(
        detuning_sigma_rad_s=2 * np.pi * 130e3,   # 130 kHz static detuning offsets
        amplitude_sigma=0.01,                     # 1% fractional Rabi noise
    )
    print(noise)

    system = RydbergSystem(
        level_structure=level_structure("rb87_7_mp", enable_rydberg_decay=True),
        register=Register.chain(2, spacing_um=3.0),
        protocol=TOProtocol(
            phase_amplitude=X_TO_SHORT[0], frequency_ratio=X_TO_SHORT[1],
            phase_offset=X_TO_SHORT[2], detuning_ratio=X_TO_SHORT[3],
            duration_ratio=X_TO_SHORT[5],
            blackman=False,  # the shortened window is below the Blackman rise
        ),
    )
    theta = X_TO_SHORT[4]

    # One shot evolves all three CZ basis states under the same realization.
    shots = 8
    ens = simulate_ensemble(
        system,
        [["0", "0"], ["0", "1"], ["1", "1"]],
        noise=noise,
        shots=shots,
        seed=11,
        backend="exact_ode",
    )

    # Nielsen CZ infidelity per shot, written out from the raw overlaps.
    s0, s1 = np.eye(7, dtype=complex)[0], np.eye(7, dtype=complex)[1]
    kets = [np.kron(s0, s0), np.kron(s0, s1), np.kron(s1, s1)]
    corrections = [1.0, np.exp(-1j * theta), np.exp(-2j * theta - 1j * np.pi)]
    infidelities = []
    for shot in ens.results:
        a00, a01, a11 = (
            corr * complex(np.vdot(ket, res.final_state))
            for corr, ket, res in zip(corrections, kets, shot)
        )
        avg_f = (1 / 20) * (
            abs(a00 + 2 * a01 + a11) ** 2
            + abs(a00) ** 2 + 2 * abs(a01) ** 2 + abs(a11) ** 2
        )
        infidelities.append(1.0 - avg_f)
    infidelities = np.asarray(infidelities)

    offsets = [f"{r['detuning_rad_s']:+.2e}" for r in ens.realizations]
    print(f"sampled detuning offsets (rad/s): {offsets}")
    print(f"mean infidelity over {shots} shots: {infidelities.mean():.4e} "
          f"(std {infidelities.std():.2e})")


if __name__ == "__main__":
    main()
