#!/usr/bin/env python3
"""NoiseModel demo: quasi-static laser noise ensembles with the statistics inline.

Runs a shortened TO gate window under shot-to-shot laser amplitude + frequency
noise via ``simulate_ensemble`` and computes the mean/std CZ infidelity from the
raw shot results (the library returns raw ensembles; aggregation is yours). The
declarative ``NoiseModel`` names the physical laser groups the protocol drives
("420" / "1013"); each shot rescales those Rabi amplitudes and adds a per-laser
frequency offset.

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

# Canonical rb87_7_mp (σ⁻/σ⁺) laser parameters, passed explicitly to the protocol
# (P19/P20). RISE_SHORT is shrunk from the nominal 20 ns so the Blackman window
# still fits inside the shortened (duration_ratio=0.13) gate.
OMEGA_420_MAX = 2 * np.pi * 491e6    # rad/s
OMEGA_1013_MAX = 2 * np.pi * 185e6   # rad/s
DELTA_DARK = 2 * np.pi * 9.1e9       # rad/s
RISE_SHORT = 5e-9                    # Blackman rise/fall (s)


def main() -> None:
    # Quasi-static Gaussian noise on the two laser groups the TO pulse drives:
    # 1% fractional Rabi (intensity) noise on each leg + a 130 kHz frequency
    # offset on the 420 nm laser (the closest analog of a static detuning offset).
    noise = NoiseModel(
        laser_amplitude_sigma={"420": 0.01, "1013": 0.01},
        laser_frequency_sigma_rad_s={"420": 2 * np.pi * 130e3},
    )

    system = RydbergSystem(
        level_structure=level_structure("rb87_7_mp"),
        register=Register.chain(2, spacing_um=3.0),
        protocol=TOProtocol(
            intermediate_detuning_rad_s=DELTA_DARK,
            omega_420_max_rad_s=OMEGA_420_MAX,
            omega_1013_max_rad_s=OMEGA_1013_MAX,
            rise_time_s=RISE_SHORT,
            phase_amplitude_rad=X_TO_SHORT[0],
            modulation_frequency_ratio=X_TO_SHORT[1],
            phase_offset_rad=X_TO_SHORT[2],
            frequency_offset_ratio=X_TO_SHORT[3],
            duration_ratio=X_TO_SHORT[5],
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

    # Nielsen CZ infidelity per shot, written out from the raw overlaps. Each
    # entry of ens.results is the tuple of three per-state EvolutionResults; the
    # phase corrections remove the ideal single-qubit Rz phases.
    corrections = [1.0, np.exp(-1j * theta), np.exp(-2j * theta - 1j * np.pi)]
    kets = [["0", "0"], ["0", "1"], ["1", "1"]]
    infidelities = []
    for shot in ens.results:
        a00, a01, a11 = (
            corr * res.amplitude(ket)
            for corr, ket, res in zip(corrections, kets, shot)
        )
        avg_f = (1 / 20) * (
            abs(a00 + 2 * a01 + a11) ** 2
            + abs(a00) ** 2 + 2 * abs(a01) ** 2 + abs(a11) ** 2
        )
        infidelities.append(1.0 - avg_f)
    infidelities = np.asarray(infidelities)

    offsets = [f"{r['laser_frequency_offsets_rad_s']['420']:+.2e}" for r in ens.realizations]
    print(f"sampled 420 nm frequency offsets (rad/s): {offsets}")
    print(f"mean infidelity over {shots} shots: {infidelities.mean():.4e} "
          f"(std {infidelities.std():.2e})")


if __name__ == "__main__":
    main()
