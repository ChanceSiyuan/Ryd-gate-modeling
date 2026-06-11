#!/usr/bin/env python3
"""NoiseModel demo: declarative noise onto the exact Monte Carlo runner.

Uses a shortened TO gate window so the run stays in seconds; the workflow is
identical for the full benchmark parameters.

Usage:
    OMP_NUM_THREADS=1 uv run python examples/demo_noise_model.py
"""

import numpy as np

from ryd_gate import NoiseModel, Register, RydbergSystem, configure_monte_carlo_runner
from ryd_gate.backends.exact import MonteCarloRunner, SparseExpmBackend
from ryd_gate.gates import TOProtocol

X_TO_SHORT = [-0.6989301339711643, 1.0296229082590798, 0.3759232324550267,
              1.5710180991068543, 1.4454279613697887, 0.13]


def main() -> None:
    noise = NoiseModel(
        runs=8,
        detuning_sigma_rad_per_us=2 * np.pi * 130e3 / 1e6,   # 130 kHz
        amp_sigma=0.01,
        rydberg_decay=True,
    )
    print(noise.summary())
    print("serialized:", noise.to_dict()["schema"])

    system = RydbergSystem.from_lattice(
        Register.chain(2, spacing_um=3.0), "rb87_7", param_set="our",
        blackmanflag=False,
        **noise.physical_kwargs(),       # decay enters at construction time
    )
    runner = MonteCarloRunner(
        system.with_protocol(TOProtocol()), X_TO_SHORT, backend=SparseExpmBackend(n_steps=24)
    )
    configure_monte_carlo_runner(runner, noise)   # quasi-static noise, exact units

    result = runner.run_gate_fidelity(n_shots=noise.runs, seed=11)
    print(f"mean infidelity over {noise.runs} shots: {result.mean_infidelity:.4e} "
          f"(std {result.std_infidelity:.2e})")


if __name__ == "__main__":
    main()
