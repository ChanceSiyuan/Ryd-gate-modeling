"""Local addressing simulation: pinning error vs global noise and local RIN.

Sweeps a global laser from -2π×15 MHz to +2π×15 MHz over 1.5 μs while
pinning Atom A with a local 784nm laser (-2π×12 MHz light shift).
Scans σ_detuning and σ_RIN, computes pinning error via Monte Carlo.
"""

import numpy as np
import matplotlib.pyplot as plt

from ryd_gate.core.atomic_system import create_atomic_system, build_sss_state_map
from ryd_gate.protocols.local_sweep import SweepAddressingProtocol
from ryd_gate.solvers.monte_carlo import AddressingMCEngine
from ryd_gate.analysis.addressing_metrics import AddressingEvaluator


def main():
    # --- System setup ---
    system = create_atomic_system(param_set="our", detuning_sign=1)

    # --- Protocol: global sweep + local pinning ---
    omega = system.rabi_eff  # use effective Rabi as global drive
    delta_start = -2 * np.pi * 15e6  # -15 MHz
    delta_end = +2 * np.pi * 15e6    # +15 MHz
    t_gate = 1.5e-6                   # 1.5 μs
    local_detuning_A = -2 * np.pi * 12e6  # -12 MHz pinning
    local_scattering_rate = 35.0      # Hz

    protocol = SweepAddressingProtocol(
        omega=omega,
        delta_start=delta_start,
        delta_end=delta_end,
        t_gate=t_gate,
        local_detuning_A=local_detuning_A,
        local_scattering_rate=local_scattering_rate,
    )

    # --- Initial state: |11⟩ (both atoms in qubit ground state) ---
    states = build_sss_state_map()
    initial_state = states["11"]

    # --- Parameter scan grid ---
    sigma_detunings = np.linspace(10e3, 500e3, 5)   # 10 kHz to 500 kHz
    sigma_rins = np.linspace(0.001, 0.05, 5)         # 0.1% to 5%
    n_shots = 50

    pinning_errors = np.zeros((len(sigma_rins), len(sigma_detunings)))

    print(f"Scanning {len(sigma_detunings)} x {len(sigma_rins)} = "
          f"{len(sigma_detunings) * len(sigma_rins)} grid points, "
          f"{n_shots} shots each")
    print(f"Protocol: sweep {delta_start/2/np.pi/1e6:.1f} -> {delta_end/2/np.pi/1e6:.1f} MHz "
          f"over {t_gate*1e6:.1f} μs")
    print(f"Local pinning: {local_detuning_A/2/np.pi/1e6:.1f} MHz, "
          f"scattering: {local_scattering_rate} Hz")
    print()

    for i, sigma_rin in enumerate(sigma_rins):
        for j, sigma_det in enumerate(sigma_detunings):
            print(f"Grid [{i},{j}]: σ_det={sigma_det/1e3:.0f} kHz, σ_RIN={sigma_rin*100:.1f}%")

            engine = AddressingMCEngine(
                system, protocol,
                sigma_detuning=sigma_det,
                sigma_local_rin=sigma_rin,
            )
            final_states = engine.run(initial_state, n_shots=n_shots, seed=42 + i * 100 + j)

            evaluator = AddressingEvaluator(final_states)
            pe = evaluator.pinning_error()
            ct = evaluator.crosstalk_error()
            ll = evaluator.leakage_loss()
            pinning_errors[i, j] = pe

            print(f"  -> Pinning error: {pe:.6f}, Crosstalk: {ct:.6f}, Leakage: {ll:.6f}")
            print()

    # --- Heatmap ---
    plt.figure(figsize=(8, 6))
    plt.imshow(
        pinning_errors,
        origin="lower",
        aspect="auto",
        extent=[
            sigma_detunings[0] / 1e3, sigma_detunings[-1] / 1e3,
            sigma_rins[0] * 100, sigma_rins[-1] * 100,
        ],
        cmap="hot",
    )
    plt.colorbar(label="Pinning Error")
    plt.xlabel("Global Phase Noise σ_detuning (kHz)")
    plt.ylabel("Local RIN σ_RIN (%)")
    plt.title("Pinning Error vs. Noise Parameters")
    plt.tight_layout()
    plt.savefig("addressing_pinning_heatmap.png", dpi=150)
    print("Saved heatmap to addressing_pinning_heatmap.png")
    plt.show()


if __name__ == "__main__":
    main()
