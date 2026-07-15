#!/usr/bin/env python3
"""CZ gate demo: the flagship benchmark point, gate metrics written out.

Reproduces the TO dark-detuning benchmark (infidelity ~6e-5 on current atomic
data; 3.7e-7 when optimized): evolve the CZ basis states |00>, |01>, |11>,
apply the ideal single-qubit Rz corrections, and score the phase-corrected
overlaps with the Nielsen formula. The library ships no gate-report API, so the
fidelity / conditional-phase / leakage formulas are written inline here and the
overlaps are read off the result with ``result.amplitude([...])``. Runtime is
three exact 49-dim solves on the adaptive exact_ode solver (~3 min
single-threaded; the GHz optical phases are resolved with error control).

Usage:
    OMP_NUM_THREADS=1 uv run python examples/demo_cz_gate.py
"""

import numpy as np

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.protocols import TOProtocol

# Historical layout [A, omega/Omega_eff, phi0, delta/Omega_eff, theta, T/T_scale];
# theta = x[4] is the ideal single-qubit Rz scoring parameter, not pulse shape.
X_TO_DARK = [-0.6894097925886826, 1.040962607910546, 0.3277877211544321,
             1.5639989822346387, 0.6689846026179691, 1.3407418093368753]

# Canonical rb87_7_mp (σ⁻/σ⁺) laser parameters. These were baked into the preset
# and are now passed explicitly to the protocol (P19/P20). The two peak Rabis are
# the canonical σ⁻ 420 nm / σ⁺ 1013 nm single-photon values (derivable from laser
# power + beam area via ryd_gate.physics.rb87_7_mp_rabi_frequencies); DELTA_DARK
# is the +9.1 GHz "dark" intermediate detuning (was detuning_sign=+1).
OMEGA_420_MAX = 2 * np.pi * 491e6    # rad/s
OMEGA_1013_MAX = 2 * np.pi * 185e6   # rad/s
DELTA_DARK = 2 * np.pi * 9.1e9       # rad/s
RISE_TIME = 20e-9                    # Blackman rise/fall (s)


def main() -> None:
    # The TO pulse is fully specified at construction from the historical x layout
    # plus the explicit physical scales; theta = x[4] is the ideal single-qubit Rz
    # phase (a scoring parameter, not pulse shape), so it stays a local variable.
    pulse = TOProtocol(
        intermediate_detuning_rad_s=DELTA_DARK,
        omega_420_max_rad_s=OMEGA_420_MAX,
        omega_1013_max_rad_s=OMEGA_1013_MAX,
        rise_time_s=RISE_TIME,
        phase_amplitude_rad=X_TO_DARK[0],
        modulation_frequency_ratio=X_TO_DARK[1],
        phase_offset_rad=X_TO_DARK[2],
        frequency_offset_ratio=X_TO_DARK[3],
        duration_ratio=X_TO_DARK[5],
    )
    theta = X_TO_DARK[4]

    system = RydbergSystem(
        level_structure=level_structure("rb87_7_mp"),
        register=Register.chain(2, spacing_um=3.0),
        protocol=pulse,
    )

    # Evolve |00>, |01>, |11> (batched: the compiled propagators are shared).
    # Leakage populations are requested as named observable expressions and
    # recorded at the endpoint (t_eval=None -> times == [t_gate]).
    labels = [["0", "0"], ["0", "1"], ["1", "1"]]
    leak_levels = ("e1", "e2", "e3", "r", "r_garb")
    obs = system.observables
    observables = {
        f"n_{lvl}": sum(obs.n(lvl, i) for i in range(system.N)) for lvl in leak_levels
    }
    results = simulate(system, labels, observables=observables)
    res00, res01, res11 = results

    # Phase-corrected overlaps read off each final state: e^{-i theta} (|01>) and
    # e^{-2i theta - i pi} (|11>) remove the ideal single-qubit Rz phases, so a
    # perfect CZ gives a00 == a01 == a11 == 1.
    a00 = res00.amplitude(["0", "0"])
    a01 = np.exp(-1j * theta) * res01.amplitude(["0", "1"])
    a11 = np.exp(-2j * theta - 1j * np.pi) * res11.amplitude(["1", "1"])

    # Nielsen average gate fidelity (d = 4; |10> folded into |01> by symmetry).
    avg_f = (1 / 20) * (
        abs(a00 + 2 * a01 + a11) ** 2 + abs(a00) ** 2 + 2 * abs(a01) ** 2 + abs(a11) ** 2
    )
    infidelity = 1.0 - avg_f
    # Residual conditional-phase error: second difference of the corrected
    # overlap phases, wrapped to (-pi, pi].
    phase_error = (np.angle(a11) - 2 * np.angle(a01) + np.angle(a00) + np.pi) % (
        2 * np.pi
    ) - np.pi

    # Residual leakage populations, averaged over the three trajectories.
    # Endpoint expectations are real shape-(1,) arrays: index [0].
    residuals = {
        lvl: float(np.mean([res.expectation(f"n_{lvl}")[0] for res in results]))
        for lvl in leak_levels
    }

    print(f"protocol:        {type(pulse).__name__} (TO dark point)")
    print(f"infidelity:      {infidelity:.3e}")
    print(f"fidelity:        {1.0 - infidelity:.7f}")
    print(f"phase error:     {phase_error:+.3e} rad")
    print(f"theta:           {theta:.6f} rad")
    print("residual leakage populations:")
    for level, value in residuals.items():
        print(f"   {level:9s} {value:.3e}")


if __name__ == "__main__":
    main()
