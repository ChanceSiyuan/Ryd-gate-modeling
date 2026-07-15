#!/usr/bin/env python3
"""CZ gate demo: the flagship benchmark point, gate metrics written out.

Reproduces the TO dark-detuning benchmark (infidelity ~6e-5 on current atomic
data; 3.7e-7 when optimized): evolve the CZ
basis states |00>, |01>, |11>, apply the ideal single-qubit Rz corrections,
and score the phase-corrected overlaps with the Nielsen formula. Runtime is
three exact 49-dim solves on the adaptive exact_ode solver (~3 min
single-threaded; the GHz optical phases are resolved with error control).

Usage:
    OMP_NUM_THREADS=1 uv run python examples/demo_cz_gate.py
"""

import numpy as np

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.protocols import TOProtocol

X_TO_DARK = [-0.6894097925886826, 1.040962607910546, 0.3277877211544321,
             1.5639989822346387, 0.6689846026179691, 1.3407418093368753]


def main() -> None:
    system = RydbergSystem(
        level_structure=level_structure("rb87_7_mp", detuning_sign=1),
        register=Register.chain(2, spacing_um=3.0),
    )

    # The TO pulse is fully specified from the historical x layout; theta =
    # x[4] is the ideal single-qubit Rz phase (a scoring parameter, not pulse
    # shape), so it stays a local variable here.
    pulse = TOProtocol(
        phase_amplitude=X_TO_DARK[0], frequency_ratio=X_TO_DARK[1],
        phase_offset=X_TO_DARK[2], detuning_ratio=X_TO_DARK[3],
        duration_ratio=X_TO_DARK[5],
    )
    theta = X_TO_DARK[4]
    bound = system.with_protocol(pulse)

    # Evolve |00>, |01>, |11> (batched: the compiled propagators are shared).
    # Leakage populations are requested as named observable expressions and
    # recorded at the endpoint (t_eval=None -> times == [t_gate]).
    labels = [["0", "0"], ["0", "1"], ["1", "1"]]
    leak_levels = ("e1", "e2", "e3", "r", "r_garb")
    observables = {
        f"n_{lvl}": system.observables.level_sum(lvl) for lvl in leak_levels
    }
    results = simulate(bound, labels, observables=observables)

    # Phase-corrected overlaps: e^{-i theta} (|01>) and e^{-2i theta - i pi}
    # (|11>) remove the ideal single-qubit Rz phases, so a perfect CZ gives
    # a00 == a01 == a11 == 1.
    s0, s1 = np.eye(7, dtype=complex)[0], np.eye(7, dtype=complex)[1]
    kets = [np.kron(s0, s0), np.kron(s0, s1), np.kron(s1, s1)]
    finals = [res.final_state for res in results]
    a00 = complex(np.vdot(kets[0], finals[0]))
    a01 = np.exp(-1j * theta) * complex(np.vdot(kets[1], finals[1]))
    a11 = np.exp(-2j * theta - 1j * np.pi) * complex(np.vdot(kets[2], finals[2]))

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
    # Endpoint expectations are shape-(1,) complex arrays: index [0], take .real.
    residuals = {
        lvl: float(np.mean([res.expectation(f"n_{lvl}")[0].real for res in results]))
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
