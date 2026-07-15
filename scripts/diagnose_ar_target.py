"""Target-soundness + gate-quality diagnostic for the CZ infidelity on rb87_7_mp.

The 3-state Nielsen objective evolves |00>, |01>, |11> and reuses |01> for
the |10> contribution, assuming 01<->10 exchange symmetry, with a single
single-qubit phase `theta = x[-1]`. This script verifies the target is
faithful on the rb87_7_mp (σ⁻/σ⁺) system (where the AR optimizer gets stuck
at 0.451), so we can rule out a target bug before blaming the optimizer.

For the TO dark point and the AR points on rb87_7_mp it reports, from the raw
computational-basis overlaps r_kk = <kk|U|kk>:

- **|a01 - a10|** -- the quantity the 3-state formula assumes is 0 (exact
  01<->10 symmetry).
- **leakage** 1 - |r_kk|^2 per state -- whether population stays computational.
- **conditional phase** arg(r11) - 2*arg(r01) + arg(r00), wrapped; a CZ needs
  this = pi. This is the entangling content.
- **nielsen infidelity** at x's theta, and the best achievable over theta (so a
  bad theta cannot be mistaken for a bad gate).

Key lesson baked in: a 4-state Bell-state infidelity can be *small even when
the gate is far from CZ*, because the Bell basis is blind to the
|01>-vs-|00> relative phase. The 3-state Nielsen value is the faithful one.

Read-only. Each case evolves the four computational states on the adaptive
exact_ode solver (~4 min per case single-threaded). Usage:
    OMP_NUM_THREADS=1 uv run python scripts/diagnose_ar_target.py
"""

import json
import time
from pathlib import Path

import numpy as np

from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.backends.exact import simulate
from ryd_gate.protocols.gate_cz import ARProtocol, TOProtocol

X_TO_DARK = [
    -0.6894097925886826, 1.040962607910546, 0.3277877211544321,
    1.5639989822346387, 0.6689846026179691, 1.3407418093368753,
]

_S0 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=complex)
_S1 = np.array([0, 1, 0, 0, 0, 0, 0], dtype=complex)
_ST = {
    "00": np.kron(_S0, _S0), "01": np.kron(_S0, _S1),
    "10": np.kron(_S1, _S0), "11": np.kron(_S1, _S1),
}


def _checkpoint_x(name: str):
    path = Path("results") / "cz_gate" / "ar_optimization" / f"ar_opt_{name}.json"
    return json.loads(path.read_text())["best_x"]


def _build_pulse(Proto, x):
    """Concrete TO/AR pulse from the non-theta entries of the optimizer x-vector.

    x layouts: TO [A, w, phi0, d, theta, T] (theta = x[4]);
    AR [w, A1, p1, A2, p2, d, T, theta] (theta = x[7]).
    """
    x = [float(v) for v in x]
    if Proto is TOProtocol:
        return TOProtocol(phase_amplitude=x[0], frequency_ratio=x[1],
                          phase_offset=x[2], detuning_ratio=x[3],
                          duration_ratio=x[5])
    return ARProtocol(frequency_ratio=x[0], phase_amplitude_1=x[1],
                      phase_offset_1=x[2], phase_amplitude_2=x[3],
                      phase_offset_2=x[4], detuning_ratio=x[5],
                      duration_ratio=x[6])


def _raw_overlaps(system, Proto, x):
    """r_kk = <kk|U|kk> for kk in 00,01,10,11 (no theta correction)."""
    bound = system.with_protocol(_build_pulse(Proto, x))
    return {k: complex(np.vdot(_ST[k], simulate(bound, _ST[k]).final_state))
            for k in ("00", "01", "10", "11")}


def _nielsen_inf_at_theta(r, theta):
    a00 = r["00"]
    a01 = np.exp(-1j * theta) * r["01"]
    a11 = np.exp(-2j * theta - 1j * np.pi) * r["11"]
    avg_f = (1 / 20) * (abs(a00 + 2 * a01 + a11) ** 2
                        + abs(a00) ** 2 + 2 * abs(a01) ** 2 + abs(a11) ** 2)
    return 1 - avg_f


def main() -> None:
    system = RydbergSystem(
        level_structure=level_structure("rb87_7_mp"),
        register=Register.chain(2, spacing_um=3.0),
    )
    cases = [
        ("TO dark  (X_TO_DARK)", TOProtocol, X_TO_DARK),
        ("AR stuck (mp)", ARProtocol, _checkpoint_x("mp")),
        ("AR pm-opt on mp", ARProtocol, _checkpoint_x("pm")),
    ]
    thetas = np.linspace(-np.pi, np.pi, 4001)

    print(f"{'case':22s} {'|a01-a10|':>10s} {'maxleak':>8s} "
          f"{'cphase-pi':>9s} {'nielsen':>11s} {'best-theta':>11s} {'sec':>5s}")
    print("-" * 84)
    for label, Proto, x in cases:
        t0 = time.perf_counter()
        r = _raw_overlaps(system, Proto, x)
        theta_x = float(x[4] if Proto is TOProtocol else x[7])  # theta: TO x[4], AR x[7]
        dsym = abs(r["01"] - r["10"])
        max_leak = max(1 - abs(r[k]) ** 2 for k in ("00", "01", "11"))
        cphase = np.angle(r["11"]) - np.angle(r["01"]) - np.angle(r["10"]) + np.angle(r["00"])
        cw = (cphase + np.pi) % (2 * np.pi) - np.pi
        cphase_dev = min(abs(cw - np.pi), abs(cw + np.pi))  # distance to +-pi
        nielsen = _nielsen_inf_at_theta(r, theta_x)
        best = min(_nielsen_inf_at_theta(r, t) for t in thetas)
        dt = time.perf_counter() - t0
        print(f"{label:22s} {dsym:10.2e} {max_leak:8.4f} "
              f"{cphase_dev:9.4f} {nielsen:11.4e} {best:11.4e} {dt:5.1f}")

    print("-" * 84)
    print("Reading:")
    print("  |a01-a10|~0  => exact 01<->10 symmetry: the 3-state Nielsen target is faithful.")
    print("  TO dark: nielsen ~1e-6 with cphase-pi~0 => target sound at a true CZ.")
    print("  AR stuck: low leakage but cphase-pi large and best-theta still bad => a genuine")
    print("            NON-entangling local minimum (wrong conditional phase), not a target bug")
    print("            and not a leakage/blockade wall. A good 'our' AR gate should exist;")
    print("            the single-start optimizer simply did not find the right entangling phase.")


if __name__ == "__main__":
    main()
