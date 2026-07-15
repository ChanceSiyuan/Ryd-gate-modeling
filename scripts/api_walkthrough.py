"""End-to-end walkthrough of the ryd-gate public API.

A runnable, narrated tour of the user-facing surface: register geometry ->
``RydbergSystem`` + a continuous-time protocol -> ``simulate(...)`` -> reading
results (expectations, sampling, final state) -> ``NoiseModel`` -> a CZ gate
fidelity computed with the Nielsen formula written inline. Everything here
runs on the base (exact) install. The CZ demo evolves three 7-level two-atom
states on the adaptive exact_ode solver, ~3 min single-threaded.

    OMP_NUM_THREADS=1 python scripts/api_walkthrough.py

This script doubles as living documentation of the reframed API: protocols are
fully specified at construction, observables are ``ObservableExpr`` expressions
built from ``system.observables`` and evaluated for you at the requested
``t_eval`` times, and the returned ``EvolutionResult`` samples itself.
"""

from __future__ import annotations

import numpy as np

from ryd_gate import (
    NoiseModel,
    Register,
    RydbergSystem,
    TFIMQuenchProtocol,
    level_structure,
    simulate,
)
from ryd_gate.protocols import TOProtocol


def many_body_quench() -> None:
    """2D TFIM quench on a Rydberg lattice; read observables off the result."""
    print("== Many-body: TFIM quench ==")
    # 1. Geometry: shape-named constructors return a Register (positions in um).
    register = Register.square(2, spacing_um=9.0)

    # 2. A continuous-time protocol is the control surface; bind it to a system.
    protocol = TFIMQuenchProtocol(hx=2 * np.pi * 1e6, t_gate=0.5e-6)
    system = RydbergSystem(
        level_structure=level_structure("1r"), register=register, protocol=protocol
    )

    # 3. simulate(): the default initial state is every site in the preset's
    #    initial level (|1> here). Observables are named expressions built from
    #    the system's read-only factory (``system.observables``); an explicit
    #    ``t_eval`` sets the measurement times (it never changes the endpoint).
    obs = system.observables
    result = simulate(
        system,
        t_eval=np.linspace(0.0, system.t_gate, 5),
        observables={"n_r": obs.level_sum("r"), "norm": obs.identity()},
    )

    # 4. Expectations come back as complex arrays over ``result.times`` (raw
    #    <psi|O|psi>, no norm division) -- take ``.real`` explicitly for
    #    populations and index the endpoint explicitly.
    n_r = result.expectation("n_r").real
    print(f"   <n_r>(t) trace:       {np.array2string(n_r, precision=3)}")
    print(f"   <n_r> after quench:   {n_r[-1]:.3f}")
    print(f"   sampled bitstrings:   {result.sample(1000, seed=0).most_common(3)}")
    print(f"   survival <psi|psi>:   {result.expectation('norm')[-1].real:.6f}")
    print(f"   final-state norm:     {np.linalg.norm(result.final_state):.6f}")


def noise_model() -> None:
    """Declarative NoiseModel: serializable data describing requested noise."""
    print("== Noise ==")
    noise = NoiseModel(detuning_sigma_rad_s=1e5, amplitude_sigma=0.01)
    print(f"  {noise}")


def cz_fidelity_demo() -> None:
    """Microscopic CZ gate: fidelity + phase diagnostics, formulas inline."""
    print("== Gate: CZ fidelity (rb87_7_mp, time-optimal) ==")
    x_to_dark = [
        -0.6894097925886826, 1.040962607910546, 0.3277877211544321,
        1.5639989822346387, 0.6689846026179691, 1.3407418093368753,
    ]
    system = RydbergSystem(
        level_structure=level_structure("rb87_7_mp"),
        register=Register.chain(2, spacing_um=3.0),
    )

    # TOProtocol is fully specified at construction (x layout:
    # [A, w, phi0, d, theta, T]); theta = x[4] is the ideal single-qubit Rz
    # phase (a scoring parameter, not pulse shape).
    pulse = TOProtocol(
        phase_amplitude=x_to_dark[0], frequency_ratio=x_to_dark[1],
        phase_offset=x_to_dark[2], detuning_ratio=x_to_dark[3],
        duration_ratio=x_to_dark[5],
    )
    theta = x_to_dark[4]
    bound = system.with_protocol(pulse)

    # Evolve the CZ basis states |00>, |01>, |11> (7-level atoms, batched so
    # the compiled propagators are shared) and phase-correct the overlaps: a
    # perfect CZ gives a00 == a01 == a11 == 1.
    labels = [["0", "0"], ["0", "1"], ["1", "1"]]
    results = simulate(bound, labels)
    s0, s1 = np.eye(7, dtype=complex)[0], np.eye(7, dtype=complex)[1]
    kets = [np.kron(s0, s0), np.kron(s0, s1), np.kron(s1, s1)]
    a00, a01, a11 = (
        np.vdot(kets[0], results[0].final_state),
        np.exp(-1j * theta) * np.vdot(kets[1], results[1].final_state),
        np.exp(-2j * theta - 1j * np.pi) * np.vdot(kets[2], results[2].final_state),
    )

    # Nielsen average gate fidelity (d = 4; |10> folded into |01> by symmetry).
    avg_f = (1 / 20) * (
        abs(a00 + 2 * a01 + a11) ** 2 + abs(a00) ** 2 + 2 * abs(a01) ** 2 + abs(a11) ** 2
    )
    # Residual conditional-phase error: second difference of the corrected
    # overlap phases, wrapped to (-pi, pi].
    phase_error = (np.angle(a11) - 2 * np.angle(a01) + np.angle(a00) + np.pi) % (
        2 * np.pi
    ) - np.pi
    print(f"   fidelity={avg_f:.7f}  phase_error={phase_error:.2e} rad")


def main() -> None:
    many_body_quench()
    noise_model()
    cz_fidelity_demo()


if __name__ == "__main__":
    main()
