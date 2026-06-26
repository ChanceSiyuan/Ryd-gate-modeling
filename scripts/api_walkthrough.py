"""End-to-end walkthrough of the ryd-gate public API.

A runnable, narrated tour of the user-facing surface: register geometry ->
``RydbergSystem`` + a continuous-time protocol -> ``simulate(...)`` -> reading
results (expectations, sampling, final state) -> ``NoiseModel`` -> a CZ gate
report. Everything here runs on the base (exact) install.

    OMP_NUM_THREADS=1 python scripts/api_walkthrough.py

This script doubles as living documentation of the reframed API: ``x`` is
optional for schedule-on-the-protocol cases, ``observables=`` is evaluated for
you, and the returned ``EvolutionResult`` measures/samples itself.
"""

from __future__ import annotations

import numpy as np

from ryd_gate import (
    NoiseModel,
    Register,
    RydbergSystem,
    TFIMQuenchProtocol,
    simulate,
)
from ryd_gate.gates import TOProtocol, cz_gate_report


def many_body_quench() -> None:
    """2D TFIM quench on a Rydberg lattice; read observables off the result."""
    print("== Many-body: TFIM quench ==")
    # 1. Geometry: shape-named constructors return a Register (positions in um).
    register = Register.square(2, spacing_um=9.0)

    # 2. A continuous-time protocol is the control surface; bind it to a system.
    protocol = TFIMQuenchProtocol(hx=2 * np.pi * 1e6, t_gate=0.5e-6)
    system = RydbergSystem.set_atom_level("1r").set_atom_geom(register).set_protocol(protocol)

    # 3. simulate(): x is optional (this protocol takes none); request observables.
    result = simulate(system, psi0="all_1", observables=["sum_nr"])

    # 4. Read results straight off the result object -- no manual re-measuring.
    print(f"   <n_r> after quench:   {result.expectation('sum_nr'):.3f}")
    print(f"   sampled bitstrings:   {result.sample(1000, seed=0).most_common(3)}")
    print(f"   final-state norm:     {np.linalg.norm(result.final_state):.6f}")


def noise_model() -> None:
    """Declarative NoiseModel: serializable data describing requested noise."""
    print("== Noise ==")
    noise = NoiseModel(runs=8, amp_sigma=0.01, detuning_sigma_rad_per_us=0.1)
    print("  " + noise.summary().replace("\n", "\n  "))


def cz_gate_report_demo() -> None:
    """Microscopic CZ gate: one call returns fidelity + phase diagnostics."""
    print("== Gate: CZ report (rb87_7, time-optimal) ==")
    x_to_dark = [
        -0.6989301339711643, 1.0296229082590798, 0.3759232324550267,
        1.5710180991068543, 1.4454279613697887, 1.3406239758422793,
    ]
    system = (
        RydbergSystem.set_atom_level("rb87_7", param_set="our")
        .set_atom_geom(Register.chain(2, spacing_um=3.0))
        .build()
    )
    report = cz_gate_report(system, TOProtocol(), x_to_dark, include_error_budget=False)
    print(f"   fidelity={report.fidelity:.7f}  phase_error={report.phase_error_rad:.2e} rad")


def main() -> None:
    many_body_quench()
    noise_model()
    cz_gate_report_demo()


if __name__ == "__main__":
    main()
