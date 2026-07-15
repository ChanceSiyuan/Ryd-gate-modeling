"""End-to-end walkthrough of the ryd-gate public API.

A runnable, narrated tour of the user-facing surface: a ``Register`` geometry ->
a fully specified continuous-time protocol -> ``RydbergSystem`` -> ``simulate(...)``
-> reading results (expectations, sampling, basis amplitudes) -> a ``NoiseModel``
ensemble -> a CZ gate fidelity computed with the Nielsen formula written inline.
Everything here runs on the base (exact) install. The CZ demo evolves three
7-level two-atom states on the adaptive exact_ode solver, ~3 min single-threaded.

    OMP_NUM_THREADS=1 python scripts/api_walkthrough.py

This script doubles as living documentation of the reframed API: protocols are
fully specified at construction (no scale is read back from a preset), observables
are ``ObservableExpr`` expressions built from ``system.observables`` and evaluated
for you at the requested ``t_eval`` times, and the returned ``EvolutionResult``
exposes only physical readouts (``expectation``/``amplitude``/``sample``).
"""

from __future__ import annotations

import numpy as np

from ryd_gate import (
    NoiseModel,
    Register,
    RydbergSystem,
    level_structure,
    simulate,
    simulate_ensemble,
)
from ryd_gate.physics import arc_pair_c6_rad_s_um6
from ryd_gate.protocols import SweepProtocol, TOProtocol


def many_body_quench() -> None:
    """Global |1>-|r> quench on a 2x2 lattice == transverse-field Ising quench."""
    print("== Many-body: TFIM quench ==")
    # 1. Geometry: shape-named constructors return a Register (positions in um).
    register = Register.square(2, spacing_um=9.0)
    ryd_level = 70

    # 2. Transverse-field Ising mapping (n_r = |r><r|): the global drive is the
    #    transverse field h_x = Omega/2 = omega_half, and the van der Waals tail
    #    sets the Ising couplings J_ij = V_ij/4. The mapping is computed locally
    #    from the register geometry and the ARC S-state C6 (the same coefficient
    #    the 1r preset uses for its interaction).
    h_x = 2 * np.pi * 1.0e6
    C6 = arc_pair_c6_rad_s_um6(
        n1=ryd_level, l1=0, j1=0.5, mj1=-0.5, mj2=-0.5, theta=0.0, phi=0.0, degenerate=False
    )
    coords = register.coords
    r_nn = float(np.hypot(*(coords[1] - coords[0])))  # nearest-neighbour spacing
    J_nn = (C6 / r_nn**6) / 4.0
    print(f"   h_x/2pi = {h_x / (2 * np.pi) / 1e6:.3f} MHz   J_nn/2pi = {J_nn / (2 * np.pi) / 1e6:.3f} MHz")

    # 3. A continuous-time protocol is the control surface, fully specified at
    #    construction; bind it in a RydbergSystem. omega_half_rad_s is already
    #    Omega/2 (no extra 1/2, P10); detuning 0 -> a pure quench.
    system = RydbergSystem(
        level_structure=level_structure("1r", ryd_level=ryd_level),
        register=register,
        protocol=SweepProtocol(
            t_gate_s=0.5e-6,
            omega_half_rad_s=lambda t: h_x,
            detuning_rad_s=lambda t: 0.0,
        ),
    )

    # 4. simulate(): the default initial state is every site in |1> (E27).
    #    Observables are named ObservableExpr built from the read-only factory
    #    system.observables; an explicit t_eval sets the measurement times only
    #    (it never moves the endpoint). Total Rydberg population is a site sum.
    n_r_total = sum(system.observables.n("r", i) for i in range(system.N))
    result = simulate(
        system,
        t_eval=np.linspace(0.0, system.t_gate, 5),
        observables={"n_r": n_r_total},
    )

    # 5. expectation(name) is a real float64 array over result.times; amplitude()
    #    and sample() read the endpoint state.
    n_r = result.expectation("n_r")
    print(f"   <n_r>(t) trace:      {np.array2string(n_r, precision=3)}")
    print(f"   <n_r> after quench:  {n_r[-1]:.3f}")
    print(f"   sampled bitstrings:  {result.sample(shots=1000, seed=0).most_common(3)}")
    print(f"   P(return to 1111):   {abs(result.amplitude(['1', '1', '1', '1'])) ** 2:.4f}")


def noise_ensemble() -> None:
    """Declarative NoiseModel + simulate_ensemble: shot-major noisy realizations."""
    print("== Noise: quasi-static position-jitter ensemble ==")
    system = RydbergSystem(
        level_structure=level_structure("1r", ryd_level=70),
        register=Register.chain(3, spacing_um=8.0),
        protocol=SweepProtocol(
            t_gate_s=0.3e-6,
            omega_half_rad_s=lambda t: 2 * np.pi * 1.0e6,
            detuning_rad_s=lambda t: 0.0,
        ),
    )
    # Zero-mean 3D Gaussian atom-position jitter (sigma_z is out-of-plane thermal
    # motion of the 2D array); it perturbs only the pair distances -> the vdW
    # interaction. Laser-group amplitude/frequency noise are the other channels.
    noise = NoiseModel(position_sigma_um=(0.1, 0.1, 0.3))
    n_r_total = sum(system.observables.n("r", i) for i in range(system.N))
    ens = simulate_ensemble(
        system, noise=noise, shots=4, seed=0, observables={"n_r": n_r_total}
    )
    per_shot = np.array([float(r.expectation("n_r")[-1]) for r in ens.results])
    print(f"   position_sigma_um = {noise.position_sigma_um}")
    print(f"   seed={ens.seed}  shots={len(ens.results)}")
    print(f"   <n_r>(t_gate) per shot: {np.array2string(per_shot, precision=3)}")


def cz_fidelity_demo() -> None:
    """Microscopic CZ gate: fidelity + phase/leakage diagnostics, formulas inline."""
    print("== Gate: CZ fidelity (rb87_7_mp, time-optimal) ==")
    # Canonical sigma-/sigma+ (rb87_7_mp, "our") laser scales: 70S Rydberg level,
    # intermediate detuning Delta_e = 2pi*9.1 GHz, peak single-photon Rabis
    # Omega_420 = 2pi*491 MHz, Omega_1013 = 2pi*185 MHz. The CZ protocols take
    # these explicitly now (rad/s) -- nothing is read back from the preset.
    delta_e = 2 * np.pi * 9.1e9
    omega_420_max = 2 * np.pi * 491e6
    omega_1013_max = 2 * np.pi * 185e6

    # x layout: [phase_amplitude, frequency_ratio, phase_offset, detuning_ratio,
    #            theta, duration_ratio]; theta = x[4] is the ideal single-qubit Rz
    # phase (a scoring parameter your analysis owns, not pulse shape).
    x_to_dark = [
        -0.6894097925886826, 1.040962607910546, 0.3277877211544321,
        1.5639989822346387, 0.6689846026179691, 1.3407418093368753,
    ]
    theta = x_to_dark[4]

    # t_gate = duration_ratio * 2pi/Omega_eff is resolved as system.t_gate once
    # bound; the family carries a Blackman rise, and the x_to_dark optimum was
    # found at the canonical rb87_7 rise time of 20 ns.
    pulse = TOProtocol(
        intermediate_detuning_rad_s=delta_e,
        omega_420_max_rad_s=omega_420_max,
        omega_1013_max_rad_s=omega_1013_max,
        rise_time_s=20e-9,
        phase_amplitude_rad=x_to_dark[0],
        modulation_frequency_ratio=x_to_dark[1],
        phase_offset_rad=x_to_dark[2],
        frequency_offset_ratio=x_to_dark[3],
        duration_ratio=x_to_dark[5],
    )
    system = RydbergSystem(
        level_structure=level_structure("rb87_7_mp"),
        register=Register.chain(2, spacing_um=3.0),
        protocol=pulse,
    )

    # Evolve the CZ basis states |00>, |01>, |11> (7-level atoms, batched so the
    # compiled propagators are shared) and phase-correct the diagonal overlaps
    # <ini|psi(t_gate)>: a perfect CZ gives a00 == a01 == a11 == 1.
    results = simulate(system, [["0", "0"], ["0", "1"], ["1", "1"]])
    a00 = results[0].amplitude(["0", "0"])
    a01 = np.exp(-1j * theta) * results[1].amplitude(["0", "1"])
    a11 = np.exp(-2j * theta - 1j * np.pi) * results[2].amplitude(["1", "1"])

    # Nielsen average gate fidelity (d = 4; |10> folded into |01> by symmetry).
    avg_f = (1 / 20) * (
        abs(a00 + 2 * a01 + a11) ** 2 + abs(a00) ** 2 + 2 * abs(a01) ** 2 + abs(a11) ** 2
    )
    # Residual conditional-phase error: second difference of the corrected
    # overlap phases, wrapped to (-pi, pi].
    phase_error = (np.angle(a11) - 2 * np.angle(a01) + np.angle(a00) + np.pi) % (
        2 * np.pi
    ) - np.pi
    # Leakage of the evolved |11> out of the two-qubit computational subspace,
    # read straight off the basis amplitudes.
    comp = [["0", "0"], ["0", "1"], ["1", "0"], ["1", "1"]]
    leak_11 = 1.0 - sum(abs(results[2].amplitude(k)) ** 2 for k in comp)
    print(f"   fidelity={avg_f:.7f}  phase_error={phase_error:.2e} rad  leak(11)={leak_11:.2e}")


def main() -> None:
    many_body_quench()
    noise_ensemble()
    cz_fidelity_demo()


if __name__ == "__main__":
    main()
