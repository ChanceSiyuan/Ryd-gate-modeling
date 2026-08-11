"""Shared physical model for the CZ error-budget script and notebook."""
from __future__ import annotations


def eval_cz_point(cfg: dict) -> dict:
    """Evaluate one CZ error-budget point from a picklable scalar config."""
    import numpy as np

    import ryd_gate as rg
    from ryd_gate.lattice import Register
    from ryd_gate.physics import rb87_7_mp_rabi_frequencies
    from ryd_gate.protocols import CZProtocol, phase_from_chirp

    def wrap(angle):
        return float(np.angle(np.exp(1j * angle)))

    def envelope(t, t_gate, ramp=0.15):
        scaled = float(np.clip(t / t_gate, 0.0, 1.0))

        def smoothstep(value):
            value = np.clip(value, 0.0, 1.0)
            return 10 * value**3 - 15 * value**4 + 6 * value**5

        if scaled < ramp:
            return float(smoothstep(scaled / ramp))
        if scaled > 1 - ramp:
            return float(smoothstep((1 - scaled) / ramp))
        return 1.0

    t_gate = cfg["t_gate"]
    delta_e = 2 * np.pi * cfg["Delta_e_Hz"]
    d_sweep = 2 * np.pi * cfg["D_sweep_Hz"]
    beam_area = cfg["beam_factor"] * cfg["spacing_um"]
    optics_loss = cfg["optics_loss"]
    omega_420, omega_1013 = rb87_7_mp_rabi_frequencies(
        max(cfg["p420_w"], 0.0) * (1 - optics_loss),
        max(cfg["p1013_w"], 0.0) * (1 - optics_loss),
        beam_area,
        ryd_level=cfg["ryd_level"],
    )

    env = lambda t: envelope(t, t_gate)
    base_chirp = lambda t: -d_sweep * np.cos(2.0 * np.pi * t / t_gate)
    d1_nom = -(4.0 / 3.0) * omega_420**2 / (4.0 * delta_e)
    dr_nom = -omega_1013**2 / (4.0 * delta_e)

    def chirp(t):
        amplitude = np.sqrt(env(t))
        return base_chirp(t) + (dr_nom - d1_nom) * amplitude**2

    phase = phase_from_chirp(
        chirp, t_gate_s=t_gate, n_samples=4 * cfg["n_steps"] + 1)
    clip = lambda t: float(np.clip(t, 0.0, t_gate))
    protocol = CZProtocol(
        t_gate_s=t_gate,
        intermediate_detuning_rad_s=delta_e,
        omega_420_max_rad_s=omega_420,
        omega_1013_max_rad_s=omega_1013,
        envelope_420=lambda t: np.sqrt(env(clip(t))),
        phase_420_rad=lambda t: phase(clip(t)),
        envelope_1013=lambda t: np.sqrt(env(clip(t))),
        phase_1013_rad=lambda t: 0.0,
    )
    system = rg.RydbergSystem(
        level_structure=rg.level_structure(
            "rb87_7_mp", ryd_level=cfg["ryd_level"], magnetic_field_G=20.0),
        register=Register.chain(2, spacing_um=cfg["spacing_um"]),
        protocol=protocol,
    )
    t_eval = np.linspace(0.0, t_gate, cfg["n_eval"])
    observables = system.observables
    n_atoms = system.N
    measured = {
        "n_e": sum(
            observables.n("e1", i) + observables.n("e2", i)
            + observables.n("e3", i)
            for i in range(n_atoms)),
        "n_r": sum(observables.n("r", i) for i in range(n_atoms)),
        "n_rg": sum(observables.n("r_garb", i) for i in range(n_atoms)),
    }
    states = ("00", "01", "10", "11")
    results = rg.simulate(
        system,
        [list(state) for state in states],
        t_eval=t_eval,
        observables=measured,
        backend=cfg["backend"],
        backend_options={
            "rtol": cfg.get("rtol", 1e-8),
            "atol": cfg.get("atol", 1e-12),
        },
    )

    decay = system.level_structure.decay_rates_per_s
    gamma_e = float(decay["e1"]["total"])
    gamma_r = float(decay["r"]["total"])
    gamma_rg = float(decay["r_garb"]["total"])
    phases = {}
    returns = {}
    leakage = {}
    p_mid = []
    p_ryd = []
    p_r_garb = []
    for state, result in zip(states, results):
        overlap = result.amplitude(list(state))
        phases[state] = float(np.angle(overlap))
        returns[state] = float(abs(overlap) ** 2)
        leakage[state] = float(
            1.0 - sum(abs(result.amplitude(list(other))) ** 2
                      for other in states))
        times = np.asarray(result.times, float)
        p_mid.append(np.trapezoid(
            gamma_e * result.expectation("n_e"), times))
        p_ryd.append(np.trapezoid(
            gamma_r * result.expectation("n_r"), times))
        p_r_garb.append(np.trapezoid(
            gamma_rg * result.expectation("n_rg"), times))

    p_mid = np.asarray(p_mid)
    p_ryd = np.asarray(p_ryd)
    p_r_garb = np.asarray(p_r_garb)
    p_total = p_mid + p_ryd + p_r_garb
    zz_phase = wrap(
        phases["11"] - phases["01"] - phases["10"] + phases["00"])
    phase_error = abs(wrap(zz_phase - np.pi))
    omega_effective = omega_420 * omega_1013 / (2 * abs(delta_e))
    k_effective = 0.5 * omega_effective
    max_leakage = max(leakage.values())
    score = (100.0 * phase_error**2 + 10.0 * max_leakage
             + 1.0e3 * float(p_total.max()))
    return {
        "Delta_e_Hz": cfg["Delta_e_Hz"],
        "D_sweep_Hz": cfg["D_sweep_Hz"],
        "p420_w": cfg["p420_w"],
        "p1013_w": cfg["p1013_w"],
        "spacing_um": cfg["spacing_um"],
        "T": t_gate,
        "Omega_420": omega_420,
        "Omega_1013": omega_1013,
        "Omega_eff_phys": omega_effective,
        "K_eff": k_effective,
        "zz_phase": zz_phase,
        "phase_err": phase_error,
        "max_leakage": max_leakage,
        "min_return_prob": min(returns.values()),
        "p_mid_max": float(p_mid.max()),
        "p_ryd_max": float(p_ryd.max()),
        "p_r_garb_max": float(p_r_garb.max()),
        "p_loss_total_max": float(p_total.max()),
        "score": score,
    }
