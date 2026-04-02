#!/usr/bin/env python
"""Visualize the breakdown of Rydberg blockade via local addressing.

Produces three figures:

  Figure 1 — Rabi dynamics (resonant driving, no chirp):
    (a) Global excitation: both atoms oscillate collectively at sqrt(2)*Omega_eff
        with P_r max ~ 0.5 due to Rydberg blockade.
    (b) Atom A addressed: P_r^(A) = 0, P_r^(B) oscillates at Omega_eff
        reaching ~1.0 (no blockade). Dashed norm shows scattering decay.

  Figure 2 — Final state populations (adiabatic sweep):
    (a) Global sweep: ~50/50 in |gr> and |rg>, |rr> ~ 0.
    (b) Atom A addressed: ~100% in |gr>.

  Figure 3 — AC Stark feed-forward compensation comparison:
    (a) Without compensation: parasitic detuning from Blackman envelope
        ramp distorts the adiabatic sweep.
    (b) With compensation: feed-forward correction cancels the AC Stark
        shift, restoring clean adiabatic passage.
"""

import matplotlib.pyplot as plt
import numpy as np

from scipy.constants import pi

from ryd_gate import simulate
from ryd_gate.analysis.observable_metrics import measure_trajectory, norm_squared
from ryd_gate.core.models.analog_3level import Analog3LevelModel
from ryd_gate.core.operators import build_product_state_map
from ryd_gate.protocols.sweep import SweepProtocol

N = 3  # 3-level system: |g>=0, |e>=1, |r>=2

# Pinning parameters scaled for Omega_eff ~ 13 MHz (equal-Rabi regime).
LOCAL_DETUNING = -2 * pi * 50e6    # rad/s  (AC Stark shift on |r>_A)
LOCAL_SCATTER = 150.0              # Hz     (scales with laser power)

# Sweep range: |delta| >> Omega_eff for adiabatic passage.
DELTA_START = -2 * pi * 40e6       # rad/s
DELTA_END = 2 * pi * 40e6          # rad/s
T_GATE = 1.5e-6                    # s

# Product state map: gg, ge, gr, eg, ee, er, rg, re, rr
PRODUCT_STATES = build_product_state_map(n_levels=N)
RYDBERG_KEYS = ["gg", "gr", "rg", "rr"]


# ======================================================================
# Figure 1: Rabi dynamics (resonant driving)
# ======================================================================

def figure_rabi_dynamics(model, initial_state):
    """Two stacked subplots: blockaded vs free Rabi oscillations."""
    system = model.system
    n_cycles = 5
    t_rabi = n_cycles * system.time_scale
    x_rabi = [0.0, 0.0, t_rabi / system.time_scale]
    n_points = 500
    t_eval = np.linspace(0, t_rabi, n_points)

    protocol_free = SweepProtocol()
    protocol_addr = SweepProtocol(addressing={0: LOCAL_DETUNING},
                                   scatter_rate=LOCAL_SCATTER)

    obs_names = ["pop_A_r", "pop_B_r"]

    # (a) No pinning -- blockade active
    result1 = simulate(system, protocol_free, x_rabi, initial_state,
                       t_eval=t_eval)
    obs1 = measure_trajectory(model, result1.states, obs_names)
    prA1, prB1 = obs1["pop_A_r"], obs1["pop_B_r"]

    # (b) Atom A addressed -- blockade irrelevant
    result2 = simulate(system, protocol_addr, x_rabi, initial_state,
                       t_eval=t_eval)
    obs2 = measure_trajectory(model, result2.states, obs_names)
    prA2, prB2 = obs2["pop_A_r"], obs2["pop_B_r"]
    norm2 = np.array([norm_squared(result2.states[:, k])
                       for k in range(result2.states.shape[1])])

    # -- Plot --
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    t1_us = result1.times * 1e6
    t2_us = result2.times * 1e6

    axes[0].plot(t1_us, prA1, color="#2ca02c", ls="-", lw=1.5,
                 label=r"$P_r^{(A)}$")
    axes[0].plot(t1_us, prB1, color="#1f77b4", ls="--", lw=1.5,
                 label=r"$P_r^{(B)}$")
    axes[0].set_ylabel("Population")
    axes[0].set_title(r"(a) Global excitation — blockade active "
                      r"($\sqrt{2}\,\Omega_{\rm eff}$ oscillation)")
    axes[0].legend(loc="upper right")
    axes[0].set_ylim(-0.05, 1.05)

    axes[1].plot(t2_us, prA2, color="#2ca02c", ls="-", lw=1.5,
                 label=r"$P_r^{(A)}$")
    axes[1].plot(t2_us, prB2, color="#1f77b4", ls="--", lw=1.5,
                 label=r"$P_r^{(B)}$")
    axes[1].plot(t2_us, norm2, color="gray", ls=":", lw=1.2,
                 label="Total norm")
    axes[1].set_xlabel(r"Time ($\mu$s)")
    axes[1].set_ylabel("Population")
    axes[1].set_title(r"(b) Atom A addressed — free single-atom oscillation "
                      r"($\Omega_{\rm eff}$)")
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(-0.05, 1.05)

    fig.suptitle("Rabi dynamics: Rydberg blockade breakdown via local addressing",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig("fig1_rabi_dynamics.png", dpi=150)
    print("Saved fig1_rabi_dynamics.png")


# ======================================================================
# Figure 2: Final state populations (adiabatic sweep)
# ======================================================================

def figure_final_populations(model, initial_state):
    """Two side-by-side bar charts of final {|gg>,|gr>,|rg>,|rr>}."""
    system = model.system
    protocol_free = SweepProtocol()
    protocol_addr = SweepProtocol(addressing={0: LOCAL_DETUNING},
                                   scatter_rate=LOCAL_SCATTER)
    x_sweep = [
        DELTA_START / system.rabi_eff,
        DELTA_END / system.rabi_eff,
        T_GATE / system.time_scale,
    ]

    joint_keys = RYDBERG_KEYS

    # (a) Global sweep -- no pinning
    result1 = simulate(system, protocol_free, x_sweep, initial_state)
    pops1 = {k: model.observables.measure(f"pop_{k}", result1.psi_final) for k in joint_keys}

    # (b) Addressed sweep
    result2 = simulate(system, protocol_addr, x_sweep, initial_state)
    pops2 = {k: model.observables.measure(f"pop_{k}", result2.psi_final) for k in joint_keys}

    # -- Plot --
    keys = list(pops1.keys())
    label_map = {"gg": "|gg>", "gr": "|gr>", "rg": "|rg>", "rr": "|rr>"}
    tick_labels = [label_map.get(k, f"|{k}>") for k in keys]
    vals1 = [pops1[k] for k in keys]
    vals2 = [pops2[k] for k in keys]
    x_pos = np.arange(len(keys))
    bar_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    bars1 = axes[0].bar(x_pos, vals1, color=bar_colors, edgecolor="black", lw=0.8)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(tick_labels, fontsize=11)
    axes[0].set_ylabel("Population")
    axes[0].set_title("(a) Global sweep — no pinning")
    axes[0].set_ylim(0, 1.1)
    for bar, val in zip(bars1, vals1):
        if val > 0.01:
            axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                         f"{val:.2f}", ha="center", fontsize=10)

    bars2 = axes[1].bar(x_pos, vals2, color=bar_colors, edgecolor="black", lw=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(tick_labels, fontsize=11)
    axes[1].set_title("(b) Atom A addressed — pinned to |g>")
    axes[1].set_ylim(0, 1.1)
    for bar, val in zip(bars2, vals2):
        if val > 0.01:
            axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                         f"{val:.2f}", ha="center", fontsize=10)

    fig.suptitle("Final state populations after adiabatic sweep", fontsize=13)
    fig.tight_layout()
    fig.savefig("fig2_final_populations.png", dpi=150)
    print("Saved fig2_final_populations.png")


# ======================================================================
# Figure 3: AC Stark feed-forward compensation comparison
# ======================================================================

def figure_stark_compensation(model, initial_state):
    """Compare adiabatic sweep with and without AC Stark feed-forward.

    The 420nm Blackman envelope A(t) creates a time-dependent AC Stark
    shift Delta_AC(t) = Omega_420^2 * A^2(t) / (4*|delta|) on the ground
    state. Without compensation, this parasitic detuning distorts the
    sweep. Feed-forward subtracts it from the 420nm phase.
    """
    system = model.system

    # AC Stark shift at peak amplitude: Omega_420^2 / (4*|delta|)
    ac_stark_peak = system.rabi_420 ** 2 / (4 * abs(system.Delta))
    print(f"  AC Stark shift at peak: {ac_stark_peak / (2*pi) / 1e6:.2f} MHz")

    x_sweep = [
        DELTA_START / system.rabi_eff,
        DELTA_END / system.rabi_eff,
        T_GATE / system.time_scale,
    ]
    n_points = 500
    t_gate_phys = x_sweep[2] * system.time_scale
    t_eval = np.linspace(0, t_gate_phys, n_points)

    # (a) Without compensation -- raw chirp, no Stark correction
    protocol_raw = SweepProtocol(ac_stark_shift=0.0)
    result_raw = simulate(system, protocol_raw, x_sweep, initial_state,
                          t_eval=t_eval)

    # (b) With feed-forward compensation
    protocol_comp = SweepProtocol(ac_stark_shift=ac_stark_peak)
    result_comp = simulate(system, protocol_comp, x_sweep, initial_state,
                           t_eval=t_eval)

    # Measure per-atom Rydberg populations
    obs_names = ["pop_A_r", "pop_B_r", "pop_r"]
    obs_raw = measure_trajectory(model, result_raw.states, obs_names)
    obs_comp = measure_trajectory(model, result_comp.states, obs_names)

    # Final state overlaps
    joint_keys = RYDBERG_KEYS
    pops_raw = {k: model.observables.measure(f"pop_{k}", result_raw.psi_final) for k in joint_keys}
    pops_comp = {k: model.observables.measure(f"pop_{k}", result_comp.psi_final) for k in joint_keys}

    print(f"  Without compensation: P(gg)={pops_raw['gg']:.3f}, "
          f"P(gr)={pops_raw['gr']:.3f}, P(rg)={pops_raw['rg']:.3f}, "
          f"P(rr)={pops_raw['rr']:.3f}")
    print(f"  With compensation:    P(gg)={pops_comp['gg']:.3f}, "
          f"P(gr)={pops_comp['gr']:.3f}, P(rg)={pops_comp['rg']:.3f}, "
          f"P(rr)={pops_comp['rr']:.3f}")

    # -- Plot (2 rows x 2 cols) --
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    t_us = result_raw.times * 1e6

    # Row 0: Rydberg populations during sweep
    for col, (obs, label) in enumerate([
        (obs_raw, "(a) Without compensation"),
        (obs_comp, "(b) With feed-forward compensation"),
    ]):
        ax = axes[0, col]
        ax.plot(t_us, obs["pop_A_r"], color="#2ca02c", ls="-", lw=1.5,
                label=r"$P_r^{(A)}$")
        ax.plot(t_us, obs["pop_B_r"], color="#1f77b4", ls="--", lw=1.5,
                label=r"$P_r^{(B)}$")
        ax.plot(t_us, obs["pop_r"] / 2, color="gray", ls=":", lw=1.2,
                label=r"$\langle n_r \rangle / 2$")
        ax.set_xlabel(r"Time ($\mu$s)")
        ax.set_ylabel("Population")
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(-0.05, 1.05)

    # Row 1: final state bar charts
    bar_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    x_pos = np.arange(len(joint_keys))
    tick_labels = [f"|{k}>" for k in joint_keys]

    for col, (pops, label) in enumerate([
        (pops_raw, "(c) Final pops — no compensation"),
        (pops_comp, "(d) Final pops — compensated"),
    ]):
        ax = axes[1, col]
        vals = [pops[k] for k in joint_keys]
        bars = ax.bar(x_pos, vals, color=bar_colors, edgecolor="black", lw=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tick_labels, fontsize=11)
        ax.set_ylabel("Population")
        ax.set_title(label)
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, vals):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                        f"{val:.3f}", ha="center", fontsize=9)

    fig.suptitle(
        r"AC Stark feed-forward compensation  "
        f"($\\Delta_{{AC}}^{{peak}}$ = {ac_stark_peak/(2*pi)/1e6:.1f} MHz,  "
        f"$\\Omega_{{eff}}$ = {system.rabi_eff/(2*pi)/1e6:.1f} MHz)",
        fontsize=13)
    fig.tight_layout()
    fig.savefig("fig3_stark_compensation.png", dpi=150)
    print("Saved fig3_stark_compensation.png")


# ======================================================================
# Figure 4: Decay channels comparison
# ======================================================================

def figure_decay_comparison(initial_state):
    """Compare adiabatic sweep with different decay channel configurations.

    Four scenarios:
      (a) No decay -- ideal unitary evolution
      (b) Intermediate-state decay only (Gamma_e ~ 1/110 ns)
      (c) Rydberg decay only (Gamma_r ~ 1/152 us)
      (d) Both decays enabled

    Non-Hermitian Hamiltonian: H -> H - i*Gamma/2 * |level><level|
    Norm loss tracks population leaked out of the 3-level subspace.
    """
    configs = [
        {"label": "(a) No decay (unitary)",
         "kwargs": {}},
        {"label": r"(b) Intermediate decay ($\Gamma_e$)",
         "kwargs": {"enable_intermediate_decay": True}},
        {"label": r"(c) Rydberg decay ($\Gamma_r$)",
         "kwargs": {"enable_rydberg_decay": True}},
        {"label": "(d) Both decays",
         "kwargs": {"enable_intermediate_decay": True,
                    "enable_rydberg_decay": True}},
    ]

    n_points = 500
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    summary = []
    for idx, cfg in enumerate(configs):
        model = Analog3LevelModel.from_defaults(detuning_sign=1, **cfg["kwargs"])
        system = model.system

        x_sweep = [
            DELTA_START / system.rabi_eff,
            DELTA_END / system.rabi_eff,
            T_GATE / system.time_scale,
        ]
        t_gate_phys = x_sweep[2] * system.time_scale
        t_eval = np.linspace(0, t_gate_phys, n_points)

        result = simulate(system, SweepProtocol(), x_sweep, initial_state,
                          t_eval=t_eval)

        obs = measure_trajectory(model, result.states,
                                 ["pop_A_r", "pop_B_r", "pop_r"])
        norms = np.array([norm_squared(result.states[:, k])
                          for k in range(result.states.shape[1])])

        pops = {k: model.observables.measure(f"pop_{k}", result.psi_final)
                for k in RYDBERG_KEYS}
        final_norm = norm_squared(result.psi_final)

        summary.append({
            "label": cfg["label"],
            "pops": pops,
            "norm": final_norm,
        })

        row, col = divmod(idx, 2)
        ax = axes[row, col]
        t_us = result.times * 1e6

        ax.plot(t_us, obs["pop_A_r"], color="#2ca02c", ls="-", lw=1.5,
                label=r"$P_r^{(A)}$")
        ax.plot(t_us, obs["pop_B_r"], color="#1f77b4", ls="--", lw=1.5,
                label=r"$P_r^{(B)}$")
        ax.plot(t_us, norms, color="gray", ls=":", lw=1.2,
                label="Norm")
        ax.set_xlabel(r"Time ($\mu$s)")
        ax.set_ylabel("Population")
        ax.set_title(cfg["label"])
        ax.legend(loc="upper left", fontsize=7)
        ax.set_ylim(-0.05, 1.05)

        # Annotate final populations
        txt = (f"|gr>={pops['gr']:.3f}  |rg>={pops['rg']:.3f}\n"
               f"Norm={final_norm:.4f}")
        ax.text(0.98, 0.55, txt, transform=ax.transAxes,
                fontsize=7, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8))

    fig.suptitle(
        "Effect of spontaneous decay on adiabatic preparation\n"
        r"($\Gamma_e^{-1}=110\,$ns intermediate,  "
        r"$\Gamma_r^{-1}=152\,\mu$s Rydberg,  "
        f"$T_{{gate}}$={T_GATE*1e6:.1f} $\\mu$s)",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("fig4_decay_comparison.png", dpi=150)
    print("Saved fig4_decay_comparison.png")

    # Print summary table
    print("\n  Decay channel comparison (final state):")
    print(f"  {'Config':<38s} {'P(gr)':>7s} {'P(rg)':>7s} "
          f"{'P(rr)':>7s} {'Norm':>7s} {'Lost':>7s}")
    for s in summary:
        p = s["pops"]
        lost = 1.0 - s["norm"]
        print(f"  {s['label']:<38s} {p['gr']:7.4f} {p['rg']:7.4f} "
              f"{p['rr']:7.4f} {s['norm']:7.4f} {lost:7.4f}")


# ======================================================================
# Figure 5: Two-atom Landau-Zener / Quantum Kibble-Zurek
# ======================================================================

def _fit_lz_envelope(sweep_rates, p_defect, Omega_eff):
    """Fit P = exp(-C * Omega_eff^2 / v) to the data."""
    from scipy.optimize import curve_fit
    mask = (p_defect > 0.005) & (p_defect < 0.95)

    def lz_model(v, C):
        return np.exp(np.clip(-C * Omega_eff ** 2 / v, -500, 0))

    if mask.sum() >= 3:
        try:
            popt, pcov = curve_fit(lz_model, sweep_rates[mask], p_defect[mask],
                                   p0=[1.0], maxfev=5000)
            C_err = float(np.sqrt(pcov[0, 0])) if pcov[0, 0] > 0 else 0.0
            return popt[0], C_err
        except RuntimeError:
            pass
    return np.pi, 0.0


def figure_landau_zener(model, initial_state):
    """Two-atom Landau-Zener with optimized sweep range (+-20 Omega_eff).

    Uses the 3-level Raman simulation (square pulse) with sweep bounds
    wide enough to converge C to pi within ~1%.

    Subfigures:
      (a) P_defect vs sweep rate with LZ fit and theory
      (b) Dynamics at three representative sweep rates
      (c) Energy diagram of the avoided crossing
      (d) C_fit with error bar vs theoretical pi
    """
    system = model.system
    Omega_eff = system.rabi_eff

    # Use +-20 Omega_eff sweep range (converged per issue #41 diagnosis)
    ds = -20 * Omega_eff
    de = 20 * Omega_eff
    delta_range = de - ds

    t_gates = np.logspace(np.log10(0.05e-6), np.log10(3e-6), 25)
    sweep_rates = delta_range / t_gates

    print("  Scanning 25 sweep rates (3-level, +-20 Omega_eff)...")
    p_defect = np.empty(len(t_gates))
    p_af = np.empty(len(t_gates))
    for i, t_g in enumerate(t_gates):
        x = [ds / Omega_eff, de / Omega_eff, t_g / system.time_scale]
        result = simulate(system, SweepProtocol(), x, initial_state)
        psi_f = result.psi_final
        p_defect[i] = model.observables.measure("pop_gg", psi_f)
        p_af[i] = (model.observables.measure("pop_gr", psi_f)
                   + model.observables.measure("pop_rg", psi_f))

    C_fit, C_err = _fit_lz_envelope(sweep_rates, p_defect, Omega_eff)
    C_rel_err = abs(C_fit - np.pi) / np.pi * 100

    def lz_curve(v, C):
        return np.exp(np.clip(-C * Omega_eff ** 2 / v, -500, 0))

    v_fine = np.logspace(np.log10(sweep_rates.min()),
                         np.log10(sweep_rates.max()), 200)
    t_fine_us = (delta_range / v_fine) * 1e6

    print(f"  C_fit = {C_fit:.3f} +/- {C_err:.3f}  "
          f"(theory pi = {np.pi:.3f}, error = {C_rel_err:.1f}%)")

    # Three representative dynamics
    dynamics_cases = [
        (0.1e-6, "Fast (100 ns)", "#d62728"),
        (0.5e-6, "Medium (500 ns)", "#ff7f0e"),
        (2.0e-6, "Slow (2 $\\mu$s)", "#2ca02c"),
    ]

    # -- Plot (2x2) --
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) P_defect vs T_gate with LZ fit and theory
    ax = axes[0, 0]
    ax.plot(t_gates * 1e6, p_defect, 'ko-', ms=4, lw=0.8, label="3-level simulation")
    ax.plot(t_fine_us, lz_curve(v_fine, C_fit), 'r-', lw=1.5, alpha=0.7,
            label=f"LZ fit $C={C_fit:.2f}$")
    ax.plot(t_fine_us, lz_curve(v_fine, np.pi), 'b--', lw=1.2, alpha=0.5,
            label=r"Theory $C=\pi$")
    ax.set_xlabel(r"$T_{\rm gate}$ ($\mu$s)")
    ax.set_ylabel(r"$P_{\rm defect} = P(|gg\rangle)$")
    ax.set_title(r"(a) Defect probability — LZ fit vs theory")
    ax.legend(fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, max(0.4, p_defect.max() * 1.3))

    # (b) Dynamics at three sweep rates
    ax = axes[0, 1]
    for t_g, case_label, color in dynamics_cases:
        x = [ds / Omega_eff, de / Omega_eff, t_g / system.time_scale]
        t_eval = np.linspace(0, t_g * (1 - 1e-10), 400)
        result = simulate(system, SweepProtocol(), x, initial_state, t_eval=t_eval)
        obs = measure_trajectory(model, result.states, ["pop_A_r"])
        ax.plot(result.times / t_g, obs["pop_A_r"], color=color, lw=1.2,
                label=case_label)
    ax.set_xlabel(r"Normalized time $t / T_{\rm gate}$")
    ax.set_ylabel(r"$P_r^{(A)}$")
    ax.set_title("(b) Rydberg population dynamics")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 0.65)

    # (c) Energy diagram
    ax = axes[1, 0]
    t_norm = np.linspace(0, 1, 200)
    delta_mhz = (ds + delta_range * t_norm) / (2 * pi * 1e6)
    gap_mhz = Omega_eff / (2 * pi * 1e6)
    E_plus = np.sqrt(delta_mhz ** 2 + gap_mhz ** 2) / 2
    ax.plot(t_norm, E_plus, 'b-', lw=1.5, label="Upper adiabatic")
    ax.plot(t_norm, -E_plus, 'r-', lw=1.5, label="Lower adiabatic")
    ax.plot(t_norm, delta_mhz / 2, 'k--', lw=0.8, alpha=0.5,
            label=r"$|gg\rangle$ diabatic")
    ax.plot(t_norm, -delta_mhz / 2, 'k:', lw=0.8, alpha=0.5,
            label=r"$|W\rangle$ diabatic")
    ax.annotate(f"Gap = {gap_mhz:.1f} MHz",
                xy=(0.5, 0), xytext=(0.65, gap_mhz * 1.5),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"),
                ha="center")
    ax.set_xlabel(r"Normalized time $t / T_{\rm gate}$")
    ax.set_ylabel("Energy (MHz)")
    ax.set_title(r"(c) Avoided crossing ($\Delta=0$ at midpoint)")
    ax.legend(fontsize=7, loc="upper left")

    # (d) C_fit with error bar vs pi
    ax = axes[1, 1]
    ax.errorbar(1, C_fit, yerr=C_err, fmt='ko', ms=10, capsize=8,
                capthick=2, elinewidth=2, label=f"$C_{{fit}} = {C_fit:.3f} \\pm {C_err:.3f}$")
    ax.axhline(np.pi, color="red", ls="--", lw=2,
               label=f"$\\pi = {np.pi:.3f}$")
    ax.fill_between([0.5, 1.5], np.pi * 0.99, np.pi * 1.01,
                    color="red", alpha=0.1, label=r"$\pm 1\%$ band")
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(C_fit - max(0.5, 3 * C_err), C_fit + max(0.5, 3 * C_err))
    ax.set_xticks([1])
    ax.set_xticklabels(["3-level\nsimulation"], fontsize=10)
    ax.set_ylabel("Fitted $C$")
    ax.set_title(f"(d) $C$ vs theory  (error = {C_rel_err:.1f}%)")
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        r"Two-atom Landau-Zener: $P_{\rm defect} = \exp(-C\,\Omega_{\rm eff}^2 / v)$"
        "\n"
        f"$\\Omega_{{\\rm eff}}/(2\\pi)$ = {Omega_eff/(2*pi)/1e6:.1f} MHz,  "
        f"Sweep: $\\pm 20\\,\\Omega_{{\\rm eff}}$,  "
        "square pulse (no Blackman)",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig("fig5_landau_zener.png", dpi=150)
    print("Saved fig5_landau_zener.png")


# ======================================================================

def main():
    initial_state = PRODUCT_STATES["gg"]

    model_no_decay = Analog3LevelModel.from_defaults(detuning_sign=1)
    figure_rabi_dynamics(model_no_decay, initial_state)
    figure_final_populations(model_no_decay, initial_state)
    figure_stark_compensation(model_no_decay, initial_state)
    figure_decay_comparison(initial_state)

    model_sq = Analog3LevelModel.from_defaults(detuning_sign=1, blackmanflag=False)
    figure_landau_zener(model_sq, initial_state)
    plt.show()


if __name__ == "__main__":
    main()
