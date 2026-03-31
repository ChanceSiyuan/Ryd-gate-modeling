#!/usr/bin/env python
"""Visualize the breakdown of Rydberg blockade via local addressing.

Produces two figures:

  Figure 1 — Rabi dynamics (resonant driving, no chirp):
    (a) Global excitation: both atoms oscillate collectively at √2·Ω_eff
        with P_r max ≈ 0.5 due to Rydberg blockade.
    (b) Atom A addressed: P_r^(A) = 0, P_r^(B) oscillates at Ω_eff
        reaching ~1.0 (no blockade). Dashed norm shows scattering decay.

  Figure 2 — Final state populations (adiabatic sweep):
    (a) Global sweep: ~50/50 in |gr⟩ and |rg⟩, |rr⟩ ≈ 0.
    (b) Atom A addressed: ~100% in |gr⟩.
"""

import matplotlib.pyplot as plt
import numpy as np

from scipy.constants import pi

from ryd_gate.core.atomic_system import (
    build_atom_a_projector,
    build_atom_b_projector,
    build_sss_state_map,
    create_analog_system,
)
from ryd_gate.protocols.sweep import SweepProtocol
from ryd_gate.solvers.schrodinger import solve_gate

N = 3  # 3-level system: |g⟩=0, |e⟩=1, |r⟩=2

# Pinning parameters scaled for Ω_eff ≈ 13 MHz (equal-Rabi regime).
# Need |local_detuning| >> Ω_eff for clean pinning.
LOCAL_DETUNING = -2 * pi * 50e6    # rad/s  (AC Stark shift on |r⟩_A)
LOCAL_SCATTER = 150.0              # Hz     (scales with laser power)

# Sweep range must satisfy |δ| >> Ω_eff for adiabatic passage.
DELTA_START = -2 * pi * 40e6       # rad/s
DELTA_END = 2 * pi * 40e6          # rad/s
T_GATE = 1.5e-6                    # s


def _build_joint_projectors():
    """Build |gg⟩, |gr⟩, |rg⟩, |rr⟩ two-atom projectors."""
    sg = np.zeros(N, dtype=complex); sg[0] = 1.0
    sr = np.zeros(N, dtype=complex); sr[2] = 1.0
    states = {
        "gg": np.kron(sg, sg),
        "gr": np.kron(sg, sr),
        "rg": np.kron(sr, sg),
        "rr": np.kron(sr, sr),
    }
    return {k: np.outer(v, v.conj()) for k, v in states.items()}


def _evolve(system, protocol, x, initial_state, n_points=500,
            ham_const_override=None):
    """Evolve and return time array + state trajectory."""
    params = protocol.unpack_params(x, system)
    t_gate = params["t_gate"]
    t_eval = np.linspace(0, t_gate, n_points)
    psi_traj = solve_gate(
        system, protocol, x, initial_state,
        t_eval=t_eval, ham_const_override=ham_const_override,
    )
    return t_eval, psi_traj


def _rydberg_pops(psi_traj):
    """Extract P_r^(A) and P_r^(B) from trajectory."""
    proj_rA = build_atom_a_projector(2, N)
    proj_rB = build_atom_b_projector(2, N)
    n_t = psi_traj.shape[1]
    prA = np.zeros(n_t)
    prB = np.zeros(n_t)
    for k in range(n_t):
        psi = psi_traj[:, k]
        prA[k] = np.real(np.vdot(psi, proj_rA @ psi))
        prB[k] = np.real(np.vdot(psi, proj_rB @ psi))
    return prA, prB


def _state_norm(psi_traj):
    """Compute total state norm at each timestep."""
    return np.array([np.real(np.vdot(psi_traj[:, k], psi_traj[:, k]))
                     for k in range(psi_traj.shape[1])])


def _joint_populations(psi):
    """Compute {|gg⟩, |gr⟩, |rg⟩, |rr⟩} populations for a single state."""
    projs = _build_joint_projectors()
    return {k: np.real(np.vdot(psi, P @ psi)) for k, P in projs.items()}


# ══════════════════════════════════════════════════════════════════════
# Figure 1: Rabi dynamics (resonant driving)
# ══════════════════════════════════════════════════════════════════════

def figure_rabi_dynamics(system, initial_state):
    """Two stacked subplots: blockaded vs free Rabi oscillations."""
    n_cycles = 5
    t_rabi = n_cycles * system.time_scale
    x_rabi = [0.0, 0.0, t_rabi / system.time_scale]  # resonant, no chirp

    protocol_free = SweepProtocol()
    protocol_addr = SweepProtocol(addressing={0: LOCAL_DETUNING},
                                   scatter_rate=LOCAL_SCATTER)

    # (a) No pinning — blockade active
    t1, psi1 = _evolve(system, protocol_free, x_rabi, initial_state)
    prA1, prB1 = _rydberg_pops(psi1)

    # (b) Atom A addressed — blockade irrelevant
    ham_addr = system.tq_ham_const + protocol_addr.get_ham_const_additions()
    t2, psi2 = _evolve(system, protocol_addr, x_rabi, initial_state,
                        ham_const_override=ham_addr)
    prA2, prB2 = _rydberg_pops(psi2)
    norm2 = _state_norm(psi2)

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    t1_us, t2_us = t1 * 1e6, t2 * 1e6

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


# ══════════════════════════════════════════════════════════════════════
# Figure 2: Final state populations (adiabatic sweep)
# ══════════════════════════════════════════════════════════════════════

def figure_final_populations(system, initial_state):
    """Two side-by-side bar charts of final {|gg⟩,|gr⟩,|rg⟩,|rr⟩}."""
    protocol_free = SweepProtocol()
    protocol_addr = SweepProtocol(addressing={0: LOCAL_DETUNING},
                                   scatter_rate=LOCAL_SCATTER)
    x_sweep = [
        DELTA_START / system.rabi_eff,
        DELTA_END / system.rabi_eff,
        T_GATE / system.time_scale,
    ]

    # (a) Global sweep — no pinning
    _, psi1 = _evolve(system, protocol_free, x_sweep, initial_state)
    pops1 = _joint_populations(psi1[:, -1])

    # (b) Addressed sweep
    ham_addr = system.tq_ham_const + protocol_addr.get_ham_const_additions()
    _, psi2 = _evolve(system, protocol_addr, x_sweep, initial_state,
                       ham_const_override=ham_addr)
    pops2 = _joint_populations(psi2[:, -1])

    # ── Plot ──────────────────────────────────────────────────────
    keys = list(pops1.keys())
    tick_labels = [f"|{k}⟩" for k in keys]
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
    axes[1].set_title("(b) Atom A addressed — pinned to |g⟩")
    axes[1].set_ylim(0, 1.1)
    for bar, val in zip(bars2, vals2):
        if val > 0.01:
            axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                         f"{val:.2f}", ha="center", fontsize=10)

    fig.suptitle("Final state populations after adiabatic sweep", fontsize=13)
    fig.tight_layout()
    fig.savefig("fig2_final_populations.png", dpi=150)
    print("Saved fig2_final_populations.png")


# ══════════════════════════════════════════════════════════════════════

def main():
    system = create_analog_system(detuning_sign=1)
    initial_state = build_sss_state_map(n_levels=3)["00"]
    figure_rabi_dynamics(system, initial_state)
    figure_final_populations(system, initial_state)
    plt.show()


if __name__ == "__main__":
    main()
