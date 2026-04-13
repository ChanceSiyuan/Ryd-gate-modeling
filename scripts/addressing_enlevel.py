# scripts/addressing_enlevel.py

"""
This script plots the two-atom Rydberg energy levels as a function of detuning.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations


O420  = 135    # Ω₄₂₀/(2π) [MHz]      (range: 10–500)
O1013 = 135    # Ω₁₀₁₃/(2π) [MHz]     (range: 10–500)
Dint  = 2.4    # Δ_int/(2π) [GHz]      (range: 0.5–15)
dA    = 10     # δ_A/(2π) [MHz]        (range: 0–80, pinning)
d420  = 0      # δ₄₂₀/(2π) [MHz]      (range: -40–40, 420 phase shift)
Vr    = 213    # V_ryd/(2π) [MHz]      (range: 0–500, interaction, calculated from 874e3/4**6 for 4µm separation and Rb87 70S state)
sh    = 40     # Sweep half-range [MHz] (range: 5–100)
cd    = 0      # Current Δ/(2π) [MHz]  (range: -sh to sh)

# ── Physics ──────────────────────────────────────────────────────
Oeff = O420 * O1013 / (2 * Dint * 1e3)
dtot = dA + d420
rr_cross = dtot + Vr
N = 500

def build_H2_vec(delta_arr, Oeff, dtot, Vr):
    N = len(delta_arr)
    H = np.zeros((N, 4, 4))
    h = Oeff / 2
    H[:, 0, 1] = H[:, 1, 0] = h
    H[:, 0, 2] = H[:, 2, 0] = h
    H[:, 1, 3] = H[:, 3, 1] = h
    H[:, 2, 3] = H[:, 3, 2] = h
    H[:, 1, 1] = -delta_arr
    H[:, 2, 2] = -delta_arr + dtot
    H[:, 3, 3] = -2 * delta_arr + dtot + Vr
    return H


# ── 2-Atom (vectorized eigvalsh) ──
Delta2 = np.linspace(-sh, sh, N)
E_gg, E_gr, E_rg = np.zeros(N), -Delta2, -Delta2 + dtot
E_rr_arr = -2 * Delta2 + dtot + Vr
adiab = np.linalg.eigvalsh(build_H2_vec(Delta2, Oeff, dtot, Vr))
marker2y = np.linalg.eigvalsh(build_H2_vec(np.array([cd]), Oeff, dtot, Vr)[0])[0]


def nearest_diabatic_labels(adiab_y, diab_y, labels):
    """Assign each adiabatic branch the nearest diabatic label."""
    best_perm = min(
        permutations(range(len(labels))),
        key=lambda perm: sum(abs(adiab_y[i] - diab_y[perm[i]]) for i in range(len(labels))),
    )
    return [labels[j] for j in best_perm]


# ── Plot ──
fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
fig.suptitle(
    f'$\\Omega_{{\\mathrm{{eff}}}}/(2\\pi) = {Oeff:.2f}\\ \\mathrm{{MHz}}$    '
    f'$\\delta_{{\\mathrm{{1}}}}/(2\\pi) = {dtot:.1f}$    '
    f'$|rr\\rangle\\ \\mathrm{{crossing}}:\\ \\Delta = {rr_cross:.0f}\\ \\mathrm{{MHz}}$',
    fontsize=11, y=0.98
)

# Two atoms
dc = ['steelblue', 'seagreen', 'orange', 'crimson']
dy = [E_gg, E_gr, E_rg, E_rr_arr]
ac = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
ax.axvspan(-sh, sh, color='lightgray', alpha=0.15)
ax.axvline(cd, color='gray', ls=':', lw=1)
for j in range(3):
    ax.plot(Delta2, adiab[:, j], color=ac[j], lw=2.5, zorder=2)
for j in range(3):
    ax.plot(Delta2, dy[j], '--', color=dc[j], lw=1.5, alpha=0.7, zorder=3)
ax.plot(cd, marker2y, 'o', ms=12, color='black', mec='yellow', mew=2)
ax.set(
    xlabel=r'Detuning $\Delta(t)/(2\pi)$ [MHz]',
    ylabel=r'Energy / $(2\pi)$ [MHz]',
    title='Adiabatic energy level diagram of addressed two Rydberg atoms',
)

diab_labels = ['gg', 'gr', 'rg']
left_labels = nearest_diabatic_labels(adiab[0, :3], [E_gg[0], E_gr[0], E_rg[0]], diab_labels)
right_labels = nearest_diabatic_labels(adiab[-1, :3], [E_gg[-1], E_gr[-1], E_rg[-1]], diab_labels)
x_pad = 1.2
for j, label in enumerate(left_labels):
    ax.text(
        Delta2[0] + x_pad, adiab[0, j], label,
        color=ac[j], fontsize=10, ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85),
    )
for j, label in enumerate(right_labels):
    ax.text(
        Delta2[-1] - x_pad, adiab[-1, j], label,
        color=ac[j], fontsize=10, ha='right', va='center',
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85),
    )

fig.tight_layout()
plt.savefig("docs/figures/addressing_enlevel.png")