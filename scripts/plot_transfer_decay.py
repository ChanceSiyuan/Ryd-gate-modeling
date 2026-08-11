"""Plot the transfer-test fidelity decay curves (companion to zxz_transfer_test.py).

Reads results/zxz_direct_qoc/transfer/transfer_metrics.npz and draws
(a) fidelity vs chain length N (log y) with the 3x3 2D points for contrast,
(b) the <Z_i> max profile deviation vs N (the local-observable robustness).
N=3 baselines are the validate-gate numbers printed by the sanity gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zxz_direct_qoc import RESULTS_DIR  # noqa: E402

TRANSFER = RESULTS_DIR / "transfer"
DATA = np.load(TRANSFER / "transfer_metrics.npz")

# N=3 baselines from the sanity gate (== validate f_ode_zoh; see zxz_transfer log)
BASE = {
    "pulse1": {"f_unitary": 0.92757, "f_ground": 0.8977, "f_cols_mean": 0.9313, "z_maxdev": 0.309},
    "pulse2": {"f_unitary": 0.99219, "f_ground": 1.0000, "f_cols_mean": 0.9948, "z_maxdev": 0.000},
}
CHAIN_N = [4, 5, 6, 8, 10]
STYLE = {"pulse1": ("tab:blue", "pulse1 (T=1.2 us)"), "pulse2": ("tab:red", "pulse2 (T=3.6 us)")}


def series(tag, field):
    ns = [3] + CHAIN_N
    vals = [BASE[tag][field]] + [
        float(DATA[f"{tag}_chain{n}__zxz__{field}"]) for n in CHAIN_N
    ]
    return np.array(ns), np.array(vals)


fig, (ax_f, ax_z) = plt.subplots(1, 2, figsize=(10.0, 4.2))

for tag, (color, label) in STYLE.items():
    ns, f_ground = series(tag, "f_ground")
    _, f_cols = series(tag, "f_cols_mean")
    _, f_unit = series(tag, "f_unitary")
    ax_f.plot(ns, f_ground, "o-", color=color, label=f"{label}: F_ground")
    ax_f.plot(ns, f_cols, "s--", color=color, alpha=0.6, label=f"{label}: F_basis-avg")
    have_unit = ~np.isnan(f_unit)
    ax_f.plot(ns[have_unit], f_unit[have_unit], "*", color=color, markersize=13,
              linestyle="none", label=f"{label}: F_unitary (N<=6)")

# geometric-decay guide fitted to pulse1 F_ground between N=3 and N=10
ns_p1, fg_p1 = series("pulse1", "f_ground")
rate = (fg_p1[-1] / fg_p1[0]) ** (1.0 / (ns_p1[-1] - ns_p1[0]))
guide_n = np.linspace(3, 10, 50)
ax_f.plot(guide_n, fg_p1[0] * rate ** (guide_n - 3), ":", color="gray",
          label=f"geometric guide ({rate:.3f}/atom)")

# 3x3 2D contrast points at N=9
for tag, (color, _label) in STYLE.items():
    for target, marker in (("rows_zxz", "x"), ("cluster2d", "+")):
        val = float(DATA[f"{tag}_square3__{target}__f_cols_mean"])
        ax_f.plot([9], [val], marker, color=color, markersize=10, markeredgewidth=2.5)
ax_f.annotate("3x3 2D lattice\n(x rows-ZXZ, + cluster)", xy=(8.85, 3.5e-3), xytext=(3.1, 7e-4),
              fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))

ax_f.set_yscale("log")
ax_f.set_ylim(2e-4, 1.4)
ax_f.set_xlabel("number of atoms N")
ax_f.set_ylabel("fidelity vs exp(-i*0.8*H_ZXZ)")
ax_f.set_title("Transfer of the 3-atom-optimized global pulse")
ax_f.grid(alpha=0.25, which="both")

for tag, (color, label) in STYLE.items():
    ns, dz = series(tag, "z_maxdev")
    ax_z.plot(ns, dz, "o-", color=color, label=label)
ax_z.set_xlabel("number of atoms N")
ax_z.set_ylabel(r"max$_i$ |<Z$_i$>$_{pulse}$ - <Z$_i$>$_{target}$|")
ax_z.set_title("Local-observable deviation (ground-state evolution)")
ax_z.set_ylim(-0.05, 1.25)
ax_z.legend(fontsize=8)
ax_z.grid(alpha=0.25)

handles, labels = ax_f.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, framealpha=0.9)
fig.tight_layout(rect=[0.0, 0.09, 1.0, 1.0])
for ext in ("png", "pdf"):
    fig.savefig(TRANSFER / f"transfer_decay.{ext}", dpi=200)
print(f"wrote {TRANSFER}/transfer_decay.png|pdf")
