"""Run the Route-2 lattice cell from plus_state_preparation.ipynb (headless)."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from itertools import product

from ryd_gate import RydbergSystem, simulate
from ryd_gate.core.rydberg_system import InteractionSpec
from ryd_gate.lattice import make_square_lattice
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol, Segment

OMEGA_R = 2 * np.pi * 200 * 1e6
OMEGA_HF = 2 * np.pi * 200 * 1e6

t_pi_R = np.pi / OMEGA_R
t_pi2_R = np.pi / (2 * OMEGA_R)
t_pi_hf = np.pi / OMEGA_HF
t_pi2_hf = np.pi / (2 * OMEGA_HF)

Lx, Ly = 3, 3
spacing = 10.0
geom = make_square_lattice(Lx, Ly, spacing_um=spacing)
N = geom.N

proto_direct = DigitalAnalogProtocol(
    [Segment(duration=t_pi2_R, omega_R=OMEGA_R)],
    n_steps=200,
)
system = RydbergSystem.from_lattice(
    geom, level_structure="01r",
    interaction=InteractionSpec(), protocol=proto_direct,
)
psi0 = system.product_state(["1"] * N)
result = simulate(system, [], psi0, t_eval=True)
times = np.concatenate([[0.0], result.times])
states = [psi0, *result.states]

levels = tuple(system.basis.local_levels)
pops_site = {
    i: {
        lvl: np.array([system.expectation(f"n_{lvl}_{i}", psi) for psi in states])
        for lvl in levels
    }
    for i in range(N)
}

psi_final = states[-1]
psi_plus = sum(
    (-1j) ** sum(c == "r" for c in cfg) * system.product_state(list(cfg))
    for cfg in product(["1", "r"], repeat=N)
) / (2 ** (N / 2))
fidelity = np.abs(np.vdot(psi_plus, psi_final)) ** 2
print(f"N={N}, fidelity to (|1⟩ - i|r⟩)/√2^{N}: {fidelity:.4f}")

out_dir = __file__.rsplit("/", 1)[0]
colors = {"0": "tab:gray", "1": "tab:blue", "r": "tab:red"}
fig, axes = plt.subplots(Ly, Lx, figsize=(3.5 * Lx, 3 * Ly), sharex=True)
for i in range(N):
    ix, iy = i // Ly, i % Ly
    ax = axes[iy, ix]
    for lvl in ("0", "1", "r"):
        ax.plot(times, pops_site[i][lvl], color=colors[lvl], lw=2, label=fr"$\langle n_{lvl}\rangle$")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_title(f"({ix}, {iy})")
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(loc="upper right", ncol=3)
for ax in axes[-1, :]:
    ax.set_xlabel("time (Rabi periods)")
fig.suptitle(r"Route 2: single $\pi/2$ Rydberg pulse ($3\times3$ lattice)")
fig.savefig(f"{out_dir}/route2_lattice_pops.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_dir}/route2_lattice_pops.png")

_proto_fields = (
    ("omega_R", r"$\Omega_R$"),
    ("omega_hf", r"$\Omega_{\rm hf}$"),
    ("delta_R", r"$\Delta_R$"),
    ("delta_hf", r"$\Delta_{\rm hf}$"),
)
_t_edges = [0.0]
_sched = {f: [] for f, _ in _proto_fields}
for seg in proto_direct.segments:
    for f, _ in _proto_fields:
        _sched[f].append(getattr(seg, f))
    _t_edges.append(_t_edges[-1] + seg.duration)
fig_p, axes_p = plt.subplots(4, 1, figsize=(12, 7), sharex=True)
t_max = max(times[-1], proto_direct._t_gate)
for ax_p, (field, label) in zip(axes_p, _proto_fields):
    t_pts, y_pts = [], []
    for j, seg in enumerate(proto_direct.segments):
        v = getattr(seg, field)
        t_pts.extend([_t_edges[j], _t_edges[j + 1]])
        y_pts.extend([v, v])
    ax_p.step(t_pts, y_pts, where="post", lw=2, color="tab:blue")
    ax_p.axhline(0.0, color="k", ls=":", lw=0.8)
    ax_p.set_ylabel(label)
    ax_p.grid(alpha=0.3)
    ax_p.set_xlim(0.0, t_max)
axes_p[-1].set_xlabel("time (s)")
axes_p[-1].axvline(times[-1], color="tab:orange", ls="--", lw=1, alpha=0.8)
fig_p.suptitle("Digital–analog protocol schedule", y=1.01)
fig_p.savefig(f"{out_dir}/route2_protocol_schedule.png", dpi=120, bbox_inches="tight")
plt.close(fig_p)
print(f"Saved {out_dir}/route2_protocol_schedule.png")
