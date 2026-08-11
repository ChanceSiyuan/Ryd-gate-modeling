"""Transfer test: do the 3-atom-optimized global ZXZ pulses transfer to larger lattices?

Applies the best stored direct pulses (ZOH midpoint waveform, exactly as the
validate gate replays them) to N-atom 1D chains and a 3x3 square lattice via
simulate(backend="exact_ode"), and compares against exp(-i*0.8*H_target):

- 1D chain: H_ZXZ = sum_{j bulk} Z_{j-1} X_j Z_{j+1}
- 2D 3x3:   (a) rows-ZXZ (one bulk triple per row); (b) 2D cluster
            H = sum_j X_j prod_{k in NN(j)} Z_k (all sites, NN by distance)

Metrics per case: full unitary fidelity |Tr(V^H U)|^2/D^2 when dim <= 64
(all basis columns evolved); otherwise mean per-column state fidelity over
the all-ground column plus M random basis columns; always the ground-state
fidelity and the <Z_i> profile deviation at final time.

One-off analysis script (2026-07-29); artifacts in results/zxz_direct_qoc/transfer/.
Run on the DGX with HOME=/tmp/arc297home (ARC).
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zxz_direct_qoc import DT_US, RESULTS_DIR, RYD_LEVEL, SPACING_UM, TAU_JEFF  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from ryd_gate import Register, RydbergSystem, level_structure, simulate  # noqa: E402
from ryd_gate.protocols import SweepProtocol  # noqa: E402

OUT_DIR = RESULTS_DIR / "transfer"
_Z = np.diag([1.0, -1.0]).astype(complex)  # |1> -> +1, |r> -> -1
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def load_best(tag):
    summary = json.loads((RESULTS_DIR / f"direct_{tag}_summary.json").read_text())
    seed = int(summary["best"]["seed"])
    data = np.load(RESULTS_DIR / f"direct_{tag}_seed{seed}.npz")
    return data["u_omega"], data["u_delta"], int(data["K"]), seed


def zoh_protocol(u_om, u_de, k):
    t_total_us = k * DT_US
    mid_om = 0.5 * (u_om[:-1] + u_om[1:])
    mid_de = 0.5 * (u_de[:-1] + u_de[1:])

    def at(t_s, mids):
        t_us = min(max(t_s * 1e6, 0.0), t_total_us - 1e-12)
        return float(mids[int(t_us / DT_US)])

    return SweepProtocol(
        t_gate_s=t_total_us * 1e-6,
        omega_half_rad_s=lambda t: at(t, mid_om) * 1e6,
        detuning_rad_s=lambda t: -at(t, mid_de) * 1e6,
    )


def all_labels(n):
    return [list(p) for p in product("1r", repeat=n)]


def op_site(n, site, mat):
    out = np.array([[1.0 + 0.0j]])
    for i in range(n):
        out = np.kron(out, mat if i == site else np.eye(2, dtype=complex))
    return out


def nn_pairs(register):
    coords = register.coords
    n = register.N
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(coords[i] - coords[j]) < 1.15 * SPACING_UM:
                pairs.append((i, j))
    return pairs


def chain_triples(register):
    """Bulk (j-1, j, j+1) triples along x (per row for a grid)."""
    coords = register.coords
    rows = {}
    for i in range(register.N):
        rows.setdefault(round(coords[i][1], 6), []).append(i)
    triples = []
    for _, sites in sorted(rows.items()):
        ordered = sorted(sites, key=lambda i: coords[i][0])
        for a, b, c in zip(ordered[:-2], ordered[1:-1], ordered[2:]):
            triples.append((a, b, c))
    return triples


def h_rows_zxz(register):
    n = register.N
    h = np.zeros((2**n, 2**n), dtype=complex)
    for a, b, c in chain_triples(register):
        h += op_site(n, a, _Z) @ op_site(n, b, _X) @ op_site(n, c, _Z)
    return h


def h_cluster(register):
    n = register.N
    h = np.zeros((2**n, 2**n), dtype=complex)
    neighbors = {i: [] for i in range(n)}
    for i, j in nn_pairs(register):
        neighbors[i].append(j)
        neighbors[j].append(i)
    for j in range(n):
        term = op_site(n, j, _X)
        for k in neighbors[j]:
            term = term @ op_site(n, k, _Z)
        h += term
    return h


def evolve_columns(register, protocol, column_labels):
    system = RydbergSystem(
        level_structure=level_structure("1r", ryd_level=RYD_LEVEL),
        register=register,
        protocol=protocol,
    )
    results = simulate(system, column_labels, backend="exact_ode")
    labels = all_labels(register.N)
    out = np.zeros((len(labels), len(column_labels)), dtype=complex)
    for j, res in enumerate(results):
        out[:, j] = [res.amplitude(lab) for lab in labels]
    return out


def z_profile(psi, n):
    probs = np.abs(psi) ** 2
    tensor = probs.reshape([2] * n)
    out = []
    for site in range(n):
        p = np.moveaxis(tensor, site, 0).reshape(2, -1).sum(axis=1)
        out.append(float(p[0] - p[1]))  # index 0 = "1", 1 = "r"
    return np.array(out)


def column_index(labels_flat):
    idx = 0
    for lab in labels_flat:
        idx = 2 * idx + (1 if lab == "r" else 0)
    return idx


def run_case(name, register, protocol, h_targets, n_sample, rng):
    """Evolve columns once, compare against each named target Hamiltonian."""
    n = register.N
    dim = 2**n
    full = dim <= 64
    ground = ["1"] * n
    if full:
        cols = all_labels(n)
    else:
        picks = rng.choice(dim, size=min(n_sample, dim - 1), replace=False)
        cols = [ground] + [all_labels(n)[int(p)] for p in picks if int(p) != 0]
    u_cols = evolve_columns(register, protocol, cols)
    col_idx = [column_index(c) for c in cols]

    out = {}
    for target_name, h_t in h_targets.items():
        v = expm(-1j * TAU_JEFF * h_t)
        v_cols = v[:, col_idx]
        overlaps = np.array([np.vdot(v_cols[:, j], u_cols[:, j]) for j in range(len(cols))])
        f_ground = float(abs(overlaps[0]) ** 2) if not full else float(abs(overlaps[col_idx.index(0)]) ** 2)
        f_cols_mean = float(np.mean(np.abs(overlaps) ** 2))
        f_unitary = float(abs(np.sum(overlaps)) ** 2 / dim**2) if full else np.nan
        g_col = col_idx.index(0)
        z_pulse = z_profile(u_cols[:, g_col], n)
        z_target = z_profile(v_cols[:, g_col], n)
        out[target_name] = {
            "f_unitary": f_unitary,
            "f_ground": f_ground,
            "f_cols_mean": f_cols_mean,
            "z_pulse": z_pulse,
            "z_target": z_target,
            "z_maxdev": float(np.max(np.abs(z_pulse - z_target))),
            "n_cols": len(cols),
            "full_unitary": full,
        }
        fu = f"{f_unitary:.4f}" if full else "  -   "
        print(
            f"[{name} vs {target_name}] N={n} F_unitary={fu} "
            f"F_ground={out[target_name]['f_ground']:.4f} "
            f"F_cols({len(cols)})={f_cols_mean:.4f} dZ_max={out[target_name]['z_maxdev']:.3f}"
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the whole ladder (default: N=3 sanity only)")
    parser.add_argument("--n-sample", type=int, default=32)
    args = parser.parse_args()
    rng = np.random.default_rng(20260729)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pulses = {}
    for tag in ("pulse1", "pulse2"):
        u_om, u_de, k, seed = load_best(tag)
        pulses[tag] = (u_om, u_de, k, seed)
        print(f"loaded {tag}: seed {seed}, K={k}")

    # sanity gate: N=3 must reproduce the validate baseline
    baselines = {}
    for tag, (u_om, u_de, k, _seed) in pulses.items():
        case = run_case(
            f"{tag} chain", Register.chain(3, spacing_um=SPACING_UM),
            zoh_protocol(u_om, u_de, k), {"zxz": h_rows_zxz(Register.chain(3, spacing_um=SPACING_UM))},
            args.n_sample, rng,
        )
        baselines[tag] = case["zxz"]["f_unitary"]
        ref = np.load(RESULTS_DIR / f"validate_{tag}.npz")["f_ode_zoh"]
        if abs(case["zxz"]["f_unitary"] - float(ref)) > 1e-3:
            sys.exit(f"SANITY FAIL: {tag} N=3 F={case['zxz']['f_unitary']:.5f} vs validate {float(ref):.5f}")
        print(f"sanity OK: {tag} N=3 matches validate ({baselines[tag]:.5f})")
    if not args.full:
        return

    records = {}
    for tag, (u_om, u_de, k, seed) in pulses.items():
        protocol = zoh_protocol(u_om, u_de, k)
        for n in (4, 5, 6, 8, 10):
            reg = Register.chain(n, spacing_um=SPACING_UM)
            case = run_case(f"{tag} chain", reg, protocol, {"zxz": h_rows_zxz(reg)}, args.n_sample, rng)
            records[f"{tag}_chain{n}"] = case
        reg = Register.square(3, spacing_um=SPACING_UM)
        case = run_case(
            f"{tag} 3x3", reg, protocol,
            {"rows_zxz": h_rows_zxz(reg), "cluster2d": h_cluster(reg)},
            args.n_sample, rng,
        )
        records[f"{tag}_square3"] = case

    flat = {}
    for key, case in records.items():
        for target_name, metrics in case.items():
            for field, value in metrics.items():
                flat[f"{key}__{target_name}__{field}"] = value
    np.savez(OUT_DIR / "transfer_metrics.npz", **flat)
    print(f"wrote {OUT_DIR / 'transfer_metrics.npz'}")


if __name__ == "__main__":
    main()
