#!/usr/bin/env python3
"""Rigorous 53P3/2 pair-interaction channel structure for the 297 nm TO gate.

The calibration models the Rydberg pair interaction as diagonal channel numbers
V_cc' |cc'><cc'| (README "问题与模型").  Rigorously, second-order dipole-dipole
gives a *matrix-valued* C6 operator on the 16-dimensional degenerate
|m_J1, m_J2> manifold of the 53P3/2 pair.  This script reconstructs that matrix
from ARC's degenerate perturbation theory (same n_range / energyDelta as the
production `arc_pair_c6_rad_s_um6`), diagonalizes it alone (B = 0) and together
with the Zeeman term at the working point (B = 20 G, R = 3 um, theta = pi/2)
plus the B = 160 G comparison point, and records how the bare rr and r-r_garb
channels decompose.  Deterministic, no RNG; writes
results/297_to_calibration/pair_channels.json.

On the DGX, ARC needs the HOME=/tmp/arc297home prefix (see the zxz README).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

ROOT = Path(
    os.environ.get("RYD_GATE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, os.fspath(ROOT / "src"))

from ryd_gate.physics import zeeman_shift_rad_s  # noqa: E402

OUT = ROOT / "results" / "297_to_calibration" / "pair_channels.json"

N, L, J = 53, 1, 1.5
THETA, PHI = math.pi / 2, 0.0
R_UM = 3.0
N_RANGE, ENERGY_DELTA_HZ = 5, 30e9          # arc_pair_c6_rad_s_um6 defaults
OMEGA_MAX_MHZ = 16.614                      # to_297.json:fixed target-leg Rabi
WEAK_SHIFT_THRESHOLD_MHZ = 5.0 * OMEGA_MAX_MHZ
MJS = (-1.5, -0.5, 0.5, 1.5)
BARE_CHANNELS = {"rr": (-1.5, -1.5), "r_rgarb": (-1.5, -0.5)}
B_FIELDS_G = (20.0, 160.0)


EV_TO_GHZ = 241798.9


def assemble_pair_hamiltonian(calc, spacing_um: float):
    """Assemble ARC's sparse pair Hamiltonian at one spacing, in GHz."""
    matrix = calc.matDiagonal.copy()
    distance_m = spacing_um * 1e-6
    for power, term in enumerate(calc.matR, start=3):
        matrix = matrix + term / distance_m**power
    return matrix.tocsr()


def find_basis_state_index(basis_states, quantum_numbers) -> int:
    """Return the unique ARC basis index matching eight pair-state numbers."""
    matches = [
        i
        for i, state in enumerate(basis_states)
        if tuple(state[:8]) == tuple(quantum_numbers)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one basis state; found {len(matches)}")
    return matches[0]


def extract_local_eigenpairs(
    hamiltonian,
    reference_ghz: float,
    bare_index: int,
    weak_threshold_mhz: float,
    *,
    initial_k: int,
    max_k: int,
    capture_target: float,
):
    """Extract a complete weak-shift window around one bare pair channel."""
    dimension = hamiltonian.shape[0]
    eigenpair_cap = min(int(max_k), dimension - 2)
    if eigenpair_cap < 1:
        raise ValueError("pair Hamiltonian must have at least three basis states")
    k = min(max(1, int(initial_k)), eigenpair_cap)
    v0 = np.linspace(1.0, 2.0, dimension)
    v0 /= np.linalg.norm(v0)

    while True:
        eigenvalues, eigenvectors = eigsh(
            hamiltonian,
            k=k,
            sigma=reference_ghz,
            which="LM",
            tol=1e-9,
            v0=v0,
        )
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        shifts_mhz = (eigenvalues - reference_ghz) * 1e3
        overlaps = np.abs(eigenvectors[bare_index, :]) ** 2
        window_bracketed = bool(
            shifts_mhz[0] < -weak_threshold_mhz
            and shifts_mhz[-1] > weak_threshold_mhz
        )
        captured_overlap = float(overlaps.sum())
        capture_converged = captured_overlap >= capture_target
        meta = {
            "eigenpairs": int(k),
            "window_bracketed": window_bracketed,
            "capture_converged": bool(capture_converged),
            "captured_overlap": captured_overlap,
        }
        if window_bracketed and capture_converged:
            return eigenvalues, eigenvectors, meta
        if k == eigenpair_cap:
            if not window_bracketed:
                raise RuntimeError(
                    f"{k} eigenpairs did not bracket the ±{weak_threshold_mhz:g} MHz window"
                )
            return eigenvalues, eigenvectors, meta
        k = min(2 * k, eigenpair_cap)


def _half_integer_label(value: float) -> str:
    twice = int(round(2 * value))
    if twice % 2 == 0:
        return f"{twice // 2:+d}"
    return f"{twice:+d}/2"


def _pair_state_label(state) -> str:
    letters = "SPDFGHIKLMNOQRTUVWXYZ"

    def single(n, l, j, mj) -> str:
        return f"{int(n)}{letters[int(l)]}{_half_integer_label(j)}(m={_half_integer_label(mj)})"

    return f"{single(*state[:4])}+{single(*state[4:8])}"


def summarize_eigenpairs(
    eigenvalues_ghz,
    eigenvectors,
    *,
    reference_ghz: float,
    bare_index: int,
    basis_states,
    target_manifold_indices,
    weak_threshold_mhz: float,
    report_overlap_cutoff: float,
) -> dict:
    """Summarize spectral weight reachable from one bare pair channel."""
    eigenvalues_ghz = np.asarray(eigenvalues_ghz)
    eigenvectors = np.asarray(eigenvectors)
    shifts_mhz = (eigenvalues_ghz - reference_ghz) * 1e3
    overlaps = np.abs(eigenvectors[bare_index, :]) ** 2
    target_weights = (
        np.abs(eigenvectors[np.asarray(target_manifold_indices), :]) ** 2
    ).sum(axis=0)

    states = []
    for k in np.argsort(-overlaps):
        if overlaps[k] < report_overlap_cutoff:
            break
        component_weights = np.abs(eigenvectors[:, k]) ** 2
        components = [
            {
                "state": _pair_state_label(basis_states[i]),
                "weight": float(component_weights[i]),
            }
            for i in np.argsort(-component_weights)[:4]
        ]
        states.append(
            {
                "overlap": float(overlaps[k]),
                "e_mhz": float(eigenvalues_ghz[k] * 1e3),
                "shift_mhz": float(shifts_mhz[k]),
                "target_manifold_weight": float(target_weights[k]),
                "top_components": components,
            }
        )

    weak = np.abs(shifts_mhz) < weak_threshold_mhz
    return {
        "bare_reference_mhz": float(reference_ghz * 1e3),
        "spectral_range_mhz": [float(shifts_mhz[0]), float(shifts_mhz[-1])],
        "captured_overlap": float(overlaps.sum()),
        "weak_shift_weight": float(overlaps[weak].sum()),
        "states": states,
    }


def intermediate_channel_inventory(atom) -> list[dict]:
    """Pair channels (n1 l1 j1, n2 l2 j2) reachable by one E1 transition per atom.

    E1 from 53P3/2 allows l' in {S, D} per atom, so the pair channels split into
    the (S,S) / (S,D) / (D,D) families the C6 sum runs over.  Channels are ranked
    by the second-order scale (R1 R2)^2 / |delta| with R_i the radial matrix
    element <53 P3/2 | r | n_i l_i j_i> and delta the pair defect — angular
    factors are channel-dependent and deliberately omitted, so the weights order
    the channels but are not the C6 decomposition itself.
    """
    e_p = atom.getEnergy(N, L, J)
    singles = []
    for l, js in ((0, (0.5,)), (2, (1.5, 2.5))):
        for n in range(N - N_RANGE, N + N_RANGE + 1):
            for j in js:
                singles.append((n, l, j, atom.getRadialMatrixElement(N, L, J, n, l, j),
                                atom.getEnergy(n, l, j)))
    names = {0: "S", 2: "D"}
    chans = []
    for i, (n1, l1, j1, r1, e1) in enumerate(singles):
        for n2, l2, j2, r2, e2 in singles[i:]:
            delta_ghz = (e1 + e2 - 2.0 * e_p) * EV_TO_GHZ
            if abs(delta_ghz) > ENERGY_DELTA_HZ / 1e9:
                continue
            chans.append({
                "pair": f"{n1}{names[l1]}{j1}+{n2}{names[l2]}{j2}",
                "family": f"({names[l1]},{names[l2]})",
                "defect_ghz": delta_ghz,
                "weight": (r1 * r2) ** 2 / abs(delta_ghz),
            })
    chans.sort(key=lambda c: -c["weight"])
    top = max(c["weight"] for c in chans)
    for c in chans:
        c["weight"] = c["weight"] / top
    return [c for c in chans if c["weight"] >= 0.01]


def main() -> None:
    from arc import PairStateInteractions, Rubidium87
    from arc.calculations_atom_pairstate import compositeState, singleAtomState

    calc = PairStateInteractions(Rubidium87(), N, L, J, N, L, J, -1.5, -1.5)
    vals, vecs = calc.getC6perturbatively(
        THETA, PHI, N_RANGE, ENERGY_DELTA_HZ, degeneratePerturbation=True)
    vals = np.asarray(vals)
    vecs = np.asarray(vecs)
    if max(np.abs(vals.imag).max(), np.abs(vecs.imag).max()) > 1e-9:
        raise RuntimeError("unexpected complex C6 eigen-decomposition at phi=0")
    vals, vecs = vals.real, vecs.real        # GHz um^6, ARC sign V = -C6/R^6

    def basis_vec(m1: float, m2: float) -> np.ndarray:
        return np.real(compositeState(singleAtomState(J, m1),
                                      singleAtomState(J, m2)).flatten())

    # Repo-sign interaction matrix at R, V/2pi in MHz (repo: V = +C6_repo/R^6,
    # C6_repo = -arc * 2pi * 1e9 — see physics.arc_pair_c6_rad_s_um6).
    v_mat = -((vecs.T * vals) @ vecs) * 1e3 / R_UM ** 6

    labels, index = [""] * 16, {}
    for m1 in MJS:
        for m2 in MJS:
            i = int(np.argmax(np.abs(basis_vec(m1, m2))))
            labels[i] = f"({m1:+.1f},{m2:+.1f})"
            index[(m1, m2)] = i

    def zeeman_diag(b_gauss: float) -> np.ndarray:
        dz = zeeman_shift_rad_s(b_gauss, l=L, j=J, delta_mj=1.0) / (2e6 * math.pi)
        zee = np.zeros(16)
        for (m1, m2), i in index.items():
            zee[i] = ((m1 + 1.5) + (m2 + 1.5)) * dz
        return zee

    out = {
        "params": {
            "n": N, "l": L, "j": J, "theta_rad": THETA, "phi_rad": PHI,
            "spacing_um": R_UM, "n_range": N_RANGE,
            "energy_delta_hz": ENERGY_DELTA_HZ,
            "delta_z_mhz_per_mj": {str(b): zeeman_shift_rad_s(
                b, l=L, j=J, delta_mj=1.0) / (2e6 * math.pi) for b in B_FIELDS_G},
            "omega_max_mhz": OMEGA_MAX_MHZ,
            "weak_shift_threshold_mhz": WEAK_SHIFT_THRESHOLD_MHZ,
        },
        "channels": {},
        "exchange_r_rgarb_mhz": float(
            basis_vec(-1.5, -0.5) @ v_mat @ basis_vec(-0.5, -1.5)),
        "intermediate_channels": intermediate_channel_inventory(Rubidium87()),
    }

    for name, (m1, m2) in BARE_CHANNELS.items():
        bare = basis_vec(m1, m2)
        rec = {"bare_expectation_mhz": float(bare @ v_mat @ bare)}

        ov0 = (vecs @ bare) ** 2
        rec["b0_eigenchannels"] = [
            {"overlap": float(ov0[k]),
             "v_at_r_mhz": float(-vals[k] * 1e3 / R_UM ** 6)}
            for k in np.argsort(-ov0) if ov0[k] >= 0.01]

        rec["dressed"] = {}
        for b in B_FIELDS_G:
            zee = zeeman_diag(b)
            w, u = np.linalg.eigh(v_mat + np.diag(zee))
            ovz = (u.T @ bare) ** 2
            shift = w - zee[index[(m1, m2)]]
            states = []
            for k in np.argsort(-ovz):
                if ovz[k] < 0.01:
                    break
                comp = np.argsort(-np.abs(u[:, k]))[:3]
                states.append({
                    "overlap": float(ovz[k]), "e_mhz": float(w[k]),
                    "shift_mhz": float(shift[k]),
                    "top_components": {labels[i]: float(u[i, k] ** 2)
                                       for i in comp}})
            rec["dressed"][str(b)] = {
                "states": states,
                "weak_shift_weight": float(
                    ovz[np.abs(shift) < WEAK_SHIFT_THRESHOLD_MHZ].sum()),
            }
        out["channels"][name] = rec

    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    for name, rec in out["channels"].items():
        print(f"\n### {name}: <bare|V|bare> = {rec['bare_expectation_mhz']:+.1f} MHz")
        for e in rec["b0_eigenchannels"]:
            print(f"  B=0   overlap {e['overlap']:.3f}  V = {e['v_at_r_mhz']:+8.1f} MHz")
        for b, d in rec["dressed"].items():
            for s in d["states"]:
                print(f"  B={b:>3s} overlap {s['overlap']:.3f}  "
                      f"shift = {s['shift_mhz']:+8.1f} MHz  {s['top_components']}")
            print(f"  B={b:>3s} weak-shift weight {d['weak_shift_weight']:.4f}")
    print(f"\nexchange <r,rg|V|rg,r> = {out['exchange_r_rgarb_mhz']:+.2f} MHz")
    print("\nintermediate pair channels (weight = (R1 R2)^2/|delta|, top = 1):")
    for c in out["intermediate_channels"][:10]:
        print(f"  {c['pair']:24s} {c['family']}  delta = {c['defect_ghz']:+8.2f} GHz  "
              f"weight {c['weight']:.3f}")


if __name__ == "__main__":
    main()
