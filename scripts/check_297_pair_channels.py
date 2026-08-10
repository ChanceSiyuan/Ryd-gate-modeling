#!/usr/bin/env python3
"""Explicit 53P3/2 pair-state spectra for the 297 nm target-only gate.

The authoritative calculation uses ARC's explicit truncated pair basis and
diagonalizes H_A(B) + H_B(B) + V_dd(R, theta), so the linear Zeeman shift enters
every retained pair state, including the near-resonant S/S Förster channels.
The former zero-field second-order C6 matrix plus a hand-added Zeeman term on
the 16-state 53P+53P manifold is retained only as a labeled comparison.

Deterministic, no RNG; writes results/297_to_calibration/pair_channels.json.

On the DGX, ARC needs the HOME=/tmp/arc297home prefix (see the zxz README).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from importlib.metadata import version
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
N_RANGE, L_MAX, ENERGY_DELTA_HZ = 5, 2, 30e9
OMEGA_MAX_MHZ = 16.614                      # to_297.json:fixed target-leg Rabi
WEAK_SHIFT_THRESHOLD_MHZ = 5.0 * OMEGA_MAX_MHZ
INITIAL_EIGENPAIRS, MAX_EIGENPAIRS = 32, 256
CAPTURE_TARGET, REPORT_OVERLAP_CUTOFF = 0.99, 0.001
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


def radial_defect_ranking(atom) -> dict:
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
    return {
        "diagnostic_only": True,
        "weight_definition": "(R1*R2)^2/abs(pair_defect)",
        "omits_angular_factors": True,
        "omits_denominator_sign": True,
        "channels": [c for c in chans if c["weight"] >= 0.01],
    }


def calculate_full_pair_field(atom, b_gauss: float) -> dict:
    """Build and diagonalize ARC's explicit pair basis at one magnetic field."""
    from arc import PairStateInteractions

    started = time.perf_counter()
    calc = PairStateInteractions(
        atom,
        N,
        L,
        J,
        N,
        L,
        J,
        -1.5,
        -1.5,
        interactionsUpTo=1,
    )
    calc.defineBasis(
        THETA,
        PHI,
        N_RANGE,
        L_MAX,
        ENERGY_DELTA_HZ,
        Bz=b_gauss * 1e-4,
    )
    hamiltonian = assemble_pair_hamiltonian(calc, R_UM)
    build_s = time.perf_counter() - started

    target_manifold_indices = [
        i
        for i, state in enumerate(calc.basisStates)
        if tuple(state[:3]) == (N, L, J)
        and tuple(state[4:7]) == (N, L, J)
    ]
    if len(target_manifold_indices) != 16:
        raise RuntimeError(
            f"expected 16 states in the 53P3/2 pair manifold; found "
            f"{len(target_manifold_indices)}"
        )

    diagonal_ghz = np.asarray(calc.matDiagonal.diagonal()).real
    channels = {}
    diagonalize_s = 0.0
    for name, (m1, m2) in BARE_CHANNELS.items():
        bare_index = find_basis_state_index(
            calc.basisStates, (N, L, J, m1, N, L, J, m2)
        )
        reference_ghz = float(diagonal_ghz[bare_index])
        diag_started = time.perf_counter()
        eigenvalues, eigenvectors, extraction = extract_local_eigenpairs(
            hamiltonian,
            reference_ghz,
            bare_index,
            WEAK_SHIFT_THRESHOLD_MHZ,
            initial_k=INITIAL_EIGENPAIRS,
            max_k=MAX_EIGENPAIRS,
            capture_target=CAPTURE_TARGET,
        )
        channel_diag_s = time.perf_counter() - diag_started
        diagonalize_s += channel_diag_s
        record = summarize_eigenpairs(
            eigenvalues,
            eigenvectors,
            reference_ghz=reference_ghz,
            bare_index=bare_index,
            basis_states=calc.basisStates,
            target_manifold_indices=target_manifold_indices,
            weak_threshold_mhz=WEAK_SHIFT_THRESHOLD_MHZ,
            report_overlap_cutoff=REPORT_OVERLAP_CUTOFF,
        )
        record.update(extraction)
        record["diagonalize_s"] = channel_diag_s
        channels[name] = record

    return {
        "b_gauss": float(b_gauss),
        "b_tesla": float(b_gauss * 1e-4),
        "basis_dimension": len(calc.basisStates),
        "target_manifold_dimension": len(target_manifold_indices),
        "hamiltonian_nnz": int(hamiltonian.nnz),
        "build_s": build_s,
        "diagonalize_s": diagonalize_s,
        "channels": channels,
    }


def calculate_effective_c6_comparison(atom) -> dict:
    """Return the former B=0 C6 plus PP-manifold Zeeman approximation."""
    from arc import PairStateInteractions
    from arc.calculations_atom_pairstate import compositeState, singleAtomState

    calc = PairStateInteractions(atom, N, L, J, N, L, J, -1.5, -1.5)
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
        "model": "zero_field_second_order_c6_plus_pp_linear_zeeman",
        "limitations": [
            "intermediate pair states are perturbatively eliminated at B=0",
            "finite-field Zeeman shifts are added only inside the 53P3/2+53P3/2 manifold",
        ],
        "channels": {},
        "exchange_r_rgarb_mhz": float(
            basis_vec(-1.5, -0.5) @ v_mat @ basis_vec(-0.5, -1.5)),
    }

    for name, (m1, m2) in BARE_CHANNELS.items():
        bare = basis_vec(m1, m2)
        rec = {"bare_expectation_mhz": float(bare @ v_mat @ bare)}

        ov0 = np.abs(vecs @ bare) ** 2
        rec["b0_eigenchannels"] = [
            {"overlap": float(ov0[k]),
             "v_at_r_mhz": float(-vals[k] * 1e3 / R_UM ** 6)}
            for k in np.argsort(-ov0) if ov0[k] >= 0.01]

        rec["pp_zeeman_dressed"] = {}
        for b in B_FIELDS_G:
            zee = zeeman_diag(b)
            w, u = np.linalg.eigh(v_mat + np.diag(zee))
            ovz = np.abs(u.T @ bare) ** 2
            shift = w - zee[index[(m1, m2)]]
            states = []
            for k in np.argsort(-ovz):
                if ovz[k] < 0.01:
                    break
                comp = np.argsort(-np.abs(u[:, k]))[:3]
                states.append({
                    "overlap": float(ovz[k]), "e_mhz": float(w[k]),
                    "shift_mhz": float(shift[k]),
                    "top_components": {labels[i]: float(abs(u[i, k]) ** 2)
                                       for i in comp}})
            rec["pp_zeeman_dressed"][str(b)] = {
                "states": states,
                "weak_shift_weight": float(
                    ovz[np.abs(shift) < WEAK_SHIFT_THRESHOLD_MHZ].sum()),
            }
        out["channels"][name] = rec
    return out


def build_output(atom) -> dict:
    """Build the schema-versioned authoritative and comparison result."""
    effective = calculate_effective_c6_comparison(atom)
    effective["authoritative"] = False
    return {
        "schema_version": 2,
        "params": {
            "arc_version": version("arc-alkali-rydberg-calculator"),
            "n": N,
            "l": L,
            "j": J,
            "theta_rad": THETA,
            "phi_rad": PHI,
            "spacing_um": R_UM,
            "n_range": N_RANGE,
            "l_max": L_MAX,
            "energy_delta_hz": ENERGY_DELTA_HZ,
            "interactions_up_to": 1,
            "b_fields_gauss": list(B_FIELDS_G),
            "delta_z_mhz_per_mj": {
                str(b): zeeman_shift_rad_s(
                    b, l=L, j=J, delta_mj=1.0
                ) / (2e6 * math.pi)
                for b in B_FIELDS_G
            },
            "omega_max_mhz": OMEGA_MAX_MHZ,
            "weak_shift_threshold_mhz": WEAK_SHIFT_THRESHOLD_MHZ,
            "initial_eigenpairs": INITIAL_EIGENPAIRS,
            "max_eigenpairs": MAX_EIGENPAIRS,
            "capture_target": CAPTURE_TARGET,
            "report_overlap_cutoff": REPORT_OVERLAP_CUTOFF,
            "approximations": [
                "pair basis truncated by n range, l maximum, and zero-field energy window",
                "dipole-dipole coupling only",
                "ARC linear paramagnetic Zeeman shifts; diamagnetic term omitted",
                "hyperfine structure and magnetic mixing between j manifolds omitted",
            ],
        },
        "full_pair": {
            "authoritative": True,
            "model": "explicit_truncated_pair_basis_with_linear_zeeman",
            "fields": {
                str(b): calculate_full_pair_field(atom, b) for b in B_FIELDS_G
            },
        },
        "effective_c6_comparison": effective,
        "radial_defect_ranking": radial_defect_ranking(atom),
    }


def main() -> None:
    from arc import Rubidium87

    started = time.perf_counter()
    out = build_output(Rubidium87())
    out["params"]["elapsed_s"] = time.perf_counter() - started
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")
    for b, field in out["full_pair"]["fields"].items():
        print(
            f"\n## full pair B={b} G: dimension={field['basis_dimension']} "
            f"nnz={field['hamiltonian_nnz']}"
        )
        for name, channel in field["channels"].items():
            print(
                f"  {name}: weak weight={channel['weak_shift_weight']:.6f}, "
                f"captured={channel['captured_overlap']:.6f}, "
                f"k={channel['eigenpairs']}"
            )
            for state in channel["states"]:
                print(
                    f"    overlap {state['overlap']:.4f}  "
                    f"shift {state['shift_mhz']:+9.3f} MHz  "
                    f"PP weight {state['target_manifold_weight']:.4f}"
                )
    comparison = out["effective_c6_comparison"]
    print(
        "\neffective-C6 comparison exchange "
        f"<r,rg|V|rg,r> = {comparison['exchange_r_rgarb_mhz']:+.2f} MHz"
    )
    print("\nradial/defect diagnostic ranking (no angular factors):")
    for c in out["radial_defect_ranking"]["channels"][:10]:
        print(f"  {c['pair']:24s} {c['family']}  delta = {c['defect_ghz']:+8.2f} GHz  "
              f"weight {c['weight']:.3f}")
    print(f"\nelapsed {out['params']['elapsed_s']:.1f} s")


if __name__ == "__main__":
    main()
