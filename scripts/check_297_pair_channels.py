#!/usr/bin/env python3
"""Explicit 53P3/2 pair-state spectra for the 297 nm target-only gate.

The authoritative calculation uses ARC's explicit truncated pair basis and
diagonalizes H_A(B) + H_B(B) + V_dd(R, theta), so the linear Zeeman shift enters
every retained pair state, including the near-resonant S/S Förster channels.
The former zero-field second-order C6 matrix plus a hand-added Zeeman term on
the 16-state 53P+53P manifold is retained only as a labeled comparison.

The ``--pair-potentials`` mode scans 53P and a 70S benchmark over field,
direction, and distance with exact diagonalization of each truncated basis;
``--plot-only`` replays its figures from pair_potential_curves.json.
``--manuscript-comparison`` renders the selected 20 G, three-angle comparison.

Deterministic, no RNG; writes under results/297_to_calibration/.

ARC may update its normal per-user atomic-data cache while defining a pair basis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

ROOT = Path(
    os.environ.get("RYD_GATE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, os.fspath(ROOT / "src"))

from ryd_gate.physics import zeeman_shift_rad_s  # noqa: E402

OUT = ROOT / "results" / "297_to_calibration" / "pair_channels.json"
PAIR_POTENTIAL_OUT = (
    ROOT / "results" / "297_to_calibration" / "pair_potential_curves.json"
)
MANUSCRIPT_COMPARISON_OUT = (
    ROOT
    / "manuscripts"
    / "figures"
    / "pair_spectrum_53P_70S_B20G.pdf"
)

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

POTENTIAL_FIELDS_G = (20.0, 40.0, 60.0)
POTENTIAL_THETA_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
POTENTIAL_PHI_RAD = 0.0
POTENTIAL_BRANCH_COUNT = 5
POTENTIAL_MATCH_TARGET = 0.25
POTENTIAL_EIGENSYSTEM_RESIDUAL_MHZ = 1e-6
POTENTIAL_N_RANGE = 3
POTENTIAL_ENERGY_DELTA_HZ = 10e9
POTENTIAL_SPECTRUM_WEIGHT_CUTOFF = 1e-5
POTENTIAL_BRANCH_WEIGHT_FLOOR = 1e-6
POTENTIAL_MANIFOLDS = {
    "53P3_2": {
        "n": 53,
        "l": 1,
        "j": 1.5,
        "mj": -1.5,
        "label": r"$53P_{3/2},\,m_j=-3/2$",
        "scale_group": "53P3_2",
    },
    "53P3_2_mj_m1_2": {
        "n": 53,
        "l": 1,
        "j": 1.5,
        "mj": -0.5,
        "label": r"$53P_{3/2},\,m_j=-1/2$",
        "scale_group": "53P3_2",
    },
    "53P3_2_mj_p1_2": {
        "n": 53,
        "l": 1,
        "j": 1.5,
        "mj": 0.5,
        "label": r"$53P_{3/2},\,m_j=+1/2$",
        "scale_group": "53P3_2",
    },
    "53P3_2_mj_p3_2": {
        "n": 53,
        "l": 1,
        "j": 1.5,
        "mj": 1.5,
        "label": r"$53P_{3/2},\,m_j=+3/2$",
        "scale_group": "53P3_2",
    },
    "70S1_2": {
        "n": 70,
        "l": 0,
        "j": 0.5,
        "mj": -0.5,
        "label": r"$70S_{1/2},\,m_j=-1/2$",
        "scale_group": "70S1_2",
    },
}


EV_TO_GHZ = 241798.9


def arc_degeneracy_offsets_ghz(calc) -> np.ndarray:
    """Return ARC's artificial per-channel diagonal tie breakers."""
    offsets = np.zeros(len(calc.basisStates))
    for start, stop in zip(calc.index[:-1], calc.index[1:]):
        start, stop = int(start), int(stop)
        offsets[start:stop] = 1e-8 * np.arange(1, stop - start + 1)
    return offsets


def assemble_pair_hamiltonian(
    calc,
    spacing_um: float,
    *,
    remove_arc_degeneracy_offsets: bool = False,
):
    """Assemble ARC's sparse pair Hamiltonian at one spacing, in GHz."""
    matrix = calc.matDiagonal.copy()
    if remove_arc_degeneracy_offsets:
        matrix = matrix - diags(
            arc_degeneracy_offsets_ghz(calc), offsets=0, format="csr"
        )
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


def potential_distance_grid() -> np.ndarray:
    """Return the 41-point potential grid with an exact 3 micrometre anchor."""
    left = np.linspace(2.5, 3.0, 9)
    right = np.linspace(3.0, 8.0, 33)
    return np.concatenate((left, right[1:]))


def match_eigenbranches(previous_vectors, candidate_vectors):
    """Match branch vectors to candidates by maximum total squared overlap."""
    overlaps = np.abs(previous_vectors.conj().T @ candidate_vectors) ** 2
    rows, columns = linear_sum_assignment(-overlaps)
    assignment = np.empty(previous_vectors.shape[1], dtype=int)
    qualities = np.empty(previous_vectors.shape[1], dtype=float)
    assignment[rows] = columns
    qualities[rows] = overlaps[rows, columns]
    return assignment, qualities


def extract_curve_eigenpairs(
    hamiltonian,
    bare_index: int,
):
    """Exactly diagonalize one truncated pair Hamiltonian."""
    if hamiltonian.shape[0] < 1:
        raise ValueError("pair Hamiltonian must contain at least one basis state")
    eigenvalues, eigenvectors = eigh(
        hamiltonian.toarray(),
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    rr_overlap = np.abs(eigenvectors[bare_index, :]) ** 2
    captured = float(rr_overlap.sum())
    residual_vectors = hamiltonian @ eigenvectors - (
        eigenvectors * eigenvalues[None, :]
    )
    max_residual_mhz = float(
        np.max(np.linalg.norm(residual_vectors, axis=0)) * 1e3
    )
    if not math.isclose(captured, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"exact eigensystem captured rr overlap {captured:.12f}")
    if max_residual_mhz > POTENTIAL_EIGENSYSTEM_RESIDUAL_MHZ:
        raise RuntimeError(
            f"maximum eigensystem residual {max_residual_mhz:.6g} MHz exceeds "
            f"{POTENTIAL_EIGENSYSTEM_RESIDUAL_MHZ:.6g} MHz"
        )
    return eigenvalues, eigenvectors, {
        "eigenpairs": int(hamiltonian.shape[0]),
        "captured_rr_overlap": captured,
        "max_eigensystem_residual_mhz": max_residual_mhz,
    }


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


def track_rr_branches(
    distances_um,
    solve,
    *,
    bare_index: int,
    reference_ghz: float,
    basis_states,
    branch_count: int,
    anchor_um: float = 3.0,
) -> dict:
    """Seed the most ``rr``-bright states at the anchor and track both ways."""
    requested_branch_count = int(branch_count)
    distances = np.asarray(distances_um, dtype=float)
    anchor_matches = np.flatnonzero(distances == float(anchor_um))
    if anchor_matches.size != 1:
        raise ValueError("distance grid must contain the anchor exactly once")
    anchor_index = int(anchor_matches[0])
    point_count = distances.size

    eigenpair_counts = np.zeros(point_count, dtype=int)
    captured_overlaps = np.zeros(point_count, dtype=float)
    eigensystem_residuals = np.zeros(point_count, dtype=float)
    weak_weights = np.zeros(point_count, dtype=float)
    spectrum_shifts = [None] * point_count
    spectrum_overlaps = [None] * point_count
    branch_shifts = np.zeros((branch_count, point_count), dtype=float)
    branch_overlaps = np.zeros((branch_count, point_count), dtype=float)
    match_overlaps = np.zeros((branch_count, point_count), dtype=float)

    def record_candidates(index, eigenvalues, eigenvectors, meta):
        shifts = (np.asarray(eigenvalues) - reference_ghz) * 1e3
        overlaps = np.abs(eigenvectors[bare_index, :]) ** 2
        eigenpair_counts[index] = int(meta["eigenpairs"])
        captured_overlaps[index] = float(meta["captured_rr_overlap"])
        eigensystem_residuals[index] = float(
            meta["max_eigensystem_residual_mhz"]
        )
        weak_weights[index] = float(
            overlaps[np.abs(shifts) < WEAK_SHIFT_THRESHOLD_MHZ].sum()
        )
        visible = overlaps >= POTENTIAL_SPECTRUM_WEIGHT_CUTOFF
        spectrum_shifts[index] = np.asarray(shifts[visible], dtype=float).tolist()
        spectrum_overlaps[index] = np.asarray(
            overlaps[visible], dtype=float
        ).tolist()
        return shifts, overlaps

    eigenvalues, eigenvectors, meta = solve(float(anchor_um))
    shifts, overlaps = record_candidates(
        anchor_index, eigenvalues, eigenvectors, meta
    )
    bright_order = np.argsort(-overlaps)
    bright_order = bright_order[
        overlaps[bright_order] >= POTENTIAL_BRANCH_WEIGHT_FLOOR
    ]
    seed_columns = bright_order[:requested_branch_count]
    branch_count = int(seed_columns.size)
    if branch_count == 0:
        raise RuntimeError("rr has no resolved spectral weight at the anchor")
    branch_shifts = branch_shifts[:branch_count]
    branch_overlaps = branch_overlaps[:branch_count]
    match_overlaps = match_overlaps[:branch_count]
    branch_shifts[:, anchor_index] = shifts[seed_columns]
    branch_overlaps[:, anchor_index] = overlaps[seed_columns]
    match_overlaps[:, anchor_index] = 1.0
    anchor_vectors = eigenvectors[:, seed_columns]

    def track(indices, initial_vectors):
        previous_vectors = initial_vectors
        for index in indices:
            eigenvalues, eigenvectors, meta = solve(float(distances[index]))
            shifts, overlaps = record_candidates(
                index, eigenvalues, eigenvectors, meta
            )
            selected, qualities = match_eigenbranches(
                previous_vectors, eigenvectors
            )
            if np.any(qualities < POTENTIAL_MATCH_TARGET):
                raise RuntimeError(
                    f"minimum adjacent branch match {float(np.min(qualities)):.6f} "
                    f"< {POTENTIAL_MATCH_TARGET:.6f}"
                )
            branch_shifts[:, index] = shifts[selected]
            branch_overlaps[:, index] = overlaps[selected]
            match_overlaps[:, index] = qualities
            previous_vectors = eigenvectors[:, selected]

    track(range(anchor_index + 1, point_count), anchor_vectors)
    track(range(anchor_index - 1, -1, -1), anchor_vectors)

    branches = []
    for rank, _column in enumerate(seed_columns, start=1):
        component_weights = np.abs(anchor_vectors[:, rank - 1]) ** 2
        top_components = [
            {
                "state": _pair_state_label(basis_states[index]),
                "weight": float(component_weights[index]),
            }
            for index in np.argsort(-component_weights)[:4]
        ]
        branches.append(
            {
                "anchor_rank": rank,
                "anchor_shift_mhz": float(branch_shifts[rank - 1, anchor_index]),
                "anchor_rr_overlap": float(
                    branch_overlaps[rank - 1, anchor_index]
                ),
                "anchor_top_components": top_components,
                "shift_mhz": branch_shifts[rank - 1].tolist(),
                "rr_overlap": branch_overlaps[rank - 1].tolist(),
                "adjacent_match_overlap": match_overlaps[rank - 1].tolist(),
                "min_adjacent_match_overlap": float(
                    np.min(match_overlaps[rank - 1])
                ),
            }
        )

    return {
        "anchor_um": float(anchor_um),
        "branch_count": int(branch_count),
        "requested_branch_count": requested_branch_count,
        "distance_um": distances.tolist(),
        "eigenpairs": eigenpair_counts.tolist(),
        "captured_rr_overlap": captured_overlaps.tolist(),
        "unresolved_rr_overlap": np.maximum(
            0.0, 1.0 - captured_overlaps
        ).tolist(),
        "max_eigensystem_residual_mhz": eigensystem_residuals.tolist(),
        "weak_shift_weight": weak_weights.tolist(),
        "spectrum_shift_mhz": spectrum_shifts,
        "spectrum_rr_overlap": spectrum_overlaps,
        "branches": branches,
    }


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


def calculate_pair_potential_case(
    atom,
    manifold: dict,
    b_gauss: float,
    theta_deg: float,
) -> dict:
    """Calculate one state/field/direction pair-potential data set."""
    from arc import PairStateInteractions

    n = int(manifold["n"])
    l = int(manifold["l"])
    j = float(manifold["j"])
    mj = float(manifold["mj"])
    theta_rad = math.radians(theta_deg)
    started = time.perf_counter()
    calc = PairStateInteractions(
        atom,
        n,
        l,
        j,
        n,
        l,
        j,
        mj,
        mj,
        interactionsUpTo=1,
    )
    calc.defineBasis(
        theta_rad,
        POTENTIAL_PHI_RAD,
        POTENTIAL_N_RANGE,
        L_MAX,
        POTENTIAL_ENERGY_DELTA_HZ,
        Bz=b_gauss * 1e-4,
    )
    build_s = time.perf_counter() - started
    bare_index = find_basis_state_index(
        calc.basisStates, (n, l, j, mj, n, l, j, mj)
    )
    degeneracy_offsets_ghz = arc_degeneracy_offsets_ghz(calc)
    reference_ghz = float(
        np.real(calc.matDiagonal[bare_index, bare_index])
        - degeneracy_offsets_ghz[bare_index]
    )

    def solve(distance_um):
        return extract_curve_eigenpairs(
            assemble_pair_hamiltonian(
                calc,
                distance_um,
                remove_arc_degeneracy_offsets=True,
            ),
            bare_index,
        )

    diagonalize_started = time.perf_counter()
    curves = track_rr_branches(
        potential_distance_grid(),
        solve,
        bare_index=bare_index,
        reference_ghz=reference_ghz,
        basis_states=calc.basisStates,
        branch_count=POTENTIAL_BRANCH_COUNT,
        anchor_um=R_UM,
    )
    diagonalize_s = time.perf_counter() - diagonalize_started
    anchor_hamiltonian = assemble_pair_hamiltonian(
        calc, R_UM, remove_arc_degeneracy_offsets=True
    )
    return {
        "b_gauss": float(b_gauss),
        "theta_deg": float(theta_deg),
        "theta_rad": float(theta_rad),
        "phi_rad": float(POTENTIAL_PHI_RAD),
        "basis_dimension": len(calc.basisStates),
        "hamiltonian_nnz": int(anchor_hamiltonian.nnz),
        "bare_reference_mhz": reference_ghz * 1e3,
        "removed_arc_degeneracy_offset_max_mhz": float(
            np.max(degeneracy_offsets_ghz) * 1e3
        ),
        "build_s": build_s,
        "diagonalize_s": diagonalize_s,
        "curves": curves,
    }


def _pair_potential_params() -> dict:
    """Return the complete fingerprint for resumable pair-potential data."""
    return {
        "arc_version": version("arc-alkali-rydberg-calculator"),
        "manifold_definitions": {
            key: {name: manifold[name] for name in ("n", "l", "j", "mj")}
            for key, manifold in POTENTIAL_MANIFOLDS.items()
        },
        "b_fields_gauss": list(POTENTIAL_FIELDS_G),
        "theta_deg": list(POTENTIAL_THETA_DEG),
        "phi_rad": POTENTIAL_PHI_RAD,
        "distance_um": potential_distance_grid().tolist(),
        "anchor_um": R_UM,
        "branch_count": POTENTIAL_BRANCH_COUNT,
        "branch_weight_floor": POTENTIAL_BRANCH_WEIGHT_FLOOR,
        "spectrum_weight_cutoff": POTENTIAL_SPECTRUM_WEIGHT_CUTOFF,
        "eigensolver": "scipy.linalg.eigh(driver='evd')",
        "match_target": POTENTIAL_MATCH_TARGET,
        "eigensystem_residual_target_mhz": POTENTIAL_EIGENSYSTEM_RESIDUAL_MHZ,
        "weak_shift_threshold_mhz": WEAK_SHIFT_THRESHOLD_MHZ,
        "n_range": POTENTIAL_N_RANGE,
        "l_max": L_MAX,
        "energy_delta_hz": POTENTIAL_ENERGY_DELTA_HZ,
        "interactions_up_to": 1,
        "arc_degeneracy_offsets_removed": True,
        "reference_convention": "E_k minus the physical bare rr diagonal at the same B",
        "scope_note": (
            "10 GHz visualization/benchmark scan; the separate 53P "
            "single-distance audit retains its 30 GHz basis"
        ),
        "approximations": [
            "explicit ARC pair basis truncated by n range, l maximum, and energy window",
            "dipole-dipole coupling only",
            "ARC linear paramagnetic Zeeman shifts; diamagnetic term omitted",
            "hyperfine structure and magnetic mixing between j manifolds omitted",
        ],
    }


def _new_pair_potential_study() -> dict:
    return {
        "schema_version": 1,
        "status": "running",
        "params": _pair_potential_params(),
        "manifolds": {},
    }


def _validate_pair_potential_config(result: dict) -> None:
    if result.get("schema_version") != 1:
        raise ValueError("pair-potential data have the wrong schema")
    params = result.get("params", {})
    mismatches = [
        key
        for key, expected in _pair_potential_params().items()
        if params.get(key) != expected
    ]
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(
            f"pair-potential configuration mismatch ({joined}); regenerate with "
            "--pair-potentials without --resume"
        )


def _pair_potential_case_is_complete(
    case,
    b_gauss: float,
    theta_deg: float,
) -> bool:
    curves = case.get("curves", {}) if isinstance(case, dict) else {}
    branches = curves.get("branches", [])
    distances = potential_distance_grid()
    point_count = distances.size
    try:
        arrays_complete = all(
            len(curves[key]) == point_count
            for key in (
                "eigenpairs",
                "captured_rr_overlap",
                "unresolved_rr_overlap",
                "max_eigensystem_residual_mhz",
                "weak_shift_weight",
                "spectrum_shift_mhz",
                "spectrum_rr_overlap",
            )
        )
        branches_complete = all(
            branch["anchor_rr_overlap"] >= POTENTIAL_BRANCH_WEIGHT_FLOOR
            and branch["min_adjacent_match_overlap"] >= POTENTIAL_MATCH_TARGET
            and all(
                len(branch[key]) == point_count
                for key in ("shift_mhz", "rr_overlap", "adjacent_match_overlap")
            )
            for branch in branches
        )
        spectrum_rows_complete = all(
            len(shifts) == len(overlaps)
            for shifts, overlaps in zip(
                curves["spectrum_shift_mhz"],
                curves["spectrum_rr_overlap"],
            )
        )
        numerical_gates = (
            min(curves["captured_rr_overlap"]) >= 1.0 - 1e-9
            and max(curves["max_eigensystem_residual_mhz"])
            <= POTENTIAL_EIGENSYSTEM_RESIDUAL_MHZ
        )
        return (
            math.isclose(case["b_gauss"], b_gauss, abs_tol=1e-12)
            and math.isclose(case["theta_deg"], theta_deg, abs_tol=1e-12)
            and math.isclose(case["phi_rad"], POTENTIAL_PHI_RAD, abs_tol=1e-12)
            and curves["requested_branch_count"] == POTENTIAL_BRANCH_COUNT
            and 0 < curves["branch_count"] <= POTENTIAL_BRANCH_COUNT
            and len(branches) == curves["branch_count"]
            and np.array_equal(np.asarray(curves["distance_um"]), distances)
            and arrays_complete
            and spectrum_rows_complete
            and branches_complete
            and numerical_gates
        )
    except (KeyError, TypeError, ValueError):
        return False


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=1) + "\n")
    temporary.replace(path)


def calculate_pair_potential_study(
    atom,
    *,
    checkpoint_path: Path | None = None,
    existing: dict | None = None,
) -> dict:
    """Calculate all state/field/theta cases, checkpointing after each case."""
    result = existing if existing is not None else _new_pair_potential_study()
    _validate_pair_potential_config(result)
    result["status"] = "running"
    started = time.perf_counter()
    completed_now = 0
    completed_total = sum(
        _pair_potential_case_is_complete(
            result.get("manifolds", {})
            .get(state_key, {})
            .get("fields", {})
            .get(str(b_gauss), {})
            .get("angles", {})
            .get(str(theta_deg)),
            b_gauss,
            theta_deg,
        )
        for state_key in POTENTIAL_MANIFOLDS
        for b_gauss in POTENTIAL_FIELDS_G
        for theta_deg in POTENTIAL_THETA_DEG
    )
    result["params"]["completed_cases"] = completed_total
    for state_key, manifold in POTENTIAL_MANIFOLDS.items():
        state_record = result["manifolds"].setdefault(
            state_key,
            {
                "n": manifold["n"],
                "l": manifold["l"],
                "j": manifold["j"],
                "mj": manifold["mj"],
                "label": manifold["label"],
                "fields": {},
            },
        )
        for b_gauss in POTENTIAL_FIELDS_G:
            field_record = state_record["fields"].setdefault(
                str(b_gauss), {"angles": {}}
            )
            for theta_deg in POTENTIAL_THETA_DEG:
                theta_key = str(theta_deg)
                prior = field_record["angles"].get(theta_key)
                if _pair_potential_case_is_complete(
                    prior, b_gauss, theta_deg
                ):
                    print(
                        f"resume: {state_key}, B={b_gauss:g} G, "
                        f"theta={theta_deg:g} deg already complete"
                    )
                    continue
                print(
                    f"calculate: {state_key}, B={b_gauss:g} G, "
                    f"theta={theta_deg:g} deg",
                    flush=True,
                )
                case = calculate_pair_potential_case(
                    atom, manifold, b_gauss, theta_deg
                )
                field_record["angles"][theta_key] = case
                completed_now += 1
                completed_total += 1
                result["params"]["elapsed_s"] = float(
                    result["params"].get("elapsed_s", 0.0)
                    + case["build_s"]
                    + case["diagonalize_s"]
                )
                result["params"]["completed_cases"] = completed_total
                if checkpoint_path is not None:
                    _write_json_atomic(checkpoint_path, result)
    result["status"] = "complete"
    result["params"]["completed_cases_this_run"] = completed_now
    result["params"]["wall_s_this_run"] = time.perf_counter() - started
    if checkpoint_path is not None:
        _write_json_atomic(checkpoint_path, result)
    return result


def _pair_potential_y_limits(result: dict) -> dict[str, float]:
    state_limits = {}
    for state_key, state in result["manifolds"].items():
        values = []
        for field in state["fields"].values():
            for case in field["angles"].values():
                values.extend(
                    value
                    for row in case["curves"]["spectrum_shift_mhz"]
                    for value in row
                    if np.isfinite(value)
                )
                values.extend(
                    value
                    for branch in case["curves"]["branches"]
                    for value in branch["shift_mhz"]
                    if np.isfinite(value)
                )
        limit = max((abs(value) for value in values), default=1.0)
        state_limits[state_key] = limit

    group_limits = {}
    for state_key, limit in state_limits.items():
        group = POTENTIAL_MANIFOLDS[state_key]["scale_group"]
        group_limits[group] = max(group_limits.get(group, 0.0), limit)
    return {
        state_key: 1.04 * group_limits[
            POTENTIAL_MANIFOLDS[state_key]["scale_group"]
        ]
        for state_key in state_limits
    }


def _overlap_marker_area(overlap):
    """Map bare-state overlap to scatter-marker area in points squared."""
    return 2.0 + 58.0 * np.asarray(overlap, dtype=float)


def _plot_curve_panel(
    ax,
    case: dict,
    y_limit: float,
    *,
    show_spectrum: bool,
    branch_linewidth: float = 1.45,
):
    from matplotlib import colormaps

    curves = case["curves"]
    distances = np.asarray(curves["distance_um"])
    if show_spectrum:
        spectrum_x = []
        spectrum_y = []
        spectrum_p = []
        for distance, shifts, overlaps in zip(
            distances,
            curves["spectrum_shift_mhz"],
            curves["spectrum_rr_overlap"],
        ):
            spectrum_x.extend([distance] * len(shifts))
            spectrum_y.extend(shifts)
            spectrum_p.extend(overlaps)
        probabilities = np.asarray(spectrum_p)
        ax.scatter(
            spectrum_x,
            spectrum_y,
            color="0.55",
            s=_overlap_marker_area(probabilities),
            alpha=0.38,
            linewidths=0.0,
            rasterized=True,
            zorder=1,
        )

    colours = colormaps["tab10"]
    for index, branch in enumerate(curves["branches"]):
        shifts = np.asarray(branch["shift_mhz"])
        overlaps = np.asarray(branch["rr_overlap"])
        colour = colours(index)
        ax.plot(
            distances,
            shifts,
            color=colour,
            lw=branch_linewidth,
            zorder=3,
        )
        sample = np.arange(0, distances.size, 4)
        ax.scatter(
            distances[sample],
            shifts[sample],
            s=_overlap_marker_area(overlaps[sample]),
            color=colour,
            edgecolors="white",
            linewidths=0.25,
            zorder=4,
        )

    ax.axhspan(
        -WEAK_SHIFT_THRESHOLD_MHZ,
        WEAK_SHIFT_THRESHOLD_MHZ,
        color="0.85",
        alpha=0.35,
        zorder=0,
    )
    ax.axhline(0.0, color="0.35", lw=0.65, zorder=0)
    ax.axvline(R_UM, color="0.35", lw=0.65, ls="--", zorder=0)
    ax.set_yscale("symlog", linthresh=WEAK_SHIFT_THRESHOLD_MHZ)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlim(float(distances[0]), float(distances[-1]))
    ax.grid(alpha=0.16, lw=0.5)


def _render_state_field_potential(
    result: dict,
    output_dir: Path,
    state_key: str,
    b_gauss: float,
    y_limit: float,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.lines import Line2D

    state = result["manifolds"][state_key]
    field = state["fields"][str(b_gauss)]
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(15.8, 8.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    for ax, theta_deg in zip(flat_axes, POTENTIAL_THETA_DEG):
        case = field["angles"][str(theta_deg)]
        _plot_curve_panel(ax, case, y_limit, show_spectrum=True)
        branch_count = case["curves"]["branch_count"]
        suffix = (
            ""
            if branch_count == POTENTIAL_BRANCH_COUNT
            else rf"; {branch_count} nonzero $rr$ line"
            + ("s" if branch_count != 1 else "")
        )
        ax.set_title(rf"$\theta={theta_deg:g}^\circ${suffix}")
    flat_axes[-1].axis("off")
    rank_handles = [
        Line2D(
            [0],
            [0],
            color=colormaps["tab10"](index),
            marker="o",
            lw=1.5,
            label=f"anchor rank {index + 1}",
        )
        for index in range(POTENTIAL_BRANCH_COUNT)
    ]
    rank_legend = flat_axes[-1].legend(
        handles=rank_handles,
        loc="upper center",
        frameon=False,
        title=r"branch at $R=3\,\mu$m",
    )
    flat_axes[-1].add_artist(rank_legend)
    size_handles = [
        Line2D(
            [0],
            [0],
            color="0.55",
            marker="o",
            linestyle="none",
            markersize=float(np.sqrt(_overlap_marker_area(overlap))),
            label=rf"$p_k={overlap:.1f}$",
        )
        for overlap in (0.1, 0.5, 1.0)
    ]
    flat_axes[-1].legend(
        handles=size_handles,
        loc="lower center",
        frameon=False,
        title=r"marker area: $p_k$",
    )
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Delta_k/h$ (MHz)")
    for ax in axes[-1, :3]:
        ax.set_xlabel(r"$R$ ($\mu$m)")
    fig.suptitle(
        rf"{state['label']} pair spectrum and up to five $rr$-bright branches, "
        rf"$B={b_gauss:g}$ G, $\phi=0$"
    )
    path = output_dir / f"pair_potential_{state_key}_B{b_gauss:g}G.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def render_pair_potential_figures(result: dict, output_dir: Path) -> list[Path]:
    """Render seven-angle pair spectra for every state and field."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_pair_potential_config(result)
    if result.get("status") != "complete":
        raise ValueError("pair-potential data are incomplete")
    incomplete = [
        (state_key, b_gauss, theta_deg)
        for state_key in POTENTIAL_MANIFOLDS
        for b_gauss in POTENTIAL_FIELDS_G
        for theta_deg in POTENTIAL_THETA_DEG
        if not _pair_potential_case_is_complete(
            result["manifolds"][state_key]["fields"][str(b_gauss)]["angles"].get(
                str(theta_deg)
            ),
            b_gauss,
            theta_deg,
        )
    ]
    if incomplete:
        raise ValueError(f"pair-potential data contain incomplete cases: {incomplete}")
    y_limits = _pair_potential_y_limits(result)
    paths = []
    for state_key in POTENTIAL_MANIFOLDS:
        for b_gauss in POTENTIAL_FIELDS_G:
            paths.append(
                _render_state_field_potential(
                    result,
                    output_dir,
                    state_key,
                    b_gauss,
                    y_limits[state_key],
                )
            )
    return paths


def render_manuscript_pair_spectrum_comparison(
    result: dict, output_path: Path
) -> Path:
    """Render the selected 20 G, two-state, three-angle comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _validate_pair_potential_config(result)
    if result.get("status") != "complete":
        raise ValueError("pair-potential data are incomplete")

    b_gauss = 20.0
    state_keys = ("53P3_2", "70S1_2")
    theta_values = (0.0, 45.0, 90.0)
    for state_key in state_keys:
        for theta_deg in theta_values:
            case = result["manifolds"][state_key]["fields"][str(b_gauss)][
                "angles"
            ].get(str(theta_deg))
            if not _pair_potential_case_is_complete(
                case, b_gauss, theta_deg
            ):
                raise ValueError(
                    f"incomplete manuscript panel: {state_key}, "
                    f"B={b_gauss:g} G, theta={theta_deg:g} deg"
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_limits = _pair_potential_y_limits(result)
    row_labels = (
        r"$53P_{3/2},\,m_J=-3/2$",
        r"$70S_{1/2},\,m_J=-1/2$",
    )
    with plt.rc_context(
        {
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    ):
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(7.2, 4.7),
            sharex=True,
            sharey="row",
            constrained_layout=True,
        )
        for row, (state_key, row_label) in enumerate(
            zip(state_keys, row_labels)
        ):
            field = result["manifolds"][state_key]["fields"][str(b_gauss)]
            for column, theta_deg in enumerate(theta_values):
                ax = axes[row, column]
                case = field["angles"][str(theta_deg)]
                _plot_curve_panel(
                    ax,
                    case,
                    y_limits[state_key],
                    show_spectrum=True,
                    branch_linewidth=2.0,
                )
                if row == 0:
                    ax.set_title(rf"$\theta={theta_deg:g}^\circ$")
                if row == 1:
                    ax.set_xlabel(r"$R$ ($\mu$m)")
            axes[row, 0].set_ylabel(
                row_label + "\n" + r"$\Delta_k/h$ (MHz)"
            )
        fig.align_ylabels(axes[:, 0])
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return output_path


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--pair-potentials",
        action="store_true",
        help="calculate 53P and 70S field/angle pair-potential data and figures",
    )
    mode.add_argument(
        "--plot-only",
        action="store_true",
        help="render pair-potential figures from the existing JSON",
    )
    mode.add_argument(
        "--manuscript-comparison",
        action="store_true",
        help="render the selected 20 G comparison into manuscripts/figures",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume completed pair-potential cases from the checkpoint JSON",
    )
    return parser.parse_args(argv)


def _run_pair_potentials(*, resume: bool) -> None:
    from arc import Rubidium87

    existing = None
    if resume and PAIR_POTENTIAL_OUT.exists():
        existing = json.loads(PAIR_POTENTIAL_OUT.read_text())
    result = calculate_pair_potential_study(
        Rubidium87(),
        checkpoint_path=PAIR_POTENTIAL_OUT,
        existing=existing,
    )
    paths = render_pair_potential_figures(result, PAIR_POTENTIAL_OUT.parent)
    print(f"wrote {PAIR_POTENTIAL_OUT}")
    for path in paths:
        print(f"wrote {path}")


def _render_existing_pair_potentials() -> None:
    result = json.loads(PAIR_POTENTIAL_OUT.read_text())
    paths = render_pair_potential_figures(result, PAIR_POTENTIAL_OUT.parent)
    for path in paths:
        print(f"wrote {path}")


def _render_manuscript_comparison() -> None:
    result = json.loads(PAIR_POTENTIAL_OUT.read_text())
    path = render_manuscript_pair_spectrum_comparison(
        result, MANUSCRIPT_COMPARISON_OUT
    )
    print(f"wrote {path}")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.resume and not args.pair_potentials:
        raise SystemExit("--resume requires --pair-potentials")
    if args.plot_only:
        _render_existing_pair_potentials()
        return
    if args.manuscript_comparison:
        _render_manuscript_comparison()
        return
    if args.pair_potentials:
        _run_pair_potentials(resume=args.resume)
        return
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
