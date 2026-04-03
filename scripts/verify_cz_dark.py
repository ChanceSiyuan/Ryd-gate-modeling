"""Verify that the optimized dark-detuning TO parameters produce a valid CZ gate.

Checks:
1. Near-zero infidelity for average, SSS, and Bell fidelity metrics
2. Correct CZ phase structure: |00⟩→|00⟩, |01⟩→e^{iθ}|01⟩, |10⟩→e^{iθ}|10⟩, |11⟩→-e^{2iθ}|11⟩
3. Unitarity of the gate in the computational subspace
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

from ryd_gate.core.atomic_system import create_our_system
from ryd_gate.protocols.gate_cz_to import TOProtocol
from ryd_gate.solvers.schrodinger import solve_gate
from ryd_gate.analysis.gate_metrics import average_gate_infidelity, sss_infidelity, bell_infidelity

X_TO_OUR_DARK = [
    -0.6989301339711643, 1.0296229082590798, 0.3759232324550267,
    1.5710180991068543, 1.4454279613697887, 1.3406239758422793,
]

system = create_our_system(blackmanflag=True, detuning_sign=1)
protocol = TOProtocol()

theta = X_TO_OUR_DARK[4]

# --- 1. Fidelity metrics ---
print("=" * 60)
print("  1. Gate fidelity (all metrics)")
print("=" * 60)
_fid_funcs = {"average": average_gate_infidelity, "sss": sss_infidelity, "bell": bell_infidelity}
for ft, fn in _fid_funcs.items():
    val = fn(system, protocol, X_TO_OUR_DARK)
    print(f"  {ft:>8s} infidelity = {val:.4e}")

# --- 2. Phase structure ---
print(f"\n{'=' * 60}")
print("  2. CZ phase structure")
print("=" * 60)
print(f"  theta (single-qubit Z) = {theta:.6f} rad")

basis = {
    "|00⟩": np.kron([1+0j,0,0,0,0,0,0], [1+0j,0,0,0,0,0,0]),
    "|01⟩": np.kron([1+0j,0,0,0,0,0,0], [0,1+0j,0,0,0,0,0]),
    "|10⟩": np.kron([0,1+0j,0,0,0,0,0], [1+0j,0,0,0,0,0,0]),
    "|11⟩": np.kron([0,1+0j,0,0,0,0,0], [0,1+0j,0,0,0,0,0]),
}

# Use theta=0 placeholder to inspect raw CZ phase structure before single-qubit Z rotation
x_no_theta = [X_TO_OUR_DARK[0], X_TO_OUR_DARK[1], X_TO_OUR_DARK[2],
              X_TO_OUR_DARK[3], 0.0, X_TO_OUR_DARK[5]]

overlaps = {}
for label, ini in basis.items():
    res = solve_gate(system, protocol, x_no_theta, ini)
    overlap = ini.conj().dot(res.T)
    overlaps[label] = overlap
    phase = np.angle(overlap)
    pop = abs(overlap) ** 2
    print(f"  {label} → {label}:  |⟨out|in⟩|² = {pop:.10f},  phase = {phase:+.6f} rad")

# Check CZ structure: phases relative to |00⟩
phi00 = np.angle(overlaps["|00⟩"])
phi01 = np.angle(overlaps["|01⟩"])
phi10 = np.angle(overlaps["|10⟩"])
phi11 = np.angle(overlaps["|11⟩"])

# After removing single-qubit Z (theta per qubit):
# CZ phase on |11⟩ should be π relative to |00⟩+|01⟩+|10⟩
cz_phase = phi11 - 2 * theta - phi00
# Wrap to [-π, π]
cz_phase = (cz_phase + np.pi) % (2 * np.pi) - np.pi

print(f"\n  Relative phases (removing single-qubit Z = {theta:.4f} rad/qubit):")
print(f"    φ(|00⟩)          = {phi00:+.6f} rad")
print(f"    φ(|01⟩) - θ      = {phi01 - theta:+.6f} rad")
print(f"    φ(|10⟩) - θ      = {phi10 - theta:+.6f} rad")
print(f"    φ(|11⟩) - 2θ     = {phi11 - 2*theta:+.6f} rad")
print(f"    CZ phase (|11⟩ - 2θ - |00⟩) = {cz_phase:+.6f} rad  (expect ±π)")
print(f"    |CZ phase - π| = {abs(abs(cz_phase) - np.pi):.4e} rad")

# --- 3. Leakage check ---
print(f"\n{'=' * 60}")
print("  3. Leakage out of computational subspace")
print("=" * 60)
for label, ini in basis.items():
    res = solve_gate(system, protocol, x_no_theta, ini)
    # Population in computational subspace
    comp_pop = 0.0
    for other_ini in basis.values():
        comp_pop += abs(other_ini.conj().dot(res.T)) ** 2
    leakage = 1.0 - comp_pop
    print(f"  {label}: leakage = {leakage:.4e}")

# --- 4. Comparison with bright ---
print(f"\n{'=' * 60}")
print("  4. Comparison: dark vs bright")
print("=" * 60)
X_TO_OUR_BRIGHT = [
    -1.7370398295694707, 0.7988774460188806, 2.3116588890406224,
    0.5186261498956248, 0.900066116155231, 1.2415235064066774,
]
system_bright = create_our_system(blackmanflag=True, detuning_sign=-1)
print(f"  {'metric':<12s} {'dark':>12s} {'bright':>12s}")
print(f"  {'-'*12} {'-'*12} {'-'*12}")
for ft, fn in _fid_funcs.items():
    vd = fn(system, protocol, X_TO_OUR_DARK)
    vb = fn(system_bright, protocol, X_TO_OUR_BRIGHT)
    print(f"  {ft:<12s} {vd:>12.4e} {vb:>12.4e}")

print(f"\n  Gate times:")
print(f"    dark:   {X_TO_OUR_DARK[5] * system.time_scale * 1e9:.2f} ns")
print(f"    bright: {X_TO_OUR_BRIGHT[5] * system_bright.time_scale * 1e9:.2f} ns")

print("\nAll checks passed." if all(
    fn(system, protocol, X_TO_OUR_DARK) < 1e-6
    for fn in [average_gate_infidelity, sss_infidelity, bell_infidelity]
) else "\nWARNING: Some fidelity checks failed!")
