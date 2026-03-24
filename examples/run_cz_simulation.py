"""Example: run a CZ gate simulation using the modular ryd_gate package.

Demonstrates both the backward-compatible CZGateSimulator facade
and the new direct module imports.
"""

# === Approach 1: CZGateSimulator facade (backward-compatible) ===

from ryd_gate import CZGateSimulator

sim = CZGateSimulator(param_set='our', strategy='TO')

# Known good TO pulse parameters: [A, w/Omega_eff, phi_0, delta/Omega_eff, theta, T/T_scale]
X_TO = [0.1122, 1.0431, -0.72565603, 0.0, 0.452, 1.219096]

infidelity = sim.gate_fidelity(X_TO)
print(f"TO gate infidelity: {infidelity:.2e}")

# AR strategy
sim_AR = CZGateSimulator(param_set='our', strategy='AR')
X_AR = [0.85973359, 0.39146974, 0.99181418, 0.1924498,
        -1.17123748, -0.00826712, 1.67429728, 0.28527346]
infidelity_AR = sim_AR.gate_fidelity(X_AR)
print(f"AR gate infidelity: {infidelity_AR:.2e}")

# Population diagnostics
print("\nPopulation diagnostics for |11> (TO):")
mid_pop, ryd_pop, ryd_garb_pop = sim.diagnose_run(X_TO, '11')
print(f"  Peak intermediate population: {mid_pop.max():.4f}")
print(f"  Peak Rydberg population:      {ryd_pop.max():.4f}")

# === Approach 2: Direct module imports (recommended for new code) ===

print("\n--- Direct module imports ---")

from ryd_gate.core.atomic_system import create_atomic_system
from ryd_gate.protocols.gate_cz_to import TOProtocol
from ryd_gate.analysis.gate_metrics import average_gate_infidelity

system = create_atomic_system(param_set="our")
protocol = TOProtocol()
infidelity_direct = average_gate_infidelity(system, protocol, X_TO)
print(f"TO gate infidelity (direct): {infidelity_direct:.2e}")
