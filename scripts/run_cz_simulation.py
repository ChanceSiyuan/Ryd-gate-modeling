"""Example: run a CZ gate simulation using the modular ryd_gate package."""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

from ryd_gate.core.atomic_system import create_our_system
from ryd_gate.protocols.gate_cz_to import TOProtocol
from ryd_gate.protocols.gate_cz_ar import ARProtocol
from ryd_gate.analysis.gate_metrics import average_gate_infidelity, population_evolution

# Known good TO pulse parameters: [A, w/Omega_eff, phi_0, delta/Omega_eff, theta, T/T_scale]
X_TO = [0.1122, 1.0431, -0.72565603, 0.0, 0.452, 1.219096]

system = create_our_system()
protocol_to = TOProtocol()

infidelity = average_gate_infidelity(system, protocol_to, X_TO)
print(f"TO gate infidelity: {infidelity:.2e}")

# AR strategy
protocol_ar = ARProtocol()
X_AR = [0.85973359, 0.39146974, 0.99181418, 0.1924498,
        -1.17123748, -0.00826712, 1.67429728, 0.28527346]
infidelity_AR = average_gate_infidelity(system, protocol_ar, X_AR)
print(f"AR gate infidelity: {infidelity_AR:.2e}")

# Population diagnostics
print("\nPopulation diagnostics for |11> (TO):")
pops = population_evolution(system, protocol_to, X_TO, "11")
mid_pop = pops["e1"] + pops["e2"] + pops["e3"]
ryd_pop = pops["ryd"]
print(f"  Peak intermediate population: {mid_pop.max():.4f}")
print(f"  Peak Rydberg population:      {ryd_pop.max():.4f}")
