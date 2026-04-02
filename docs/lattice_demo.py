"""
This script demonstrates how to use the ryd_gate package to simulate a CZ gate in a lattice system.
"""

import numpy as np
from ryd_gate import create_lattice_system, SweepProtocol, simulate

system = create_lattice_system(Lx=4, Ly=4, V_nn=24.0, Omega=1.0)
protocol = SweepProtocol(
    addressing={0: -4.0, 5: -4.0},   # Setting local detuning for atoms 0: (0, 0) and 5: (1, 1)
    omega_ramp_frac=0.1,
    n_steps=200,
)

x = [-3.0, 2.5, 55.0]   # [delta_start, delta_end, t_sweep] for LatticeSystem
dim = 2 ** system.N
psi0 = np.zeros(dim, dtype=complex)
psi0[0] = 1.0           # all in the ground state |g>

t_eval = np.linspace(0, x[2], 201)
result = simulate(system, protocol, x, psi0, t_eval=t_eval)

psi_final = result.psi_final
times = result.times
states = result.states