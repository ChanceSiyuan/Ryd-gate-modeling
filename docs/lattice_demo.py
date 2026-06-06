"""
This script demonstrates how to use the ryd_gate package to simulate a CZ gate in a lattice system.
"""

import numpy as np
from ryd_gate.model.system import InteractionSpec

from ryd_gate import RydbergSystem, SweepProtocol
from ryd_gate.backends.exact import simulate
from ryd_gate.lattice import make_square_lattice

delta_start = -3.0
delta_end = 2.5
t_sweep = 55.0
omega = 1.0
omega_ramp_frac = 0.1
addressing = {0: -4.0, 5: -4.0}   # local detuning for atoms 0 and 5


def omega_half_t(t):
    ramp_time = omega_ramp_frac * t_sweep
    return 0.5 * omega if ramp_time == 0 else 0.5 * omega * min(1.0, t / ramp_time)


def delta_t(t):
    frac = np.clip(t / t_sweep, 0.0, 1.0)
    return delta_start + (delta_end - delta_start) * frac


def address_t(t, i):
    return addressing.get(i, 0.0)


protocol = SweepProtocol(
    t_gate=t_sweep,
    omega_half_fn=omega_half_t,
    delta_fn=delta_t,
    address_fn=address_t,
    n_steps=200,
)
geometry = make_square_lattice(4, 4, spacing_um=1.0)
system = RydbergSystem.from_lattice(
    geometry,
    "1r",
    interaction=InteractionSpec(C6=24.0, mode="nn"),
    protocol=protocol,
    Omega=1.0,
)

dim = 2 ** system.N
psi0 = np.zeros(dim, dtype=complex)
psi0[0] = 1.0           # all in the ground state |g>

t_eval = np.linspace(0, t_sweep, 201)
result = simulate(system, [], psi0, t_eval=t_eval)

psi_final = result.psi_final
times = result.times
states = result.states
