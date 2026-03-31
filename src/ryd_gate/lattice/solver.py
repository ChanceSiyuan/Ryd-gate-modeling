"""Solver for 2-level lattice systems.

Wraps the existing evolve_sweep / evolve_constant_H into a unified
interface matching the solve_gate pattern.
"""

from __future__ import annotations

import numpy as np

from ryd_gate.lattice.evolution import evolve_constant_H, evolve_sweep


def solve_lattice(
    system,
    protocol,
    x: list[float],
    initial_state: np.ndarray,
    t_eval: np.ndarray | None = None,
) -> np.ndarray:
    """Evolve a lattice system under the given sweep protocol.

    Parameters
    ----------
    system : LatticeSystem
        Lattice system with precomputed operators.
    protocol : SweepProtocol
        Sweep protocol with detuning ramp parameters.
    x : list of float
        Parameter vector [Delta_i, Delta_f, t_sweep].
    initial_state : ndarray, shape (2^N,)
        Initial state vector.
    t_eval : ndarray or None
        If given, hold at final detuning and return trajectory.

    Returns
    -------
    ndarray
        Final state (2^N,) or trajectory (n_times, 2^N).
    """
    protocol.validate_params(x)
    params = protocol.unpack_params(x, system)

    pin_deltas = protocol.get_pin_deltas(system.N)

    ops = {
        "sum_X": system.sum_X,
        "sum_n": system.sum_n,
        "n_list": system.n_list,
        "H_vdw": system.H_vdw,
    }

    psi_final = evolve_sweep(
        initial_state,
        Delta_i=params["delta_start"],
        Delta_f=params["delta_end"],
        t_sweep=params["t_gate"],
        n_steps=protocol.n_steps,
        pin_deltas=pin_deltas,
        ops=ops,
        omega_ramp_frac=protocol.omega_ramp_frac,
    )

    if t_eval is not None:
        from ryd_gate.lattice.operators import build_hamiltonian
        H_hold = build_hamiltonian(
            system.Omega, params["delta_end"], pin_deltas, ops,
        )
        t_total = t_eval[-1] - t_eval[0]
        _, states = evolve_constant_H(psi_final, H_hold, t_total, len(t_eval))
        return states

    return psi_final
