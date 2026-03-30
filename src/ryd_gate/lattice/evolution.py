"""Time evolution routines for many-body Rydberg systems."""

import numpy as np
from scipy.sparse.linalg import expm_multiply

from .operators import build_hamiltonian_base


def evolve_constant_H(psi0, H, t_total, n_points):
    """Evolve under a time-independent Hamiltonian using batch expm_multiply.

    Parameters
    ----------
    psi0 : ndarray
        Initial state vector.
    H : sparse matrix
        Time-independent Hamiltonian.
    t_total : float
        Total evolution time.
    n_points : int
        Number of output time points.

    Returns
    -------
    times : ndarray, shape (n_points,)
    states : ndarray, shape (n_points, dim)
    """
    states = expm_multiply(-1j * H, psi0,
                           start=0, stop=t_total, num=n_points,
                           endpoint=True)
    times = np.linspace(0, t_total, n_points, endpoint=True)
    return times, states


def evolve_sweep(psi0, Delta_i, Delta_f, t_sweep, n_steps, pin_deltas, ops,
                 omega_ramp_frac=0.1):
    """Evolve under a time-dependent linear sweep Hamiltonian.

    Precomputes H_base (interactions + pinning) so each step only needs
    two scalar-times-sparse additions for the time-dependent Omega and Delta.

    Parameters
    ----------
    psi0 : ndarray
        Initial state vector.
    Delta_i, Delta_f : float
        Start and end detuning values.
    t_sweep : float
        Total sweep duration.
    n_steps : int
        Number of piecewise-constant time steps.
    pin_deltas : ndarray
        Per-site local detunings during the sweep.
    ops : dict
        Operator cache from build_operators.
    omega_ramp_frac : float
        Fraction of t_sweep over which Omega ramps from 0 to 1.
    """
    Omega = 1.0
    H_base = build_hamiltonian_base(pin_deltas, ops)
    psi = psi0.copy()
    dt = t_sweep / n_steps
    ramp_time = omega_ramp_frac * t_sweep

    for k in range(n_steps):
        t_mid = (k + 0.5) * dt
        frac = np.clip(t_mid / t_sweep, 0, 1)
        Delta_t = Delta_i + (Delta_f - Delta_i) * frac
        Omega_t = Omega if ramp_time == 0 else Omega * min(1.0, t_mid / ramp_time)
        H = (Omega_t / 2) * ops['sum_X'] - Delta_t * ops['sum_n'] + H_base
        psi = expm_multiply(-1j * dt * H, psi)
        psi /= np.linalg.norm(psi)
    return psi
