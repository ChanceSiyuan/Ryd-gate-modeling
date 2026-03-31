"""Time evolution routines for many-body Rydberg systems (2-level and 3-level)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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


# ── 3-level evolution ────────────────────────────────────────────────


@dataclass
class EvolutionResult:
    """Results from a 3-level lattice evolution."""

    psi_final: np.ndarray
    times: np.ndarray | None = None
    states: np.ndarray | None = None


def evolve_3level_sweep(
    psi0: np.ndarray,
    ops,
    delta_start: float,
    delta_end: float,
    t_sweep: float,
    n_steps: int,
    pin_atoms: list[int] | None = None,
    pin_detuning: float = 0.0,
    amplitude_fn: Callable[[float], float] | None = None,
    store_every: int = 0,
    verbose: bool = False,
) -> EvolutionResult:
    """Evolve under a 3-level sweep Hamiltonian with quadratic phase.

    H(t) = H_static + A(t)[e^{-iφ(t)} H_420 + h.c.]
    where φ(t) = ω₀t + αt²/2.
    """
    H_static = ops.H_const + ops.H_1013 + ops.H_1013_dag
    if pin_atoms:
        for i in pin_atoms:
            H_static = H_static - pin_detuning * ops.n_r_list[i]

    psi = psi0.copy().astype(complex)
    dt = t_sweep / n_steps
    omega_0 = delta_start
    chirp = (delta_end - delta_start) / t_sweep

    stored_times = []
    stored_states = []

    for k in range(n_steps):
        t_mid = (k + 0.5) * dt
        phi = omega_0 * t_mid + 0.5 * chirp * t_mid**2
        phase = np.exp(-1j * phi)
        A_t = amplitude_fn(t_mid) if amplitude_fn is not None else 1.0

        H = H_static + A_t * (phase * ops.H_420_uniform
                               + phase.conj() * ops.H_420_uniform_dag)
        psi = expm_multiply(-1j * dt * H, psi)
        psi /= np.linalg.norm(psi)

        if store_every > 0 and (k + 1) % store_every == 0:
            stored_times.append((k + 1) * dt)
            stored_states.append(psi.copy())

        if verbose and (k + 1) % max(1, n_steps // 10) == 0:
            print(f"  Step {k+1}/{n_steps}")

    result = EvolutionResult(psi_final=psi)
    if stored_states:
        result.times = np.array(stored_times)
        result.states = np.array(stored_states)
    return result
