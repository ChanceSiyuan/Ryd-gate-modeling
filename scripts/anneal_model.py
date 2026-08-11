"""Shared TFIM anneal construction for sweep and 3x3 calibration scripts."""
from __future__ import annotations

import numpy as np

from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.physics import arc_pair_c6_rad_s_um6
from ryd_gate.protocols import SweepProtocol

RYD_LEVEL = 70
A_UM = 6.0


def rydberg_c6() -> float:
    return float(arc_pair_c6_rad_s_um6(
        n1=RYD_LEVEL, l1=0, j1=0.5, mj1=-0.5, mj2=-0.5,
        theta=0.0, phi=0.0, degenerate=False))


def build_anneal_system(
    lx: int,
    ly: int,
    c6: float,
    *,
    hz_i_w0: float,
    hx_peak_w0: float,
    t_hold_w0: float,
):
    """Build the nearest-neighbour TFIM protocol with boundary-field pins."""
    geometry = Register.rectangle(lx, ly, spacing_um=A_UM)
    cutoff_um = 1.1 * A_UM
    coords = geometry.coords
    n_sites = geometry.N
    interactions = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            distance = float(np.hypot(*(coords[j] - coords[i])))
            if distance <= cutoff_um * (1 + 1e-9):
                interactions[i, j] = interactions[j, i] = c6 / distance**6
    shift = 0.25 * interactions.sum(axis=1)
    shift_reference = float(shift.mean())
    pins = 2.0 * (shift - shift_reference)

    v_nn = c6 / A_UM**6
    w0 = v_nn / 24.0
    hx_peak = hx_peak_w0 * w0
    hz_initial, hz_final = hz_i_w0 * w0, 0.0
    t_rise = 2.0 / w0
    t_hold = t_hold_w0 / w0
    t_fall = 2.0 / w0
    t_gate = t_rise + t_hold + t_fall

    def hx(t):
        if t < t_rise:
            return hx_peak * (t / t_rise)
        if t < t_rise + t_hold:
            return hx_peak
        return hx_peak * max(
            0.0, 1.0 - (t - t_rise - t_hold) / t_fall)

    def hz(t):
        if t < t_rise:
            return hz_initial
        if t < t_rise + t_hold:
            return hz_initial + (hz_final - hz_initial) * (t - t_rise) / t_hold
        return hz_final

    protocol = SweepProtocol(
        t_gate_s=t_gate,
        omega_half_rad_s=lambda t: hx(t),
        detuning_rad_s=lambda t: 2.0 * (shift_reference - hz(t)),
        local_detuning_rad_s=lambda t, i, pins=pins: float(pins[i]),
    )
    system = RydbergSystem(
        level_structure=level_structure("1r", ryd_level=RYD_LEVEL),
        register=geometry,
        protocol=protocol,
        interaction_cutoff_um=cutoff_um,
    )
    lattice_coords = np.round(coords / A_UM).astype(int)
    sublattice = np.array([
        (-1.0) ** (x + y) for x, y in lattice_coords])
    return system, n_sites, t_gate, w0, sublattice


def peps_options(
    *,
    dt_w0: float,
    w0: float,
    bond_dimension: int,
    measurement_method: str,
) -> dict:
    return {
        "time_step_s": dt_w0 / w0,
        "bond_dimension": bond_dimension,
        "svd_tolerance": 1e-8,
        "ntu_max_iterations": 20,
        "ntu_iteration_tolerance": 1e-10,
        "measurement_method": measurement_method,
        "environment_bond_dimension": 32,
        "environment_tolerance": 1e-8,
        "environment_max_iterations": 50,
        "device": "cpu",
    }


def staggered_magnetization(n_r_t: np.ndarray, sublattice: np.ndarray):
    return (sublattice[:, None] * (2.0 * n_r_t - 1.0)).mean(axis=0)
