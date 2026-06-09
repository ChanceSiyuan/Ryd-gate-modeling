"""Gate fidelity metrics, error budget, and diagnostic analysis.

Unified implementations replacing _fidelity_avg (TO) and _avg_fidelity_AR (AR),
plus error_budget, state_infidelity, and population_evolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import integrate, interpolate

from ryd_gate.core.operators import build_occ_operator, build_sss_state_map

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ryd_gate.protocols.base import Protocol


def _is_rydberg_system(system) -> bool:
    from ryd_gate.core.system import RydbergSystem

    return isinstance(system, RydbergSystem)


def _system_value(system, name: str, default=None):
    if hasattr(system, "meta"):
        return system.meta(name, default)
    return getattr(system, name, default)


def _solve_state(
    system,
    protocol: "Protocol",
    x: list[float],
    state: np.ndarray,
    *,
    t_eval: "NDArray[np.floating] | None" = None,
    ham_const_override: "NDArray[np.complexfloating] | None" = None,
    amplitude_scale: float = 1.0,
) -> "NDArray[np.complexfloating]":
    if _is_rydberg_system(system):
        if ham_const_override is not None:
            raise NotImplementedError(
                "ham_const_override is only supported by the historical AtomicSystem path. "
                "Use MonteCarloRunner to add perturbation terms to compiled IR."
            )
        from ryd_gate.backends.exact import simulate

        bound = system.with_protocol(protocol).with_amplitude_scale(amplitude_scale)
        result = simulate(bound, x, state, t_eval=t_eval)
        return result.states if t_eval is not None else result.psi_final

    from ryd_gate.backends.exact.legacy import solve_gate

    return solve_gate(
        system,
        protocol,
        x,
        state,
        t_eval=t_eval,
        ham_const_override=ham_const_override,
        amplitude_scale=amplitude_scale,
    )


def _observable_population(system, name: str, psi: np.ndarray, fallback_op) -> float:
    if _is_rydberg_system(system) and system.observables.has(name):
        return system.observables.measure(name, psi)
    return float(np.real(np.vdot(psi, fallback_op @ psi)))


def _solve_trajectory(system, protocol, x, state, t_eval):
    """Return ``(state_seq, times)`` for a trajectory as a list of state vectors.

    Normalizes the two backend conventions onto row-major iteration: the new
    RydbergSystem exact backend returns states as ``(n_t, dim)`` and reports the
    times it actually recorded (a piecewise backend may record fewer points than
    requested); the legacy AtomicSystem path returns column-major ``(dim, n_t)``
    interpolated to ``t_eval``.
    """
    if _is_rydberg_system(system):
        from ryd_gate.backends.exact import simulate

        res = simulate(system.with_protocol(protocol), x, state, t_eval=t_eval)
        return list(np.asarray(res.states)), np.asarray(res.times)
    res_list = np.asarray(_solve_state(system, protocol, x, state, t_eval=t_eval))
    return [res_list[:, k] for k in range(res_list.shape[1])], np.asarray(t_eval)


def average_gate_infidelity(
    system,
    protocol: "Protocol",
    x: list[float],
    return_residuals: bool = False,
    ham_const_override: "NDArray[np.complexfloating] | None" = None,
    amplitude_scale: float = 1.0,
) -> "float | tuple[float, dict[str, float]]":
    """Compute average gate infidelity using the Nielsen formula.

    Unified implementation replacing _fidelity_avg (TO) and _avg_fidelity_AR (AR).
    Evolves |00⟩, |01⟩, |11⟩ and computes overlaps with ideal CZ + Rz output.

    Parameters
    ----------
    system
        Atomic system.
    protocol : Protocol
        Pulse protocol.
    x : list of float
        Pulse parameters.
    return_residuals : bool
        If True, return (infidelity, residuals_dict).
    ham_const_override : ndarray or None
        Perturbed Hamiltonian for MC shots.
    amplitude_scale : float
        Multiplicative scale on 420nm laser amplitude (default 1.0).

    Returns
    -------
    float or (float, dict)
        Average gate infidelity, optionally with residual populations.
    """
    params = protocol.unpack_params(x, system)
    theta = params["theta"]

    solve_kw = dict(ham_const_override=ham_const_override, amplitude_scale=amplitude_scale)

    if return_residuals:
        occ_ops = {
            "e1": build_occ_operator(2),
            "e2": build_occ_operator(3),
            "e3": build_occ_operator(4),
            "ryd": build_occ_operator(5),
            "ryd_garb": build_occ_operator(6),
        }
        residuals_accum = {key: 0.0 for key in occ_ops}

    # |00⟩
    ini_state_00 = np.kron(
        [1 + 0j, 0, 0, 0, 0, 0, 0], [1 + 0j, 0, 0, 0, 0, 0, 0]
    )
    res = _solve_state(system, protocol, x, ini_state_00, **solve_kw)
    a00 = ini_state_00.conj().dot(res.T)

    if return_residuals:
        for key, op in occ_ops.items():
            residuals_accum[key] += np.real(res.conj() @ op @ res)

    # |01⟩
    ini_state = np.kron(
        [1 + 0j, 0, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
    )
    res = _solve_state(system, protocol, x, ini_state, **solve_kw)
    a01 = np.exp(-1.0j * theta) * ini_state.conj().dot(res.T)

    if return_residuals:
        for key, op in occ_ops.items():
            residuals_accum[key] += np.real(res.conj() @ op @ res)

    # |11⟩
    ini_state = np.kron(
        [0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
    )
    res = _solve_state(system, protocol, x, ini_state, **solve_kw)
    a11 = np.exp(-2.0j * theta - 1.0j * np.pi) * ini_state.conj().dot(res.T)

    if return_residuals:
        for key, op in occ_ops.items():
            residuals_accum[key] += np.real(res.conj() @ op @ res)

    # Nielsen formula
    avg_F = (1 / 20) * (
        abs(a00 + 2 * a01 + a11) ** 2
        + abs(a00) ** 2
        + 2 * abs(a01) ** 2
        + abs(a11) ** 2
    )
    infidelity = 1 - avg_F

    if return_residuals:
        residuals = {key: val / 3.0 for key, val in residuals_accum.items()}
        return float(infidelity), residuals

    return infidelity


def sss_infidelity(
    system,
    protocol: "Protocol",
    x: list[float],
) -> float:
    """Average state infidelity over 12 SSS states."""
    return sum(
        state_infidelity(system, protocol, x, f"SSS-{i}")
        for i in range(12)
    ) / 12


def bell_infidelity(
    system,
    protocol: "Protocol",
    x: list[float],
) -> float:
    """Average state infidelity over 4 Bell states."""
    s0 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=complex)
    s1 = np.array([0, 1, 0, 0, 0, 0, 0], dtype=complex)
    st00, st01 = np.kron(s0, s0), np.kron(s0, s1)
    st10, st11 = np.kron(s1, s0), np.kron(s1, s1)
    bell_states = [
        (st00 + st11) / np.sqrt(2),
        (st00 - st11) / np.sqrt(2),
        (st01 + st10) / np.sqrt(2),
        (st01 - st10) / np.sqrt(2),
    ]
    return sum(state_infidelity(system, protocol, x, b) for b in bell_states) / 4


def state_infidelity(
    system,
    protocol: "Protocol",
    x: list[float],
    initial_state,
) -> float:
    """Compute state infidelity for a specific initial state.

    Parameters
    ----------
    initial_state : str or ndarray
        State label or 49-D state vector.
    """
    params = protocol.unpack_params(x, system)
    theta = params["theta"]

    if isinstance(initial_state, str):
        sss_states = build_sss_state_map()
        if initial_state not in sss_states:
            raise ValueError(f"Unsupported initial state: '{initial_state}'")
        ini_state = sss_states[initial_state]
    else:
        ini_state = np.asarray(initial_state, dtype=complex)

    s0 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=complex)
    s1 = np.array([0, 1, 0, 0, 0, 0, 0], dtype=complex)
    state_00 = np.kron(s0, s0)
    state_01 = np.kron(s0, s1)
    state_10 = np.kron(s1, s0)
    state_11 = np.kron(s1, s1)

    res = _solve_state(system, protocol, x, ini_state)

    c00 = np.vdot(state_00, ini_state)
    c01 = np.vdot(state_01, ini_state)
    c10 = np.vdot(state_10, ini_state)
    c11 = np.vdot(state_11, ini_state)

    psi_ideal = (
        c00 * state_00
        + c01 * np.exp(+1j * theta) * state_01
        + c10 * np.exp(+1j * theta) * state_10
        + c11 * np.exp(+1j * (2 * theta + np.pi)) * state_11
    )

    fid = np.abs(np.vdot(res, psi_ideal)) ** 2
    return 1.0 - fid


def population_evolution(
    system,
    protocol: "Protocol",
    x: list[float],
    initial_state: str,
) -> "dict[str, NDArray[np.floating]]":
    """Run gate simulation and return per-level population time series.

    Returns
    -------
    dict
        Keys: 't_list', 'e1', 'e2', 'e3', 'ryd', 'ryd_garb'.
    """
    sss_states = build_sss_state_map()
    if initial_state not in sss_states:
        raise ValueError(f"Unsupported initial state: '{initial_state}'")
    ini_state = sss_states[initial_state]

    params = protocol.unpack_params(x, system)
    t_gate = params["t_gate"]
    t_eval = np.linspace(0, t_gate, 1000)

    states, t_list = _solve_trajectory(system, protocol, x, ini_state, t_eval)

    occ_specs = {
        "e1": ("pop_e1", build_occ_operator(2)),
        "e2": ("pop_e2", build_occ_operator(3)),
        "e3": ("pop_e3", build_occ_operator(4)),
        "ryd": ("pop_r", build_occ_operator(5)),
        "ryd_garb": ("pop_r_garb", build_occ_operator(6)),
    }

    result = {"t_list": np.asarray(t_list)}
    for key, (obs_name, op) in occ_specs.items():
        result[key] = np.array([
            _observable_population(system, obs_name, psi, op) for psi in states
        ])

    return result


def residuals_to_branching(
    system,
    residuals: dict[str, float],
) -> dict[str, float]:
    """Convert per-level residual populations to XYZ/AL/LG error components."""
    xyz = 0.0
    al = 0.0
    lg = 0.0

    al += residuals["ryd"] + residuals["ryd_garb"]

    for F, key in [(1, "e1"), (2, "e2"), (3, "e3")]:
        mid_res = residuals[key]
        mbr = _system_value(system, "mid_branch")[F]
        xyz += mid_res * (mbr["to_0"] + mbr["to_1"])
        lg += mid_res * (mbr["to_L0"] + mbr["to_L1"])

    return {"XYZ": xyz, "AL": al, "LG": lg}


def error_budget(
    system,
    protocol: "Protocol",
    x: list[float],
    initial_states: list[str] | None = None,
) -> dict:
    """Compute error budget with XYZ/AL/LG breakdown by error source."""
    if initial_states is None:
        initial_states = ["01", "11"]

    budget_accum = {
        "rydberg_decay": {"XYZ": 0.0, "AL": 0.0, "LG": 0.0},
        "intermediate_decay": {"XYZ": 0.0, "AL": 0.0, "LG": 0.0},
        "polarization_leakage": {"XYZ": 0.0, "AL": 0.0, "LG": 0.0},
    }

    for init_state in initial_states:
        pops = population_evolution(system, protocol, x, init_state)
        t_list = pops["t_list"]

        # --- Rydberg decay ---
        ryd_occ = pops["ryd"]
        ryd_garb_occ = pops["ryd_garb"]

        rd_decay = decay_integrate(t_list, ryd_occ, _system_value(system, "ryd_RD_rate"))[0, -1]
        bbr_decay = decay_integrate(t_list, ryd_occ, _system_value(system, "ryd_BBR_rate"))[0, -1]
        ryd_residual = ryd_occ[-1]

        br = _system_value(system, "ryd_branch")
        xyz_frac = br["to_0"] + br["to_1"]
        lg_frac = br["to_L0"] + br["to_L1"]

        budget_accum["rydberg_decay"]["XYZ"] += rd_decay * xyz_frac
        budget_accum["rydberg_decay"]["LG"] += rd_decay * lg_frac
        budget_accum["rydberg_decay"]["AL"] += bbr_decay + ryd_residual

        # --- Polarization leakage ---
        garb_rd_decay = decay_integrate(t_list, ryd_garb_occ, _system_value(system, "ryd_RD_rate"))[0, -1]
        garb_bbr_decay = decay_integrate(t_list, ryd_garb_occ, _system_value(system, "ryd_BBR_rate"))[0, -1]
        garb_residual = ryd_garb_occ[-1]

        budget_accum["polarization_leakage"]["XYZ"] += garb_rd_decay * xyz_frac
        budget_accum["polarization_leakage"]["LG"] += garb_rd_decay * lg_frac
        budget_accum["polarization_leakage"]["AL"] += garb_bbr_decay + garb_residual

        # --- Intermediate decay ---
        mid_xyz = 0.0
        mid_lg = 0.0
        for F, level_key in [(1, "e1"), (2, "e2"), (3, "e3")]:
            mid_occ = pops[level_key]
            mid_decay_total = decay_integrate(
                t_list, mid_occ, _system_value(system, "mid_state_decay_rate"),
            )[0, -1]
            mid_residual = mid_occ[-1]

            mbr = _system_value(system, "mid_branch")[F]
            m_xyz_frac = mbr["to_0"] + mbr["to_1"]
            m_lg_frac = mbr["to_L0"] + mbr["to_L1"]

            mid_xyz += mid_decay_total * m_xyz_frac
            mid_lg += mid_decay_total * m_lg_frac
            mid_xyz += mid_residual * m_xyz_frac
            mid_lg += mid_residual * m_lg_frac

        budget_accum["intermediate_decay"]["XYZ"] += mid_xyz
        budget_accum["intermediate_decay"]["LG"] += mid_lg

    n = len(initial_states)
    result = {}
    for source, errors in budget_accum.items():
        total = sum(errors.values()) / n
        result[source] = {
            "total": total,
            "XYZ": errors["XYZ"] / n,
            "AL": errors["AL"] / n,
            "LG": errors["LG"] / n,
        }
    return result


def decay_integrate(
    t_list: "NDArray[np.floating]",
    occ_list: "NDArray[np.floating]",
    decay_rate: float,
) -> "NDArray[np.floating]":
    """Integrate decay probability given time-dependent occupation."""
    poly_interpolation = interpolate.CubicSpline(t_list, occ_list)
    args = (poly_interpolation, decay_rate)

    def fun(t, _y, poly_interpolation, decay_rate):
        diff = decay_rate * poly_interpolation.__call__(t)
        return np.array([diff])

    t_span = [0, t_list[-1]]
    result = integrate.solve_ivp(
        fun,
        t_span,
        np.array([0]),
        t_eval=t_list,
        args=args,
        method="DOP853",
        rtol=1e-8,
        atol=1e-12,
    )
    return np.array(result.y)
