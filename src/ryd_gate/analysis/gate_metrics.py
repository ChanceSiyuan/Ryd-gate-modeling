"""Gate fidelity metrics, error budget, and diagnostic analysis.

Unified implementations replacing _fidelity_avg (TO) and _avg_fidelity_AR (AR),
plus error_budget, state_infidelity, and population_evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import integrate, interpolate, optimize

from ryd_gate.core.operators import build_occ_operator, build_sss_state_map

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ryd_gate.protocols.base import Protocol


def _require_rydberg_system(system) -> None:
    from ryd_gate.core.system import RydbergSystem

    if not isinstance(system, RydbergSystem):
        raise TypeError(
            "gate metrics require a RydbergSystem. Build one with "
            "RydbergSystem.set_atom_level(...).set_atom_geom(...).set_protocol(...)."
        )


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
    _require_rydberg_system(system)
    if ham_const_override is not None:
        raise NotImplementedError(
            "ham_const_override is not supported on the RydbergSystem path. "
            "Use MonteCarloRunner to add perturbation terms to compiled IR."
        )
    from ryd_gate.backends.exact import simulate

    bound = system.with_protocol(protocol).with_amplitude_scale(amplitude_scale)
    result = simulate(bound, x, state, t_eval=t_eval)
    return result.states if t_eval is not None else result.psi_final


def _observable_population(system, name: str, psi: np.ndarray, fallback_op) -> float:
    _require_rydberg_system(system)
    if system.observables.has(name):
        return system.observables.measure(name, psi)
    return float(np.real(np.vdot(psi, fallback_op @ psi)))


def _solve_trajectory(system, protocol, x, state, t_eval):
    """Return ``(state_seq, times)`` for a trajectory as a list of state vectors.

    Normalizes exact backend output onto row-major iteration: states as
    ``(n_t, dim)`` plus the times actually recorded (a piecewise backend may
    record fewer points than requested).
    """
    _require_rydberg_system(system)
    from ryd_gate.backends.exact import simulate

    res = simulate(system.with_protocol(protocol), x, state, t_eval=t_eval)
    return list(np.asarray(res.states)), np.asarray(res.times)


def _bind_cz(system, protocol, x) -> "tuple[Protocol, list[float], float]":
    """Resolve ``(pulse, eval_x, theta)`` for a metric evaluation.

    CZ *builders* (TO/AR, exposing ``build``) construct a fresh concrete
    :class:`~ryd_gate.protocols.gate_cz.CZProtocol` from ``x`` and take the
    single-qubit Z ``theta`` from ``x[theta_index]`` (theta is a scoring param,
    not part of the pulse).  Direct protocols pass through with their own theta.
    """
    if hasattr(protocol, "build"):
        pulse = protocol.build(list(x), system)
        ti = getattr(protocol, "theta_index", None)
        theta = float(x[ti]) if ti is not None else 0.0
        return pulse, [], theta
    params = protocol.unpack_params(list(x), system)
    return protocol, list(x), float(params.get("theta", 0.0))


def _cz_overlaps(
    system,
    protocol: "Protocol",
    x: list[float],
    *,
    amplitude_scale: float = 1.0,
    ham_const_override: "NDArray[np.complexfloating] | None" = None,
    collect_residuals: bool = False,
) -> "tuple[dict[str, complex], float, dict[str, float] | None]":
    """Evolve |00⟩, |01⟩, |11⟩ once and return phase-corrected CZ overlaps.

    Shared core for :func:`average_gate_infidelity` and the Stage 5 gate
    report. The corrections ``e^{-iθ}`` (|01⟩) and ``e^{-2iθ-iπ}`` (|11⟩)
    remove the ideal single-qubit Rz phases, so a perfect CZ gives
    ``a00 == a01 == a11 == 1``. Returns ``(overlaps, theta, residuals)``;
    ``residuals`` is the per-level population average over the three
    trajectories when *collect_residuals*, else ``None``.
    """
    pulse, eval_x, theta = _bind_cz(system, protocol, x)
    solve_kw = dict(ham_const_override=ham_const_override, amplitude_scale=amplitude_scale)

    basis = {
        "00": np.kron([1 + 0j, 0, 0, 0, 0, 0, 0], [1 + 0j, 0, 0, 0, 0, 0, 0]),
        "01": np.kron([1 + 0j, 0, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]),
        "11": np.kron([0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]),
    }
    corrections = {
        "00": 1.0,
        "01": np.exp(-1.0j * theta),
        "11": np.exp(-2.0j * theta - 1.0j * np.pi),
    }

    occ_ops = (
        {
            "e1": build_occ_operator(2),
            "e2": build_occ_operator(3),
            "e3": build_occ_operator(4),
            "ryd": build_occ_operator(5),
            "ryd_garb": build_occ_operator(6),
        }
        if collect_residuals
        else {}
    )
    residuals_accum = {key: 0.0 for key in occ_ops}

    overlaps: dict[str, complex] = {}
    for label, ini_state in basis.items():
        res = _solve_state(system, pulse, eval_x, ini_state, **solve_kw)
        overlaps[f"a{label}"] = corrections[label] * ini_state.conj().dot(res.T)
        for key, op in occ_ops.items():
            residuals_accum[key] += np.real(res.conj() @ op @ res)

    residuals = (
        {key: val / 3.0 for key, val in residuals_accum.items()}
        if collect_residuals
        else None
    )
    return overlaps, float(theta), residuals


def _nielsen_infidelity(overlaps: "dict[str, complex]") -> float:
    """Nielsen average-gate-infidelity from the phase-corrected CZ overlaps."""
    a00, a01, a11 = overlaps["a00"], overlaps["a01"], overlaps["a11"]
    avg_F = (1 / 20) * (
        abs(a00 + 2 * a01 + a11) ** 2
        + abs(a00) ** 2
        + 2 * abs(a01) ** 2
        + abs(a11) ** 2
    )
    return 1 - avg_F


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
    overlaps, _theta, residuals = _cz_overlaps(
        system,
        protocol,
        x,
        amplitude_scale=amplitude_scale,
        ham_const_override=ham_const_override,
        collect_residuals=return_residuals,
    )
    infidelity = _nielsen_infidelity(overlaps)

    if return_residuals:
        return float(infidelity), residuals

    return infidelity


@dataclass(frozen=True)
class CZOptimizeResult:
    """Outcome of :func:`optimize_cz_parameters`.

    ``x`` is the optimized parameter vector and ``infidelity`` its average gate
    infidelity. ``seed_infidelity`` / ``theta_infidelity`` expose the
    theta-projection warm start (infidelity at the seed, and after re-fitting
    the single-qubit phase alone) so callers can report the recovery.
    """

    x: list[float]
    infidelity: float
    seed_infidelity: float
    theta_infidelity: float
    n_eval: int


def optimize_cz_parameters(
    system,
    protocol: "Protocol",
    x0,
    *,
    polish: bool = True,
    maxiter: int = 4000,
    xatol: float = 1e-8,
    fatol: float = 1e-13,
) -> CZOptimizeResult:
    """Minimize ``average_gate_infidelity`` with a theta-projection warm start.

    Under the explicit |0> model the single-qubit Z correction
    ``theta = x[protocol.theta_index]`` is hyper-sensitive to the pulse timing:
    |0> sits at the 6.835 GHz clock splitting, so ``d(theta*)/d(T)`` ~ that
    splitting and the optimal theta winds rapidly with the gate time. That
    severely ill-conditions a *joint* search -- a wrong theta dominates the
    objective (~1e-2) and masks the ~1e-6 leakage / conditional-phase gradients,
    trapping Nelder-Mead far from the optimum.

    So this projects theta out first: a cheap 1-D bounded minimization snaps
    onto the optimal-theta ridge (typically 1e-2 -> 1e-6 in ~20 evaluations),
    then a Nelder-Mead polish refines all parameters from that warm start.
    Protocols without a single-qubit phase (``theta_index is None``) skip the
    projection and go straight to the joint polish.
    """
    x = [float(v) for v in x0]
    f = lambda xv: float(average_gate_infidelity(system, protocol, list(xv)))
    seed_infidelity = f(x)

    ti = protocol.theta_index
    if ti is not None:
        res_t = optimize.minimize_scalar(
            lambda t: f([*x[:ti], float(t), *x[ti + 1:]]),
            bounds=(x[ti] - np.pi, x[ti] + np.pi),
            method="bounded",
            options={"xatol": 1e-10},
        )
        x = [*x[:ti], float(res_t.x), *x[ti + 1:]]
        theta_infidelity = float(res_t.fun)
    else:
        theta_infidelity = seed_infidelity

    if not polish:
        return CZOptimizeResult(
            x=x, infidelity=theta_infidelity, seed_infidelity=seed_infidelity,
            theta_infidelity=theta_infidelity, n_eval=0,
        )

    res = optimize.minimize(
        f, x, method="Nelder-Mead",
        options={"xatol": xatol, "fatol": fatol, "maxiter": maxiter},
    )
    return CZOptimizeResult(
        x=res.x.tolist(), infidelity=float(res.fun), seed_infidelity=seed_infidelity,
        theta_infidelity=theta_infidelity, n_eval=int(res.nfev),
    )


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
    pulse, eval_x, theta = _bind_cz(system, protocol, x)

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

    res = _solve_state(system, pulse, eval_x, ini_state)

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

    pulse, eval_x, _ = _bind_cz(system, protocol, x)
    t_gate = pulse.unpack_params(eval_x, system)["t_gate"]
    t_eval = np.linspace(0, t_gate, 1000)

    states, t_list = _solve_trajectory(system, pulse, eval_x, ini_state, t_eval)

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
