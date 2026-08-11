"""Two-atom `01r` adiabatic pulse optimization — the computation behind
`results/01r_adiabatic_optimization/result.json`.

This is the orchestration source of truth for the study: a bounded 17-coordinate
spline pulse from `scripts/one_r_control.py` optimized through the `qoc` GRAPE
seam at fixed duration, each
accepted endpoint re-validated with the public `exact_ode` backend, then
continued down a fixed-step duration ladder.

`scripts/notebooks/06_01r_adiabatic_optimization.ipynb` imports from this module
and only reads the artifact and plots; it holds no computation of its own.

One-shot reproduction of the committed artifact (3 branches x 16 durations =
48 stages, 46 accepted, ~75 min measured on the DGX — the 48 `exact_ode`
validations dominate at ~57 min of that, the optimizer itself is ~18 min):

    uv run python scripts/adiabatic_01r_optimization.py --force

Verified 2026-07-30: a full rerun reproduces every recorded number in
`result.json` bit for bit (worst relative deviation 0.0 across config, all four
metric blocks, exposures, and selected parameters; every discrete field equal).

Default behaviour without `--force` is a no-op when a schema-matching artifact
is already present, so the command is safe to re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

ROOT = next(
    path for path in (Path(__file__).resolve(), *Path(__file__).resolve().parents)
    if (path / "pyproject.toml").exists()
)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import qoc  # noqa: E402
from qoc import grape  # noqa: E402
from ryd_gate import (  # noqa: E402
    Register,
    RydbergSystem,
    bilinear_control_model,
    level_structure,
    simulate,
)
from ryd_gate.physics import arc_pair_c6_rad_s_um6  # noqa: E402
from ryd_gate.protocols import DigitalAnalogProtocol, SweepProtocol, phase_from_chirp  # noqa: E402
from one_r_control import (  # noqa: E402
    A_MAX,
    BASIS,
    CHI_MAX,
    ControlBasis as ControlBasis,
    EDGE_CHIRP_MIN,
    LOGICAL_LABELS,
    MHz,
    RYD_LEVEL,
    SEED_SPECS,
    SPACING_UM,
)

RESULT_PATH = ROOT / "results" / "01r_adiabatic_optimization" / "result.json"

# Acceptance criteria.
L_MAX_ACCEPT = 1e-4
C_PHI_ACCEPT = 0.5
S_PHI_MIN = C_PHI_ACCEPT**2

# Search and validation settings. The 1 ns piecewise-constant midpoint step
# reproduces the previous CF4 engine's endpoint amplitudes to ~2e-5 on the
# seed pool, far below both acceptance thresholds.
INITIAL_DURATION_S = 2.0e-6
LADDER_STEP_S = 0.1e-6
LADDER_MIN_DURATION_S = 0.5e-6
PW_MAX_STEP_S = 1.0e-9
PHASE_SAMPLES = 4001
EXACT_TRAJECTORY_POINTS = 301
EXACT_OPTIONS = {"hamiltonian_format": "dense", "rtol": 1e-10, "atol": 1e-12}
OPTIMIZER_OPTIONS = {"maxiter": 50, "maxfun": 100, "gtol": 1e-8, "ftol": 1e-14, "maxls": 20}
SCHEMA_VERSION = 4

REGISTER = Register.chain(2, spacing_um=SPACING_UM)


# --------------------------------------------------------------------------
# Physical two-atom model
# --------------------------------------------------------------------------
def resolve_interaction():
    """Signed ARC 70S pair interaction (rad/s). Called only on a fresh compute."""
    c6 = arc_pair_c6_rad_s_um6(
        n1=RYD_LEVEL, l1=0, j1=0.5, mj1=-0.5, mj2=-0.5,
        theta=0.0, phi=0.0, degenerate=False,
    )
    return float(c6), float(c6) / SPACING_UM**6


def build_search_model():
    """Search-side bilinear control model exported from ryd_gate (ADR-0024).

    The reference protocol carries zero coefficients: it only fixes the
    channel structure (the E[r,1] quadratures and the E[r,r] detuning
    channel). The search drives the x quadrature with A/2 and the diagonal
    channel with -chi; h0 carries the geometry-resolved blockade, so the same
    compiler that serves exact validation supplies the search matrices.
    """
    reference = SweepProtocol(
        t_gate_s=INITIAL_DURATION_S,
        omega_half_rad_s=lambda t: 0.0,
        detuning_rad_s=lambda t: 0.0,
    )
    system = RydbergSystem(
        level_structure=level_structure("01r", ryd_level=RYD_LEVEL),
        register=REGISTER, protocol=reference)
    h0, channels, states = bilinear_control_model(system, states=list(LOGICAL_LABELS))
    initial_states = [states[tuple(labels)] for labels in LOGICAL_LABELS]
    return {
        "h0": h0,
        "controls": {name: channels[name] for name in ("E[r,1]:x", "E[r,r]")},
        "initial_states": initial_states,
        "indices": [int(np.argmax(np.abs(vec))) for vec in initial_states],
        "dim": int(h0.shape[0]),
    }


# --------------------------------------------------------------------------
# Bounded 17-coordinate pulse
# --------------------------------------------------------------------------
def pulse_arrays(parameters, duration_s, basis=BASIS, n_points=501):
    t = np.linspace(0.0, duration_s, n_points)
    amplitude, chirp = basis.controls(parameters, t / duration_s)
    phase = np.r_[0.0, cumulative_trapezoid(chirp, t)]
    return t, amplitude, chirp, phase


def project_coordinates(parameters, old_basis, new_basis):
    """Least-squares re-projection of the amplitude/chirp coordinate functions.

    Used by the optional spline-capacity check: retry a failed duration with a
    larger K to separate a smooth-ansatz limit from the physics.
    """
    s = np.linspace(0.0, 1.0, 401)
    old_B = old_basis.matrix(s)
    new_B = new_basis.matrix(s)
    old_p = np.asarray(parameters[: old_basis.n_coeffs])
    old_d = np.asarray(parameters[old_basis.n_coeffs : 2 * old_basis.n_coeffs])
    p_new = np.linalg.lstsq(new_B, old_B @ old_p, rcond=None)[0]
    d_new = np.linalg.lstsq(new_B, old_B @ old_d, rcond=None)[0]
    return np.r_[
        np.clip(p_new, 0.0, 1.0),
        np.clip(d_new, -3.0, 3.0),
        parameters[-1],
    ]


# --------------------------------------------------------------------------
# Finite-V_rr spline-GRAPE through the qoc seam
# --------------------------------------------------------------------------
def propagation_grid(duration_s, basis, max_step_s=PW_MAX_STEP_S):
    normalized_edges = np.unique(np.r_[0.0, 0.15, 0.85, 1.0, basis.knots])
    pieces = []
    for left, right in zip(normalized_edges[:-1], normalized_edges[1:]):
        n_steps = max(1, int(np.ceil(duration_s * (right - left) / max_step_s)))
        pieces.append(np.linspace(left, right, n_steps + 1)[:-1])
    return duration_s * np.r_[np.concatenate(pieces), 1.0]


def bisect_grid(grid):
    grid = np.asarray(grid, dtype=float)
    midpoints = 0.5 * (grid[:-1] + grid[1:])
    refined = np.empty(grid.size + midpoints.size)
    refined[0::2] = grid
    refined[1::2] = midpoints
    return refined


def search_evaluation(parameters, duration_s, basis, grid, model, *, gradient):
    """Gate metrics (and 17-coordinate gradient) through qoc's GRAPE engine.

    Controls are sampled at slice midpoints: u = A/2 on the E[r,1]:x
    quadrature and u = -chi on the E[r,r] diagonal channel; the spline chain
    rule lives entirely in the pullback.
    """
    grid = np.asarray(grid, dtype=float)
    midpoints_s = 0.5 * (grid[:-1] + grid[1:]) / duration_s
    captured = {}

    def control_map(named):
        amplitude, chirp = basis.controls(np.asarray(named["theta"], dtype=float), midpoints_s)
        return {"E[r,1]:x": 0.5 * amplitude, "E[r,r]": -chirp}

    def control_pullback(named, channel_gradients):
        _amp, _chirp, d_amplitude, d_chirp = basis.controls(
            np.asarray(named["theta"], dtype=float), midpoints_s, jacobian=True)
        return {"theta": 0.5 * channel_gradients["E[r,1]:x"] @ d_amplitude
                - channel_gradients["E[r,r]"] @ d_chirp}

    def terminal_objective(final_states):
        amplitudes = np.array([final_states[j][model["indices"][j]] for j in range(4)])
        metrics = gate_metrics(amplitudes)
        captured["metrics"] = metrics
        return metrics["cost"], objective_costates(amplitudes, metrics, model)

    named = {"theta": np.asarray(parameters, dtype=float)}
    engine = dict(
        h0=model["h0"], controls=model["controls"],
        initial_states=model["initial_states"], time_grid=grid,
        control_map=control_map, terminal_objective=terminal_objective)
    if gradient:
        _value, named_gradient = grape.value_and_grad(
            named, control_pullback=control_pullback, **engine)
        return captured["metrics"], named_gradient["theta"]
    grape.value(named, **engine)
    return captured["metrics"], None


# --------------------------------------------------------------------------
# Endpoint objective
# --------------------------------------------------------------------------
def gate_metrics(amplitudes):
    amplitudes = np.asarray(amplitudes, dtype=complex)
    losses = 1.0 - np.abs(amplitudes) ** 2
    q = amplitudes[0] * amplitudes[3] * np.conj(amplitudes[1]) * np.conj(amplitudes[2])
    q_abs = np.sqrt(abs(q) ** 2 + 1e-30)
    s_phi = 0.5 * (q_abs - q.real)
    hinge = max(0.0, 1.0 - s_phi / S_PHI_MIN)
    phi_zz = float(np.angle(q))
    return {
        "amplitudes": amplitudes,
        "losses": losses,
        "q": q,
        "L_mean": float(np.mean(losses)),
        "L_max": float(np.max(losses)),
        "Phi_ZZ": phi_zz,
        "C_phi": float(abs(np.sin(0.5 * phi_zz))),
        "S_phi": float(s_phi),
        "phase_penalty": float(hinge * hinge),
        "cost": float(np.mean(losses) + hinge * hinge),
    }


def objective_costates(amplitudes, metrics, model):
    """Terminal costates dJ/d(conj psi_b) of J = L_mean + hinge^2.

    Each logical loss contributes -a_b/4; the phase-penalty part goes through
    the Wirtinger derivatives of S_phi = (|q| - Re q)/2 with
    q = a00 a11 conj(a01) conj(a10). Verified against central finite
    differences of the full seam chain (relative error ~1e-8).
    """
    a00, a01, a10, a11 = amplitudes
    d_abar = -amplitudes / 4.0
    hinge = max(0.0, 1.0 - metrics["S_phi"] / S_PHI_MIN)
    if hinge > 0.0:
        q = metrics["q"]
        q_abs = np.sqrt(abs(q) ** 2 + 1e-30)
        d_s_dq = 0.5 * (np.conj(q) / (2.0 * q_abs) - 0.5)
        d_q_dabar = np.array([0.0, a00 * a11 * np.conj(a10), a00 * a11 * np.conj(a01), 0.0])
        d_qbar_dabar = np.array([np.conj(a11) * a01 * a10, 0.0, 0.0, np.conj(a00) * a01 * a10])
        d_abar = d_abar + (-2.0 * hinge / S_PHI_MIN) * (
            d_s_dq * d_q_dabar + np.conj(d_s_dq) * d_qbar_dabar)
    costates = []
    for value, index in zip(d_abar, model["indices"]):
        costate = np.zeros(model["dim"], dtype=complex)
        costate[index] = value
        costates.append(costate)
    return costates


def gate_passes(metrics):
    return metrics["L_max"] <= L_MAX_ACCEPT and metrics["C_phi"] >= C_PHI_ACCEPT


# --------------------------------------------------------------------------
# One fixed-duration optimization stage
# --------------------------------------------------------------------------
def scalar_metrics(metrics):
    return {
        key: float(metrics[key])
        for key in ("cost", "L_mean", "L_max", "Phi_ZZ", "C_phi", "S_phi", "phase_penalty")
    }


def append_history(history, *, branch, duration_s, stage_index, stage_iteration,
                   parameters, metrics, point_kind, grid_level, wall_time_s):
    branch_iteration = sum(entry["branch"] == branch for entry in history)
    history.append({
        "global_iteration": len(history),
        "branch_iteration": branch_iteration,
        "stage_iteration": int(stage_iteration),
        "branch": branch,
        "duration_s": float(duration_s),
        "stage_index": int(stage_index),
        "point_kind": point_kind,
        "grid_level": grid_level,
        "wall_time_s": float(wall_time_s),
        "parameters": np.asarray(parameters, dtype=float).tolist(),
        **scalar_metrics(metrics),
    })


def optimize_fixed_duration(parameters_seed, duration_s, *, branch, stage_index,
                            basis, history, grid, grid_level, model):
    started = time.perf_counter()
    cache = {"key": None, "metrics": None, "gradient": None}

    seed_metrics, _ = search_evaluation(
        parameters_seed, duration_s, basis, grid, model, gradient=False)
    append_history(
        history, branch=branch, duration_s=duration_s, stage_index=stage_index,
        stage_iteration=0, parameters=parameters_seed, metrics=seed_metrics,
        point_kind="seed", grid_level=grid_level, wall_time_s=0.0)

    # One propagation serves both the loss and the gradient: qoc never caches
    # (ADR-0023), so this explicit study-side memo shares the single
    # value_and_grad evaluation between qoc's separate loss/gradient calls.
    def evaluate(named):
        theta = np.asarray(named["theta"], dtype=float)
        key = theta.tobytes()
        if cache["key"] != key:
            metrics, gradient_value = search_evaluation(
                theta, duration_s, basis, grid, model, gradient=True)
            cache.update(key=key, metrics=metrics, gradient=gradient_value)
        return cache

    iteration = 0

    def iteration_callback(named):
        nonlocal iteration
        iteration += 1
        theta = np.asarray(named["theta"], dtype=float)
        if cache["key"] == theta.tobytes():
            metrics = cache["metrics"]
        else:
            metrics, _ = search_evaluation(
                theta, duration_s, basis, grid, model, gradient=False)
        append_history(
            history, branch=branch, duration_s=duration_s, stage_index=stage_index,
            stage_iteration=iteration, parameters=theta, metrics=metrics,
            point_kind="iteration", grid_level=grid_level,
            wall_time_s=time.perf_counter() - started)

    lower = np.array([pair[0] for pair in basis.bounds()])
    upper = np.array([pair[1] for pair in basis.bounds()])
    result = qoc.minimize(
        lambda named: evaluate(named)["metrics"]["cost"],
        {"theta": np.asarray(parameters_seed, dtype=float)},
        method="l-bfgs-b",
        bounds={"theta": (lower, upper)},
        options={
            **OPTIMIZER_OPTIONS,
            "gradient": lambda named: {"theta": evaluate(named)["gradient"]},
            "iteration_callback": iteration_callback,
        },
    )
    optimized_parameters = np.asarray(result.best_parameters["theta"], dtype=float)
    optimized_metrics, _ = search_evaluation(
        optimized_parameters, duration_s, basis, grid, model, gradient=False)
    optimizer_record = {
        "success": bool(result.success),
        "message": str(result.message),
        "nit": int(result.n_iterations),
        "nfev": int(result.n_evaluations),
        "wall_time_s": float(time.perf_counter() - started),
    }
    return {
        "seed_parameters": np.asarray(parameters_seed).copy(),
        "seed_metrics": seed_metrics,
        "optimized_parameters": optimized_parameters,
        "optimized_metrics": optimized_metrics,
        "optimizer": optimizer_record,
    }


def choose_two_points(search, *, fallback=False):
    candidates = (
        ("seed", search["seed_parameters"], search["seed_metrics"]),
        ("optimized", search["optimized_parameters"], search["optimized_metrics"]),
    )
    feasible = [c for c in candidates if gate_passes(c[2])]
    if feasible:
        return min(feasible, key=lambda c: c[2]["L_max"])
    if fallback:
        # Ladder mode never stalls: when neither point is feasible, carry the
        # lower-objective point forward (it is still exact-validated and the
        # stage is marked failed).
        return min(candidates, key=lambda c: c[2]["cost"])
    return None


# --------------------------------------------------------------------------
# Public exact-ODE validation and Rydberg exposure
# --------------------------------------------------------------------------
def physical_protocol(parameters, duration_s, basis):
    def amplitude(t_s):
        return basis.controls(parameters, t_s / duration_s)[0]

    def chirp(t_s):
        return basis.controls(parameters, t_s / duration_s)[1]

    phase = phase_from_chirp(chirp, duration_s, n_samples=PHASE_SAMPLES)
    return DigitalAnalogProtocol(
        t_gate_s=duration_s,
        coupling_r1_rad_s=lambda t_s: 0.5 * amplitude(t_s) * np.exp(-1j * phase(t_s)),
    )


def exact_validate(parameters, duration_s, basis):
    protocol = physical_protocol(parameters, duration_s, basis)
    system = RydbergSystem(
        level_structure=level_structure("01r", ryd_level=RYD_LEVEL),
        register=REGISTER, protocol=protocol)
    n_r_total = sum(system.observables.n("r", site) for site in range(system.N))
    t_eval = np.linspace(0.0, duration_s, EXACT_TRAJECTORY_POINTS)
    results = simulate(
        system, list(LOGICAL_LABELS), backend="exact_ode", t_eval=t_eval,
        observables={"n_r_total": n_r_total}, backend_options=EXACT_OPTIONS)

    amplitudes = np.array([
        result.amplitude(labels) for result, labels in zip(results, LOGICAL_LABELS)])
    metrics = gate_metrics(amplitudes)
    # Adaptive ODE roundoff can make 1-|a|^2 negative by ~machine precision;
    # exact validation reports the corresponding physical loss as zero.
    physical_losses = np.maximum(metrics["losses"], 0.0)
    metrics["losses"] = physical_losses
    metrics["L_mean"] = float(np.mean(physical_losses))
    metrics["L_max"] = float(np.max(physical_losses))
    metrics["cost"] = metrics["L_mean"] + metrics["phase_penalty"]

    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    exposures = np.array([
        trapezoid(result.expectation("n_r_total"), result.times) for result in results])
    metrics["exposures_s"] = exposures
    metrics["exposure_mean_s"] = float(np.mean(exposures))
    metrics["exposure_max_s"] = float(np.max(exposures))
    return metrics


def run_stage(parameters_seed, duration_s, *, branch, stage_index, basis,
              history, model, select_fallback=False):
    history_start = sum(entry["branch"] == branch for entry in history)
    grid = propagation_grid(duration_s, basis)
    search = optimize_fixed_duration(
        parameters_seed, duration_s, branch=branch, stage_index=stage_index,
        basis=basis, history=history, grid=grid, grid_level="N", model=model)
    selected = choose_two_points(search, fallback=select_fallback)

    stage = {
        "branch": branch,
        "stage_index": int(stage_index),
        "duration_s": float(duration_s),
        "n_coeffs": int(basis.n_coeffs),
        "grid_steps": int(len(grid) - 1),
        "optimizer": search["optimizer"],
        "history_start": history_start,
        "seed_parameters": search["seed_parameters"],
        "optimized_parameters": search["optimized_parameters"],
        "seed_metrics": search["seed_metrics"],
        "optimized_metrics": search["optimized_metrics"],
        "accepted": False,
    }
    if selected is None:
        stage["failure"] = "neither seed nor optimizer endpoint is feasible on the propagation grid"
        return stage

    source, parameters, coarse_metrics = selected
    fine_grid = bisect_grid(grid)
    fine_metrics, _ = search_evaluation(
        parameters, duration_s, basis, fine_grid, model, gradient=False)
    grid_error = {
        "steps": int(len(fine_grid) - 1),
        "delta_L_max": float(fine_metrics["L_max"] - coarse_metrics["L_max"]),
        "delta_C_phi": float(fine_metrics["C_phi"] - coarse_metrics["C_phi"]),
    }
    exact_metrics = exact_validate(parameters, duration_s, basis)
    stage.update({
        "selected_source": source,
        "selected_parameters": np.asarray(parameters).copy(),
        "search_metrics": fine_metrics,
        "grid_error": grid_error,
        "exact_metrics": exact_metrics,
        "history_end": sum(entry["branch"] == branch for entry in history) - 1,
        "accepted": bool(gate_passes(exact_metrics)),
    })
    if not stage["accepted"]:
        stage["failure"] = "public exact_ode validation failed"
    return stage


# --------------------------------------------------------------------------
# Result schema, caching, and atomic persistence
# --------------------------------------------------------------------------
def complex_record(value):
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def metrics_record(metrics, *, exact):
    record = {
        **scalar_metrics(metrics),
        "amplitudes": [complex_record(v) for v in metrics["amplitudes"]],
        "losses": np.asarray(metrics["losses"]).tolist(),
        "q": complex_record(metrics["q"]),
    }
    if exact:
        record["exposures_s"] = np.asarray(metrics["exposures_s"]).tolist()
        record["exposure_mean_s"] = float(metrics["exposure_mean_s"])
        record["exposure_max_s"] = float(metrics["exposure_max_s"])
    return record


def serialize_stage(stage):
    """The one canonical JSON-safe stage record (fresh and cached share it)."""
    record = {
        "branch": stage["branch"],
        "stage_index": int(stage["stage_index"]),
        "duration_s": float(stage["duration_s"]),
        "n_coeffs": int(stage["n_coeffs"]),
        "grid_steps": int(stage["grid_steps"]),
        "optimizer": stage["optimizer"],
        "history_start": int(stage["history_start"]),
        "seed_parameters": np.asarray(stage["seed_parameters"], dtype=float).tolist(),
        "optimized_parameters": np.asarray(stage["optimized_parameters"], dtype=float).tolist(),
        "seed_metrics": metrics_record(stage["seed_metrics"], exact=False),
        "optimized_metrics": metrics_record(stage["optimized_metrics"], exact=False),
        "accepted": bool(stage["accepted"]),
    }
    if "failure" in stage:
        record["failure"] = stage["failure"]
    if "selected_parameters" in stage:
        record["selected_source"] = stage["selected_source"]
        record["selected_parameters"] = np.asarray(
            stage["selected_parameters"], dtype=float).tolist()
        record["search_metrics"] = metrics_record(stage["search_metrics"], exact=False)
        record["grid_error"] = stage["grid_error"]
        record["exact_metrics"] = metrics_record(stage["exact_metrics"], exact=True)
        record["history_end"] = int(stage["history_end"])
    return record


def build_config(c6_rad_s, v_rr_rad_s, *, run_continuation,
                 ladder_min_duration_s=LADDER_MIN_DURATION_S):
    return {
        "ryd_level": RYD_LEVEL,
        "spacing_um": SPACING_UM,
        "register_coords_um": REGISTER.coords.tolist(),
        "C6_rad_s_um6": float(c6_rad_s),
        "V_RR_rad_s": float(v_rr_rad_s),
        "amplitude_max_rad_s": float(A_MAX),
        "chirp_max_rad_s": float(CHI_MAX),
        "endpoint_chirp_min_rad_s": float(EDGE_CHIRP_MIN),
        "acceptance": {"L_max": L_MAX_ACCEPT, "C_phi": C_PHI_ACCEPT},
        "s_phi_min": S_PHI_MIN,
        "initial_duration_s": INITIAL_DURATION_S,
        "basis_n_coeffs": int(BASIS.n_coeffs),
        "basis_degree": int(BASIS.degree),
        "basis_knots": BASIS.knots.tolist(),
        "search_engine": "qoc.grape discrete adjoint (piecewise-constant midpoint slices)",
        "pw_max_step_s": PW_MAX_STEP_S,
        "phase_samples": PHASE_SAMPLES,
        "exact_trajectory_points": EXACT_TRAJECTORY_POINTS,
        "exact_options": EXACT_OPTIONS,
        "optimizer_options": OPTIMIZER_OPTIONS,
        "run_continuation": bool(run_continuation),
        # Continuation semantics provenance: the duration ladder is the only mode.
        "continuation_mode": "ladder",
        "ladder_step_s": LADDER_STEP_S,
        "ladder_min_duration_s": float(ladder_min_duration_s),
        "seed_specs": [dict(spec) for spec in SEED_SPECS],
    }


def build_payload(config, history, stages, complete):
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "two_atom_01r_computational_return_pulse",
        "complete": bool(complete),
        "config": config,
        "history": list(history),
        "stages": list(stages),
    }


def write_result(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def load_result(path):
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------
# Duration-ladder continuation
# --------------------------------------------------------------------------
def ladder_branch(baseline_record, *, basis, history, stages, model, save,
                  min_duration_s=LADDER_MIN_DURATION_S):
    """Fixed-step duration ladder; proceeds regardless of acceptance (never stalls)."""
    branch = baseline_record["branch"]
    if "selected_parameters" not in baseline_record:
        print(f"[{branch}] baseline has no selected point; skipping ladder")
        return
    current_parameters = baseline_record["selected_parameters"]
    n_steps = int(round((INITIAL_DURATION_S - min_duration_s) / LADDER_STEP_S))

    for stage_index in range(1, n_steps + 1):
        trial_duration = INITIAL_DURATION_S - stage_index * LADDER_STEP_S
        print(f'[{branch}] ladder stage {stage_index}: T={trial_duration / 1e-6:.4f} us')
        trial = run_stage(
            current_parameters, trial_duration, branch=branch,
            stage_index=stage_index, basis=basis, history=history,
            model=model, select_fallback=True)
        record = serialize_stage(trial)
        stages.append(record)
        save(complete=False)
        exact = record["exact_metrics"]
        print(f'  {"accepted" if record["accepted"] else "failed"}: '
              f'Lmax={exact["L_max"]:.3e}, Cphi={exact["C_phi"]:.6f}')
        current_parameters = record["selected_parameters"]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _report(record):
    if record.get("accepted") and "exact_metrics" in record:
        e = record["exact_metrics"]
        print(f'  accepted: Lmax={e["L_max"]:.3e}, Cphi={e["C_phi"]:.6f}, '
              f'PhiZZ={e["Phi_ZZ"]:+.6f} rad')
    elif "exact_metrics" in record:
        e = record["exact_metrics"]
        print(f'  rejected by exact_ode: Lmax={e["L_max"]:.3e}, Cphi={e["C_phi"]:.6f}')
    else:
        print("  failed:", record.get("failure", "unknown"))


def compute_result(result_path, *, run_continuation=True,
                   ladder_min_duration_s=LADDER_MIN_DURATION_S):
    """Run the study end to end, persisting after every stage."""
    c6, v_rr = resolve_interaction()
    model = build_search_model()
    # The exported h0 must carry the same signed ARC blockade as the recorded
    # config; the (0,1,r) local order puts |rr> at the last basis index.
    rr_index = model["dim"] - 1
    assert abs(model["h0"][rr_index, rr_index] - v_rr) <= 1e-9 * abs(v_rr)
    config = build_config(c6, v_rr, run_continuation=run_continuation,
                          ladder_min_duration_s=ladder_min_duration_s)
    history, stages = [], []

    def save(*, complete):
        write_result(result_path, build_payload(config, history, stages, complete))

    for spec in SEED_SPECS:
        print(f'[{spec["branch"]}] optimizing T = {INITIAL_DURATION_S / 1e-6:.3f} us')
        stage = run_stage(
            BASIS.seed(spec["amplitude_MHz"], spec["edge_chirp_MHz"]),
            INITIAL_DURATION_S, branch=spec["branch"], stage_index=0,
            basis=BASIS, history=history, model=model)
        record = serialize_stage(stage)
        stages.append(record)
        save(complete=False)
        _report(record)

    if run_continuation:
        for baseline in [s for s in stages if s["stage_index"] == 0]:
            ladder_branch(baseline, basis=BASIS, history=history, stages=stages,
                          model=model, save=save,
                          min_duration_s=ladder_min_duration_s)

    save(complete=True)
    return config, history, stages


def load_or_compute(result_path=RESULT_PATH, *, force=False, run_continuation=True,
                    ladder_min_duration_s=LADDER_MIN_DURATION_S):
    """Return (config, history, stages), reusing a schema-matching artifact.

    This is the entry point the notebook uses: with the committed artifact in
    place it performs no optimizer / exact_ode / ARC work.
    """
    result_path = Path(result_path)
    payload = None
    if result_path.exists() and not force:
        candidate = load_result(result_path)
        if candidate.get("schema_version") == SCHEMA_VERSION:
            payload = candidate
        else:
            print(f'Cached result has schema {candidate.get("schema_version")}; '
                  f'recomputing with schema {SCHEMA_VERSION}.')

    if payload is not None:
        state = "complete" if payload.get("complete") else "PARTIAL (interrupted)"
        print(f"Loaded cached {state} result from {result_path}")
        print(f'  {len(payload["stages"])} stage(s), '
              f'{len(payload["history"])} optimizer iteration(s); '
              f"no minimize / exact_ode / ARC calls.")
        return payload["config"], payload["history"], payload["stages"]

    if result_path.exists() and force:
        print("--force: replacing the cached result with a new run.")
    return compute_result(result_path, run_continuation=run_continuation,
                          ladder_min_duration_s=ladder_min_duration_s)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=RESULT_PATH,
                        help="artifact path (default: the committed result.json)")
    parser.add_argument("--force", action="store_true",
                        help="recompute even when a schema-matching artifact exists")
    parser.add_argument("--no-continuation", action="store_true",
                        help="only the three T=2 us baselines, no duration ladder")
    parser.add_argument("--min-duration-us", type=float,
                        default=LADDER_MIN_DURATION_S / 1e-6,
                        help="ladder floor in us (default: 0.5)")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    config, history, stages = load_or_compute(
        args.out, force=args.force,
        run_continuation=not args.no_continuation,
        ladder_min_duration_s=args.min_duration_us * 1e-6)
    print(f"V_rr/2pi = {config['V_RR_rad_s'] / MHz:.6f} MHz")
    print(f"{len(stages)} stage(s), {len(history)} history entries, "
          f"{sum(s['accepted'] for s in stages)} accepted")
    print(f"total wall time: {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
