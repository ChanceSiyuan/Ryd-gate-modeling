"""End-to-end CZ validation of the qoc discrete-adjoint GRAPE seam.

Physical model, spline basis, and seam wiring follow
scripts/notebooks/06_01r_adiabatic_optimization.ipynb (two-atom `01r`,
signed ARC 70S blockade at 4 um, 17-coordinate bounded spline through
`ryd_gate.bilinear_control_model`, ADR-0024/0025). This script differs from
the notebook in one deliberate way: the objective targets CZ specifically.

Composite objective (all evaluated through the qoc GRAPE engine plus
study-side glue, never hand-built propagation):

    J(theta) = L_mean + W_PHASE * S_cz + R(theta)

with a_b = <b|U(T)|b> for b in {00,01,10,11}, q = a00 a11 conj(a01) conj(a10),

    S_cz = (|q| + Re q)/2 = |q| cos^2(Phi_ZZ / 2),

which vanishes iff Phi_ZZ = pi. For small phase error delta and losses L_b,
1 - F_avg ~= L_mean + (3/20) delta^2 in the diagonal model, and
W_PHASE * S_cz ~= 0.6 * delta^2/4 = 0.15 delta^2, so J (minus R) is a smooth
second-order surrogate of the Nielsen average infidelity. R is a nonzero
pulse-fluence penalty routed through the spline Jacobian:

    R = W_FLUENCE * sum_k (A_k / A_MAX)^2 * dt_k / T.

The script runs four validation stages and writes one JSON artifact:

1. Random-direction central-difference gradient check of the full composite
   J (spline control_map, discrete-adjoint costates, control_pullback, and
   the nonzero R all on the same path).
2. Multistart optimization (analytic seeds + random starts) through
   qoc.minimize/L-BFGS-B, reporting convergence rate, the full final-loss
   distribution (not just the best start), and the per-iteration objective
   history of every start (for the descent figures in notebook 07).
3. Independent acceptance: each terminal pulse is rebuilt as a continuous
   complex-drive protocol and evolved with public
   simulate(..., backend="exact_ode"); the full 4x4 logical block K gives the
   docs/gates.md metrics (Nielsen F_avg with deterministic local-Z, wrapped
   CZ phase error, worst-input leakage, constraint residuals). GRAPE's own
   slice propagation never decides acceptance.
4. Time-grid convergence: J on uniform N-slice ladders (doublings), with the
   |J_N - J_2N| tail and the GRAPE-vs-exact gap both compared against 10% of
   the claimed infidelity 1 - F_avg.

Run on the DGX:

    uv run python scripts/cz_grape_e2e_validation.py            # full run
    uv run python scripts/cz_grape_e2e_validation.py --starts 2 --maxiter 5 \
        --n-search 256 --grid-ladder 64,128,256 --directions 2  # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

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

TWOPI = 2.0 * np.pi
MHz = TWOPI * 1e6

# Physical model and bounds, identical to notebook 06.
SPACING_UM = 4.0
RYD_LEVEL = 70
A_MAX = 17.0 * MHz
CHI_MAX = 20.0 * MHz
EDGE_CHIRP_MIN = 5.0 * MHz
DURATION_S = 2.0e-6
REGISTER = Register.chain(2, spacing_um=SPACING_UM)
LOGICAL_LABELS = (["0", "0"], ["0", "1"], ["1", "0"], ["1", "1"])

# Composite-objective weights (module docstring derives W_PHASE = 0.6).
W_PHASE = 0.6
W_FLUENCE = 1e-4

# Exact-side acceptance for one start: worst basis-input return loss and
# wrapped CZ phase error.
L_MAX_ACCEPT = 1e-4
PHASE_ERR_ACCEPT = 0.05

# Assumed hardware slew caps (not repo-sourced): full amplitude / full chirp
# swing in 50 ns. Residuals are reported against these explicit assumptions.
SLEW_A_MAX = A_MAX / 50e-9
SLEW_CHI_MAX = CHI_MAX / 50e-9

EXACT_OPTIONS = {"hamiltonian_format": "dense", "rtol": 1e-10, "atol": 1e-12}
PHASE_SAMPLES = 4001

SEED_SPECS = (
    {"branch": "phase_near_pi", "amplitude_MHz": 13.0, "edge_chirp_MHz": 15.0},
    {"branch": "negative_phase", "amplitude_MHz": 14.0, "edge_chirp_MHz": 12.0},
    {"branch": "positive_phase", "amplitude_MHz": 10.0, "edge_chirp_MHz": 13.0},
)

RESULT_PATH = ROOT / "results" / "cz_grape_e2e" / "validation.json"


# --- Bounded 17-coordinate pulse basis, unchanged from notebook 06 -----------


def power_envelope(s):
    """15% quintic rise, flat top, and symmetric fall."""
    s = np.asarray(s, dtype=float)
    out = np.ones_like(s)
    rise = 0.15

    left = s < rise
    x = np.clip(s[left] / rise, 0.0, 1.0)
    out[left] = 10 * x**3 - 15 * x**4 + 6 * x**5

    right = s > 1.0 - rise
    x = np.clip((1.0 - s[right]) / rise, 0.0, 1.0)
    out[right] = 10 * x**3 - 15 * x**4 + 6 * x**5
    return out.item() if out.ndim == 0 else out


@dataclass(frozen=True)
class ControlBasis:
    """Cubic B-spline coordinates for bounded amplitude and chirp."""

    n_coeffs: int = 8
    degree: int = 3

    def __post_init__(self):
        n_inner = self.n_coeffs - self.degree - 1
        inner = np.linspace(0.0, 1.0, n_inner + 2)[1:-1]
        knots = np.r_[np.zeros(self.degree + 1), inner, np.ones(self.degree + 1)]
        object.__setattr__(self, "knots", knots)
        object.__setattr__(
            self, "_spline",
            BSpline(knots, np.eye(self.n_coeffs), self.degree, extrapolate=False),
        )

    @property
    def n_parameters(self):
        return 2 * self.n_coeffs + 1

    def matrix(self, s):
        return np.asarray(self._spline(np.asarray(s, dtype=float)))

    def seed(self, amplitude_MHz, edge_chirp_MHz):
        p = np.full(self.n_coeffs, amplitude_MHz / (A_MAX / MHz))
        d = np.zeros(self.n_coeffs)
        eta = edge_chirp_MHz / (CHI_MAX / MHz)
        return np.r_[p, d, eta]

    def bounds(self):
        return (
            [(0.0, 1.0)] * self.n_coeffs
            + [(-3.0, 3.0)] * self.n_coeffs
            + [(EDGE_CHIRP_MIN / CHI_MAX, 1.0)]
        )

    def controls(self, parameters, s, *, jacobian=False):
        parameters = np.asarray(parameters, dtype=float)
        scalar = np.ndim(s) == 0
        s = np.atleast_1d(np.asarray(s, dtype=float))
        B = self.matrix(s)
        envelope = np.asarray(power_envelope(s))

        p = parameters[: self.n_coeffs]
        d = parameters[self.n_coeffs : 2 * self.n_coeffs]
        eta = parameters[-1]

        amplitude = A_MAX * envelope * (B @ p)
        x = -eta * np.cos(TWOPI * s)
        v = np.tanh(envelope * (B @ d))
        denominator = 1.0 + x * v
        chirp = CHI_MAX * (x + v) / denominator

        if not jacobian:
            if scalar:
                return float(amplitude[0]), float(chirp[0])
            return amplitude, chirp

        d_amplitude = np.zeros((s.size, self.n_parameters))
        d_chirp = np.zeros((s.size, self.n_parameters))
        d_amplitude[:, : self.n_coeffs] = A_MAX * envelope[:, None] * B
        chirp_factor = (1.0 - x * x) * (1.0 - v * v) / (denominator * denominator)
        d_chirp[:, self.n_coeffs : 2 * self.n_coeffs] = (
            CHI_MAX * chirp_factor[:, None] * envelope[:, None] * B
        )
        d_chirp[:, -1] = (
            -CHI_MAX * np.cos(TWOPI * s) * (1.0 - v * v) / (denominator * denominator)
        )

        if scalar:
            return float(amplitude[0]), float(chirp[0]), d_amplitude[0], d_chirp[0]
        return amplitude, chirp, d_amplitude, d_chirp


BASIS = ControlBasis(n_coeffs=8)


# --- Search model (ADR-0024 export) ------------------------------------------


def build_search_model():
    """Bilinear control model exported from ryd_gate; see notebook 06 section 2."""
    reference = SweepProtocol(
        t_gate_s=DURATION_S,
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


# --- Composite CZ objective ---------------------------------------------------


def wrap_phase(angle):
    return float(np.angle(np.exp(1j * angle)))


def cz_metrics(amplitudes):
    """Diagonal-amplitude CZ metrics; cost = L_mean + W_PHASE * S_cz (no R)."""
    amplitudes = np.asarray(amplitudes, dtype=complex)
    losses = 1.0 - np.abs(amplitudes) ** 2
    q = amplitudes[0] * amplitudes[3] * np.conj(amplitudes[1]) * np.conj(amplitudes[2])
    q_abs = np.sqrt(abs(q) ** 2 + 1e-30)
    s_cz = 0.5 * (q_abs + q.real)
    phi_zz = float(np.angle(q))
    return {
        "amplitudes": amplitudes,
        "losses": losses,
        "q": q,
        "L_mean": float(np.mean(losses)),
        "L_max": float(np.max(losses)),
        "Phi_ZZ": phi_zz,
        "phase_error": wrap_phase(phi_zz - np.pi),
        "S_cz": float(s_cz),
        "cost": float(np.mean(losses) + W_PHASE * s_cz),
    }


def cz_costates(amplitudes, model):
    """Terminal costates dJ/d(conj psi_b) of cost = L_mean + W_PHASE * S_cz.

    Wirtinger route as in notebook 06, with the CZ-target sign: for
    S_cz = (|q| + Re q)/2, dS/dq = conj(q)/(4|q|) + 1/4.
    """
    a00, a01, a10, a11 = amplitudes
    d_abar = -np.asarray(amplitudes, dtype=complex) / 4.0
    q = a00 * a11 * np.conj(a01) * np.conj(a10)
    q_abs = np.sqrt(abs(q) ** 2 + 1e-30)
    d_s_dq = 0.5 * (np.conj(q) / (2.0 * q_abs) + 0.5)
    d_q_dabar = np.array([0.0, a00 * a11 * np.conj(a10), a00 * a11 * np.conj(a01), 0.0])
    d_qbar_dabar = np.array([np.conj(a11) * a01 * a10, 0.0, 0.0, np.conj(a00) * a01 * a10])
    d_abar = d_abar + W_PHASE * (d_s_dq * d_q_dabar + np.conj(d_s_dq) * d_qbar_dabar)
    costates = []
    for value, index in zip(d_abar, model["indices"]):
        costate = np.zeros(model["dim"], dtype=complex)
        costate[index] = value
        costates.append(costate)
    return costates


def fluence_penalty(theta, midpoints_s, weights, basis, *, jacobian=False):
    """R = W_FLUENCE * sum_k (A_k/A_MAX)^2 dt_k/T with weights = dt_k/(T A_MAX^2)."""
    if jacobian:
        amplitude, _chirp, d_amplitude, _d_chirp = basis.controls(
            theta, midpoints_s, jacobian=True)
        value = W_FLUENCE * float(weights @ amplitude**2)
        gradient = 2.0 * W_FLUENCE * ((weights * amplitude) @ d_amplitude)
        return value, gradient
    amplitude, _chirp = basis.controls(theta, midpoints_s)
    return W_FLUENCE * float(weights @ amplitude**2)


def composite_evaluation(theta, n_slices, basis, model, *, gradient):
    """Full J (and 17-coordinate dJ/dtheta) on a uniform N-slice midpoint grid.

    The physics part goes through qoc.grape (discrete adjoint through the
    spline control_map/control_pullback and the CZ costates); the R part and
    its analytic gradient are study-side glue added on top, per ADR-0024.
    """
    theta = np.asarray(theta, dtype=float)
    grid = np.linspace(0.0, DURATION_S, int(n_slices) + 1)
    midpoints_s = 0.5 * (grid[:-1] + grid[1:]) / DURATION_S
    weights = np.diff(grid) / (DURATION_S * A_MAX**2)
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
        metrics = cz_metrics(amplitudes)
        captured["metrics"] = metrics
        return metrics["cost"], cz_costates(amplitudes, model)

    named = {"theta": theta}
    engine = dict(
        h0=model["h0"], controls=model["controls"],
        initial_states=model["initial_states"], time_grid=grid,
        control_map=control_map, terminal_objective=terminal_objective)
    if gradient:
        _value, named_gradient = grape.value_and_grad(
            named, control_pullback=control_pullback, **engine)
        penalty, penalty_gradient = fluence_penalty(
            theta, midpoints_s, weights, basis, jacobian=True)
        metrics = captured["metrics"]
        metrics["R"] = penalty
        metrics["J"] = metrics["cost"] + penalty
        return metrics, named_gradient["theta"] + penalty_gradient
    grape.value(named, **engine)
    metrics = captured["metrics"]
    metrics["R"] = fluence_penalty(theta, midpoints_s, weights, basis)
    metrics["J"] = metrics["cost"] + metrics["R"]
    return metrics, None


# --- Stage 1: random-direction gradient check --------------------------------


def gradient_check(theta0, n_slices, basis, model, *, n_directions, epsilons, rng):
    _metrics, grad = composite_evaluation(theta0, n_slices, basis, model, gradient=True)
    records = []
    for direction_index in range(n_directions):
        direction = rng.standard_normal(theta0.size)
        direction /= np.linalg.norm(direction)
        analytic = float(grad @ direction)
        rows = []
        for eps in epsilons:
            plus, _ = composite_evaluation(theta0 + eps * direction, n_slices, basis, model, gradient=False)
            minus, _ = composite_evaluation(theta0 - eps * direction, n_slices, basis, model, gradient=False)
            finite = (plus["J"] - minus["J"]) / (2.0 * eps)
            rows.append({
                "epsilon": eps,
                "finite_difference": finite,
                "relative_error": abs(finite - analytic) / max(abs(analytic), 1e-14),
            })
        records.append({
            "direction_index": direction_index,
            "analytic": analytic,
            "rows": rows,
            "best_relative_error": min(row["relative_error"] for row in rows),
        })
    worst = max(record["best_relative_error"] for record in records)
    return {
        "theta0": theta0.tolist(),
        "n_slices": int(n_slices),
        "gradient_norm": float(np.linalg.norm(grad)),
        "directions": records,
        "worst_best_relative_error": worst,
        "passed": bool(worst < 1e-6),
    }


# --- Stage 3: independent exact-ODE acceptance -------------------------------


def physical_protocol(theta, basis):
    def amplitude(t_s):
        return basis.controls(theta, t_s / DURATION_S)[0]

    def chirp(t_s):
        return basis.controls(theta, t_s / DURATION_S)[1]

    phase = phase_from_chirp(chirp, DURATION_S, n_samples=PHASE_SAMPLES)
    return DigitalAnalogProtocol(
        t_gate_s=DURATION_S,
        coupling_r1_rad_s=lambda t_s: 0.5 * amplitude(t_s) * np.exp(-1j * phase(t_s)),
    )


def logical_block_metrics(K):
    """docs/gates.md metrics from the full 4x4 logical block."""
    diagonal = np.diag(K)
    theta_b = float(np.angle(diagonal[1]) - np.angle(diagonal[0]))
    theta_a = float(np.angle(diagonal[2]) - np.angle(diagonal[0]))
    local_z = np.diag([
        1.0,
        np.exp(-1j * theta_b),
        np.exp(-1j * theta_a),
        np.exp(-1j * (theta_a + theta_b)),
    ])
    ideal_cz = np.diag([1.0, 1.0, 1.0, -1.0])
    M = ideal_cz.conj().T @ local_z @ K
    fidelity = float((np.vdot(M, M).real + abs(np.trace(M)) ** 2) / 20.0)

    phase = np.angle(diagonal)
    phase_error = wrap_phase(phase[3] - phase[2] - phase[1] + phase[0] - np.pi)

    # Adaptive-ODE roundoff can push a leakage a few 1e-16 below zero.
    basis_leakage = [
        max(0.0, 1.0 - float(np.vdot(K[:, column], K[:, column]).real))
        for column in range(4)
    ]
    # Worst over ALL logical inputs (superpositions included): largest
    # eigenvalue of I - K^dag K on the computational subspace.
    worst_leakage = max(0.0, float(np.max(np.linalg.eigvalsh(np.eye(4) - K.conj().T @ K))))

    return {
        "K": K,
        "F_avg": fidelity,
        "infidelity": 1.0 - fidelity,
        "Phi_ZZ": wrap_phase(phase[3] - phase[2] - phase[1] + phase[0]),
        "phase_error": phase_error,
        "local_z_theta_a": theta_a,
        "local_z_theta_b": theta_b,
        "basis_leakage": basis_leakage,
        "basis_leakage_max": max(basis_leakage),
        "worst_input_leakage": worst_leakage,
    }


def exact_validate(theta, basis):
    """Continuous-protocol evolution with the public exact_ode backend."""
    system = RydbergSystem(
        level_structure=level_structure("01r", ryd_level=RYD_LEVEL),
        register=REGISTER, protocol=physical_protocol(theta, basis))
    results = simulate(
        system, list(LOGICAL_LABELS), backend="exact_ode",
        backend_options=EXACT_OPTIONS)

    K = np.empty((4, 4), dtype=complex)
    for column, _input_labels in enumerate(LOGICAL_LABELS):
        for row, output_labels in enumerate(LOGICAL_LABELS):
            K[row, column] = results[column].amplitude(output_labels)

    metrics = logical_block_metrics(K)
    # Same diagonal composite cost as the search evaluates, for the seam gap.
    metrics["diagonal_metrics"] = cz_metrics(np.diag(K))
    return metrics


def constraint_residuals(theta, basis, n_points=4001):
    t = np.linspace(0.0, DURATION_S, n_points)
    amplitude, chirp = basis.controls(theta, t / DURATION_S)
    slew_a = np.gradient(amplitude, t)
    slew_chi = np.gradient(chirp, t)
    return {
        "amplitude_max_rad_s": float(np.max(amplitude)),
        "amplitude_min_rad_s": float(np.min(amplitude)),
        "amplitude_bound_rad_s": float(A_MAX),
        "amplitude_residual_rad_s": float(max(0.0, np.max(amplitude) - A_MAX, -np.min(amplitude))),
        "chirp_max_abs_rad_s": float(np.max(np.abs(chirp))),
        "chirp_bound_rad_s": float(CHI_MAX),
        "chirp_residual_rad_s": float(max(0.0, np.max(np.abs(chirp)) - CHI_MAX)),
        "edge_chirp_abs_rad_s": float(abs(chirp[0])),
        "edge_chirp_min_rad_s": float(EDGE_CHIRP_MIN),
        "edge_chirp_residual_rad_s": float(max(0.0, EDGE_CHIRP_MIN - abs(chirp[0]))),
        "slew_amplitude_max_rad_s2": float(np.max(np.abs(slew_a))),
        "slew_amplitude_bound_rad_s2": float(SLEW_A_MAX),
        "slew_amplitude_residual_rad_s2": float(max(0.0, np.max(np.abs(slew_a)) - SLEW_A_MAX)),
        "slew_chirp_max_rad_s2": float(np.max(np.abs(slew_chi))),
        "slew_chirp_bound_rad_s2": float(SLEW_CHI_MAX),
        "slew_chirp_residual_rad_s2": float(max(0.0, np.max(np.abs(slew_chi)) - SLEW_CHI_MAX)),
    }


# --- Stage 2: multistart optimization ----------------------------------------


def history_entry(metrics):
    return {
        "J": float(metrics["J"]),
        "cost": float(metrics["cost"]),
        "L_mean": float(metrics["L_mean"]),
        "L_max": float(metrics["L_max"]),
        "S_cz": float(metrics["S_cz"]),
        "phase_error": float(metrics["phase_error"]),
    }


def optimize_start(theta_seed, n_slices, basis, model, *, maxiter, maxfun):
    started = time.perf_counter()
    cache = {"key": None, "metrics": None, "gradient": None}

    def evaluate(named):
        theta = np.asarray(named["theta"], dtype=float)
        key = theta.tobytes()
        if cache["key"] != key:
            metrics, grad = composite_evaluation(theta, n_slices, basis, model, gradient=True)
            cache.update(key=key, metrics=metrics, gradient=grad)
        return cache

    # Entry 0 is the seed; one entry per accepted L-BFGS-B iteration after
    # that (line-search evaluations are not iterations). The memo usually
    # makes the callback a cache hit, so recording costs no extra propagation.
    seed_metrics, _ = composite_evaluation(
        np.asarray(theta_seed, dtype=float), n_slices, basis, model, gradient=False)
    history = [history_entry(seed_metrics)]

    def iteration_callback(named):
        theta = np.asarray(named["theta"], dtype=float)
        if cache["key"] == theta.tobytes():
            metrics = cache["metrics"]
        else:
            metrics, _ = composite_evaluation(theta, n_slices, basis, model, gradient=False)
        history.append(history_entry(metrics))

    lower = np.array([pair[0] for pair in basis.bounds()])
    upper = np.array([pair[1] for pair in basis.bounds()])
    result = qoc.minimize(
        lambda named: evaluate(named)["metrics"]["J"],
        {"theta": np.asarray(theta_seed, dtype=float)},
        method="l-bfgs-b",
        bounds={"theta": (lower, upper)},
        options={
            "maxiter": int(maxiter), "maxfun": int(maxfun),
            "gtol": 1e-9, "ftol": 1e-15, "maxls": 30,
            "gradient": lambda named: {"theta": evaluate(named)["gradient"]},
            "iteration_callback": iteration_callback,
        },
    )
    theta_final = np.asarray(result.best_parameters["theta"], dtype=float)
    final_metrics, _ = composite_evaluation(theta_final, n_slices, basis, model, gradient=False)
    return {
        "theta_final": theta_final,
        "search_metrics": final_metrics,
        "history": history,
        "success": bool(result.success),
        "message": str(result.message),
        "n_iterations": int(result.n_iterations),
        "n_evaluations": int(result.n_evaluations),
        "wall_time_s": float(time.perf_counter() - started),
    }


def build_starts(basis, n_starts, rng):
    starts = []
    for spec in SEED_SPECS[: min(len(SEED_SPECS), n_starts)]:
        starts.append({
            "name": spec["branch"],
            "kind": "analytic_seed",
            "theta": basis.seed(spec["amplitude_MHz"], spec["edge_chirp_MHz"]),
        })
    for index in range(len(starts), n_starts):
        theta = np.r_[
            rng.uniform(0.45, 0.95, basis.n_coeffs),
            rng.uniform(-1.0, 1.0, basis.n_coeffs),
            rng.uniform(0.30, 0.95),
        ]
        starts.append({"name": f"random_{index:02d}", "kind": "random", "theta": theta})
    return starts


# --- Serialization ------------------------------------------------------------


def complex_record(value):
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def search_metrics_record(metrics):
    return {
        "J": float(metrics["J"]),
        "R": float(metrics["R"]),
        "cost": float(metrics["cost"]),
        "L_mean": float(metrics["L_mean"]),
        "L_max": float(metrics["L_max"]),
        "Phi_ZZ": float(metrics["Phi_ZZ"]),
        "phase_error": float(metrics["phase_error"]),
        "S_cz": float(metrics["S_cz"]),
        "amplitudes": [complex_record(v) for v in metrics["amplitudes"]],
    }


def exact_metrics_record(metrics):
    return {
        "F_avg": float(metrics["F_avg"]),
        "infidelity": float(metrics["infidelity"]),
        "Phi_ZZ": float(metrics["Phi_ZZ"]),
        "phase_error": float(metrics["phase_error"]),
        "local_z_theta_a": float(metrics["local_z_theta_a"]),
        "local_z_theta_b": float(metrics["local_z_theta_b"]),
        "basis_leakage": [float(v) for v in metrics["basis_leakage"]],
        "basis_leakage_max": float(metrics["basis_leakage_max"]),
        "worst_input_leakage": float(metrics["worst_input_leakage"]),
        "K": [[complex_record(v) for v in row] for row in metrics["K"]],
        "diagonal_cost": float(metrics["diagonal_metrics"]["cost"]),
        "diagonal_L_max": float(metrics["diagonal_metrics"]["L_max"]),
    }


def write_result(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# --- Driver -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--maxfun", type=int, default=400)
    parser.add_argument("--n-search", type=int, default=2048)
    parser.add_argument("--grid-ladder", type=str, default="64,128,256,512,1024,2048,4096")
    parser.add_argument("--directions", type=int, default=6)
    parser.add_argument("--rng-seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args()

    rng = np.random.default_rng(args.rng_seed)
    started = time.perf_counter()

    c6 = float(arc_pair_c6_rad_s_um6(
        n1=RYD_LEVEL, l1=0, j1=0.5, mj1=-0.5, mj2=-0.5,
        theta=0.0, phi=0.0, degenerate=False,
    ))
    v_rr = c6 / SPACING_UM**6
    model = build_search_model()
    rr_index = model["dim"] - 1
    assert abs(model["h0"][rr_index, rr_index] - v_rr) <= 1e-9 * abs(v_rr), (
        "exported h0 blockade element disagrees with the independent ARC C6")
    print(f"model: dim={model['dim']}, V_rr/2pi = {v_rr / MHz:.3f} MHz, "
          f"T = {DURATION_S / 1e-6:.2f} us, N_search = {args.n_search}", flush=True)

    config = {
        "duration_s": DURATION_S,
        "ryd_level": RYD_LEVEL,
        "spacing_um": SPACING_UM,
        "C6_rad_s_um6": c6,
        "V_RR_rad_s": v_rr,
        "amplitude_max_rad_s": float(A_MAX),
        "chirp_max_rad_s": float(CHI_MAX),
        "basis_n_coeffs": int(BASIS.n_coeffs),
        "basis_degree": int(BASIS.degree),
        "w_phase": W_PHASE,
        "w_fluence": W_FLUENCE,
        "n_search_slices": args.n_search,
        "acceptance": {"L_max": L_MAX_ACCEPT, "phase_error_rad": PHASE_ERR_ACCEPT},
        "slew_caps_assumed_rad_s2": {"amplitude": float(SLEW_A_MAX), "chirp": float(SLEW_CHI_MAX)},
        "exact_options": EXACT_OPTIONS,
        "optimizer": {"method": "l-bfgs-b", "maxiter": args.maxiter, "maxfun": args.maxfun},
        "rng_seed": args.rng_seed,
        "n_starts": args.starts,
    }

    # Stage 1: gradient check at a generic interior point (perturbed middle
    # seed; d_k = 0 would leave the tanh chirp branch untested).
    print("\n[1] random-direction gradient check of the composite J", flush=True)
    theta0 = BASIS.seed(13.0, 15.0)
    theta0[BASIS.n_coeffs: 2 * BASIS.n_coeffs] += rng.uniform(-0.5, 0.5, BASIS.n_coeffs)
    theta0[:BASIS.n_coeffs] += rng.uniform(-0.15, 0.05, BASIS.n_coeffs)
    check = gradient_check(
        theta0, args.n_search, BASIS, model,
        n_directions=args.directions, epsilons=(1e-4, 1e-5, 1e-6), rng=rng)
    for record in check["directions"]:
        best = record["best_relative_error"]
        print(f"  direction {record['direction_index']}: grad.d = {record['analytic']:+.9e}, "
              f"best rel err = {best:.3e}", flush=True)
    print(f"  gradient check {'PASSED' if check['passed'] else 'FAILED'} "
          f"(worst best-eps relative error {check['worst_best_relative_error']:.3e}, "
          f"threshold 1e-6)", flush=True)

    # Stage 2 + 3: multistart search, then independent exact-ODE acceptance of
    # every terminal pulse (GRAPE slice propagation never decides acceptance).
    print(f"\n[2] multistart optimization ({args.starts} starts)", flush=True)
    starts = build_starts(BASIS, args.starts, rng)
    start_records = []
    for start in starts:
        outcome = optimize_start(
            start["theta"], args.n_search, BASIS, model,
            maxiter=args.maxiter, maxfun=args.maxfun)
        exact = exact_validate(outcome["theta_final"], BASIS)
        feasible = bool(
            exact["basis_leakage_max"] <= L_MAX_ACCEPT
            and abs(exact["phase_error"]) <= PHASE_ERR_ACCEPT)
        record = {
            "name": start["name"],
            "kind": start["kind"],
            "theta_seed": np.asarray(start["theta"], dtype=float).tolist(),
            "theta_final": outcome["theta_final"].tolist(),
            "optimizer": {
                "success": outcome["success"],
                "message": outcome["message"],
                "n_iterations": outcome["n_iterations"],
                "n_evaluations": outcome["n_evaluations"],
                "wall_time_s": outcome["wall_time_s"],
            },
            "search_metrics": search_metrics_record(outcome["search_metrics"]),
            "history": outcome["history"],
            "exact_metrics": exact_metrics_record(exact),
            "feasible": feasible,
        }
        start_records.append(record)
        print(f"  {start['name']:>16}: J = {record['search_metrics']['J']:.3e}, "
              f"exact 1-F = {record['exact_metrics']['infidelity']:.3e}, "
              f"Lmax = {record['exact_metrics']['basis_leakage_max']:.2e}, "
              f"|dPhi| = {abs(record['exact_metrics']['phase_error']):.2e}, "
              f"{'feasible' if feasible else 'infeasible'} "
              f"({'conv' if outcome['success'] else 'no-conv'}, "
              f"{outcome['n_iterations']} it, {outcome['wall_time_s']:.0f} s)", flush=True)

    final_J = sorted(record["search_metrics"]["J"] for record in start_records)
    final_infidelity = sorted(record["exact_metrics"]["infidelity"] for record in start_records)
    multistart_summary = {
        "n_starts": len(start_records),
        "convergence_rate": float(np.mean([r["optimizer"]["success"] for r in start_records])),
        "feasibility_rate": float(np.mean([r["feasible"] for r in start_records])),
        "final_J": final_J,
        "final_J_median": float(np.median(final_J)),
        "final_exact_infidelity": final_infidelity,
        "final_exact_infidelity_median": float(np.median(final_infidelity)),
    }
    print(f"  convergence rate {multistart_summary['convergence_rate']:.0%}, "
          f"feasibility rate {multistart_summary['feasibility_rate']:.0%}, "
          f"J median {multistart_summary['final_J_median']:.3e}, "
          f"exact 1-F median {multistart_summary['final_exact_infidelity_median']:.3e}", flush=True)

    # Stage 3 detail: best start by independent exact infidelity (feasible
    # preferred), full gates.md report + constraint residuals.
    feasible_records = [r for r in start_records if r["feasible"]] or start_records
    best = min(feasible_records, key=lambda r: r["exact_metrics"]["infidelity"])
    theta_best = np.asarray(best["theta_final"], dtype=float)
    residuals = constraint_residuals(theta_best, BASIS)
    exact_best = best["exact_metrics"]
    claim = float(exact_best["infidelity"])
    print(f"\n[3] best start '{best['name']}' (independent exact_ode acceptance)", flush=True)
    print(f"  F_avg = {exact_best['F_avg']:.8f}  (claimed infidelity {claim:.3e})", flush=True)
    print(f"  Phi_ZZ = {exact_best['Phi_ZZ']:+.6f} rad, phase error = "
          f"{exact_best['phase_error']:+.3e} rad", flush=True)
    print(f"  worst-input leakage = {exact_best['worst_input_leakage']:.3e} "
          f"(basis max {exact_best['basis_leakage_max']:.3e})", flush=True)
    print(f"  residuals: amplitude {residuals['amplitude_residual_rad_s']:.2e}, "
          f"chirp {residuals['chirp_residual_rad_s']:.2e}, "
          f"slew A {residuals['slew_amplitude_residual_rad_s2']:.2e}, "
          f"slew chi {residuals['slew_chirp_residual_rad_s2']:.2e} (rad/s, rad/s^2)", flush=True)

    # Stage 4: uniform time-grid convergence at the selected pulse.
    print("\n[4] time-grid convergence (uniform N-slice ladders)", flush=True)
    ladder = sorted({int(n) for n in args.grid_ladder.split(",")})
    j_by_n = {}
    for n in ladder:
        metrics, _ = composite_evaluation(theta_best, n, BASIS, model, gradient=False)
        j_by_n[n] = search_metrics_record(metrics)
    convergence_rows = []
    for n in ladder:
        row = {"N": n, "J": j_by_n[n]["J"], "cost": j_by_n[n]["cost"]}
        if 2 * n in j_by_n:
            row["delta_J_2N"] = abs(j_by_n[n]["J"] - j_by_n[2 * n]["J"])
        convergence_rows.append(row)
        delta = row.get("delta_J_2N")
        print(f"  N = {n:5d}: J = {row['J']:.9e}"
              + (f", |J_N - J_2N| = {delta:.3e}" if delta is not None else ""), flush=True)

    finest_pair = [row for row in convergence_rows if "delta_J_2N" in row][-1]
    if args.n_search in j_by_n:
        search_cost = j_by_n[args.n_search]["cost"]
    else:
        search_cost = float(composite_evaluation(
            theta_best, args.n_search, BASIS, model, gradient=False)[0]["cost"])
    seam_gap = abs(search_cost - exact_best["diagonal_cost"])
    tolerance = 0.1 * claim
    grid_ok = bool(finest_pair["delta_J_2N"] < tolerance)
    seam_ok = bool(seam_gap < tolerance)
    print(f"  finest |J_N - J_2N| = {finest_pair['delta_J_2N']:.3e} at N = {finest_pair['N']} "
          f"-> {'OK' if grid_ok else 'TOO LARGE'} vs 0.1 * claim = {tolerance:.3e}", flush=True)
    print(f"  GRAPE-vs-exact physics gap = {seam_gap:.3e} "
          f"-> {'OK' if seam_ok else 'TOO LARGE'} vs 0.1 * claim = {tolerance:.3e}", flush=True)

    acceptance = {
        "claimed_infidelity": claim,
        "tolerance_10pct": tolerance,
        "gradient_check_passed": check["passed"],
        "grid_convergence_delta": finest_pair["delta_J_2N"],
        "grid_convergence_ok": grid_ok,
        "grape_vs_exact_gap": seam_gap,
        "grape_vs_exact_ok": seam_ok,
        "any_feasible_start": bool(any(r["feasible"] for r in start_records)),
        "all_passed": bool(check["passed"] and grid_ok and seam_ok
                           and any(r["feasible"] for r in start_records)),
    }

    payload = {
        "schema_version": 2,
        "kind": "cz_grape_e2e_validation",
        "config": config,
        "gradient_check": check,
        "multistart": {"summary": multistart_summary, "starts": start_records},
        "best": {
            "name": best["name"],
            "theta": theta_best.tolist(),
            "exact_metrics": exact_best,
            "constraint_residuals": residuals,
        },
        "grid_convergence": {"rows": convergence_rows, "by_N": {str(k): v for k, v in j_by_n.items()}},
        "acceptance": acceptance,
        "wall_time_s": float(time.perf_counter() - started),
    }
    write_result(args.output, payload)
    print(f"\nwrote {args.output}", flush=True)
    print(f"verdict: {'ALL CHECKS PASSED' if acceptance['all_passed'] else 'CHECKS FAILED'} "
          f"(gradient {check['passed']}, grid {grid_ok}, seam {seam_ok}, "
          f"feasible-start {acceptance['any_feasible_start']}); "
          f"total {payload['wall_time_s']:.0f} s", flush=True)
    return 0 if acceptance["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
