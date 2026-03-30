"""Gaussian-waist sensitivity & iso-fidelity surface error decomposition.

Two analyses that improve upon the uniform-relative-scaling approach in
``calibration_sensitivity.py``:

1. **Gaussian-waist 1D scan** — for each pulse parameter, sweep its value
   while keeping others at the optimum, fit a Gaussian to the fidelity
   curve to extract σ_i (beam waist).  Then use ``p_opt ± α·σ_i`` with a
   single unified coefficient α for sensitivity analysis.

2. **Iso-fidelity surface decomposition** — sample random directions in
   the σ-normalised parameter space, find the surface where infidelity
   increases by 0.1 %, and average the XYZ / AL / LG / Phase error
   fractions on that surface.

See GitHub issue #36 for motivation.

Estimated runtime (each infidelity evaluation ≈ 12 s):
  Part 1  (5 × 51 sweep + range detection): ~1 h
  Part 1b (α search):                       ~12 min
  Part 2  (100 directions × bisection):     ~7 h
  Total:                                    ~8 h
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from ryd_gate.ideal_cz import CZGateSimulator

# ---------------------------------------------------------------------------
# Optimised dark CZ gate parameters (from opt_dark.py)
# ---------------------------------------------------------------------------
X_TO_OUR_DARK = [
    -0.6989301339711643,
    1.0296229082590798,
    0.3759232324550267,
    1.5710180991068543,
    1.4454279613697887,
    1.3406239758422793,
]

PARAM_NAMES = {
    0: "A (cosine amplitude)",
    1: "ω/Ω_eff (mod. freq.)",
    2: "φ₀ (initial phase)",
    3: "δ/Ω_eff (chirp rate)",
    5: "T/T_scale (gate time)",
}
PARAM_INDICES = [0, 1, 2, 3, 5]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SWEEP_POINTS = 51                # per-parameter sweep (keep odd for centre)
INFIDELITY_CEILING = 0.01          # auto-expand sweep until this is reached
INITIAL_HALF_WIDTH_ABS = 0.02      # seed half-width for sweep range search

SIMULTANEOUS_TARGET = 0.001        # target infidelity for α search

ISO_FIDELITY_DROP = 0.001          # 0.1 % fidelity drop
N_DIRECTIONS = 100                 # random directions for iso-surface
BISECTION_TOL = 1e-4               # sufficient precision for surface location
BISECTION_MAX_ITER = 25
MAX_ALPHA_SCALE = 20.0
DIRECTION_SEED = 42

# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
sim = CZGateSimulator(
    param_set="our",
    strategy="TO",
    blackmanflag=True,
    detuning_sign=+1,
)


# ===================================================================
# Helpers
# ===================================================================

def _infidelity_at(x: list[float]) -> float:
    """Deterministic average infidelity for parameter vector *x*."""
    return sim._gate_infidelity_single(x, fid_type="average")


def _infidelity_shifted(idx: int, dp: float) -> float:
    """Infidelity when parameter *idx* is shifted by *dp* from optimum."""
    x = list(X_TO_OUR_DARK)
    x[idx] += dp
    return _infidelity_at(x)


# ===================================================================
# Part 1 — Gaussian-waist 1D scan
# ===================================================================

def _determine_sweep_range(idx: int) -> float:
    """Return half-width for the 1D sweep of parameter *idx*.

    Starts from a seed value and doubles until the endpoint infidelity
    exceeds ``INFIDELITY_CEILING``.
    """
    hw = INITIAL_HALF_WIDTH_ABS
    for _ in range(20):
        inf_plus = _infidelity_shifted(idx, +hw)
        inf_minus = _infidelity_shifted(idx, -hw)
        if inf_plus >= INFIDELITY_CEILING and inf_minus >= INFIDELITY_CEILING:
            break
        hw *= 2.0
    return hw


def _gaussian_fidelity(p: np.ndarray, f_peak: float, p0: float,
                        sigma: float) -> np.ndarray:
    """Gaussian model: F(p) = F_peak · exp(-(p - p0)² / (2σ²))."""
    return f_peak * np.exp(-(p - p0) ** 2 / (2 * sigma ** 2))


def _sweep_and_fit(idx: int, n_points: int = N_SWEEP_POINTS) -> dict:
    """1D sweep of parameter *idx* and Gaussian fit to the fidelity curve.

    Returns a dict with sweep data, fitted σ, and fit diagnostics.
    """
    p_opt = X_TO_OUR_DARK[idx]
    hw = _determine_sweep_range(idx)

    p_values = np.linspace(p_opt - hw, p_opt + hw, n_points)
    infidelities = np.array([_infidelity_shifted(idx, p - p_opt)
                             for p in p_values])
    fidelities = 1.0 - infidelities

    # --- initial guesses ---
    f_peak_guess = float(fidelities.max())
    p0_guess = float(p_values[np.argmax(fidelities)])

    # estimate σ from the width at F = F_peak / e
    threshold = f_peak_guess / np.e
    above = np.where(fidelities > threshold)[0]
    if len(above) >= 2:
        sigma_guess = (p_values[above[-1]] - p_values[above[0]]) / 2.0
    else:
        sigma_guess = hw / 3.0

    try:
        popt, pcov = curve_fit(
            _gaussian_fidelity, p_values, fidelities,
            p0=[f_peak_guess, p0_guess, max(sigma_guess, 1e-15)],
            bounds=([0.0, p_opt - hw, 1e-15],
                    [1.0 + 1e-10, p_opt + hw, hw * 3]),
            maxfev=10000,
        )
        success = True
    except (RuntimeError, ValueError):
        popt = np.array([f_peak_guess, p0_guess, sigma_guess])
        pcov = np.full((3, 3), np.nan)
        success = False

    return {
        "p_values": p_values,
        "infidelities": infidelities,
        "fidelities": fidelities,
        "sigma": abs(popt[2]),
        "p0_fit": popt[1],
        "F_peak_fit": popt[0],
        "fit_params": popt,
        "fit_cov": pcov,
        "fit_success": success,
    }


def _plot_gaussian_waists(results: dict[int, dict],
                          save_path: str = "gaussian_waist_scan.pdf") -> None:
    """Plot 1D fidelity curves with Gaussian fits for all parameters."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    for i, idx in enumerate(PARAM_INDICES):
        ax = axes_flat[i]
        r = results[idx]

        ax.semilogy(r["p_values"], r["infidelities"], "b.", ms=2,
                     label="Data")

        p_fine = np.linspace(r["p_values"][0], r["p_values"][-1], 500)
        f_fit = _gaussian_fidelity(p_fine, *r["fit_params"])
        ax.semilogy(p_fine, 1.0 - f_fit, "r-", lw=1.5, label="Gaussian fit")

        p0 = r["p0_fit"]
        sigma = r["sigma"]
        for mult, ls in [(1, "--"), (2, ":"), (3, "-.")]:
            ax.axvline(p0 - mult * sigma, color="gray", ls=ls, alpha=0.5)
            ax.axvline(p0 + mult * sigma, color="gray", ls=ls, alpha=0.5)

        ax.set_title(f"{PARAM_NAMES[idx]}\n$\\sigma$ = {sigma:.6f}")
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Infidelity")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes_flat[5].set_visible(False)

    fig.suptitle("Per-Parameter Gaussian Waist Scan", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved plot → {save_path}")


def run_gaussian_waist_scan(plot: bool = True) -> dict[int, float]:
    """Part 1: per-parameter Gaussian-waist 1D scan.

    Returns ``{param_idx: sigma_i}``.
    """
    print(f"\n{'=' * 72}")
    print("Part 1: Per-parameter Gaussian waist 1D scan")
    print(f"{'=' * 72}\n")

    baseline = _infidelity_at(X_TO_OUR_DARK)
    print(f"  Baseline infidelity: {baseline:.4e}\n")

    results: dict[int, dict] = {}
    sigmas: dict[int, float] = {}

    for idx in PARAM_INDICES:
        print(f"  Scanning {PARAM_NAMES[idx]} …")
        r = _sweep_and_fit(idx)
        results[idx] = r
        sigmas[idx] = r["sigma"]
        status = "OK" if r["fit_success"] else "FAILED"
        print(f"    σ = {r['sigma']:.8f}  p0_fit = {r['p0_fit']:.8f}  "
              f"F_peak = {r['F_peak_fit']:.10f}  fit: {status}")

    # summary table
    print(f"\n  {'Parameter':<26} {'p_opt':>12} {'σ_i':>14} "
          f"{'σ/|p|':>12}")
    print(f"  {'-' * 66}")
    for idx in PARAM_INDICES:
        p_opt = X_TO_OUR_DARK[idx]
        sig = sigmas[idx]
        rel = abs(sig / p_opt) if p_opt != 0 else float("inf")
        print(f"  {PARAM_NAMES[idx]:<26} {p_opt:>12.8f} {sig:>14.8f} "
              f"{rel:>12.6f}")

    if plot:
        _plot_gaussian_waists(results)

    return sigmas


# ===================================================================
# Part 1b — σ-based simultaneous sensitivity
# ===================================================================

def run_sigma_based_sensitivity(sigmas: dict[int, float]) -> None:
    """Find unified α such that shifting all params by ±α·σ_i reaches
    target infidelity.
    """
    print(f"\n{'=' * 72}")
    print("Part 1b: σ-based simultaneous sensitivity")
    print(f"{'=' * 72}\n")

    sigma_arr = np.array([sigmas[idx] for idx in PARAM_INDICES])

    def _inf_shifted(alpha: float, sign: int) -> float:
        x = list(X_TO_OUR_DARK)
        for j, idx in enumerate(PARAM_INDICES):
            x[idx] += sign * alpha * sigma_arr[j]
        return _infidelity_at(x)

    target = SIMULTANEOUS_TARGET
    for sign, label in [(+1, "+α"), (-1, "-α")]:
        alpha = 0.001
        lo = 0.0
        found = False
        for _ in range(60):
            if _inf_shifted(alpha, sign) >= target:
                found = True
                break
            lo = alpha
            alpha *= 2
            if alpha > 100:
                break
        if not found:
            print(f"  {label}: no solution found")
            continue
        hi = alpha
        while hi - lo > 1e-14:
            mid = (lo + hi) / 2
            if _inf_shifted(mid, sign) < target:
                lo = mid
            else:
                hi = mid
        alpha = (lo + hi) / 2
        fid = _inf_shifted(alpha, sign)
        print(f"  {label}: α = {alpha:.6e}, infidelity = {fid:.6e}")
        print(f"    Per-parameter shifts:")
        for j, idx in enumerate(PARAM_INDICES):
            dp = sign * alpha * sigma_arr[j]
            p = X_TO_OUR_DARK[idx]
            rel = abs(dp / p) * 100 if p != 0 else float("inf")
            print(f"      {PARAM_NAMES[idx]}: Δp = {dp:+.8e} "
                  f"({rel:.4f} %)")


# ===================================================================
# Part 2 — Iso-fidelity surface error decomposition
# ===================================================================

def _sample_directions(n_dirs: int, n_dim: int = 5,
                       seed: int = DIRECTION_SEED) -> np.ndarray:
    """Uniformly distributed unit vectors on the (n_dim-1)-sphere."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_dirs, n_dim))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-300)
    return raw / norms


def _find_iso_surface_point(
    direction: np.ndarray,
    sigma_arr: np.ndarray,
    target_infidelity: float,
) -> float | None:
    """Bisection for α along *direction* in σ-normalised space."""
    def _inf_at(alpha: float) -> float:
        x = list(X_TO_OUR_DARK)
        dp = alpha * sigma_arr * direction
        for j, idx in enumerate(PARAM_INDICES):
            x[idx] += dp[j]
        return _infidelity_at(x)

    # bracket
    alpha = 0.1
    lo = 0.0
    found = False
    for _ in range(30):
        if _inf_at(alpha) >= target_infidelity:
            found = True
            break
        lo = alpha
        alpha *= 2.0
        if alpha > MAX_ALPHA_SCALE:
            break
    if not found:
        return None

    hi = alpha
    for _ in range(BISECTION_MAX_ITER):
        if hi - lo < BISECTION_TOL:
            break
        mid = (lo + hi) / 2.0
        if _inf_at(mid) < target_infidelity:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def _decompose_at_point(x: list[float]) -> dict[str, float]:
    """Error decomposition at parameter point *x*."""
    infidelity, residuals = sim._gate_infidelity_single(
        x, fid_type="average", return_residuals=True,
    )
    branching = sim._residuals_to_branching(residuals)
    xyz = branching["XYZ"]
    al = branching["AL"]
    lg = branching["LG"]
    phase = max(infidelity - (xyz + al + lg), 0.0)

    total = xyz + al + lg + phase
    if total > 0:
        f_xyz = xyz / total
        f_al = al / total
        f_lg = lg / total
        f_phase = phase / total
    else:
        f_xyz = f_al = f_lg = f_phase = 0.0

    return {
        "infidelity": infidelity,
        "XYZ": xyz, "AL": al, "LG": lg, "Phase": phase,
        "f_XYZ": f_xyz, "f_AL": f_al, "f_LG": f_lg, "f_Phase": f_phase,
    }


def _plot_iso_fidelity(result: dict,
                       save_path: str = "iso_fidelity_decomposition.pdf",
                       ) -> None:
    """Visualise iso-fidelity surface decomposition."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- bar chart of mean fractions ---
    ax = axes[0]
    names = ["XYZ", "AL", "LG", "Phase"]
    frac_keys = ["f_XYZ", "f_AL", "f_LG", "f_Phase"]
    means = [result["mean_fractions"][k] for k in frac_keys]
    stds = [result["std_fractions"][k] for k in frac_keys]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors,
                  alpha=0.8)
    ax.set_ylabel("Fractional contribution")
    ax.set_title("Mean error decomposition\non iso-fidelity surface")
    ax.set_ylim(0, 1)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02, f"{m:.3f}",
                ha="center", va="bottom", fontsize=10)

    # --- histogram of α ---
    ax = axes[1]
    ax.hist(result["alphas"], bins=30, color="steelblue", alpha=0.7,
            edgecolor="black")
    ax.set_xlabel("$\\alpha$ (σ-units)")
    ax.set_ylabel("Count")
    ax.set_title("Distance to iso-fidelity surface")
    ax.axvline(result["alphas"].mean(), color="red", ls="--",
               label=f"mean = {result['alphas'].mean():.3f}")
    ax.legend()

    # --- AL vs Phase scatter ---
    ax = axes[2]
    al_fracs = [d["f_AL"] for d in result["all_decompositions"]]
    phase_fracs = [d["f_Phase"] for d in result["all_decompositions"]]
    ax.scatter(al_fracs, phase_fracs, s=10, alpha=0.6)
    ax.set_xlabel("AL fraction")
    ax.set_ylabel("Phase fraction")
    ax.set_title("AL vs Phase on iso-surface")
    ax.plot([0, 1], [1, 0], "k--", alpha=0.3)

    fig.suptitle(
        f"Iso-Fidelity Surface "
        f"($\\Delta$infidelity = {ISO_FIDELITY_DROP})",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved plot → {save_path}")


def run_iso_fidelity_decomposition(
    sigmas: dict[int, float],
    n_directions: int = N_DIRECTIONS,
    plot: bool = True,
) -> dict:
    """Part 2: iso-fidelity surface error decomposition.

    Parameters
    ----------
    sigmas : dict from Part 1, ``{param_idx: sigma_i}``.
    n_directions : number of random directions to sample.
    plot : whether to save a visualisation.

    Returns
    -------
    dict with ``mean_fractions``, ``std_fractions``, ``all_decompositions``,
    ``alphas``, ``directions``.
    """
    print(f"\n{'=' * 72}")
    print("Part 2: Iso-fidelity surface error decomposition")
    print(f"{'=' * 72}\n")

    baseline = _infidelity_at(X_TO_OUR_DARK)
    target = baseline + ISO_FIDELITY_DROP
    print(f"  Baseline infidelity: {baseline:.4e}")
    print(f"  Target infidelity:   {target:.4e}  "
          f"(baseline + {ISO_FIDELITY_DROP})")
    print(f"  Sampling {n_directions} random directions …\n")

    sigma_arr = np.array([sigmas[idx] for idx in PARAM_INDICES])
    directions = _sample_directions(n_directions, n_dim=len(PARAM_INDICES))

    decompositions: list[dict[str, float]] = []
    alphas: list[float] = []
    skipped = 0

    for i, d in enumerate(directions):
        alpha = _find_iso_surface_point(d, sigma_arr, target)
        if alpha is None:
            skipped += 1
            continue

        x = list(X_TO_OUR_DARK)
        dp = alpha * sigma_arr * d
        for j, idx in enumerate(PARAM_INDICES):
            x[idx] += dp[j]

        decomp = _decompose_at_point(x)
        decompositions.append(decomp)
        alphas.append(alpha)

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{n_directions} directions done …")

    print(f"\n  {len(decompositions)} surface points found, "
          f"{skipped} skipped.")

    if not decompositions:
        print("  ERROR: no surface points found.")
        return {}

    # --- aggregate ---
    frac_keys = ["f_XYZ", "f_AL", "f_LG", "f_Phase"]
    display = {"f_XYZ": "XYZ", "f_AL": "AL", "f_LG": "LG",
               "f_Phase": "Phase"}

    mean_fracs: dict[str, float] = {}
    std_fracs: dict[str, float] = {}
    for key in frac_keys:
        vals = np.array([d[key] for d in decompositions])
        mean_fracs[key] = float(np.mean(vals))
        std_fracs[key] = float(np.std(vals))

    print(f"\n  {'Error type':<12} {'Mean fraction':>14} {'Std':>14}")
    print(f"  {'-' * 42}")
    for key in frac_keys:
        print(f"  {display[key]:<12} {mean_fracs[key]:>14.6f} "
              f"{std_fracs[key]:>14.6f}")

    # absolute contributions
    print(f"\n  {'Error type':<12} {'Mean absolute':>14} {'Std absolute':>14}")
    print(f"  {'-' * 42}")
    for abs_key in ["XYZ", "AL", "LG", "Phase"]:
        vals = np.array([d[abs_key] for d in decompositions])
        print(f"  {abs_key:<12} {np.mean(vals):>14.4e} "
              f"{np.std(vals):>14.4e}")

    alphas_arr = np.array(alphas)
    print(f"\n  α statistics (σ-units to surface):")
    print(f"    mean = {alphas_arr.mean():.4f}  "
          f"std = {alphas_arr.std():.4f}  "
          f"min = {alphas_arr.min():.4f}  "
          f"max = {alphas_arr.max():.4f}")

    result = {
        "mean_fractions": mean_fracs,
        "std_fractions": std_fracs,
        "all_decompositions": decompositions,
        "alphas": alphas_arr,
        "directions": directions,
    }

    if plot:
        _plot_iso_fidelity(result)

    return result


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    sigmas = run_gaussian_waist_scan(plot=True)
    run_sigma_based_sensitivity(sigmas)
    run_iso_fidelity_decomposition(sigmas, plot=True)
