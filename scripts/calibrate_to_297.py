"""TO calibration for the 297 nm single-photon CZ gate (rb87_297_clock_4).

Port of the two-photon TO calibration workflow of record
(results/cz_gate/to_calibration/to_{mp,pm}.json + notebook 01):

  x = [A, w/Omega, phi0, d/Omega, theta, T*Omega/2pi]

theta is analytically optimal for fixed amplitudes (no extra simulation), so
only the 5 shape parameters are optimized. The original's batched-candidate
search (~25 shapes/round in parallel) is ported as a worker-parallel
differential evolution (two-photon shapes seeded into the initial population,
relaxed solver tolerance) followed by a Nelder-Mead polish at the default
tolerance; the end point is re-scored at rtol=1e-10 / atol=1e-13, which is
the number quoted everywhere.

Coherent average-gate infidelity against CZ (d=4, leakage included through the
sub-normalized amplitudes), with the single-qubit phase correction theta:

  F = (|a00 + 2 e^{-i theta} a01 - e^{-2 i theta} a11|^2
       + |a00|^2 + 2 |a01|^2 + |a11|^2) / 20

Outputs (all under --outdir, default results/297_to_calibration/):
  to_297.json           calibration record (same shape as to_mp.json)
  summary.json          per-state metrics + spontaneous-emission budget
  traces_297.npz        t, per-level populations, 4x4 amplitude matrix
  populations_297.png   per-level populations during the gate, per basis state
  error_budget_297.png  cumulative decay losses + final error budget
  pulse_297.png         calibrated TO waveform (envelope / phase / chirp)

Run (DGX):
  uv run python scripts/calibrate_to_297.py
  uv run python scripts/calibrate_to_297.py --force   # ignore existing record
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.optimize import differential_evolution, minimize, minimize_scalar

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.physics import (
    arc_pair_c6_rad_s_um6,
    rb87_297_clock_rabi_frequencies,
    zeeman_shift_rad_s,
)
from ryd_gate.protocols import Direct297TOProtocol

MHz = 2 * np.pi * 1e6
LEVELS = ("0", "1", "r", "r_garb")
KEYS = ("00", "01", "10", "11")
BASIS = {k: list(k) for k in KEYS}
OBJ_KEYS = ("00", "01", "11")  # |10> == |01> by symmetry of the 2-atom chain
NIELSEN_W = {"00": 0.25, "01": 0.5, "11": 0.25}
TIGHT = {"rtol": 1e-10, "atol": 1e-13}

# Level colors (categorical palette, CVD-validated): |0>, |1>, |r>, |r_garb>.
COLOR = {"0": "#0173B2", "1": "#029E73", "r": "#D55E00", "r_garb": "#8C6BB1"}
LABEL = {"0": r"$|0\rangle$", "1": r"$|1\rangle$", "r": r"$|r\rangle$",
         "r_garb": r"$|r_{\rm garb}\rangle$"}

# Two-photon TO shapes of record (results/cz_gate/to_calibration/to_{mp,pm}.json),
# as [A, w/Omega, phi0, d/Omega, T*Omega/2pi] seeds for the 297 calibration.
X_MP = [-0.6906380080019421, 1.0407458425800344, 0.32728332656414416,
        1.5648394495744864, 1.3411757826266593]
X_PM = [-0.6549329995429718, 1.0654166775251044, 0.2894876435222684,
        -0.09417546701812637, 1.321421269989839]

# Search space for [A, w/Omega, phi0, d/Omega, T*Omega/2pi] (contains both
# two-photon optima; T lower bound keeps 2*rise < t_gate at the default rise).
BOUNDS = [(-2.0, 2.0), (0.3, 1.8), (-np.pi, np.pi), (-2.0, 2.0), (0.9, 2.2)]
SEARCH_TOL = {"rtol": 1e-6, "atol": 1e-9}  # DE stage only; NM polish uses defaults

_CFG = None  # set before forking worker pools; workers read it post-fork


def _amp_one(job):
    """Worker: <s|U|s> for one basis state (runs in a forked process)."""
    x5, key, tol = job
    out = simulate(build_system(x5, _CFG), BASIS[key], backend_options=tol)
    return out.amplitude(BASIS[key])


def _obj_de(x5):
    """Worker objective for the differential-evolution stage."""
    return objective(x5, _CFG, tol=SEARCH_TOL)


def build_system(x5, cfg):
    proto = Direct297TOProtocol(
        omega_297_max_rad_s=cfg["omega"],
        rise_time_s=cfg["rise_time_s"],
        phase_amplitude_rad=float(x5[0]),
        modulation_frequency_ratio=float(x5[1]),
        phase_offset_rad=float(x5[2]),
        frequency_offset_ratio=float(x5[3]),
        duration_ratio=float(x5[4]),
    )
    return RydbergSystem(level_structure=cfg["level_structure"],
                         register=Register.chain(2, spacing_um=cfg["spacing_um"]),
                         protocol=proto)


def gate_amps(x5, cfg, backend_options=None, keys=OBJ_KEYS, pool=None):
    """<s|U|s> for the requested basis states (final time only)."""
    if pool is not None:
        return np.array(pool.map(_amp_one, [(x5, k, backend_options) for k in keys]))
    sys_ = build_system(x5, cfg)
    out = simulate(sys_, [BASIS[k] for k in keys], backend_options=backend_options)
    return np.array([out[i].amplitude(BASIS[k]) for i, k in enumerate(keys)])


def cz_infidelity(a00, a01, a11):
    """Min over theta of the CZ average-gate infidelity; returns (infid, theta)."""
    p = abs(a00) ** 2 + 2 * abs(a01) ** 2 + abs(a11) ** 2

    def fid(theta):
        s = a00 + 2 * np.exp(-1j * theta) * a01 - np.exp(-2j * theta) * a11
        return (abs(s) ** 2 + p) / 20.0

    grid = np.linspace(-np.pi, np.pi, 4097)
    vals = np.array([fid(t) for t in grid])
    i = int(np.argmax(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda t: -fid(t), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-12})
    return 1.0 - fid(res.x), float(res.x)


def objective(x5, cfg, tol=None, pool=None):
    if not np.all(np.isfinite(x5)) or x5[4] <= 0:
        return 1.0
    try:
        a00, a01, a11 = gate_amps(x5, cfg, backend_options=tol, pool=pool)
    except ValueError:  # e.g. 2*rise > t_gate for too-small duration_ratio
        return 1.0
    return cz_infidelity(a00, a01, a11)[0]


def calibrate(cfg, de_gens, nm_maxiter, workers, rng_seed=1):
    d_stark = (cfg["omega_garb"] ** 2 / (4 * cfg["garb_zeeman"])) / cfg["omega"]
    seeds = [
        list(X_PM),
        list(X_MP),
        X_PM[:3] + [0.0, X_PM[4]],
        X_PM[:3] + [+d_stark, X_PM[4]],
        X_PM[:3] + [-d_stark, X_PM[4]],
    ]
    rng = np.random.default_rng(rng_seed)
    n_pop = max(workers, 24)
    init = np.array([np.clip(s, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
                     for s in seeds]
                    + [[rng.uniform(lo, hi) for lo, hi in BOUNDS]
                       for _ in range(n_pop - len(seeds))])
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        de = differential_evolution(
            _obj_de, BOUNDS, init=init, maxiter=de_gens, tol=1e-6,
            mutation=(0.3, 1.0), recombination=0.7, seed=rng_seed,
            polish=False, updating="deferred", workers=pool.map, disp=True)
    t_de = time.time() - t0
    print(f"DE stage: 1-F = {de.fun:.3e} after {de.nfev} evals, {t_de:.0f} s "
          f"(x = {np.array2string(de.x, precision=4)})", flush=True)
    with ctx.Pool(len(OBJ_KEYS)) as pool3:
        res = minimize(lambda x: objective(x, cfg, tol=None, pool=pool3), de.x,
                       method="Nelder-Mead",
                       options={"maxiter": nm_maxiter, "xatol": 5e-5, "fatol": 1e-10})
    print(f"NM polish (default tol): {de.fun:.3e} -> {res.fun:.3e} "
          f"(nfev {res.nfev}, {time.time() - t0 - t_de:.0f} s)", flush=True)
    search_report = {
        "seeds": seeds,
        "de": {"fun": float(de.fun), "nfev": int(de.nfev), "x": [float(v) for v in de.x],
               "generations": de_gens, "population": int(n_pop),
               "search_tol": SEARCH_TOL, "rng_seed": rng_seed},
        "nm": {"fun": float(res.fun), "nfev": int(res.nfev)},
    }
    return res, search_report, time.time() - t0


def compute_traces(x5, cfg, n_eval):
    """Tight-tolerance per-level populations + 4x4 amplitude matrix."""
    sys_ = build_system(x5, cfg)
    obs = {f"n_{l}": sum(sys_.observables.n(l, i) for i in range(2)) for l in LEVELS}
    t_eval = np.linspace(0.0, sys_.t_gate, n_eval)
    out = simulate(sys_, [BASIS[k] for k in KEYS], t_eval=t_eval, observables=obs,
                   backend_options=TIGHT)
    pops = np.array([[out[i].expectation(f"n_{l}") for l in LEVELS]
                     for i in range(len(KEYS))])
    amps = np.array([[out[i].amplitude(BASIS[k]) for k in KEYS]
                     for i in range(len(KEYS))])
    return np.asarray(out[0].times), pops, amps, float(sys_.t_gate)


def se_budget(t, pops, decay):
    """Nielsen-weighted in-gate decay integrals + final residuals (no branching
    split: rb87_297_clock_4 publishes no branching_ratios)."""
    iR, iG = LEVELS.index("r"), LEVELS.index("r_garb")
    g_rd, g_bbr = decay["radiative"], decay["blackbody"]
    b = {}
    for lvl, idx in (("r", iR), ("r_garb", iG)):
        rows = {"radiative": 0.0, "blackbody": 0.0, "residual": 0.0}
        for k, w in NIELSEN_W.items():
            occ = pops[KEYS.index(k), idx]
            rows["radiative"] += w * g_rd * simpson(occ, x=t)
            rows["blackbody"] += w * g_bbr * simpson(occ, x=t)
            rows["residual"] += w * occ[-1]
        b[lvl] = rows
    b["se_total"] = sum(b[l]["radiative"] + b[l]["blackbody"] for l in ("r", "r_garb"))
    return b


def plot_populations(t, pops, amps, cfg, infid, path):
    t_ns = 1e9 * t
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), sharex=True)
    for j, k in enumerate(KEYS):
        for l in ("0", "1", "r"):
            axes[0, j].plot(t_ns, pops[j, LEVELS.index(l)] / 2, lw=1.6,
                            color=COLOR[l], label=LABEL[l])
        axes[0, j].set_ylim(-0.02, 1.05)
        axes[0, j].set_title(f"$|{k}\\rangle$: return prob = {abs(amps[j, j])**2:.6f}",
                             fontsize=9)
        axes[1, j].plot(t_ns, pops[j, LEVELS.index("r_garb")] / 2, lw=1.6,
                        color=COLOR["r_garb"], label=LABEL["r_garb"])
        axes[1, j].set_ylim(bottom=0.0)
        axes[1, j].set_xlabel("time (ns)")
    axes[0, 0].set_ylabel("population per atom")
    axes[1, 0].set_ylabel(LABEL["r_garb"] + "\nper atom")
    axes[0, 0].legend(fontsize=8, loc="center right")
    fig.suptitle(
        f"rb87_297_clock_4 TO CZ: per-level populations "
        f"(n={cfg['ryd_level']}, a={cfg['spacing_um']} um, B={cfg['magnetic_field_G']} G, "
        f"coherent 1-F = {infid:.2e})", y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_budget(t, pops, budget, coherent, path):
    t_ns = 1e9 * t
    iR, iG = LEVELS.index("r"), LEVELS.index("r_garb")
    g_tot = budget["gamma_radiative"] + budget["gamma_blackbody"]
    occ_r = sum(NIELSEN_W[k] * pops[KEYS.index(k), iR] for k in NIELSEN_W)
    occ_g = sum(NIELSEN_W[k] * pops[KEYS.index(k), iG] for k in NIELSEN_W)
    cum_r = g_tot * cumulative_trapezoid(occ_r, t, initial=0.0)
    cum_g = g_tot * cumulative_trapezoid(occ_g, t, initial=0.0)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax0.plot(t_ns, cum_r, lw=1.8, color=COLOR["r"], label=LABEL["r"] + " decay")
    ax0.plot(t_ns, cum_g, lw=1.8, color=COLOR["r_garb"], label=LABEL["r_garb"] + " decay")
    ax0.plot(t_ns, cum_r + cum_g, lw=1.8, ls="--", color="0.2", label="total")
    ax0.set_xlabel("time (ns)")
    ax0.set_ylabel("cumulative decay probability")
    ax0.set_title("Nielsen-weighted in-gate spontaneous emission", fontsize=10)
    ax0.legend(fontsize=8)

    b = budget["budget"]
    rows = [
        ("coherent (rtol 1e-10)", coherent, "0.2"),
        (LABEL["r"] + " radiative", b["r"]["radiative"], COLOR["r"]),
        (LABEL["r"] + " blackbody", b["r"]["blackbody"], COLOR["r"]),
        (LABEL["r_garb"] + " radiative", b["r_garb"]["radiative"], COLOR["r_garb"]),
        (LABEL["r_garb"] + " blackbody", b["r_garb"]["blackbody"], COLOR["r_garb"]),
        ("total (coherent + SE)", coherent + b["se_total"], "0.2"),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (name, val, color) in zip(y, rows):
        ax1.plot([val], [yi], "o", ms=8, color=color)
        ax1.hlines(yi, 0, val, lw=1.0, color=color, alpha=0.5)
        ax1.annotate(f" {val:.2e}", (val, yi), fontsize=8, va="center")
    ax1.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax1.set_xscale("log")
    ax1.set_xlabel("infidelity contribution")
    ax1.set_title("CZ error budget", fontsize=10)
    ax1.margins(x=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pulse(x5, cfg, t_gate, path):
    from ryd_gate.protocols.pulses import blackman_pulse

    A, w_ratio, phi0, d_ratio, _ = [float(v) for v in x5]
    omega, rise = cfg["omega"], cfg["rise_time_s"]
    t = np.linspace(0.0, t_gate, 1200)
    env = np.array([blackman_pulse(ti, rise, t_gate) for ti in t])
    phase = A * np.cos(w_ratio * omega * t + phi0) + d_ratio * omega * t
    chirp = np.gradient(phase, t) / MHz

    fig, axes = plt.subplots(3, 1, figsize=(8, 6.4), sharex=True)
    axes[0].plot(1e9 * t, env * omega / MHz, lw=1.8, color=COLOR["r"])
    axes[0].set_ylabel(r"$\Omega(t)/2\pi$ (MHz)")
    axes[1].plot(1e9 * t, phase, lw=1.8, color=COLOR["0"])
    axes[1].set_ylabel(r"$\phi(t)$ (rad)")
    axes[2].plot(1e9 * t, chirp, lw=1.8, color=COLOR["1"])
    axes[2].set_ylabel(r"$\dot\phi(t)/2\pi$ (MHz)")
    axes[2].set_xlabel("time (ns)")
    fig.suptitle("Calibrated 297 nm TO waveform", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--power-at-atoms-w", type=float, default=0.6,
                    help="297 nm power at the atoms (single_photon.ipynb: 3 W x 20%% optics)")
    ap.add_argument("--beam-area-um2", type=float, default=420.0)
    ap.add_argument("--ryd-level", type=int, default=53)
    ap.add_argument("--spacing-um", type=float, default=3.0)
    ap.add_argument("--magnetic-field-g", type=float, default=20.0,
                    help="preset default; also the max_leakage_297 sweep field")
    ap.add_argument("--rise-time-ns", type=float, default=20.0)
    ap.add_argument("--de-gens", type=int, default=40, help="differential-evolution generations")
    ap.add_argument("--workers", type=int, default=24, help="parallel workers for the DE stage")
    ap.add_argument("--maxiter", type=int, default=600, help="Nelder-Mead polish maxiter")
    ap.add_argument("--n-eval", type=int, default=301, help="trace time points")
    ap.add_argument("--outdir", type=Path, default=Path("results/297_to_calibration"))
    ap.add_argument("--force", action="store_true", help="re-optimize even if a done record exists")
    args = ap.parse_args()

    omega, omega_garb = rb87_297_clock_rabi_frequencies(
        args.power_at_atoms_w, args.beam_area_um2, ryd_level=args.ryd_level)
    global _CFG
    cfg = _CFG = {
        "omega": omega,
        "omega_garb": omega_garb,
        "rise_time_s": args.rise_time_ns * 1e-9,
        "spacing_um": args.spacing_um,
        "ryd_level": args.ryd_level,
        "magnetic_field_G": args.magnetic_field_g,
        "level_structure": level_structure("rb87_297_clock_4",
                                           ryd_level=args.ryd_level,
                                           magnetic_field_G=args.magnetic_field_g),
        "garb_zeeman": zeeman_shift_rad_s(args.magnetic_field_g, l=1, j=1.5, delta_mj=1.0),
    }
    v_nn = arc_pair_c6_rad_s_um6(n1=args.ryd_level, l1=1, j1=1.5, mj1=-1.5, mj2=-1.5,
                                 theta=np.pi / 2, phi=0.0) / args.spacing_um ** 6
    print(f"omega_297/2pi = {omega / MHz:.3f} MHz, omega_garb/2pi = {omega_garb / MHz:.3f} MHz")
    print(f"garb Zeeman/2pi = {cfg['garb_zeeman'] / MHz:.3f} MHz, "
          f"V_nn/2pi = {v_nn / MHz:.3f} MHz", flush=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    record_path = args.outdir / "to_297.json"

    record = None
    if record_path.exists() and not args.force:
        existing = json.loads(record_path.read_text())
        if existing.get("done"):
            print(f"loaded existing record {record_path} (use --force to re-optimize)")
            record = existing
    if record is None:
        t_eval0 = time.time()
        f_seed = objective(np.asarray(X_PM), cfg, tol=SEARCH_TOL)
        print(f"pilot objective eval (search tol): {time.time() - t_eval0:.2f} s "
              f"(pm seed 1-F = {f_seed:.3e})", flush=True)
        res, search_report, elapsed = calibrate(cfg, args.de_gens, args.maxiter,
                                                args.workers)
        a00, a01, a11 = gate_amps(res.x, cfg, backend_options=TIGHT)
        tight_infid, theta = cz_infidelity(a00, a01, a11)
        record = {
            "preset": "rb87_297_clock_4",
            "spacing_um": args.spacing_um,
            "fixed": {
                "omega_297_max_rad_s": float(omega),
                "omega_297_garb_rad_s": float(omega_garb),
                "rise_time_s": cfg["rise_time_s"],
                "power_at_atoms_w": args.power_at_atoms_w,
                "beam_area_um2": args.beam_area_um2,
                "ryd_level": args.ryd_level,
                "magnetic_field_G": args.magnetic_field_g,
            },
            "x": [float(v) for v in res.x[:4]] + [theta, float(res.x[4])],
            "coherent_infidelity": float(res.fun),
            "mode": "optimize",
            "theta": theta,
            "amps": [[float(a.real), float(a.imag)] for a in (a00, a01, a11)],
            "tight_tol_infidelity": float(tight_infid),
            "garb_zeeman_rad_s": float(cfg["garb_zeeman"]),
            "blockade_V_nn_rad_s": float(v_nn),
            "search": search_report,
            "elapsed_s": elapsed,
            "done": True,
        }
        record_path.write_text(json.dumps(record, indent=1) + "\n")
        print(f"wrote {record_path}")

    x = record["x"]
    x5 = np.array(x[:4] + [x[5]])
    theta = float(x[4])

    t, pops, amps, t_gate = compute_traces(x5, cfg, args.n_eval)
    np.savez(args.outdir / "traces_297.npz", t=t, pops=pops, amps=amps,
             x=np.asarray(x, dtype=float),
             fixed=np.array([record["fixed"][k] for k in sorted(record["fixed"])]),
             spacing_um=np.array(args.spacing_um), rtol=np.array(TIGHT["rtol"]))

    # Per-state metrics at tight tolerance (traces amps), theta from the record.
    infid, theta_traces = cz_infidelity(amps[0, 0], amps[1, 1], amps[3, 3])
    cz_phase = (np.angle(amps[3, 3]) - 2 * theta_traces - np.angle(amps[0, 0]) + np.pi) \
        % (2 * np.pi) - np.pi
    decay = {k: float(v) for k, v in
             cfg["level_structure"].decay_rates_per_s["r"].items()}
    budget = se_budget(t, pops, decay)
    summary = {
        "coherent_infidelity_tight": float(infid),
        "theta": float(theta_traces),
        "cz_phase_dist_from_pi": float(abs(abs(cz_phase) - np.pi)),
        "t_gate_ns": 1e9 * t_gate,
        "per_state": {
            k: {"return_prob": float(abs(amps[j, j]) ** 2),
                "raw_phase_rad": float(np.angle(amps[j, j])),
                "leakage": float(1 - np.sum(np.abs(amps[j]) ** 2))}
            for j, k in enumerate(KEYS)
        },
        "gamma_radiative_per_s": decay["radiative"],
        "gamma_blackbody_per_s": decay["blackbody"],
        "se_budget": budget,
        "total_incl_se": float(infid + budget["se_total"]),
        "note": "residual rows are final populations already counted in the "
                "coherent infidelity; se_total = decay integrals only",
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    plot_populations(t, pops, amps, cfg, infid, args.outdir / "populations_297.png")
    plot_error_budget(t, pops,
                      {"budget": budget, "gamma_radiative": decay["radiative"],
                       "gamma_blackbody": decay["blackbody"]},
                      infid, args.outdir / "error_budget_297.png")
    plot_pulse(x5, cfg, t_gate, args.outdir / "pulse_297.png")

    print(f"\nt_gate = {1e9 * t_gate:.1f} ns, theta = {theta_traces:+.6f} rad "
          f"(record theta {theta:+.6f})")
    print(f"coherent 1-F (rtol 1e-10) = {infid:.3e}")
    print(f"SE decay total            = {budget['se_total']:.3e}")
    print(f"total incl. SE            = {infid + budget['se_total']:.3e}")
    for j, k in enumerate(KEYS):
        s = summary["per_state"][k]
        print(f" |{k}>: return {s['return_prob']:.6f}  phase {s['raw_phase_rad']:+.5f}  "
              f"leakage {s['leakage']:.2e}")
    print(f"wrote plots + summary under {args.outdir}")


if __name__ == "__main__":
    main()
