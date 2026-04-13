#!/usr/bin/env python3
"""Unified local addressing analysis: wavelength optimization and noise sensitivity.

Subcommands:
    wavelength   Scan AC Stark shift, scattering, and FOM vs laser wavelength.
    noise        Sweep noise sources (detuning, RIN, amplitude) and measure addressing quality.

Usage:
  # Re-plot exp1 (t_gate = 4.5 μs run)                                                                                                                             
  uv run --active python scripts/scan_local_addressing.py optimize-plot --prefix exp1                                                                              
                                                                                                                                                                   
  # Re-plot exp2 (t_gate = 9 μs run)                                                                                                                               
  uv run --active python scripts/scan_local_addressing.py optimize-plot --prefix exp2                                                                              
   
 # To run a brand-new third experiment with its own prefix, use the same flag on the optimize subcommand:                                                           
  uv run --active python scripts/scan_local_addressing.py optimize --prefix exp3 --t-gate-us 6  
"""

import argparse
import os
import time as _time

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi

from ryd_gate.analysis.local_addressing import (
    BASELINE_AMP,
    BASELINE_DETUNING_HZ,
    BASELINE_RIN,
    COMBINED_SCALE_MAX,
    DEFAULT_LOCAL_DETUNING,
    DEFAULT_LOCAL_SCATTER,
    default_sweep_x,
    evaluate_addressing,
)
from ryd_gate.core.atomic_system import (
    LAMBDA_D2,
    LAMBDA_PAPER,
    POWER_REF_UW,
    build_product_state_map,
    compute_shift_scatter,
    create_analog_system,
)
from ryd_gate.protocols.sweep import SweepProtocol

# ═══════════════════════════════════════════════════════════════════════
# Subcommand: noise
# ═══════════════════════════════════════════════════════════════════════

def cmd_noise(args):
    """Sweep noise parameters and measure addressing quality."""
    print("=" * 60)
    print("  Addressing Noise Sensitivity Scan")
    print("=" * 60)

    system = create_analog_system(detuning_sign=1)
    initial_state = build_product_state_map(n_levels=3)["gg"]
    protocol = SweepProtocol(
        addressing={0: DEFAULT_LOCAL_DETUNING},
        scatter_rate=DEFAULT_LOCAL_SCATTER,
    )
    x = default_sweep_x(system)

    scans = {
        "Global detuning": {
            "param": "sigma_detuning",
            "values": np.linspace(0, 500e3, args.n_points),
            "xlabel": "Detuning noise $\\sigma_\\Delta$ (kHz)",
            "xscale": 1e-3,
        },
        "Local RIN": {
            "param": "sigma_local_rin",
            "values": np.linspace(0, 0.05, args.n_points),
            "xlabel": "RIN $\\sigma_{\\mathrm{RIN}}$ (%)",
            "xscale": 100,
        },
        "Amplitude noise": {
            "param": "sigma_amplitude",
            "values": np.linspace(0, 0.05, args.n_points),
            "xlabel": "Amplitude noise $\\sigma_\\Omega$ (%)",
            "xscale": 100,
        },
    }

    results = {}
    for name, cfg in scans.items():
        print(f"\n  Scanning: {name}")
        t0 = _time.time()
        pin_errs, xtalk_errs, leak_errs = [], [], []
        for val in cfg["values"]:
            kwargs = {"sigma_detuning": 0.0, "sigma_local_rin": 0.0, "sigma_amplitude": 0.0}
            kwargs[cfg["param"]] = val
            pin, xtalk, leak = evaluate_addressing(
                system, initial_state, protocol, x, kwargs, args.n_mc)
            pin_errs.append(pin)
            xtalk_errs.append(xtalk)
            leak_errs.append(leak)
        print(f"    Done in {_time.time() - t0:.1f}s")
        results[name] = (np.array(pin_errs), np.array(xtalk_errs), np.array(leak_errs))

    # Optional combined sweep
    if args.combined:
        print("\n  Scanning: Combined (all noise sources)")
        scale_factors = np.linspace(0, COMBINED_SCALE_MAX, args.n_points)
        t0 = _time.time()
        pin_comb, xtalk_comb, leak_comb = [], [], []
        for scale in scale_factors:
            pin, xtalk, leak = evaluate_addressing(
                system, initial_state, protocol, x,
                {"sigma_detuning": BASELINE_DETUNING_HZ * scale,
                 "sigma_local_rin": BASELINE_RIN * scale,
                 "sigma_amplitude": BASELINE_AMP * scale},
                args.n_mc)
            pin_comb.append(pin)
            xtalk_comb.append(xtalk)
            leak_comb.append(leak)
        print(f"    Done in {_time.time() - t0:.1f}s")

    # Plot
    os.makedirs(FIGDIR, exist_ok=True)
    n_panels = 4 if args.combined else 3
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flat

    for ax, (name, cfg) in zip(axes_flat, scans.items()):
        pin, xtalk, leak = results[name]
        xvals = cfg["values"] * cfg["xscale"]
        ax.plot(xvals, pin, "o-", color="red", label="Pinning", markersize=5)
        ax.plot(xvals, xtalk, "s-", color="blue", label="Crosstalk", markersize=5)
        ax.plot(xvals, leak, "^-", color="orange", label="Leakage", markersize=5)
        ax.set_xlabel(cfg["xlabel"])
        ax.set_ylabel("Error")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    if args.combined:
        ax = axes_flat[3]
        ax.plot(scale_factors, pin_comb, "o-", color="red", label="Pinning", markersize=5)
        ax.plot(scale_factors, xtalk_comb, "s-", color="blue", label="Crosstalk", markersize=5)
        ax.plot(scale_factors, leak_comb, "^-", color="orange", label="Leakage", markersize=5)
        ax.axvline(1.0, color="green", ls=":", alpha=0.7, label="Baseline")
        ax.set_xlabel("Noise scale factor")
        ax.set_ylabel("Error")
        ax.set_title("Combined noise (all sources)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        axes_flat[3].set_visible(False)

    fig.suptitle("Addressing Noise Sensitivity Analysis", fontsize=14)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "addressing_noise_sensitivity.png")
    fig.savefig(path, dpi=150)
    print(f"\nFigure saved to {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Subcommand: optimize
# ═══════════════════════════════════════════════════════════════════════

# Per-process state for parallel grid scan workers.
_OPT_WORKER = {}


def _simulate_grid_point(task):
    """Simulate one (wl, power, Δ_max, T_gate) point. Returns (idx, row tuple).

    The worker takes the *sweep window* (``delta_half_mhz``) and *gate time*
    (``t_gate_us``) per task so the same scan loop can iterate over either
    axis (legacy ``optimize`` collapses them to a single value).
    """
    from ryd_gate import simulate

    idx, wl, power_uw, delta_half_mhz, t_gate_us = task
    model = _OPT_WORKER["model"]
    system = _OPT_WORKER["system"]
    psi0 = _OPT_WORKER["psi0"]
    eta = _OPT_WORKER["eta"]
    ac_stark_peak = _OPT_WORKER["ac_stark_peak"]

    t_gate = t_gate_us * 1e-6
    x_sweep = [
        2 * pi * (-delta_half_mhz) * 1e6 / system.rabi_eff,
        2 * pi *   delta_half_mhz  * 1e6 / system.rabi_eff,
        t_gate / system.time_scale,
    ]

    shift_ref, scatter_ref = compute_shift_scatter(wl)
    scale = power_uw / POWER_REF_UW
    delta_A = 2 * pi * float(shift_ref) * scale
    scatter_A = float(scatter_ref) * scale
    delta_B = eta * delta_A

    protocol = SweepProtocol(addressing={0: delta_A, 1: delta_B},
                             ac_stark_shift=ac_stark_peak)
    result = simulate(system, protocol, x_sweep, psi0)
    psi_f = result.psi_final

    P_gg = model.observables.measure("pop_gg", psi_f)
    P_gr = model.observables.measure("pop_gr", psi_f)
    P_rg = model.observables.measure("pop_rg", psi_f)
    P_rr = model.observables.measure("pop_rr", psi_f)
    pinning_leak = P_rg + P_rr
    crosstalk = P_gg
    scatter_pen = 1.0 - np.exp(-scatter_A * t_gate)
    total_cost = pinning_leak + crosstalk + scatter_pen

    row = (
        wl, power_uw,
        delta_A / (2 * pi * 1e6), delta_B / (2 * pi * 1e6),
        scatter_A,
        P_gg, P_gr, P_rg, P_rr,
        pinning_leak, crosstalk, scatter_pen, total_cost,
        P_gr, 1.0 - (P_gg + P_gr + P_rg + P_rr),
        abs(delta_A) / system.rabi_eff,
        abs(delta_A) / system.v_ryd,
        # Issue #44 follow-up: trailing columns identifying which (Δ_max, T)
        # cell this row corresponds to. Constant in legacy `optimize` mode,
        # varies along an axis when --sweep-delta or --sweep-tgate is set.
        delta_half_mhz, t_gate_us,
    )
    return idx, row


def _run_optimize_scan(args):
    """Run the 2D grid scan and return (grid, metadata dict)."""
    from ryd_gate import simulate
    from ryd_gate.core.models.analog_3level import Analog3LevelModel

    print("=" * 60)
    print("  Local Addressing: Wavelength x Power Optimization")
    print("=" * 60)

    model = Analog3LevelModel.from_defaults(
        detuning_sign=1, blackmanflag=True,
        distance_um=args.distance_um,
        Delta_Hz=2.4e9,       # paper: ~2.4 GHz intermediate detuning
        rabi_420_Hz=135e6,    # paper: balanced Rabi, Omega_eff ≈ 3.8 MHz
        rabi_1013_Hz=135e6,
    )
    system = model.system
    psi0 = np.zeros(9, dtype=complex)
    psi0[0] = 1.0  # |gg>

    # AC Stark feed-forward: compensate Blackman-envelope light shift.
    # The textbook Ω²/(4Δ) is correct for the analog 3-level model because
    # |e⟩ is in H_420 explicitly (not adiabatically eliminated), so the
    # second-order dynamical shift on |g⟩ has exactly this magnitude. See
    # the issue #44 correction comment for the analog vs 7-level distinction.
    ac_stark_peak = system.rabi_420 ** 2 / (4 * abs(system.Delta))
    if getattr(args, "no_ac_stark", False):
        ac_stark_peak = 0.0
        print("  AC-Stark feed-forward DISABLED via --no-ac-stark "
              "(diagnostic mode)")

    # Gaussian tail factor
    eta = np.exp(-2 * (args.distance_um / args.waist_um) ** 2)
    print(f"  Gaussian tail: eta = exp(-2*(a/w)^2) = {eta:.4e}")
    print(f"  Omega_eff/(2pi) = {system.rabi_eff/(2*pi)/1e6:.2f} MHz")
    print(f"  AC Stark peak = {ac_stark_peak/(2*pi)/1e6:.2f} MHz")
    print(f"  V_ryd/(2pi) = {system.v_ryd/(2*pi)/1e6:.1f} MHz")
    print()

    # ── Build the four scan axes (legacy mode collapses Δ_max and T_gate
    #     to single values; --fix-wl/--fix-power-uw collapse the (λ,P) axes;
    #     --sweep-delta/--sweep-tgate replace those single values with N-point
    #     ranges).  See issue #44 for the rationale behind the (Δ_max, T_gate)
    #     scan capability.
    if args.fix_wl is not None:
        wls = np.array([float(args.fix_wl)])
        print(f"  λ axis pinned to {wls[0]:.3f} nm")
    else:
        wls = np.linspace(args.wl_min, args.wl_max, args.n_wl)

    # Adiabatic threshold: |delta_A| > 4*Omega_eff per wavelength.
    adiabatic_ratio = 4.0
    adiabatic_power_min = {}  # wl -> min power (uW) for adiabatic condition
    print(f"  Adiabatic condition: |delta_A| > {adiabatic_ratio:.0f} Omega_eff "
          f"= {adiabatic_ratio * system.rabi_eff/(2*pi)/1e6:.1f} MHz")
    print(f"  {'WL (nm)':>10} {'|shift|/P (MHz/uW)':>20} {'P_min_adiab (uW)':>18}")
    for wl in wls:
        shift_ref, _ = compute_shift_scatter(wl)
        shift_per_uw = abs(float(shift_ref)) / POWER_REF_UW  # Hz per uW
        p_adiab = adiabatic_ratio * system.rabi_eff / (2 * pi) / shift_per_uw
        adiabatic_power_min[wl] = p_adiab
        print(f"  {wl:10.2f} {shift_per_uw/1e6*1e0:20.4f} {p_adiab:18.1f}")

    # Auto-set power range if not specified by user
    p_adiab_worst = max(adiabatic_power_min.values())
    if args.power_min_uw is None:
        args.power_min_uw = POWER_REF_UW
        print(f"\n  Auto power_min = {args.power_min_uw:.1f} uW (reference power)")
    if args.power_max_uw is None:
        args.power_max_uw = max(2.0 * p_adiab_worst, 3.0 * args.power_min_uw)
        print(f"  Auto power_max = {args.power_max_uw:.1f} uW")
    print()

    if args.fix_power_uw is not None:
        powers = np.array([float(args.fix_power_uw)])
        print(f"  P axis pinned to {powers[0]:.1f} μW")
    else:
        powers = np.linspace(args.power_min_uw, args.power_max_uw, args.n_power)

    if args.sweep_delta is not None:
        lo, hi, n = args.sweep_delta
        delta_halfs = np.linspace(float(lo), float(hi), int(n))
        print(f"  Δ_max axis: {len(delta_halfs)} values from "
              f"{delta_halfs[0]:.1f} to {delta_halfs[-1]:.1f} MHz (symmetric ±)")
    else:
        delta_halfs = np.array([abs(args.delta_end_mhz)])
        print(f"  Δ_max axis pinned to ±{delta_halfs[0]:.1f} MHz")

    if args.sweep_tgate is not None:
        lo, hi, n = args.sweep_tgate
        t_gates_us = np.linspace(float(lo), float(hi), int(n))
        print(f"  T_gate axis: {len(t_gates_us)} values from "
              f"{t_gates_us[0]:.2f} to {t_gates_us[-1]:.2f} μs")
    else:
        t_gates_us = np.array([float(args.t_gate_us)])
        print(f"  T_gate axis pinned to {t_gates_us[0]:.2f} μs")

    n_total = len(wls) * len(powers) * len(delta_halfs) * len(t_gates_us)

    # Result arrays. Two trailing columns (delta_half_mhz, t_gate_us) added
    # in issue #44 follow-up so the new sweep modes can identify which cell
    # of the (Δ_max, T_gate) plane each row corresponds to. Legacy CSVs are
    # 17 columns; the new dtype is 19. cmd_optimize_plot dispatches on count.
    dtype = [
        ("wl", float), ("power_uw", float),
        ("delta_A_mhz", float), ("delta_B_mhz", float),
        ("scatter_A_hz", float),
        ("P_gg", float), ("P_gr", float), ("P_rg", float), ("P_rr", float),
        ("pinning_leak", float), ("crosstalk", float),
        ("scatter_penalty", float), ("total_cost", float),
        ("P_target", float), ("other_residual", float),
        ("delta_A_over_Omega", float), ("delta_A_over_V", float),
        ("delta_half_mhz", float), ("t_gate_us", float),
    ]
    grid = np.empty(n_total, dtype=dtype)

    n_jobs = args.n_jobs if args.n_jobs and args.n_jobs > 0 else (os.cpu_count() or 1)
    n_jobs = min(n_jobs, n_total)
    print(f"  Scanning {len(wls)}×{len(powers)}×{len(delta_halfs)}×"
          f"{len(t_gates_us)} = {n_total} grid points "
          f"on {n_jobs} workers...")
    t0 = _time.time()

    # 4-fold Cartesian product over (λ, P, Δ_max, T_gate). In legacy mode the
    # last two axes are length-1 so this collapses to the original 2D scan.
    tasks = [(i, float(wl), float(p), float(d), float(t))
             for i, (wl, p, d, t) in enumerate(
                 (wl, p, d, t)
                 for wl in wls
                 for p in powers
                 for d in delta_halfs
                 for t in t_gates_us)]

    # Pre-populate worker state in the *parent* process. The model/system
    # objects were already built above, so workers inherit them via fork
    # instead of each calling Rubidium87() — which hits a SQLite race in
    # ARC's literature-values cache when many workers start in parallel.
    # x_sweep and t_gate are now per-task (issue #44 sweep-axis support).
    _OPT_WORKER.update(
        model=model, system=system, psi0=psi0,
        ac_stark_peak=ac_stark_peak, eta=eta,
    )

    def _print_point(done, row):
        # row layout: (wl, power_uw, dA, dB, scatter_A, P_gg, P_gr, P_rg, P_rr,
        #              pinning_leak, crosstalk, scatter_pen, total_cost,
        #              ..., delta_half_mhz, t_gate_us)
        wl, power_uw = row[0], row[1]
        P_gg, P_gr = row[5], row[6]
        decay = row[11]  # 1 - exp(-Γ_A T_gate)
        dh, tg = row[17], row[18]
        print(f"  [{done:>5}/{n_total}] λ={wl:7.2f} nm  P={power_uw:6.1f} μW  "
              f"±{dh:5.1f} MHz  T={tg:5.2f} μs  "
              f"P_gg={P_gg:.4f}  P_gr={P_gr:.4f}  decay={decay:.4e}",
              flush=True)

    if n_jobs == 1:
        for task in tasks:
            i, row = _simulate_grid_point(task)
            grid[i] = row
            _print_point(i + 1, row)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        # Force fork start method so workers inherit _OPT_WORKER (and the
        # ARC atom database connection in cached form) from this process.
        # Required on platforms where the default is "spawn".
        mp_ctx = mp.get_context("fork")

        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=mp_ctx,
        ) as ex:
            done = 0
            # chunksize=1 streams every result back as soon as it's ready, so
            # the user sees progress immediately. For ~seconds-per-point work
            # the per-task IPC overhead is negligible compared to the compute.
            for i, row in ex.map(_simulate_grid_point, tasks, chunksize=1):
                grid[i] = row
                done += 1
                _print_point(done, row)

    elapsed = _time.time() - t0
    print(f"  Grid scan done in {elapsed:.1f}s ({elapsed/n_total:.2f}s per point)")

    # Find best
    best_idx = np.argmin(grid["total_cost"])
    best = grid[best_idx]
    print(f"\n  Best point: lambda={best['wl']:.1f} nm, P={best['power_uw']:.0f} uW, "
          f"±{best['delta_half_mhz']:.1f} MHz, T={best['t_gate_us']:.2f} μs")
    print(f"    delta_A = {best['delta_A_mhz']:.2f} MHz "
          f"({best['delta_A_over_Omega']:.1f} Omega_eff, "
          f"{best['delta_A_over_V']:.3f} V_ryd)")
    print(f"    P_gg={best['P_gg']:.4f}, P_gr={best['P_gr']:.4f}, "
          f"P_rg={best['P_rg']:.4f}, P_rr={best['P_rr']:.6f}")
    print(f"    pinning_leak={best['pinning_leak']:.4f}, "
          f"crosstalk={best['crosstalk']:.4f}, scatter={best['scatter_penalty']:.4f}")
    print(f"    TOTAL COST = {best['total_cost']:.6f}")

    # Save results. Embed experiment parameters as `# key=value` comment lines
    # in the CSV header so re-plotting can auto-recover them. np.loadtxt skips
    # any line starting with `#` by default, so the data section stays clean.
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, _prefixed("addressing_opt_grid.csv", args.prefix))

    meta_kv = {
        "distance_um": args.distance_um,
        "waist_um": args.waist_um,
        "t_gate_us": args.t_gate_us,
        "delta_start_mhz": args.delta_start_mhz,
        "delta_end_mhz": args.delta_end_mhz,
        "wl_min": args.wl_min,
        "wl_max": args.wl_max,
        "n_wl": args.n_wl,
        "power_min_uw": args.power_min_uw,
        "power_max_uw": args.power_max_uw,
        "n_power": args.n_power,
        # Reproducibility (added per issue #44 metadata-completeness fix):
        "rabi_420_hz": float(system.rabi_420 / (2 * pi)),
        "rabi_1013_hz": float(system.rabi_1013 / (2 * pi)),
        "Delta_hz": float(system.Delta / (2 * pi)),
        "ac_stark_peak_mhz": float(ac_stark_peak / (2 * pi) / 1e6),
        "ac_stark_off": bool(getattr(args, "no_ac_stark", False)),
        # Sweep-axis support (issue #44 follow-up). When the legacy axes are
        # collapsed, these record the single value used; when --sweep-delta /
        # --sweep-tgate are active they record the actual range.
        "sweep_delta_min_mhz": float(delta_halfs[0]),
        "sweep_delta_max_mhz": float(delta_halfs[-1]),
        "n_delta": int(len(delta_halfs)),
        "sweep_tgate_min_us": float(t_gates_us[0]),
        "sweep_tgate_max_us": float(t_gates_us[-1]),
        "n_tgate": int(len(t_gates_us)),
        "fix_wl": (float(args.fix_wl) if args.fix_wl is not None else "None"),
        "fix_power_uw": (float(args.fix_power_uw)
                         if args.fix_power_uw is not None else "None"),
    }
    header_lines = [f"{k}={v}" for k, v in meta_kv.items()]
    header_lines.append(",".join(grid.dtype.names))
    np.savetxt(csv_path, grid, fmt="%.6e",
               header="\n".join(header_lines), delimiter=",")
    print(f"\n  Grid saved to {csv_path}")

    # --- Top-K candidates ---
    top_k = min(args.top_k, n_total)
    ranking = np.argsort(grid["total_cost"])[:top_k]
    print(f"\n  Top-{top_k} candidates:")
    print(f"  {'Rank':>4} {'WL(nm)':>8} {'P(uW)':>8} {'dA(MHz)':>9} "
          f"{'P_gr':>7} {'Leak':>7} {'Xtalk':>7} {'Scat':>7} {'Total':>8}")
    for rank, ri in enumerate(ranking):
        r = grid[ri]
        print(f"  {rank+1:4d} {r['wl']:8.1f} {r['power_uw']:8.0f} "
              f"{r['delta_A_mhz']:9.2f} {r['P_gr']:7.4f} "
              f"{r['pinning_leak']:7.4f} {r['crosstalk']:7.4f} "
              f"{r['scatter_penalty']:7.4f} {r['total_cost']:8.5f}")

    # --- Hybrid validation of top-K with scatter in Hamiltonian ---
    # Each candidate now has its own (Δ_max, T_gate) cell, so reconstruct
    # x_sweep and t_gate from the per-row trailing columns rather than a
    # single shared x_sweep (issue #44 sweep-axis support).
    if top_k > 0:
        print(f"\n  Hybrid validation (scatter in Hamiltonian)...")
        for rank, ri in enumerate(ranking):
            r = grid[ri]
            delta_A = r["delta_A_mhz"] * 2 * pi * 1e6
            delta_B = eta * delta_A
            scatter_A = r["scatter_A_hz"]
            scatter_B = eta * scatter_A
            dh = r["delta_half_mhz"]
            tg = r["t_gate_us"] * 1e-6
            x_sweep_r = [
                2 * pi * (-dh) * 1e6 / system.rabi_eff,
                2 * pi *   dh  * 1e6 / system.rabi_eff,
                tg / system.time_scale,
            ]
            proto_scat = SweepProtocol(
                addressing={0: delta_A, 1: delta_B},
                scatter_rates={0: scatter_A, 1: scatter_B},
                ac_stark_shift=ac_stark_peak,
            )
            res = simulate(system, proto_scat, x_sweep_r, psi0)
            P_gr_dyn = model.observables.measure("pop_gr", res.psi_final)
            norm = float(np.real(np.vdot(res.psi_final, res.psi_final)))
            print(f"    #{rank+1}: P_gr={P_gr_dyn:.4f}, norm={norm:.6f}, "
                  f"norm_loss={1-norm:.4e}")

    # Reconstruct x_sweep at the best cell for meta consistency (CSV, replot).
    best_dh = best["delta_half_mhz"]
    best_tg = best["t_gate_us"] * 1e-6
    x_sweep = [
        2 * pi * (-best_dh) * 1e6 / system.rabi_eff,
        2 * pi *   best_dh  * 1e6 / system.rabi_eff,
        best_tg / system.time_scale,
    ]

    meta = {
        "system": system, "model": model, "eta": eta, "x_sweep": x_sweep,
        "ac_stark_peak": ac_stark_peak, "wls": wls, "powers": powers,
        "delta_halfs": delta_halfs, "t_gates_us": t_gates_us,
        "best": best, "best_idx": best_idx,
        "adiabatic_power_min": adiabatic_power_min,
    }
    return grid, meta


# ── shared helpers ────────────────────────────────────────────────────

def _make_adiabatic_boundary(meta):
    """Return a callable that overlays the adiabatic condition boundary on an axis."""
    adiabatic_power_min = meta["adiabatic_power_min"]
    powers = meta["powers"]
    adiab_wls = np.array(sorted(adiabatic_power_min.keys()))
    adiab_pows = np.array([adiabatic_power_min[w] for w in adiab_wls])

    def _add(ax):
        p_lo, p_hi = powers[0], powers[-1]
        mask = (adiab_pows >= p_lo) & (adiab_pows <= p_hi)
        if mask.any():
            ax.plot(adiab_pows[mask], adiab_wls[mask], "r--", lw=2, alpha=0.9,
                    label=r"$|\delta_A| = 4\,\Omega_{\rm eff}$")
    return _add


def _gaussian_noise_summary(meta, sigma_pts, ref_power_uw):
    """Map Gaussian smoothing width (grid points) to equivalent power noise."""
    powers = np.asarray(meta["powers"], dtype=float)
    if len(powers) < 2:
        return None
    power_step_uw = float(np.mean(np.diff(np.sort(powers))))
    sigma_power_uw = float(sigma_pts) * power_step_uw
    frac = sigma_power_uw / ref_power_uw if ref_power_uw > 0 else np.nan
    return {
        "power_step_uw": power_step_uw,
        "sigma_power_uw": sigma_power_uw,
        "sigma_frac": frac,
    }

# ── individual plot functions ──────────────────────────────────────────

def _plot_components(grid, meta, args):
    """2×2 cost component overview → addressing_opt_components.png"""
    wls = meta["wls"]; powers = meta["powers"]; best = meta["best"]
    distance_um = getattr(args, "distance_um", None)
    waist_um = getattr(args, "waist_um", None)
    t_gate_us = getattr(args, "t_gate_us", None)
    add_boundary = _make_adiabatic_boundary(meta)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (field, title, cmap) in zip(axes.flat, [
        ("pinning_leak", "Pinning leak $P_{rg}+P_{rr}$", "Reds"),
        ("crosstalk", "Crosstalk $P_{gg}$", "Blues"),
        ("scatter_penalty", "Scatter $1-e^{-\\Gamma T}$", "Oranges"),
        ("total_cost", "Total cost $E_{total}$", "viridis_r"),
    ]):
        data = grid[field].reshape(args.n_wl, args.n_power)
        im = ax.pcolormesh(powers, wls, data, cmap=cmap, shading="auto")
        add_boundary(ax)
        ax.plot(best["power_uw"], best["wl"], "k*", ms=12)
        ax.set_xlabel("Power ($\\mu$W)")
        ax.set_ylabel("Wavelength (nm)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.suptitle(
        "Cost decomposition\n"
        rf"selected raw best: $\lambda={best['wl']:.2f}\,$nm, "
        rf"$P={best['power_uw']:.1f}\,\mu$W, "
        rf"$\delta_A={best['delta_A_mhz']:.2f}$ MHz"
        + (rf", $a={distance_um:.2f}\,\mu$m" if distance_um is not None else "")
        + (rf", $w={waist_um:.2f}\,\mu$m" if waist_um is not None else "")
        + (rf", $T_{{\rm gate}}={t_gate_us:.2f}\,\mu$s" if t_gate_us is not None else ""),
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(args.outdir, _prefixed("addressing_opt_components.png", args.prefix))
    fig.savefig(path, dpi=150)
    print(f"  Components saved to {path}")
    plt.close(fig)


def _plot_smoothed(grid, meta, args):
    """Gaussian-smoothed cost landscape → addressing_opt_smoothed.png"""
    from scipy.ndimage import gaussian_filter1d

    wls = meta["wls"]; powers = meta["powers"]; best = meta["best"]
    distance_um = getattr(args, "distance_um", None)
    waist_um = getattr(args, "waist_um", None)
    t_gate_us = getattr(args, "t_gate_us", None)
    add_boundary = _make_adiabatic_boundary(meta)

    sigma_smooth = float(getattr(args, "smooth_sigma", OPT_SMOOTH_SIGMA))
    cost_2d_raw = grid["total_cost"].reshape(args.n_wl, args.n_power)
    cost_2d_smooth = gaussian_filter1d(cost_2d_raw, sigma=sigma_smooth, axis=1)

    smooth_flat_idx = np.argmin(cost_2d_smooth)
    smooth_wi, smooth_pi = np.unravel_index(smooth_flat_idx, cost_2d_smooth.shape)
    best_smooth_wl = wls[smooth_wi]
    best_smooth_pow = powers[smooth_pi]
    best_smooth_cost = cost_2d_smooth[smooth_wi, smooth_pi]
    noise = _gaussian_noise_summary(meta, sigma_smooth, best_smooth_pow)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(powers, wls, cost_2d_smooth, cmap="viridis_r", shading="auto")
    add_boundary(ax)
    ax.plot(best["power_uw"], best["wl"], "r*", ms=14, mew=2, mfc="none",
            label=(rf"Raw best: $\lambda={best['wl']:.2f}$ nm, "
                   rf"$P={best['power_uw']:.1f}\,\mu$W"))
    ax.plot(best_smooth_pow, best_smooth_wl, "r*", ms=14, mew=2,
            label=(rf"Smoothed best: $\lambda={best_smooth_wl:.2f}$ nm, "
                   rf"$P={best_smooth_pow:.1f}\,\mu$W"))
    ax.set_xlabel("Power ($\\mu$W)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title(r"Smoothed cost (Gaussian $\sigma$="
                 f"{sigma_smooth:g} pts, washes out LZS fringes)")
    if distance_um is not None or waist_um is not None or t_gate_us is not None:
        meta_lines = []
        if distance_um is not None:
            meta_lines.append(rf"$a = {distance_um:.2f}\,\mu$m")
        if waist_um is not None:
            meta_lines.append(rf"$w = {waist_um:.2f}\,\mu$m")
        if t_gate_us is not None:
            meta_lines.append(rf"$T_{{\rm gate}} = {t_gate_us:.2f}\,\mu$s")
        ax.text(
            0.98, 0.02,
            "\n".join(meta_lines),
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.92),
        )
    fig.colorbar(im, ax=ax, label="$E_{total}$ (smoothed)")
    ax.legend(fontsize=9)
    if noise is not None:
        noise_text = (
            rf"$\sigma_{{\rm G}}={sigma_smooth:g}$ pts"
            "\n"
            rf"$\Delta P_{{\rm grid}}\approx{noise['power_step_uw']:.2f}\,\mu$W"
            "\n"
            rf"$\sigma_P\approx{noise['sigma_power_uw']:.2f}\,\mu$W"
            "\n"
            rf"$\sigma_I/I\approx{100*noise['sigma_frac']:.1f}\%$ "
            rf"@ $P={best_smooth_pow:.1f}\,\mu$W"
        )
        ax.text(
            0.02, 0.02, noise_text,
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.92),
        )
    fig.tight_layout()
    path = os.path.join(args.outdir, _prefixed("addressing_opt_smoothed.png", args.prefix))
    fig.savefig(path, dpi=150)
    print(f"  Smoothed heatmap saved to {path}")
    print(f"    Smoothed best: lambda={best_smooth_wl:.2f} nm, "
          f"P={best_smooth_pow:.1f} uW, cost={best_smooth_cost:.5f}")
    if noise is not None:
        print(f"    Gaussian sigma={sigma_smooth:g} pts  ->  "
              f"sigma_P={noise['sigma_power_uw']:.2f} uW  ->  "
              f"sigma_I/I≈{100*noise['sigma_frac']:.1f}% "
              f"at P={best_smooth_pow:.1f} uW")
    plt.close(fig)


def _detect_sweep_axes(grid):
    """Return ((xname, xvals), (yname, yvals)) for the two axes that vary.

    Looks at the four scan axes ``wl``, ``power_uw``, ``delta_half_mhz``,
    ``t_gate_us`` and picks the (up to) two with more than one unique value.
    Returns ``None`` if fewer than two axes vary.
    """
    candidates = [
        ("wl", "wavelength (nm)"),
        ("power_uw", "power (μW)"),
        ("delta_half_mhz", r"$\Delta_{\max}$ (MHz)"),
        ("t_gate_us", r"$T_{\rm gate}$ (μs)"),
    ]
    varying = []
    for col, label in candidates:
        if col in grid.dtype.names:
            uniq = np.unique(grid[col])
            if len(uniq) > 1:
                varying.append((col, label, uniq))
    if len(varying) < 2:
        return None
    if len(varying) > 2:
        # More than two axes vary — pick the two with the most points.
        varying.sort(key=lambda v: -len(v[2]))
    (xc, xl, xv), (yc, yl, yv) = varying[0], varying[1]
    return (xc, xl, xv), (yc, yl, yv)


def _plot_components_sweep(grid, meta, args):
    """Generic 2×2 components heatmap for any pair of varying axes.

    Used when ``optimize`` runs in sweep mode (e.g. ``--sweep-delta`` and/or
    ``--sweep-tgate``) where the active axes are not the legacy (λ, P).
    """
    axes_info = _detect_sweep_axes(grid)
    if axes_info is None:
        print("  (no two-axis sweep detected; skipping components-sweep plot)")
        return
    (xc, xl, xv), (yc, yl, yv) = axes_info

    # Reshape to (len(yv), len(xv)) using the dtype field as the key.
    def reshape(field):
        z = np.full((len(yv), len(xv)), np.nan)
        for r in grid:
            ix = int(np.searchsorted(xv, r[xc]))
            iy = int(np.searchsorted(yv, r[yc]))
            if 0 <= ix < len(xv) and 0 <= iy < len(yv):
                z[iy, ix] = r[field]
        return z

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        ("P_gg",            "P_gg (crosstalk)",        "Blues"),
        ("pinning_leak",    "Pinning leak P_rg+P_rr",  "Reds"),
        ("scatter_penalty", "Scatter penalty",         "Oranges"),
        ("total_cost",      "Total cost",              "viridis_r"),
    ]
    best = meta["best"]
    for ax, (field, title, cmap) in zip(axs.flat, panels):
        Z = reshape(field)
        im = ax.pcolormesh(xv, yv, Z, cmap=cmap, shading="auto")
        ax.plot(best[xc], best[yc], "k*", ms=14)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax)
    fig.suptitle(
        f"Sweep components — best at "
        f"{xc}={best[xc]:.3g}, {yc}={best[yc]:.3g}, "
        f"total_cost={best['total_cost']:.4e}",
        fontsize=12)
    fig.tight_layout()
    path = os.path.join(args.outdir,
                        _prefixed("addressing_opt_components_sweep.png", args.prefix))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Sweep components saved to {path}")


def _is_legacy_wlp_mode(meta):
    """True iff the only varying axes are wavelength and power (legacy mode).

    When True, :func:`_plot_optimize` can draw the full (λ, P) heatmaps:
    2×2 component breakdown + smoothed total cost.  Sweep modes fall back to
    :func:`_plot_components_sweep`.
    """
    return (len(meta["wls"]) > 1 and len(meta["powers"]) > 1
            and len(meta["delta_halfs"]) == 1
            and len(meta["t_gates_us"]) == 1)


def _plot_optimize(grid, meta, args):
    """Generate plots from the optimization grid data.

    Legacy (λ, P) mode (see :func:`_is_legacy_wlp_mode`): two figures —
    ``addressing_opt_components.png`` (2×2 cost breakdown) and
    ``addressing_opt_smoothed.png`` (Gaussian-smoothed total cost).

    Sweep modes (collapsed λ and/or P, or extra axes in Δ_max / T_gate): a
    single ``addressing_opt_components_sweep.png`` on the two active axes.
    """
    os.makedirs(args.outdir, exist_ok=True)

    if _is_legacy_wlp_mode(meta):
        print("\n[1/2] Cost component overview (2×2)...")
        _plot_components(grid, meta, args)

        print("\n[2/2] Smoothed cost landscape...")
        _plot_smoothed(grid, meta, args)

    else:
        print("\n[sweep] Generic components heatmap on the active axes...")
        _plot_components_sweep(grid, meta, args)


def cmd_optimize(args):
    """2D (wavelength x power) optimization of local addressing parameters."""
    grid, meta = _run_optimize_scan(args)
    _plot_optimize(grid, meta, args)


def _read_csv_metadata(csv_path):
    """Parse `# key=value` comment lines from the top of a CSV.

    Returns a dict of strings (caller is responsible for casting). Stops at the
    first non-comment line, so this is cheap even for large files.
    """
    meta = {}
    with open(csv_path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            s = line.lstrip("#").strip()
            if "=" in s:
                k, v = s.split("=", 1)
                meta[k.strip()] = v.strip()
    return meta


def cmd_optimize_plot(args):
    """Re-plot from a previously saved CSV grid (no simulation)."""
    from ryd_gate.core.models.analog_3level import Analog3LevelModel

    # If --csv was not given explicitly, derive it from --outdir + --prefix
    # so users only need to remember the experiment tag.
    if args.csv is None:
        args.csv = os.path.join(
            args.outdir, _prefixed("addressing_opt_grid.csv", args.prefix))
        print(f"  --csv not given; using {args.csv}")

    # Auto-load experiment parameters embedded in the CSV header. CLI flags
    # (when explicitly passed) take precedence over the recovered values.
    csv_meta = _read_csv_metadata(args.csv)
    if csv_meta:
        print(f"  Recovered {len(csv_meta)} parameter(s) from CSV header:")
    for key, cast in [
        ("distance_um", float), ("waist_um", float), ("t_gate_us", float),
        ("delta_start_mhz", float), ("delta_end_mhz", float),
    ]:
        if getattr(args, key, None) is None and key in csv_meta:
            try:
                setattr(args, key, cast(csv_meta[key]))
                print(f"    {key} = {getattr(args, key)}  (from CSV)")
            except ValueError:
                pass
    # Hardcoded fallbacks if neither CLI nor CSV provided a value.
    _fallbacks = dict(distance_um=4.0, waist_um=2.0, t_gate_us=4.5,
                      delta_start_mhz=-40.0, delta_end_mhz=40.0)
    for key, val in _fallbacks.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)
            print(f"    {key} = {val}  (fallback default)")

    # Informational metadata (added in issue #44 follow-up): print these so
    # the user can see what physical parameters the CSV was generated with.
    # Not used by the plotting routines.
    for key in ("rabi_420_hz", "rabi_1013_hz", "Delta_hz",
                "ac_stark_peak_mhz", "ac_stark_off"):
        if key in csv_meta:
            print(f"    {key} = {csv_meta[key]}  (from CSV, info only)")

    # Load grid. Two CSV layouts are supported (issue #44 follow-up):
    #   17 columns — legacy `optimize` runs (no Δ_max / T_gate axes)
    #   19 columns — new sweep-aware runs with trailing (delta_half_mhz,
    #                t_gate_us) per row
    dtype_legacy = [
        ("wl", float), ("power_uw", float),
        ("delta_A_mhz", float), ("delta_B_mhz", float),
        ("scatter_A_hz", float),
        ("P_gg", float), ("P_gr", float), ("P_rg", float), ("P_rr", float),
        ("pinning_leak", float), ("crosstalk", float),
        ("scatter_penalty", float), ("total_cost", float),
        ("P_target", float), ("other_residual", float),
        ("delta_A_over_Omega", float), ("delta_A_over_V", float),
    ]
    dtype_new = dtype_legacy + [
        ("delta_half_mhz", float), ("t_gate_us", float),
    ]
    raw = np.loadtxt(args.csv, delimiter=",", dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] == 17:
        dtype = dtype_legacy
    elif raw.shape[1] == 19:
        dtype = dtype_new
    else:
        raise ValueError(
            f"unexpected CSV column count {raw.shape[1]}; "
            f"expected 17 (legacy) or 19 (sweep-aware)")
    grid = np.empty(raw.shape[0], dtype=dtype)
    for i, name in enumerate(grid.dtype.names):
        grid[name] = raw[:, i]
    print(f"  Loaded {len(grid)} points from {args.csv} "
          f"({raw.shape[1]} columns)")

    # If the CSV is the 17-column legacy form, synthesize the trailing two
    # columns from the recovered metadata so the unified 19-column-aware
    # plot path can use the same code.
    if raw.shape[1] == 17:
        new_grid = np.empty(len(grid), dtype=dtype_new)
        for name in grid.dtype.names:
            new_grid[name] = grid[name]
        new_grid["delta_half_mhz"] = float(args.delta_end_mhz)
        new_grid["t_gate_us"] = float(args.t_gate_us)
        grid = new_grid

    # Reconstruct grid shape from unique values
    wls = np.unique(grid["wl"])
    powers = np.unique(grid["power_uw"])
    delta_halfs = np.unique(grid["delta_half_mhz"])
    t_gates_us = np.unique(grid["t_gate_us"])
    args.n_wl = len(wls)
    args.n_power = len(powers)

    # Rebuild model for adiabatic-boundary overlay and meta consistency
    model = Analog3LevelModel.from_defaults(
        detuning_sign=1, blackmanflag=True,
        distance_um=args.distance_um,
        Delta_Hz=2.4e9,
        rabi_420_Hz=135e6,
        rabi_1013_Hz=135e6,
    )
    system = model.system
    eta = np.exp(-2 * (args.distance_um / args.waist_um) ** 2)
    ac_stark_peak = system.rabi_420 ** 2 / (4 * abs(system.Delta))

    best_idx = np.argmin(grid["total_cost"])
    best = grid[best_idx]

    # Reconstruct x_sweep from the best cell's Δ_max / T_gate (meta / CSV).
    best_dh = float(best["delta_half_mhz"])
    best_tg = float(best["t_gate_us"]) * 1e-6
    x_sweep = [
        2 * pi * (-best_dh) * 1e6 / system.rabi_eff,
        2 * pi *   best_dh  * 1e6 / system.rabi_eff,
        best_tg / system.time_scale,
    ]

    # Adiabatic boundary
    adiabatic_ratio = 4.0
    adiabatic_power_min = {}
    for wl in wls:
        shift_ref, _ = compute_shift_scatter(wl)
        shift_per_uw = abs(float(shift_ref)) / POWER_REF_UW
        adiabatic_power_min[wl] = adiabatic_ratio * system.rabi_eff / (2 * pi) / shift_per_uw

    meta = {
        "system": system, "model": model, "eta": eta, "x_sweep": x_sweep,
        "ac_stark_peak": ac_stark_peak, "wls": wls, "powers": powers,
        "delta_halfs": delta_halfs, "t_gates_us": t_gates_us,
        "best": best, "best_idx": best_idx,
        "adiabatic_power_min": adiabatic_power_min,
    }
    os.makedirs(args.outdir, exist_ok=True)
    _plot_optimize(grid, meta, args)




# ═══════════════════════════════════════════════════════════════════════
# OPTIMIZE-COMMAND DEFAULTS  (edit here to change baseline scan settings)
# ═══════════════════════════════════════════════════════════════════════
FIGDIR = "docs/figures"

# These constants are wired into the `optimize` subparser as `default=...`.
# CLI flags still override them at runtime; this block exists so you can
# tweak the baseline experiment without hunting through argparse boilerplate.

# --- Wavelength axis (addressing laser tuning) ---------------------------
OPT_WL_MIN_NM = 781.0       # Lower edge of the AC-Stark addressing laser sweep [nm].
OPT_WL_MAX_NM = 786.0       # Upper edge of the sweep [nm]. Window straddles the
                            # 5P3/2 D2 line region where the differential
                            # ground/Rydberg light shift can be tuned.
OPT_N_WL = 60               # Number of wavelength samples (grid resolution along λ).

# --- Power axis (addressing laser intensity at the target atom) ----------
OPT_POWER_MIN_UW = 140.0    # Min single-beam optical power [μW] delivered to atom A.
                            # Should sit at/above the adiabaticity threshold
                            # |δ_A| ≳ 4·Ω_eff for the chosen wavelength.
OPT_POWER_MAX_UW = 500.0    # Max optical power [μW]. Upper bound is set by
                            # tolerable photon scattering on atom A.
OPT_N_POWER = 40            # Number of power samples along the P axis.

# --- Geometry (atom pair + addressing beam) ------------------------------
OPT_DISTANCE_UM = 4.5       # Inter-atom separation a [μm]. Chosen so V_ryd(a) ≫ δ_A
                            # to keep the pair out of the anti-blockade resonance.
OPT_WAIST_UM = 2.0          # 1/e² radius of the addressing Gaussian beam [μm].
                            # Sets the crosstalk tail η = exp(-2 (a/w)²) felt by
                            # the spectator atom B.

# --- Detuning sweep (Landau–Zener ramp on atom A) ------------------------
OPT_DELTA_START_MHZ = -80 # Initial detuning of the addressing ramp [MHz].
OPT_DELTA_END_MHZ = 80    # Final detuning of the addressing ramp [MHz].
                            # Together with T_GATE these define the sweep rate
                            # α = (Δ_end − Δ_start)/T_gate that drives the LZ pin.
OPT_T_GATE_US = 9         # Total gate duration [μs] over which Δ ramps from
                            # start → end. Larger T → more adiabatic but more
                            # scattering accumulated.

# --- Output / runtime ----------------------------------------------------
OPT_TOP_K = 5               # Number of best (λ, P) candidates to print + validate.
OPT_OUTDIR = "results"      # Where to drop CSV + plots.
OPT_PREFIX = "exp5"             # Filename prefix prepended to every CSV/PNG output
                            # (e.g. "exp1" → exp1_addressing_opt_grid.csv,
                            # exp1_addressing_opt_components.png, ...). Lets you
                            # keep multiple runs side by side without overwriting.
                            # Empty string = no prefix (default behavior).
OPT_N_JOBS = 0              # CPU workers for the parallel grid scan.
                            # 0 = use all cores, 1 = serial (debug-friendly).
OPT_SMOOTH_SIGMA = 3.0      # Gaussian smoothing width (in power-grid points) used
                            # when plotting the smoothed landscape. This can be
                            # reinterpreted as an effective quasi-static power noise.
OPT_NO_AC_STARK = False     # Diagnostic only: set True to disable the AC-Stark
                            # phase feed-forward in SweepProtocol. Issue #44
                            # showed the existing Ω²/(4Δ) formula is correct
                            # for the analog 3-level model; this flag exists so
                            # users can verify empirically by comparing P_gg
                            # with vs without compensation.


def _prefixed(filename, prefix):
    """Prepend ``prefix_`` to ``filename`` (no-op when prefix is empty)."""
    if not prefix:
        return filename
    return f"{prefix}_{filename}"



# ═══════════════════════════════════════════════════════════════════════
# Main
#═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ns = sub.add_parser("noise", help="Scan noise parameters")
    p_ns.add_argument("--n-mc", type=int, default=50,
                       help="MC shots per point (default: 50)")
    p_ns.add_argument("--n-points", type=int, default=8,
                       help="Scan points per noise source (default: 8)")
    p_ns.add_argument("--combined", action="store_true",
                       help="Also run combined noise sweep")

    # Defaults come from the OPTIMIZE-COMMAND DEFAULTS block at the top of
    # this file — edit there to change the baseline experiment, not here.
    p_opt = sub.add_parser("optimize",
                            help="2D wavelength x power optimization")
    p_opt.add_argument("--wl-min", type=float, default=OPT_WL_MIN_NM,
                       help="Lower edge of wavelength sweep [nm]")
    p_opt.add_argument("--wl-max", type=float, default=OPT_WL_MAX_NM,
                       help="Upper edge of wavelength sweep [nm]")
    p_opt.add_argument("--n-wl", type=int, default=OPT_N_WL,
                       help="Wavelength grid resolution")
    p_opt.add_argument("--power-min-uw", type=float, default=OPT_POWER_MIN_UW,
                       help="Min addressing power [μW] (≳ adiabaticity threshold)")
    p_opt.add_argument("--power-max-uw", type=float, default=OPT_POWER_MAX_UW,
                       help="Max addressing power [μW] (limited by scattering)")
    p_opt.add_argument("--n-power", type=int, default=OPT_N_POWER,
                       help="Power grid resolution")
    p_opt.add_argument("--distance-um", type=float, default=OPT_DISTANCE_UM,
                       help="Inter-atom separation [μm]; keeps V_ryd ≫ δ_A")
    p_opt.add_argument("--waist-um", type=float, default=OPT_WAIST_UM,
                       help="Addressing beam 1/e² waist [μm]; sets crosstalk η")
    p_opt.add_argument("--delta-start-mhz", type=float, default=OPT_DELTA_START_MHZ,
                       help="Initial detuning of LZ ramp [MHz]")
    p_opt.add_argument("--delta-end-mhz", type=float, default=OPT_DELTA_END_MHZ,
                       help="Final detuning of LZ ramp [MHz]")
    p_opt.add_argument("--t-gate-us", type=float, default=OPT_T_GATE_US,
                       help="Gate duration [μs] over which Δ ramps")
    p_opt.add_argument("--top-k", type=int, default=OPT_TOP_K,
                       help="Number of best candidates to print + validate")
    p_opt.add_argument("--outdir", type=str, default=OPT_OUTDIR,
                       help="Output directory for CSV + plots")
    p_opt.add_argument("--n-jobs", type=int, default=OPT_N_JOBS,
                       help="CPU workers for grid scan (0 = all cores, 1 = serial)")
    p_opt.add_argument("--prefix", type=str, default=OPT_PREFIX,
                       help="Filename prefix for all CSV/PNG outputs "
                            "(e.g. 'exp1' → exp1_addressing_opt_grid.csv, ...)")
    p_opt.add_argument("--smooth-sigma", type=float, default=OPT_SMOOTH_SIGMA,
                       help="Gaussian smoothing width in power-grid points for "
                            "the smoothed landscape")
    p_opt.add_argument("--no-ac-stark", action="store_true",
                       default=OPT_NO_AC_STARK,
                       help="Disable AC-Stark phase feed-forward in "
                            "SweepProtocol (diagnostic only — see issue #44)")
    # --- Issue #44 follow-up: 2D sweep over (Δ_max, T_gate) at fixed (λ,P) ---
    p_opt.add_argument("--sweep-delta", nargs=3, type=float,
                       metavar=("MIN", "MAX", "N"), default=None,
                       help="Replace fixed Δ-window with a sweep over N values "
                            "from MIN to MAX MHz (each value sets symmetric "
                            "±MHz). Conflicts with --delta-start/--delta-end.")
    p_opt.add_argument("--sweep-tgate", nargs=3, type=float,
                       metavar=("MIN", "MAX", "N"), default=None,
                       help="Replace fixed --t-gate-us with a sweep over N "
                            "values from MIN to MAX μs.")
    p_opt.add_argument("--fix-wl", type=float, default=None,
                       help="Pin wavelength to a single value (nm); collapses "
                            "the wavelength axis to one point.")
    p_opt.add_argument("--fix-power-uw", type=float, default=None,
                       help="Pin power to a single value (μW); collapses the "
                            "power axis to one point.")

    p_plt = sub.add_parser("optimize-plot",
                            help="Re-plot from saved CSV (no simulation)")
    p_plt.add_argument("--csv", type=str, default=None,
                        help="Path to grid CSV from optimize run "
                             "(default: <outdir>/[<prefix>_]addressing_opt_grid.csv)")
    p_plt.add_argument("--outdir", type=str, default=OPT_OUTDIR,
                        help="Output directory for re-plotted figures")
    p_plt.add_argument("--prefix", type=str, default=OPT_PREFIX,
                        help="Filename prefix for output figures (and for "
                             "auto-resolving --csv if not given explicitly)")
    p_plt.add_argument("--smooth-sigma", type=float, default=OPT_SMOOTH_SIGMA,
                       help="Gaussian smoothing width in power-grid points for "
                            "the smoothed landscape")

    args = parser.parse_args()

    if args.command == "noise":
        cmd_noise(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "optimize-plot":
        cmd_optimize_plot(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
