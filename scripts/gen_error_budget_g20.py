#!/usr/bin/env python
"""Batch generator for the CZ error-budget maps at a chosen grid resolution (default 20x20)
with the alias-free exact_ode solver, in parallel.

It shares the physical point evaluator with ``scripts/notebooks/error_buget.ipynb``
and writes
``results/error_budget/cz_gate_maps/error_budget_fig{B,C,A_De<NN>}_ode_g<N>.npz``
caches, which the
notebook then loads by setting ``ODE_GRID_N = <N>``.

Resume-friendly: a figure/slice whose cache already exists is skipped. Each point has a
wall-clock timeout so a pathological ODE leaves a blank pixel instead of stalling the sweep.

Run (detached, ~20 h for 20x20 on 40 cores):
    setsid uv run --extra dev python scripts/gen_error_budget_g20.py 20 > /tmp/g20.log 2>&1 &
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as _mp

import numpy as np

# Keep the shared evaluator importable both as a script and through importlib.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from error_budget_model import eval_cz_point

MHz = 2 * np.pi * 1e6

ODE_BACKEND = "exact_ode"
ODE_N_EVAL = 301
ODE_N_STEPS = 400
ODE_RTOL = 1e-6
ODE_ATOL = 1e-9
N_WORKERS = 40
TIMEOUT_S = 5400          # 90 min per grid point; pathological ODE -> None -> blank pixel

_GRID_KEYS = ["phase_err", "max_leakage", "p_mid_max", "p_ryd_max", "p_r_garb_max", "p_loss_total_max"]


def results_dir():
    root = os.getcwd()
    for _ in range(6):
        if os.path.isdir(os.path.join(root, "results")) or os.path.exists(os.path.join(root, "pyproject.toml")):
            break
        root = os.path.dirname(root)
    d = os.path.join(root, "results", "error_budget", "cz_gate_maps")
    os.makedirs(d, exist_ok=True)
    return d


def eval_cz_point_safe(cfg):
    """Per-point wall-clock timeout + never raise: a stalled/failed point -> None -> blank pixel."""
    def _timeout(signum, frame):
        raise TimeoutError("grid point exceeded time budget")
    prev = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(int(cfg.get("timeout_s", 0)))
    try:
        return eval_cz_point(cfg)
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def run_grid(cfgs, n_workers=N_WORKERS):
    if not cfgs:
        return []
    ctx = _mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=min(n_workers, len(cfgs)), mp_context=ctx) as ex:
        return list(ex.map(eval_cz_point_safe, cfgs))


def make_cfg(Delta_e_Hz, D_sweep_Hz, p420_w, p1013_w, *, spacing_um=3.0, t_gate=1.0e-6,
             optics_loss=0.9, ryd_level=70, beam_factor=7*20):
    return dict(Delta_e_Hz=Delta_e_Hz, D_sweep_Hz=D_sweep_Hz, p420_w=p420_w, p1013_w=p1013_w,
                spacing_um=spacing_um, t_gate=t_gate, optics_loss=optics_loss, ryd_level=ryd_level,
                beam_factor=beam_factor, backend=ODE_BACKEND, n_steps=ODE_N_STEPS, n_eval=ODE_N_EVAL,
                rtol=ODE_RTOL, atol=ODE_ATOL, timeout_s=TIMEOUT_S)


def save_sweep(path, records, axis_x, axis_y, axis_x_name, axis_y_name, grids):
    np.savez(path, records=np.array(records, dtype=object),
             axis_x=axis_x, axis_y=axis_y, axis_x_name=axis_x_name, axis_y_name=axis_y_name,
             **{f"grid_{k}": grids[k] for k in _GRID_KEYS})


def _fill(idxs, recs, grid_n):
    grids = {k: np.full((grid_n, grid_n), np.nan) for k in _GRID_KEYS}
    records = []
    for (iy, ix), r in zip(idxs, recs):
        if r is None:
            continue
        for k in grids:
            grids[k][iy, ix] = r[k]
        records.append(r)
    return grids, records, len(recs) - len(records)


def _run_and_save(path, label, cfgs, idxs, grid_n, axis_x, axis_y, axis_x_name, axis_y_name):
    """Shared tail for the fig_* generators: run the grid, fill, persist, log."""
    print(f"[{label}] grid_n={grid_n}  {len(cfgs)} points ...", flush=True)
    grids, records, nf = _fill(idxs, run_grid(cfgs), grid_n)
    save_sweep(path, records, axis_x, axis_y, axis_x_name, axis_y_name, grids)
    print(f"[{label}] saved -> {path}  ({len(records)} pts, {nf} failed)", flush=True)


def fig_B(grid_n, rdir):
    path = os.path.join(rdir, f"error_budget_figB_ode_g{grid_n}.npz")
    if os.path.exists(path):
        print(f"[Fig B] cache exists, skipping -> {path}", flush=True)
        return
    spacing_um, d_sweep_hz, optics_loss, ryd_level = 3.0, 20e6, 0.9, 70
    beam_area_um2, p420_max_w, p1013_fixed_w = 7*20*spacing_um, 6.41, 100.0
    dGHz = np.linspace(15.0, 80.0, grid_n)
    kMHz = np.linspace(1.0, 12.0, grid_n)
    from ryd_gate.physics import rb87_7_mp_rabi_frequencies
    o420_1W, o1013 = rb87_7_mp_rabi_frequencies(1.0*(1-optics_loss), p1013_fixed_w*(1-optics_loss),
                                                beam_area_um2, ryd_level=ryd_level)
    cfgs, idxs = [], []
    for ix, De in enumerate(dGHz):
        Delta_e = 2*np.pi*De*1e9
        for iy, Ke in enumerate(kMHz):
            p420 = (4.0*Delta_e*(MHz*Ke)/o1013 / o420_1W)**2
            if p420 > p420_max_w:
                continue
            cfgs.append(make_cfg(De*1e9, d_sweep_hz, p420, p1013_fixed_w))
            idxs.append((iy, ix))
    _run_and_save(path, "Fig B", cfgs, idxs, grid_n, dGHz, kMHz, "Delta_e/2pi (GHz)", "K_eff/2pi (MHz)")


def fig_C(grid_n, rdir):
    path = os.path.join(rdir, f"error_budget_figC_ode_g{grid_n}.npz")
    if os.path.exists(path):
        print(f"[Fig C] cache exists, skipping -> {path}", flush=True)
        return
    p420_c_w, p1013_c_w = 6.41, 100.0
    dGHz = np.linspace(15.0, 80.0, grid_n)
    dswMHz = np.linspace(2.0, 30.0, grid_n)
    cfgs, idxs = [], []
    for ix, De in enumerate(dGHz):
        for iy, Dsw in enumerate(dswMHz):
            cfgs.append(make_cfg(De*1e9, Dsw*1e6, p420_c_w, p1013_c_w))
            idxs.append((iy, ix))
    _run_and_save(path, "Fig C", cfgs, idxs, grid_n, dGHz, dswMHz, "Delta_e/2pi (GHz)", "D_sweep/2pi (MHz)")


def fig_A(grid_n, rdir):
    d_sweep_hz, p420_max_w = 20e6, 6.41
    slices = [20e9, 30e9, 45e9]
    p420_grid = np.linspace(0.25, p420_max_w, grid_n)
    p1013_grid = np.linspace(5.0, 100.0, grid_n)
    for De_Hz in slices:
        path = os.path.join(rdir, f"error_budget_figA_De{int(round(De_Hz/1e9))}_ode_g{grid_n}.npz")
        if os.path.exists(path):
            print(f"[Fig A De={De_Hz/1e9:.0f}] cache exists, skipping -> {path}", flush=True)
            continue
        cfgs, idxs = [], []
        for iy, p1013 in enumerate(p1013_grid):
            for ix, p420 in enumerate(p420_grid):
                cfgs.append(make_cfg(De_Hz, d_sweep_hz, p420, p1013))
                idxs.append((iy, ix))
        _run_and_save(path, f"Fig A De={De_Hz/1e9:.0f}", cfgs, idxs, grid_n, p420_grid, p1013_grid,
                      "P420 before loss (W)", "P1013 before loss (W)")


if __name__ == "__main__":
    grid_n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rdir = results_dir()
    print(f"=== error-budget g{grid_n} generation (exact_ode, {N_WORKERS} workers, "
          f"rtol={ODE_RTOL}, timeout={TIMEOUT_S}s/pt) -> {rdir} ===", flush=True)
    t0 = time.time()
    fig_B(grid_n, rdir); print(f"  [elapsed {(time.time()-t0)/60:.1f} min]", flush=True)
    fig_C(grid_n, rdir); print(f"  [elapsed {(time.time()-t0)/60:.1f} min]", flush=True)
    fig_A(grid_n, rdir); print(f"  [elapsed {(time.time()-t0)/60:.1f} min]", flush=True)
    print(f"=== DONE g{grid_n} in {(time.time()-t0)/3600:.2f} h ===", flush=True)
