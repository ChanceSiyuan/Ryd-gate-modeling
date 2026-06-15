"""Standalone 10x10 finite-PEPS quench run (rydtn backend: NTU evolve + boundary-MPS).

Reproduces the 10x10 cell of scripts/notebooks/04_quench_and_state_prep.ipynb (Part A) as a headless,
long-running job.  Physics/backend options are identical to the notebook; only the
plotting is dropped and results are saved to an .npz for later inspection.

Set RYDTN_PROGRESS=10 (env) to get a per-10-step progress line with ETA + GPU mem.
"""

from __future__ import annotations

import time
import traceback

import numpy as np

import ryd_gate as rg
from ryd_gate import InteractionSpec
from ryd_gate.lattice import Register

OUT = "scripts/peps_10x10_results.npz"

# ---- protocol parameters (identical to notebook cell 4) ----
a_um = 10                          # lattice spacing, um
C6_70s = 2 * np.pi * 874e9         # rad/s * um^6, Rb 70S typical

Omega = 2 * np.pi * 3.8e6          # rad/s
delta_start = -2 * np.pi * 10.0e6  # rad/s
delta_end = 2 * np.pi * 10.0e6     # rad/s
t_sweep = 1.5e-6                   # s


def omega_half_t(t):
    ramp_frac = 0.09
    s = np.clip(t / max(t_sweep, np.finfo(float).eps), 0.0, 1.0)

    def smoothstep5(u):
        u = np.clip(u, 0.0, 1.0)
        return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5

    if s < ramp_frac:
        env = smoothstep5(s / ramp_frac)
    elif s > 1.0 - ramp_frac:
        env = smoothstep5((1.0 - s) / ramp_frac)
    else:
        env = 1.0
    return 0.5 * Omega * env


def delta_t(t):
    s = np.clip(t / max(t_sweep, np.finfo(float).eps), 0.0, 1.0)
    delta_mid = 0.5 * (delta_start + delta_end)
    delta_amp = 0.5 * (delta_end - delta_start)
    return delta_mid - delta_amp * np.cos(2.0 * np.pi * s)


t_eval = np.linspace(0.0, t_sweep, 7)
dt_tn = 0.2 / Omega  # seconds; 0.2 * Omega^{-1}

# ---- 10x10 system (notebook cell 17) ----
Lx, Ly = 10, 10
geom = Register.rectangle(Lx, Ly, spacing_um=a_um)
protocol = rg.SweepProtocol(
    t_gate=t_sweep, omega_half_fn=omega_half_t, delta_fn=delta_t, n_steps=120,
)
system = rg.RydbergSystem.from_lattice(
    geom, "1r",
    interaction=InteractionSpec(C6=C6_70s, mode="nn"),
    protocol=protocol,
)

backend_options = {
    "chi_max": 8,
    "dt": dt_tn,
    "svd_min": 1e-8,
    "measurement_environment": "bp",
    "update_environment": "ntu",
    "max_iter": 10,
    "tol_iter": 1e-7,
    "use_cuda": True,
    "backend_name": "torch",
    "device": "cuda",
}

print(f"=== 10x10 finite-PEPS quench ===", flush=True)
print(f"Lx={Lx} Ly={Ly} N={Lx*Ly}  t_sweep={t_sweep:.3e}s  dt_tn={dt_tn:.3e}s", flush=True)
print(f"t_eval(us) = {np.round(t_eval*1e6, 4).tolist()}", flush=True)
print(f"backend_options = {backend_options}", flush=True)

t0 = time.perf_counter()
try:
    res = rg.simulate(
        system, [], "all_ground",
        backend="peps", t_eval=t_eval, observables=["n_mean", "n_i"],
        backend_options=backend_options,
    )
except Exception:
    print("!!! RUN FAILED", flush=True)
    traceback.print_exc()
    raise

elapsed = time.perf_counter() - t0
n_mean = np.asarray(res.metadata["obs"]["n_mean"])
n_i = np.asarray(res.metadata["obs"]["n_i"])

print(f"\n10x10 PEPS elapsed: {elapsed:.3f} s  ({elapsed/3600:.2f} h)", flush=True)
print("time(us)  PEPS <n_r>", flush=True)
for t, val in zip(t_eval * 1e6, n_mean):
    print(f"{t:7.3f}   {val:10.4f}", flush=True)

np.savez(
    OUT,
    times=np.asarray(res.times),
    t_eval=t_eval,
    n_mean=n_mean,
    n_i=n_i,
    Lx=Lx, Ly=Ly,
    elapsed=elapsed,
    chi_max=backend_options["chi_max"],
    dt=res.metadata.get("dt"),
    max_truncation_error=res.metadata.get("max_truncation_error"),
    accumulated_truncation_error=res.metadata.get("accumulated_truncation_error"),
)
print(f"\nSaved -> {OUT}", flush=True)
print("DONE", flush=True)
