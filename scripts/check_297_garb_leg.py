"""Garbage-leg attribution check for the 297 nm TO calibration.

Answers "is the 9.17e-4 coherent error intrinsic, or optimizer slack?" with two
deterministic experiments against the calibrated record
results/297_to_calibration/to_297.json (no RNG anywhere, so reruns reproduce):

1. Frozen-pulse B scan: B only feeds the r_garb Zeeman detuning delta_Z in
   rb87_297_clock_4 (kappa, C6 channels and decay rates are B-independent), so
   sweeping B with the pulse frozen isolates the garbage leg. The single-driven-
   atom |01> leakage collapsing faster than any power law identifies sideband-
   comb spectral leakage.
2. Re-polish at B=160 G (delta_Z x8): the same 5-parameter family + the same
   Nelder-Mead polish, restarted from the frozen optimum. Reaching ~1e-6 proves
   the pipeline has no generic ~1e-3 floor.

Writes results/297_to_calibration/garb_leg_check.json.

Run (DGX):
  uv run python scripts/check_297_garb_leg.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_to_297 as cal

from ryd_gate import level_structure
from ryd_gate.physics import rb87_297_clock_rabi_frequencies, zeeman_shift_rad_s

OUTDIR = Path("results/297_to_calibration")
SCAN_B = (20.0, 40.0, 80.0, 160.0)
REPOLISH_B = 160.0


def make_cfg(omega, omega_garb, B):
    return {
        "omega": omega, "omega_garb": omega_garb,
        "rise_time_s": 20e-9, "spacing_um": 3.0,
        "ryd_level": 53, "magnetic_field_G": B,
        "level_structure": level_structure("rb87_297_clock_4", ryd_level=53,
                                           magnetic_field_G=B),
    }


def main():
    rec = json.loads((OUTDIR / "to_297.json").read_text())
    x = rec["x"]
    x0 = np.array([x[0], x[1], x[2], x[3], x[5]])  # [A, w, phi0, d, T]
    omega, omega_garb = rb87_297_clock_rabi_frequencies(0.6, 420.0, ryd_level=53)

    scan = []
    for B in SCAN_B:
        cfg = make_cfg(omega, omega_garb, B)
        a00, a01, a11 = cal.gate_amps(x0, cfg, backend_options=cal.TIGHT)
        infid, _ = cal.cz_infidelity(a00, a01, a11)
        row = {
            "B_G": B,
            "garb_zeeman_rad_s": zeeman_shift_rad_s(B, l=1, j=1.5, delta_mj=1.0),
            "tight_tol_infidelity": float(infid),
            "leak_01": float(1 - abs(a01) ** 2),
            "leak_11": float(1 - abs(a11) ** 2),
        }
        scan.append(row)
        print(f"scan B={B:6.1f} G: 1-F {infid:.3e}, "
              f"leak01 {row['leak_01']:.3e}, leak11 {row['leak_11']:.3e}", flush=True)

    # Re-polish at REPOLISH_B. Serial warm-up in the main process first so the
    # forked workers inherit the ARC C6 cache instead of racing on its sqlite DB.
    cfg = cal._CFG = make_cfg(omega, omega_garb, REPOLISH_B)
    t0 = time.time()
    v0 = cal.objective(x0, cfg, tol=cal.SEARCH_TOL)
    print(f"re-polish warm-up (search tol): 1-F {v0:.3e}", flush=True)
    with mp.get_context("fork").Pool(3) as pool:
        res = minimize(lambda x5: cal.objective(x5, cfg, tol=cal.SEARCH_TOL, pool=pool),
                       x0, method="Nelder-Mead",
                       options={"maxiter": 200, "xatol": 5e-5, "fatol": 1e-10})
        a00, a01, a11 = cal.gate_amps(res.x, cfg, backend_options=cal.TIGHT, pool=pool)
    infid, _ = cal.cz_infidelity(a00, a01, a11)
    print(f"re-polish B={REPOLISH_B} G: NM fun {res.fun:.3e} (nfev {res.nfev}) "
          f"-> tight 1-F {infid:.3e}", flush=True)

    out = {
        "source_record": "to_297.json",
        "frozen_x5": [float(v) for v in x0],
        "tight_tol": cal.TIGHT,
        "scan": scan,
        "repolish": {
            "B_G": REPOLISH_B,
            "search_tol": cal.SEARCH_TOL,
            "nm_fun_search_tol": float(res.fun),
            "nm_nfev": int(res.nfev),
            "x5": [float(v) for v in res.x],
            "tight_tol_infidelity": float(infid),
            "leak_01": float(1 - abs(a01) ** 2),
            "leak_11": float(1 - abs(a11) ** 2),
            "elapsed_s": time.time() - t0,
        },
    }
    (OUTDIR / "garb_leg_check.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUTDIR / 'garb_leg_check.json'}", flush=True)


if __name__ == "__main__":
    main()
