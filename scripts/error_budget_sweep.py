"""Error-budget map sweep + plot for the 1r round-trip detuning sweep.

Two modes:

  * ``--mode sweep``  4-D brute-force map over (lattice size, atom spacing,
                      Delta_e, t_sweep). Checkpoint/resume; live ETA.
  * ``--mode plot``   render the saved map: one figure per lattice size, a grid
                      of (spacing rows) x (error-channel cols) 2-D colormaps with
                      x = Delta_e, y = t_sweep.

Physics model: the "1r" two-level (|1>,|r>) round-trip detuning sweep on an
Lx x Ly square lattice (exact backend, nearest-neighbour vdW). A configuration is
driven toward Rydberg and swept back; we score three per-atom error channels
(site-averaged) plus their sum:

  eps_coh  = mean_i n_r_i(T)                                   # residual (non-adiabatic) excitation
  eps_SE   = mean_i Gamma_r * INT n_r_i dt                     # Rydberg spontaneous emission
  eps_sc   = mean_i Gamma_e * INT [ (4/3)(O420(t)/2De)^2 n1_i  # intermediate-state scattering,
                                    + (O1013/2De)^2 n_r_i ] dt #   both laser legs
  eps_total = eps_coh + eps_SE + eps_sc
  eps_loss  = eta_loss * eps_sc                                # atom-loss subset of scattering

LASER INTENSITY / RABI (the reworked piece). The 420 nm (6.4 W) and 1013 nm
(100 W) beams illuminate the *whole ~200-atom array*, not a single site. At
spacing ``a`` the array footprint is a ``sqrt(N_BEAM_ATOMS)*a`` by ``6 µm`` rectangle,
so the top-hat intensity is ``I = P / (sqrt(N_BEAM_ATOMS) * a * 6 µm)`` and the
single-photon Rabis
scale as ``Omega ~ 1/a`` (Omega_eff ~ 1/a^2). The simulated Lx x Ly lattice is a
representative patch; its size sets the many-body Hilbert space but NOT the drive
strength. So Omega_eff depends on ``a`` and ``Delta_e`` only, shared across lattices.

NOTE on the model: eps_SE/eps_sc/eps_loss are post-hoc perturbative estimates on a
lossless unitary run (no Lindblad), valid because the 1r trajectory conserves norm
(n1_i = 1 - n_r_i). eps_sc is essentially quasi-static (~INT env^2 dt) and converged
at modest n_steps; eps_coh is the residual non-adiabatic excitation and is
discretization-limited (needs large n_steps) -- in the strong-drive regime here it
is sub-dominant to eps_sc, so it is reported but flagged as the least trustworthy.

NOTE on d_amp: the detuning-sweep half-amplitude is FIXED (``--d-amp-mhz``, default
20 MHz) -- the experiment can sweep only ~+-20 MHz. It is deliberately not scaled
with Omega_eff.

NOTE on the backend: we force the sparse exact solver. The auto-selector routes
small dims to a dense ``expm`` that is pathologically slow here (||dt*H|| is large
because Omega_eff ~ 100 MHz), e.g. 2x4 is ~50x slower on the dense path.

Usage:
    # full default sweep (6 lattices x 6 spacings x 100 x 100); cheap lattice first
    python scripts/error_budget_sweep.py --mode sweep --resume

    # faster preview
    python scripts/error_budget_sweep.py --mode sweep --res 40 --lattices 2x2 3x3

    # plot whatever the checkpoint has so far
    python scripts/error_budget_sweep.py --mode plot

Headless (multi-day server run):
    setsid nohup python scripts/error_budget_sweep.py --mode sweep --resume \
        > scripts/ebudget_sweep.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

import ryd_gate as rg
from ryd_gate import InteractionSpec
from ryd_gate.backends.exact.simulate import simulate as exact_simulate
from ryd_gate.backends.exact.sparse_expm import SparseExpmBackend
from ryd_gate.lattice import Register
from ryd_gate.physics import our_laser_rabis

# ---- 'our' 70S constants (rad/s unless noted) -------------------------------
P420_W = 6.4                     # 420 nm laser power (W)
P1013_W = 100.0                  # 1013 nm laser power (W)
OPTICS_PATH_LOSS = 0.10          # 10% loss in each delivery path
N_BEAM_ATOMS = 200               # beam illuminates the full ~200-atom array
BEAM_SHORT_AXIS_UM = 6.0         # fixed short axis of the rectangular top-hat
RYD_LEVEL = 70

C6 = 2 * np.pi * 874e9           # rad/s * um^6, Rb 70S van der Waals
GAMMA_R = 1.0 / 151.55e-6        # Rydberg decay rate (~6598 rad/s)
GAMMA_E = 1.0 / 110.7e-9         # intermediate-state decay rate (~9.03e6 rad/s)
ETA_LOSS = 0.614                 # branching of scattering into atom-loss channels

# np.trapz is deprecated (numpy>=2) in favour of np.trapezoid; use whichever exists.
trapz = getattr(np, "trapezoid", None) or np.trapz

# Per-channel arrays carried through the checkpoint / plots.
CHANNELS = ("eps_coh", "eps_SE", "eps_sc", "eps_loss", "eps_total", "Oeff_MHz")
PLOT_CHANNELS = ("eps_coh", "eps_SE", "eps_sc", "eps_total")
PLOT_LABELS = {
    "eps_coh": "$\\epsilon_{\\rm coh}$ (non-adiab.)\n(n_steps-limited)",
    "eps_SE": r"$\epsilon_{\rm SE}$ (Ryd. decay)",
    "eps_sc": r"$\epsilon_{\rm sc}$ (intermediate scatter)",
    "eps_total": r"$\epsilon_{\rm total}$",
}


# --------------------------------------------------------------------------- #
# physics: spacing-dependent laser Rabis and the schedule
# --------------------------------------------------------------------------- #
def _power_at_atoms(nominal_power_w: float) -> float:
    return nominal_power_w * (1.0 - OPTICS_PATH_LOSS)


@lru_cache(maxsize=None)
def laser_rabis(a_um: float) -> tuple[float, float]:
    """(Omega_420, Omega_1013) in rad/s for the beam over the array at spacing a.

    Top-hat over the array footprint: area = sqrt(N_BEAM_ATOMS) * a_um * BEAM_SHORT_AXIS_UM.
    """
    beam_length_um = float(np.sqrt(N_BEAM_ATOMS) * a_um)
    beam_area_um2 = beam_length_um * BEAM_SHORT_AXIS_UM
    return our_laser_rabis(
        _power_at_atoms(P420_W),
        _power_at_atoms(P1013_W),
        beam_area_um2,
        ryd_level=RYD_LEVEL,
    )


def omega_eff(a_um: float, delta_e: float) -> float:
    """Two-photon effective Rabi Omega_eff = O420 * O1013 / (2 Delta_e) (rad/s)."""
    o420, o1013 = laser_rabis(a_um)
    return o420 * o1013 / (2.0 * delta_e)


# Relative per-eval cost model, used only for a *stable* ETA. The sparse-expm cost
# tracks ||dt*H|| ~ Omega_eff, so cost ~ lat_base[lattice] * sqrt(Omega_eff(a,De)).
# _LAT_BASE: measured s/eval at the grid centre (a=7 um, De=2 GHz; n_steps=80, n_eval=21).
_LAT_BASE = {(2, 2): 0.10, (2, 3): 0.16, (2, 4): 0.32,
             (3, 3): 0.51, (2, 5): 0.86, (3, 4): 3.41}


def _lat_base(lx: int, ly: int) -> float:
    if (lx, ly) in _LAT_BASE:
        return _LAT_BASE[(lx, ly)]
    return 0.10 * (2 ** (lx * ly) / 16.0) ** 0.635    # power-law fallback


def smoothstep5(u):
    """Quintic smoothstep 10u^3 - 15u^4 + 6u^5, clipped to [0, 1] (vectorized)."""
    u = np.clip(u, 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def make_schedule(omega_eff_val, t_sweep, d_amp):
    """Closures (omega_half_fn, delta_fn, env_fn) for one (Omega_eff, t_sweep) point.

    omega_half(t) = 0.5 * Omega_eff * env(t);  delta(t) = -d_amp * cos(2*pi*s);
    env in [0,1] is a quintic ramp up/flat/down (s = t/t_sweep). env_fn is reused
    in the scatter integral so Omega_420(t) = Omega_420 * env(t) matches the pulse.
    """
    ramp_frac = 0.09

    def env_fn(t):
        s = np.clip(np.asarray(t, dtype=float) / t_sweep, 0.0, 1.0)
        rise = smoothstep5(s / ramp_frac)
        fall = smoothstep5((1.0 - s) / ramp_frac)
        return np.where(s < ramp_frac, rise, np.where(s > 1.0 - ramp_frac, fall, 1.0))

    def omega_half_fn(t):
        return float(0.5 * omega_eff_val * env_fn(t))

    def delta_fn(t):
        s = float(np.clip(t / t_sweep, 0.0, 1.0))
        return float(-d_amp * np.cos(2.0 * np.pi * s))

    return omega_half_fn, delta_fn, env_fn


# --------------------------------------------------------------------------- #
# one point -> budget
# --------------------------------------------------------------------------- #
def build_base_system(lx: int, ly: int, a_um: float):
    """A protocol-less 1r/nn RydbergSystem; rebind schedules with with_protocol()."""
    geom = Register.rectangle(lx, ly, spacing_um=a_um)
    return rg.RydbergSystem.from_lattice(
        geom, "1r", interaction=InteractionSpec(C6=C6, mode="nn")
    )


def evaluate(base_system, a_um, delta_e, t_sweep, *, d_amp, n_steps, n_eval):
    """forward + budget at one (a, Delta_e, t_sweep) point, reusing base_system."""
    o420, o1013 = laser_rabis(a_um)
    oeff = omega_eff(a_um, delta_e)
    omega_half_fn, delta_fn, env_fn = make_schedule(oeff, t_sweep, d_amp)

    proto = rg.SweepProtocol(
        t_gate=t_sweep, omega_half_fn=omega_half_fn, delta_fn=delta_fn, n_steps=n_steps
    )
    system = base_system.with_protocol(proto)
    N = system.N
    t_eval = np.linspace(0.0, t_sweep, n_eval)
    res = exact_simulate(
        system, [], "all_ground",
        backend=SparseExpmBackend(n_steps=n_steps), t_eval=t_eval,
    )
    n_r = np.asarray(
        [[system.expectation(f"n_r_{i}", psi) for i in range(N)] for psi in res.states]
    )                                # [n_eval, N]
    n_1 = 1.0 - n_r                  # lossless 2-level
    env = np.asarray(env_fn(t_eval), dtype=float)

    eps_coh = float(np.mean(n_r[-1, :]))
    p_r_i = GAMMA_R * trapz(n_r, t_eval, axis=0)
    leg_420 = (4.0 / 3.0) * (o420 * env[:, None] / (2.0 * delta_e)) ** 2 * n_1
    leg_1013 = (o1013 / (2.0 * delta_e)) ** 2 * n_r
    p_sc_i = GAMMA_E * trapz(leg_420 + leg_1013, t_eval, axis=0)
    eps_se = float(np.mean(p_r_i))
    eps_sc = float(np.mean(p_sc_i))
    return {
        "eps_coh": eps_coh,
        "eps_SE": eps_se,
        "eps_sc": eps_sc,
        "eps_loss": ETA_LOSS * eps_sc,
        "eps_total": eps_coh + eps_se + eps_sc,
        "Oeff_MHz": float(oeff / (2 * np.pi) / 1e6),
    }


# --------------------------------------------------------------------------- #
# checkpoint helpers
# --------------------------------------------------------------------------- #
def _fmt_dt(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "--"
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _axes(args):
    lat = np.asarray(args.lattices, dtype=int)                       # (n_lat, 2)
    spc = np.asarray(args.spacings, dtype=float)                     # (n_a,)
    de = np.linspace(args.de_min, args.de_max, args.res)             # GHz
    ts = np.linspace(args.t_min_us, args.t_max_us, args.res)         # us
    return lat, spc, de, ts


def _empty_grids(shape):
    return {k: np.full(shape, np.nan, dtype=float) for k in CHANNELS}


def _load_checkpoint(path, lat, spc, de, ts):
    """Return grids dict resumed from `path` if axes match, else fresh NaN grids."""
    shape = (len(lat), len(spc), len(de), len(ts))
    if not path.exists():
        return _empty_grids(shape)
    z = np.load(path, allow_pickle=False)
    same = (
        z["lattices"].shape == lat.shape and np.array_equal(z["lattices"], lat)
        and np.allclose(z["spacings_um"], spc)
        and z["delta_e_ghz"].shape == de.shape and np.allclose(z["delta_e_ghz"], de)
        and z["t_sweep_us"].shape == ts.shape and np.allclose(z["t_sweep_us"], ts)
    )
    if not same:
        raise SystemExit(
            f"--resume: axes in {path} differ from the requested grid. "
            "Use a new --out or matching --lattices/--spacings/--res/ranges."
        )
    grids = _empty_grids(shape)
    for k in CHANNELS:
        if k in z.files:
            grids[k][...] = z[k]
    return grids


def _save_checkpoint(path, lat, spc, de, ts, grids, meta):
    tmp = path.with_name(path.name + ".tmp")
    # pass a file handle so np.savez does not append a second ".npz" to the name
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            lattices=lat, spacings_um=spc, delta_e_ghz=de, t_sweep_us=ts,
            meta=json.dumps(meta), **grids,
        )
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #
def run_sweep(args):
    lat, spc, de, ts = _axes(args)
    d_amp = 2 * np.pi * args.d_amp_mhz * 1e6
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = out_path.with_suffix(".status.json")

    grids = (_load_checkpoint(out_path, lat, spc, de, ts)
             if args.resume else _empty_grids((len(lat), len(spc), len(de), len(ts))))

    total = len(lat) * len(spc) * len(de) * len(ts)
    done0 = int(np.isfinite(grids["eps_total"]).sum())
    meta = {
        "mode": "sweep",
        "P420_W": P420_W, "P1013_W": P1013_W, "N_BEAM_ATOMS": N_BEAM_ATOMS,
        "d_amp_mhz": args.d_amp_mhz, "n_steps": args.n_steps, "n_eval": args.n_eval,
        "de_ghz": [args.de_min, args.de_max], "t_sweep_us": [args.t_min_us, args.t_max_us],
        "res": args.res, "total": total,
    }
    print(f"sweep: {len(lat)} lattices x {len(spc)} spacings x {args.res} x {args.res} "
          f"= {total} points ({done0} already done)", flush=True)

    # cost model for a stable ETA: per-eval ~ lat_base[lattice] * sqrt(Omega_eff(a,De)),
    # self-calibrated to live wall-time (absorbs the absolute s/eval). cost_f ~ O(1).
    de_rad = 2 * np.pi * de * 1e9
    oeff_grid = np.array([[omega_eff(float(a), float(d)) for d in de_rad] for a in spc])
    cost_f = np.sqrt(oeff_grid / oeff_grid.mean())                 # (n_a, n_de)
    lat_base = np.array([_lat_base(int(lat[i, 0]), int(lat[i, 1])) for i in range(len(lat))])
    est_total = len(ts) * float(lat_base.sum()) * float(cost_f.sum())
    finite_cnt = np.isfinite(grids["eps_total"]).sum(axis=3)       # (n_lat, n_a, n_de)
    est_done0 = float((lat_base[:, None, None] * cost_f[None, :, :] * finite_cnt).sum())

    t0 = time.time()
    done_proc = 0                                     # points done in THIS process (for rate)
    est_proc = 0.0                                    # probe-seconds of work done this process
    for li in range(len(lat)):
        lx, ly = int(lat[li, 0]), int(lat[li, 1])
        for ai in range(len(spc)):
            a_um = float(spc[ai])
            base = None                               # build lazily, only if work remains
            for di in range(len(de)):
                for ti in range(len(ts)):
                    if np.isfinite(grids["eps_total"][li, ai, di, ti]):
                        continue
                    if base is None:
                        base = build_base_system(lx, ly, a_um)
                    try:
                        b = evaluate(
                            base, a_um, 2 * np.pi * float(de[di]) * 1e9, float(ts[ti]) * 1e-6,
                            d_amp=d_amp, n_steps=args.n_steps, n_eval=args.n_eval,
                        )
                        for k in CHANNELS:
                            grids[k][li, ai, di, ti] = b[k]
                    except Exception as exc:           # leave NaN; retried on --resume
                        print(f"  ! {lx}x{ly} a={a_um} De={de[di]:.3f} t={ts[ti]:.3f}us: "
                              f"{exc!r}"[:160], flush=True)
                    done_proc += 1
                    est_proc += lat_base[li] * cost_f[ai, di]

                # --- per-row progress + checkpoint cadence ---
                done_tot = done0 + done_proc
                elapsed = time.time() - t0
                rate = done_proc / elapsed if done_proc else 0.0
                cps = est_proc / elapsed if elapsed > 0 else 0.0   # probe-s per wall-s (calibration)
                eta = (est_total - est_done0 - est_proc) / cps if cps > 0 else float("inf")
                if done_proc % args.ckpt_every < len(ts):
                    _save_checkpoint(out_path, lat, spc, de, ts, grids, meta)
                status = {
                    **meta, "done": done_tot, "frac": done_tot / total,
                    "elapsed_s": round(elapsed, 1), "eta_s": round(eta, 1),
                    "eta_human": _fmt_dt(eta), "rate_per_s": round(rate, 3),
                    "current": {"lattice": f"{lx}x{ly}", "a_um": a_um,
                                "de_ghz": round(float(de[di]), 3)},
                }
                status_path.write_text(json.dumps(status, indent=1))
                print(f"[{done_tot}/{total} {100*done_tot/total:5.1f}%] {lx}x{ly} "
                      f"a={a_um:.0f} De={de[di]:.2f}GHz | {rate:5.2f}/s "
                      f"elapsed {_fmt_dt(elapsed)} ETA {_fmt_dt(eta)}", flush=True)

            # panel finished -> always checkpoint
            _save_checkpoint(out_path, lat, spc, de, ts, grids, meta)

    _save_checkpoint(out_path, lat, spc, de, ts, grids, meta)
    print(f"DONE {done0 + done_proc}/{total} -> {out_path}", flush=True)


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #
def run_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    in_path = Path(args.out)
    if not in_path.exists():
        raise SystemExit(f"no checkpoint at {in_path}; run --mode sweep first.")
    z = np.load(in_path, allow_pickle=False)
    lat, spc = z["lattices"], z["spacings_um"]
    de, ts = z["delta_e_ghz"], z["t_sweep_us"]
    grids = {k: z[k] for k in PLOT_CHANNELS}
    extent = [de[0], de[-1], ts[0], ts[-1]]           # x = Delta_e, y = t_sweep
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # shared per-channel colour scale across all lattices/spacings (comparable panels)
    norms = {}
    for k in PLOT_CHANNELS:
        v = grids[k][np.isfinite(grids[k]) & (grids[k] > 0)]
        if v.size:
            vmax = float(np.nanpercentile(v, 99.5))
            vmin = max(float(np.nanpercentile(v, 1.0)), vmax * 1e-4)
            norms[k] = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norms[k] = None

    n_a, n_ch = len(spc), len(PLOT_CHANNELS)
    written = []
    for li in range(len(lat)):
        lx, ly = int(lat[li, 0]), int(lat[li, 1])
        fig, axes = plt.subplots(
            n_a, n_ch, figsize=(3.1 * n_ch, 2.5 * n_a),
            squeeze=False, constrained_layout=True,
        )
        col_im = [None] * n_ch
        for ai in range(n_a):
            for ci, k in enumerate(PLOT_CHANNELS):
                ax = axes[ai][ci]
                data = grids[k][li, ai].T          # -> (t, De) for imshow
                col_im[ci] = ax.imshow(
                    np.ma.masked_invalid(data), origin="lower", aspect="auto",
                    extent=extent, cmap="viridis", norm=norms[k],
                )
                if ai == 0:
                    ax.set_title(PLOT_LABELS[k], fontsize=9)
                if ci == 0:
                    ax.set_ylabel(f"a={spc[ai]:.0f} um\n" + r"$t_{\rm sweep}$ ($\mu$s)",
                                  fontsize=8)
                if ai == n_a - 1:
                    ax.set_xlabel(r"$\Delta_e$ (GHz)", fontsize=8)
                ax.tick_params(labelsize=7)
        for ci in range(n_ch):                      # one shared colourbar per column
            if col_im[ci] is not None:
                fig.colorbar(col_im[ci], ax=[axes[ai][ci] for ai in range(n_a)],
                             location="right", shrink=0.92)
        done = int(np.isfinite(grids["eps_total"][li]).sum())
        tot = grids["eps_total"][li].size
        fig.suptitle(f"1r round-trip error budget  --  lattice {lx}x{ly} "
                     f"({lx*ly} atoms)   [{done}/{tot} points]   "
                     "(colour shared per column across spacings)", fontsize=11)
        out_png = fig_dir / f"ebudget_{lx}x{ly}.png"
        fig.savefig(out_png, dpi=args.dpi)
        plt.close(fig)
        written.append(str(out_png))
        print(f"wrote {out_png}  ({done}/{tot} points)", flush=True)
    print("figures:", *written, sep="\n  ")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _lattice(s: str) -> tuple[int, int]:
    lx, ly = s.lower().split("x")
    return int(lx), int(ly)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["sweep", "plot"], default="sweep")

    # axes (defaults: 6 feasible lattices cheap-first, 6 spacings, 100x100 maps)
    p.add_argument("--lattices", type=_lattice, nargs="+",
                   default=[(2, 2), (2, 3), (2, 4), (3, 3), (2, 5), (3, 4)],
                   help="lattice sizes 'LxXLy' (cheap first; 4x4 excluded -- "
                        "exact backend would take months).")
    p.add_argument("--spacings", type=float, nargs="+", default=[5, 6, 7, 8, 9, 10],
                   help="atom spacings (um); one colormap row each.")
    p.add_argument("--res", type=int, default=100, help="points per swept axis (De and t_sweep).")
    p.add_argument("--de-min", type=float, default=1.0, help="Delta_e axis min (GHz).")
    p.add_argument("--de-max", type=float, default=3.0, help="Delta_e axis max (GHz).")
    p.add_argument("--t-min-us", type=float, default=0.3, help="t_sweep axis min (us).")
    p.add_argument("--t-max-us", type=float, default=3.0, help="t_sweep axis max (us).")

    # fixed physics / propagation
    p.add_argument("--d-amp-mhz", type=float, default=20.0,
                   help="detuning-sweep half-amplitude (MHz), FIXED hardware cap.")
    p.add_argument("--n-steps", type=int, default=80,
                   help="sparse piecewise steps (eps_sc/eps_SE converged; eps_coh is "
                        "discretization-limited -- raise for a trustworthy eps_coh).")
    p.add_argument("--n-eval", type=int, default=21, help="trajectory samples for the trapz integrals.")

    # housekeeping
    p.add_argument("--out", type=str, default="data/ebudget_map.npz",
                   help="checkpoint .npz path (also <out>.status.json for live progress).")
    p.add_argument("--ckpt-every", type=int, default=500, help="checkpoint cadence (points).")
    p.add_argument("--resume", action="store_true", help="resume, skipping finished points.")
    p.add_argument("--fig-dir", type=str, default="figs", help="[plot] output directory.")
    p.add_argument("--dpi", type=int, default=130, help="[plot] figure DPI.")
    return p


def main():
    args = build_parser().parse_args()
    if args.mode == "plot":
        run_plot(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
