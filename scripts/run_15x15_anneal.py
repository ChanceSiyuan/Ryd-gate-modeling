"""Run a TFIM/Rydberg anneal through the unified TN backend API.

Default arguments run a small 4x4 smoke test.  Use ``--production-15x15`` or
explicit ``--Lx 15 --Ly 15`` for the large lattice discussed in ``main.tex``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ryd_gate.protocols.lattice_dynamics import TFIMAnnealProtocol
from tn_common import create_tn_lattice_spec, simulate_tn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a square-lattice g-r/TFIM anneal through a selected backend.",
    )
    parser.add_argument("--Lx", type=int, default=4, help="lattice rows")
    parser.add_argument("--Ly", type=int, default=4, help="lattice columns")
    parser.add_argument(
        "--production-15x15",
        action="store_true",
        help="override Lx/Ly to 15x15",
    )
    parser.add_argument(
        "--backend",
        default="tenpy",
        choices=("tenpy", "mps", "itensors", "gputn", "2dtn", "ttn", "nqs"),
        help="simulation backend",
    )
    parser.add_argument(
        "--engine-package",
        default=None,
        help="optional Python package target, e.g. yastn/quimb/pytreenet/netket/jvmc",
    )
    parser.add_argument("--V-nn", type=float, default=24.0, help="nearest-neighbor interaction strength")
    parser.add_argument(
        "--interaction-mode",
        default="nn",
        choices=("nn", "nnn"),
        help="interaction graph for paper TFIM runs; use nn for main.tex and nnn for Rydberg tails",
    )
    parser.add_argument("--hx-peak", type=float, default=1.0, help="peak TFIM transverse field")
    parser.add_argument("--hx-initial", type=float, default=0.0, help="initial TFIM transverse field")
    parser.add_argument("--hx-final", type=float, default=0.0, help="final TFIM transverse field")
    parser.add_argument("--hz-initial", type=float, default=4.0, help="initial TFIM longitudinal field")
    parser.add_argument("--hz-final", type=float, default=-4.0, help="final TFIM longitudinal field")
    parser.add_argument("--t-rise", type=float, default=2.0, help="transverse-field ramp duration")
    parser.add_argument("--t-sweep", type=float, default=8.0, help="longitudinal-field sweep duration")
    parser.add_argument("--t-fall", type=float, default=2.0, help="transverse-field fall duration")
    parser.add_argument("--t-eval-steps", type=int, default=13, help="number of recorded time points")
    parser.add_argument(
        "--observables",
        default="sigma_z,czz_centerline",
        help="comma-separated observable names",
    )
    parser.add_argument("--chi-max", type=int, default=256, help="MPS/TN bond-dimension cap")
    parser.add_argument("--chi-2d", type=int, default=None, help="2D-TN simple-update bond-dimension cap")
    parser.add_argument("--chi-2d-prime", type=int, default=None, help="2D-TN measurement pre-truncation cap")
    parser.add_argument("--dt", type=float, default=0.1, help="time step for TDVP-style backends")
    parser.add_argument("--svd-min", type=float, default=1e-10, help="singular-value cutoff")
    parser.add_argument(
        "--measurement-alg",
        default="bp",
        choices=("bp", "boundarymps", "exact"),
        help="2D-TN observable contraction algorithm",
    )
    parser.add_argument(
        "--measurement-bond-dim",
        type=int,
        default=32,
        help="BoundaryMPS bond dimension for --measurement-alg boundarymps",
    )
    parser.add_argument(
        "--normalize-tensors",
        action="store_true",
        help="request tensor normalization inside the TNQS simple-update kernel",
    )
    parser.add_argument("--samples", type=int, default=4096, help="NQS sample count metadata")
    parser.add_argument("--seed", type=int, default=0, help="random seed metadata for stochastic backends")
    parser.add_argument(
        "--rotated-basis",
        action="store_true",
        help="request rotated-basis NQS metadata",
    )
    parser.add_argument("--cnn-channels", type=int, default=16, help="NQS CNN channel metadata")
    parser.add_argument("--julia-cmd", default="julia", help="Julia executable for backend=itensors/2dtn")
    parser.add_argument("--julia-project", default=None, help="Julia project dir for backend=itensors/2dtn")
    parser.add_argument("--julia-script", default=None, help="Julia kernel script for backend=itensors/2dtn")
    parser.add_argument("--julia-timeout", type=float, default=None, help="subprocess timeout for backend=itensors/2dtn")
    parser.add_argument(
        "--julia-no-bashrc",
        action="store_true",
        help="do not fallback to sourcing ~/.bashrc when julia is not on PATH",
    )
    parser.add_argument("--julia-threads", type=int, default=None, help="JULIA_NUM_THREADS for backend=itensors/2dtn")
    parser.add_argument("--use-cuda", action="store_true", help="request CUDA metadata/checks in compatible backends")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional .npz output for recorded times, observables, and metadata",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the compiled run configuration without invoking the backend",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.production_15x15:
        args.Lx = 15
        args.Ly = 15

    spec = create_tn_lattice_spec(
        args.Lx,
        args.Ly,
        V_nn=args.V_nn,
        Omega=2.0 * args.hx_peak,
        level_structure="1r",
        interaction_mode=args.interaction_mode,
        ordering="snake",
    )
    if args.backend in {"tenpy", "mps", "itensors", "gputn"} and spec.N < 2:
        raise ValueError("TN TDVP backends require at least two lattice sites; use --Lx/--Ly with Lx*Ly >= 2.")

    protocol = TFIMAnnealProtocol(
        hx_peak=args.hx_peak,
        hx_initial=args.hx_initial,
        hx_final=args.hx_final,
        hz_initial=args.hz_initial,
        hz_final=args.hz_final,
        t_rise=args.t_rise,
        t_sweep=args.t_sweep,
        t_fall=args.t_fall,
    )
    t_eval = np.linspace(0.0, protocol.t_gate, args.t_eval_steps)
    observables = _parse_observables(args.observables)
    backend_options = _backend_options(args)

    config = {
        "Lx": spec.Lx,
        "Ly": spec.Ly,
        "N": spec.N,
        "backend": args.backend,
        "interaction_mode": args.interaction_mode,
        "backend_options": _jsonable_options(backend_options),
        "protocol": {
            "hx_peak": args.hx_peak,
            "hz_initial": args.hz_initial,
            "hz_final": args.hz_final,
            "t_gate": protocol.t_gate,
        },
        "observables": observables,
        "t_eval_steps": int(len(t_eval)),
    }
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    result = simulate_tn(
        spec,
        protocol,
        [],
        initial_state="all_ground",
        backend=args.backend,
        t_eval=t_eval,
        observables=observables,
        backend_options=backend_options,
    )
    if args.output is not None:
        _save_result(args.output, result)

    summary = {
        **config,
        "result_backend": result.metadata.get("backend"),
        "result_method": result.metadata.get("method"),
        "recorded_times": None if result.times is None else int(len(result.times)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _backend_options(args: argparse.Namespace) -> dict:
    options: dict[str, object] = {
        "chi_max": args.chi_max,
        "dt": args.dt,
        "svd_min": args.svd_min,
    }
    if args.chi_2d is not None:
        options["chi_2d"] = args.chi_2d
    if args.chi_2d_prime is not None:
        options["chi_2d_prime"] = args.chi_2d_prime
    if args.engine_package:
        options["engine_package"] = args.engine_package
    if args.backend in {"itensors", "2dtn"}:
        options.update(
            {
                "julia_cmd": args.julia_cmd,
                "source_bashrc": not args.julia_no_bashrc,
                "use_cuda": bool(args.use_cuda),
            }
        )
        if args.julia_project:
            options["project_dir"] = args.julia_project
        if args.julia_script:
            options["script_path"] = args.julia_script
        if args.julia_timeout is not None:
            options["timeout"] = args.julia_timeout
        if args.julia_threads is not None:
            options["threads"] = args.julia_threads
    if args.backend == "2dtn":
        options.update(
            {
                "measurement_alg": args.measurement_alg,
                "measurement_bond_dim": args.measurement_bond_dim,
                "normalize_tensors": bool(args.normalize_tensors),
            }
        )
    if args.backend == "nqs":
        options.update(
            {
                "samples": args.samples,
                "seed": args.seed,
                "rotated_basis": bool(args.rotated_basis),
                "cnn_channels": args.cnn_channels,
            }
        )
    return options


def _parse_observables(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _jsonable_options(options: dict) -> dict:
    return {
        key: value
        for key, value in options.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def _save_result(path: Path, result) -> None:
    arrays = {}
    if result.times is not None:
        arrays["times"] = np.asarray(result.times)
    for name, value in (result.metadata.get("obs") or {}).items():
        arrays[f"obs_{name}"] = np.asarray(value)
    arrays["metadata_json"] = np.asarray(json.dumps(_metadata_for_json(result.metadata), sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _metadata_for_json(metadata: dict) -> dict:
    jsonable = {}
    for key, value in metadata.items():
        if key == "obs":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            jsonable[key] = value
    return jsonable


if __name__ == "__main__":
    main()
